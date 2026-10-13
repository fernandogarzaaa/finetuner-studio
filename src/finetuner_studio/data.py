"""Dataset loading and normalization.

Every supported format is converted into the conversational `messages` format
([{"role": ..., "content": ...}]), which TRL's SFTTrainer renders with the
model's chat template. Evaluation splits each conversation into a prompt (all
turns before the final assistant turn) and a reference (that final turn).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .spec import DatasetFormat, DatasetSpec

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset

Message = dict[str, str]


class DataError(ValueError):
    pass


def _col(spec: DatasetSpec, canonical: str) -> str:
    return spec.columns.get(canonical, canonical)


def to_messages(row: dict[str, Any], spec: DatasetSpec) -> list[Message]:
    """Convert one raw row into a list of chat messages."""
    fmt = spec.format
    system = row.get(_col(spec, "system")) or spec.system_prompt
    msgs: list[Message] = []

    if fmt is DatasetFormat.messages:
        raw = row.get(_col(spec, "messages"))
        if not isinstance(raw, list) or not raw:
            raise DataError("row has no 'messages' list")
        for m in raw:
            role, content = m.get("role"), m.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise DataError(f"bad message: {m!r}")
            msgs.append({"role": role, "content": content})
        if system and msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": system})
    elif fmt is DatasetFormat.alpaca:
        instruction = row.get(_col(spec, "instruction"))
        extra = row.get(_col(spec, "input")) or ""
        output = row.get(_col(spec, "output"))
        if not instruction or output is None:
            raise DataError("alpaca row needs instruction and output")
        user = f"{instruction}\n\n{extra}".strip() if extra else str(instruction)
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": str(output)}]
    elif fmt is DatasetFormat.prompt_completion:
        prompt = row.get(_col(spec, "prompt"))
        completion = row.get(_col(spec, "completion"))
        if not prompt or completion is None:
            raise DataError("row needs prompt and completion")
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(completion)},
        ]
    else:  # pragma: no cover
        raise DataError(f"unknown format {fmt}")

    if msgs[-1]["role"] != "assistant":
        raise DataError("conversation must end with an assistant turn")
    return msgs


@dataclass
class EvalExample:
    prompt: list[Message]
    reference: str


def split_for_eval(messages: list[Message]) -> EvalExample:
    return EvalExample(prompt=messages[:-1], reference=messages[-1]["content"])


@dataclass
class PreparedData:
    train: Dataset
    eval: Dataset
    skipped_rows: int


def _load_raw(spec: DatasetSpec, split: str) -> Dataset:
    from datasets import load_dataset

    if spec.source == "jsonl":
        return load_dataset("json", data_files=spec.path, split="train")
    return load_dataset(spec.path, spec.config_name, split=split)


def _normalize(ds: Dataset, spec: DatasetSpec) -> tuple[Dataset, int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in ds:
        try:
            rows.append({"messages": to_messages(row, spec)})
        except DataError:
            skipped += 1
    if not rows:
        raise DataError("no usable rows after normalization; check dataset.format and columns")
    from datasets import Dataset as HFDataset

    return HFDataset.from_list(rows), skipped


def prepare(spec: DatasetSpec) -> PreparedData:
    raw = _load_raw(spec, spec.split).shuffle(seed=spec.shuffle_seed)
    if spec.max_samples:
        raw = raw.select(range(min(spec.max_samples, len(raw))))
    data, skipped = _normalize(raw, spec)

    if spec.eval_split and spec.source == "hf":
        eval_raw = _load_raw(spec, spec.eval_split)
        eval_data, eval_skipped = _normalize(eval_raw, spec)
        return PreparedData(train=data, eval=eval_data, skipped_rows=skipped + eval_skipped)

    if len(data) < 2:
        raise DataError("need at least 2 rows to hold out an eval set")
    parts = data.train_test_split(test_size=spec.eval_size, seed=spec.shuffle_seed)
    return PreparedData(train=parts["train"], eval=parts["test"], skipped_rows=skipped)
