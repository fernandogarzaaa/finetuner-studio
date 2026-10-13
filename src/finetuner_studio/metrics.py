"""Text metrics with no heavy dependencies, so they are fast and fully testable."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


GENERATION_METRICS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def score_generations(
    preds: Sequence[str], refs: Sequence[str], metric_names: Sequence[str]
) -> dict[str, float]:
    if len(preds) != len(refs):
        raise ValueError("preds and refs differ in length")
    out: dict[str, float] = {}
    for name in metric_names:
        fn = GENERATION_METRICS.get(name)
        if fn is None or not preds:
            continue
        out[name] = sum(fn(p, r) for p, r in zip(preds, refs, strict=True)) / len(preds)
    return out


def perplexity_from_loss(mean_nll: float) -> float:
    return math.exp(min(mean_nll, 50.0))
