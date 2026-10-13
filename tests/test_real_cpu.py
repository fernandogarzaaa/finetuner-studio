"""Real (non-simulated) end-to-end run on CPU with a 135M model.

Slow and needs the `train` extra plus network access, so it is opt-in:
    pytest -m slow
"""

from pathlib import Path

import pytest

from finetuner_studio.events import ListSink
from finetuner_studio.pipeline import run as run_pipeline
from finetuner_studio.spec import RunSpec

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "support_sample.jsonl"


@pytest.mark.slow
def test_real_cpu_run(tmp_path):
    pytest.importorskip("trl")
    spec = RunSpec.from_yaml(
        f"""
version: 1
name: cpu-smoke
model: {{base_model: HuggingFaceTB/SmolLM2-135M-Instruct, quantization: none, max_seq_length: 256}}
dataset: {{source: jsonl, path: {DATA}, format: messages, eval_size: 2}}
training: {{max_steps: 3, per_device_batch_size: 1, gradient_accumulation_steps: 1, logging_steps: 1}}
evaluation: {{max_samples: 2, max_new_tokens: 16, batch_size: 2}}
ship_rule: {{min_eval_samples: 1}}
"""
    )
    sink = ListSink()
    report = run_pipeline(spec, tmp_path, sink)
    assert report["verdict"]["decision"] in {"SHIP", "DONT_SHIP", "INCONCLUSIVE"}
    assert set(report["metrics"]["base"]) == {"perplexity", "rouge_l", "token_f1"}
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "eval_samples.jsonl").exists()
    assert any(e.type == "train_metrics" for e in sink.events)
