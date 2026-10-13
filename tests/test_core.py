from pathlib import Path

import pytest
from pydantic import ValidationError

from finetuner_studio.data import DataError, split_for_eval, to_messages
from finetuner_studio.events import ListSink, StdoutSink, parse_line
from finetuner_studio.metrics import exact_match, rouge_l, score_generations, token_f1
from finetuner_studio.pipeline import run as run_pipeline
from finetuner_studio.spec import DatasetSpec, Metric, RunSpec, ShipRule
from finetuner_studio.verdict import Decision, decide

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "specs"

MINIMAL = """
version: 1
name: t
model: {base_model: sshleifer/tiny-gpt2}
dataset: {path: x/y}
training: {max_steps: 5, logging_steps: 1}
"""


# --- spec -------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.stem)
def test_example_specs_validate(path):
    spec = RunSpec.from_file(path)
    assert spec.name
    RunSpec.from_yaml(spec.to_yaml())  # round-trips


def test_minimal_spec_defaults():
    spec = RunSpec.from_yaml(MINIMAL)
    assert spec.model.quantization == "4bit"
    assert spec.effective_batch_size() == 16


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        RunSpec.from_yaml(MINIMAL + "\nbogus: 1\n")


def test_ship_rule_metric_must_be_evaluated():
    bad = MINIMAL + "evaluation: {metrics: [perplexity]}\n"
    with pytest.raises(ValidationError, match="ship_rule uses metrics"):
        RunSpec.from_yaml(bad)


def test_eval_size_bounds():
    with pytest.raises(ValidationError):
        DatasetSpec(path="a/b", eval_size=1.5)


# --- data -------------------------------------------------------------------

def test_alpaca_to_messages_with_input_and_system():
    spec = DatasetSpec(path="a/b", format="alpaca", system_prompt="be brief")
    msgs = to_messages({"instruction": "Sum", "input": "1 2", "output": "3"}, spec)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "Sum\n\n1 2"


def test_column_mapping():
    spec = DatasetSpec(path="a/b", format="prompt_completion", columns={"prompt": "q", "completion": "a"})
    msgs = to_messages({"q": "hi", "a": "hello"}, spec)
    assert split_for_eval(msgs).reference == "hello"


def test_messages_must_end_with_assistant():
    spec = DatasetSpec(path="a/b")
    with pytest.raises(DataError):
        to_messages({"messages": [{"role": "user", "content": "hi"}]}, spec)


# --- metrics ----------------------------------------------------------------

def test_metrics():
    assert exact_match("The Cat.", "cat") == 1.0
    assert token_f1("a b c", "x y") == 0.0
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert 0 < rouge_l("cat sat mat", "the cat sat on a mat") < 1
    scores = score_generations(["yes"], ["yes"], ["exact_match", "token_f1"])
    assert scores == {"exact_match": 1.0, "token_f1": 1.0}


# --- verdict ----------------------------------------------------------------

RULE = ShipRule(primary_metric=Metric.rouge_l, min_improvement=0.05, min_eval_samples=10)


def test_ship_when_primary_improves_and_guardrail_holds():
    v = decide({"rouge_l": 0.30, "perplexity": 8.0}, {"rouge_l": 0.36, "perplexity": 7.5}, RULE, 50)
    assert v.decision is Decision.SHIP


def test_dont_ship_on_small_gain():
    v = decide({"rouge_l": 0.30, "perplexity": 8.0}, {"rouge_l": 0.305, "perplexity": 7.5}, RULE, 50)
    assert v.decision is Decision.DONT_SHIP
    assert "short of" in v.summary


def test_dont_ship_on_guardrail_regression():
    v = decide({"rouge_l": 0.30, "perplexity": 8.0}, {"rouge_l": 0.40, "perplexity": 10.0}, RULE, 50)
    assert v.decision is Decision.DONT_SHIP
    assert "Guardrail broken" in v.summary


def test_inconclusive_on_tiny_eval():
    v = decide({"rouge_l": 0.3, "perplexity": 8}, {"rouge_l": 0.5, "perplexity": 7}, RULE, 3)
    assert v.decision is Decision.INCONCLUSIVE


# --- events / pipeline ------------------------------------------------------

def test_event_line_roundtrip(capsys):
    from finetuner_studio.events import Event

    StdoutSink().emit(Event("verdict", {"decision": "SHIP"}))
    line = capsys.readouterr().out
    ev = parse_line(line)
    assert ev.type == "verdict" and ev.data["decision"] == "SHIP"
    assert parse_line("plain text").type == "log"


def test_simulated_pipeline(tmp_path):
    sink = ListSink()
    rep = run_pipeline(RunSpec.from_yaml(MINIMAL), tmp_path, sink, simulate=True)
    types = [e.type for e in sink.events]
    assert "train_metrics" in types and "verdict" in types
    assert rep["verdict"]["decision"] == "SHIP"
    assert (tmp_path / "report.md").read_text().startswith("# t:")
