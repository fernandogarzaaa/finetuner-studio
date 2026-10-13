import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError, type ValidateResult } from "../api";

const STARTER = `version: 1
name: my-first-run
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  quantization: 4bit
  max_seq_length: 1024
dataset:
  source: hf
  path: yahma/alpaca-cleaned
  format: alpaca
  max_samples: 2000
  eval_size: 100
training:
  epochs: 1
  learning_rate: 0.0002
ship_rule:
  primary_metric: rouge_l
  min_improvement: 0.03
`;

export default function NewRun() {
  const nav = useNavigate();
  const [yaml, setYaml] = useState(STARTER);
  const [examples, setExamples] = useState<{ name: string; spec_yaml: string }[]>([]);
  const [result, setResult] = useState<ValidateResult | null>(null);
  const [simulate, setSimulate] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const timer = useRef<number>();

  useEffect(() => {
    api.examples().then(setExamples).catch(() => {});
    api
      .hardware()
      .then((h) => !h.cuda && setSimulate(true))
      .catch(() => {});
  }, []);

  useEffect(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      api.validate(yaml).then(setResult).catch(() => setResult(null));
    }, 350);
    return () => window.clearTimeout(timer.current);
  }, [yaml]);

  const submit = async () => {
    setSubmitting(true);
    setError("");
    try {
      const run = await api.createRun(yaml, simulate);
      nav(`/runs/${run.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? JSON.stringify(e.detail) : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="grid-2">
      <div className="card">
        <div className="card-head">
          <h2>Run spec</h2>
          {examples.length > 0 && (
            <select defaultValue="" onChange={(e) => e.target.value && setYaml(e.target.value)}>
              <option value="">Load an example…</option>
              {examples.map((ex) => (
                <option key={ex.name} value={ex.spec_yaml}>
                  {ex.name}
                </option>
              ))}
            </select>
          )}
        </div>
        <textarea
          className="editor"
          spellCheck={false}
          value={yaml}
          onChange={(e) => setYaml(e.target.value)}
        />
      </div>

      <div className="card side">
        <h2>Check</h2>
        {!result && <p className="muted">Validating…</p>}
        {result?.valid && (
          <div className="ok">
            <strong>Spec is valid.</strong>
            <p className="muted">Effective batch size {result.effective_batch_size}.</p>
          </div>
        )}
        {result && !result.valid && (
          <ul className="errors">
            {result.errors.map((e, i) => (
              <li key={i}>
                {e.loc && <code>{e.loc}</code>} {e.msg}
              </li>
            ))}
          </ul>
        )}

        <label className="toggle">
          <input type="checkbox" checked={simulate} onChange={(e) => setSimulate(e.target.checked)} />
          <span>
            Simulate
            <small className="muted">Runs the full flow with synthetic numbers, no GPU or download.</small>
          </span>
        </label>

        {error && <p className="error">{error}</p>}
        <button className="primary wide" disabled={!result?.valid || submitting} onClick={submit}>
          {submitting ? "Queuing…" : "Queue run"}
        </button>
        <p className="muted small">
          The run trains a LoRA adapter, evaluates it against the base model on held-out data, and applies
          your ship rule.
        </p>
      </div>
    </div>
  );
}
