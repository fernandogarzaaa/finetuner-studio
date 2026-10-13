# Fine-Tuner Studio

[![CI](https://github.com/fernandogarzaaa/finetuner-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/finetuner-studio/actions/workflows/ci.yml) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fernandogarzaaa/finetuner-studio/blob/main/notebooks/colab_quickstart.ipynb) ![License](https://img.shields.io/badge/license-Apache--2.0-blue)

**Spec in. Fine-tuned model and a ship / don't-ship verdict out.**

Fine-Tuner Studio turns LLM fine-tuning into one reviewable artifact: a YAML spec. Submit it from the web UI, the CLI, or the REST API and the platform will:

1. **Load and normalize** your data (Hugging Face dataset or JSONL; chat `messages`, Alpaca, or prompt/completion).
2. **Fine-tune** with LoRA / QLoRA (PEFT + TRL), 4-bit by default, so 1–8B models fit on a single 16 GB GPU.
3. **Evaluate the tuned model against its own base** on a held-out set, with both models sharing one copy of the weights in VRAM.
4. **Decide**: apply your ship rule (primary metric must improve by X%, guardrail metrics may not regress more than Y%) and return `SHIP`, `DONT_SHIP`, or `INCONCLUSIVE`, with the reasons.

Every run is persisted with its spec, a live event stream, metrics, side-by-side eval samples, a Markdown report, and the trained adapter.

## Why

Most fine-tuning tools stop at "training finished". The question a team actually has to answer is *is the tuned model better than what we already have, without breaking anything?* Fine-Tuner Studio makes that question part of every run and makes the answer explicit and auditable.

## Architecture

```
            ┌──────────────┐   REST + SSE    ┌──────────────────────┐
  Browser ──▶  React UI    │ ──────────────▶ │  FastAPI  (api)      │
            └──────────────┘                 │  runs, events, specs │
                                             └──────────┬───────────┘
                                                        │ SQLite / Postgres
                                             ┌──────────▼───────────┐
                                             │  Worker (1 per GPU)  │
                                             │  claims queued runs  │
                                             └──────────┬───────────┘
                                                        │ subprocess per run
                                             ┌──────────▼───────────┐
                                             │  Pipeline            │
                                             │  data → train → eval │
                                             │  → verdict → report  │
                                             └──────────────────────┘
```

- **Process isolation:** each run executes in its own child process, so CUDA memory is fully released afterwards, a crashing run can't take down the worker, and cancel is a clean `SIGTERM` (then `SIGKILL`).
- **Crash recovery:** a worker that restarts marks runs it orphaned as failed instead of leaving them "running" forever.
- **Structured events:** the pipeline emits typed events (`stage`, `train_metrics`, `eval_result`, `verdict`, `artifact`, `error`, `log`) that are stored and streamed to the UI over Server-Sent Events with resume (`Last-Event-ID`).
- **Validation first:** specs are validated (Pydantic) before anything touches the GPU, including cross-field checks such as "the ship rule only uses metrics you actually compute".
- **Auth:** set `FTS_API_TOKEN` to require a bearer token on every API call.

## Quick start

### Local GPU (16 GB)

```bash
pip install -e ".[train]"
cd web && npm ci && npm run build && cd ..
fts serve --host 0.0.0.0 --port 8000      # API + UI + worker in one process
# open http://localhost:8000
```

Split the API and GPU worker for production:

```bash
fts serve --no-worker &
fts worker
```

### Docker (NVIDIA Container Toolkit)

```bash
docker compose up --build
```

### Colab (T4)

The CLI runs the same pipeline in the foreground; see `notebooks/colab_quickstart.ipynb`.

```bash
pip install "finetuner-studio[train] @ git+https://github.com/fernandogarzaaa/finetuner-studio"
fts run examples/specs/qwen2.5-1.5b-alpaca-t4.yaml
```

`fts run` exits `0` for SHIP or INCONCLUSIVE and `3` for DONT_SHIP, so it can gate a CI pipeline.

### No GPU? Simulate

`fts run spec.yaml --simulate` (or the *Simulate* toggle in the UI) runs the whole flow with synthetic numbers and no model download. Useful for developing the UI and integrations.

## The spec

```yaml
version: 1
name: qwen2.5-1.5b-alpaca
model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  quantization: 4bit          # 4bit | 8bit | none
  dtype: auto                 # float16 on T4, bfloat16 on Ampere+
  max_seq_length: 1024
dataset:
  source: hf                  # hf | jsonl
  path: yahma/alpaca-cleaned
  format: alpaca              # messages | alpaca | prompt_completion
  max_samples: 5000
  eval_size: 200              # fraction (<1) or row count
lora: { r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear }
training: { epochs: 1, learning_rate: 0.0002, per_device_batch_size: 2, gradient_accumulation_steps: 8 }
evaluation:
  metrics: [perplexity, rouge_l, token_f1]
  max_samples: 200
ship_rule:
  primary_metric: rouge_l
  min_improvement: 0.03       # +3% relative
  guardrails:
    - { metric: perplexity, max_regression: 0.10 }
  min_eval_samples: 50
```

Full schema: `fts schema` or `GET /api/spec/schema`. More examples in `examples/specs/`.

### Metrics

| Metric | Direction | What it measures |
|---|---|---|
| `perplexity` | lower is better | Likelihood of the reference answer given the prompt (completion tokens only) |
| `rouge_l` | higher is better | Longest-common-subsequence F1 between generation and reference |
| `token_f1` | higher is better | Bag-of-tokens F1 (SQuAD-style normalization) |
| `exact_match` | higher is better | Normalized exact match |

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/spec/validate` | Validate a spec, return errors by field |
| `POST` | `/api/runs` | Queue a run `{spec_yaml, simulate}` |
| `GET` | `/api/runs` | List runs |
| `GET` | `/api/runs/{id}` | Run status, stage, progress, verdict |
| `GET` | `/api/runs/{id}/stream` | Live events (SSE) |
| `GET` | `/api/runs/{id}/events?after=` | Events (polling) |
| `POST` | `/api/runs/{id}/cancel` | Cancel |
| `GET` | `/api/runs/{id}/report` / `report.md` | Report |
| `GET` | `/api/runs/{id}/samples` | Side-by-side base vs tuned generations |
| `GET` | `/api/runs/{id}/adapter.zip` | Trained LoRA adapter |

Interactive docs at `/docs`.

## Configuration

| Env var | Default | |
|---|---|---|
| `FTS_HOME` | `.fts` | Data directory (database, run artifacts) |
| `FTS_DB_URL` | `sqlite:///$FTS_HOME/fts.db` | Any SQLAlchemy URL |
| `FTS_API_TOKEN` | unset | Require `Authorization: Bearer <token>` |
| `FTS_CORS_ORIGINS` | unset | Comma-separated origins for a separately hosted UI |
| `FTS_SIMULATE` | `false` | Force every run to simulate (demo hosts without GPUs) |
| `HF_TOKEN` | unset | For gated models / private datasets |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
cd web && npm run dev     # Vite dev server proxies /api to :8000
```

## Roadmap

- Assistant-only loss masking and sequence packing presets
- LLM-as-judge pairwise metric
- Merge-and-export (GGUF / merged safetensors) and push to Hub
- Multiple GPU workers with device pinning
- Hyperparameter sweeps from one spec

## License

Apache-2.0
