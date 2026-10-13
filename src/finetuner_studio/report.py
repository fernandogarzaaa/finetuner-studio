"""Run report: machine-readable JSON plus a Markdown summary for humans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .spec import RunSpec
from .verdict import Verdict

_BADGE = {"SHIP": "✅ SHIP", "DONT_SHIP": "⛔ DON'T SHIP", "INCONCLUSIVE": "⚠️ INCONCLUSIVE"}


def build(spec: RunSpec, verdict: Verdict, base: dict[str, float], tuned: dict[str, float],
          extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "run": spec.name,
        "base_model": spec.model.base_model,
        "dataset": spec.dataset.path,
        "verdict": verdict.to_dict(),
        "metrics": {"base": base, "tuned": tuned},
        "spec": spec.model_dump(mode="json"),
        **(extra or {}),
    }


def to_markdown(report: dict[str, Any]) -> str:
    v = report["verdict"]
    lines = [
        f"# {report['run']}: {_BADGE.get(v['decision'], v['decision'])}",
        "",
        v["summary"],
        "",
        f"- Base model: `{report['base_model']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Eval samples: {v['eval_samples']}",
        "",
        "| Metric | Base | Tuned | Change |",
        "|---|---:|---:|---:|",
    ]
    for d in v["deltas"]:
        arrow = "↑ better" if d["higher_is_better"] else "↓ better"
        lines.append(
            f"| {d['metric']} ({arrow}) | {d['base']:.4f} | {d['tuned']:.4f} | {d['relative']:+.1%} |"
        )
    lines += ["", "## Why", ""] + [f"- {r}" for r in v["reasons"]]
    return "\n".join(lines) + "\n"


def write(run_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    j = run_dir / "report.json"
    m = run_dir / "report.md"
    j.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    m.write_text(to_markdown(report), encoding="utf-8")
    return j, m
