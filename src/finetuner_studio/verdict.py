"""The ship / don't-ship decision.

Pure function of (base metrics, tuned metrics, rule). Every decision carries the
reasons that produced it, so the report can explain itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .spec import Metric, ShipRule


class Decision(str, Enum):
    SHIP = "SHIP"
    DONT_SHIP = "DONT_SHIP"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class MetricDelta:
    metric: str
    base: float
    tuned: float
    absolute: float
    relative: float  # signed so that positive always means "better"
    higher_is_better: bool


@dataclass
class Verdict:
    decision: Decision
    summary: str
    reasons: list[str] = field(default_factory=list)
    deltas: list[MetricDelta] = field(default_factory=list)
    eval_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def _delta(metric: Metric, base: float, tuned: float) -> MetricDelta:
    hib = metric.higher_is_better
    absolute = tuned - base
    denom = abs(base) if abs(base) > 1e-12 else 1e-12
    improvement = (absolute if hib else -absolute) / denom
    return MetricDelta(
        metric=metric.value,
        base=base,
        tuned=tuned,
        absolute=absolute,
        relative=improvement,
        higher_is_better=hib,
    )


def decide(
    base: dict[str, float],
    tuned: dict[str, float],
    rule: ShipRule,
    eval_samples: int,
) -> Verdict:
    deltas: dict[str, MetricDelta] = {}
    for name in sorted(set(base) & set(tuned)):
        try:
            m = Metric(name)
        except ValueError:
            continue
        deltas[name] = _delta(m, base[name], tuned[name])

    all_deltas = list(deltas.values())

    if eval_samples < rule.min_eval_samples:
        return Verdict(
            decision=Decision.INCONCLUSIVE,
            summary=f"Only {eval_samples} eval samples; need at least {rule.min_eval_samples}.",
            reasons=["Evaluation set too small to trust the comparison."],
            deltas=all_deltas,
            eval_samples=eval_samples,
        )

    primary = deltas.get(rule.primary_metric.value)
    if primary is None:
        return Verdict(
            decision=Decision.INCONCLUSIVE,
            summary=f"Primary metric {rule.primary_metric.value} was not computed.",
            reasons=["Missing primary metric."],
            deltas=all_deltas,
            eval_samples=eval_samples,
        )

    reasons: list[str] = []
    failures: list[str] = []

    if primary.relative >= rule.min_improvement:
        reasons.append(
            f"{primary.metric} improved {primary.relative:+.1%} "
            f"(needed {rule.min_improvement:+.1%})."
        )
    else:
        msg = (
            f"{primary.metric} changed {primary.relative:+.1%}, short of the "
            f"{rule.min_improvement:+.1%} required."
        )
        failures.append(msg)
        reasons.append(msg)

    for g in rule.guardrails:
        d = deltas.get(g.metric.value)
        if d is None:
            msg = f"Guardrail metric {g.metric.value} was not computed."
            failures.append(msg)
            reasons.append(msg)
            continue
        if -d.relative > g.max_regression:
            msg = (
                f"Guardrail broken: {d.metric} regressed {-d.relative:.1%} "
                f"(limit {g.max_regression:.1%})."
            )
            failures.append(msg)
            reasons.append(msg)
        else:
            reasons.append(f"Guardrail held: {d.metric} {d.relative:+.1%} vs base.")

    if failures:
        decision = Decision.DONT_SHIP
        summary = "Don't ship: " + failures[0]
    else:
        decision = Decision.SHIP
        summary = (
            f"Ship it: {primary.metric} {primary.base:.4g} -> {primary.tuned:.4g} "
            f"({primary.relative:+.1%}) with all guardrails holding."
        )
    return Verdict(
        decision=decision,
        summary=summary,
        reasons=reasons,
        deltas=all_deltas,
        eval_samples=eval_samples,
    )
