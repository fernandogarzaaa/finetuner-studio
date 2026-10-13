import type { Decision, RunStatus } from "../api";

const DECISION_LABEL: Record<Decision, string> = {
  SHIP: "Ship",
  DONT_SHIP: "Don't ship",
  INCONCLUSIVE: "Inconclusive",
};

export function VerdictBadge({ decision, large }: { decision: Decision | null; large?: boolean }) {
  if (!decision) return <span className="muted">—</span>;
  return (
    <span className={`verdict verdict-${decision.toLowerCase()} ${large ? "large" : ""}`}>
      {DECISION_LABEL[decision]}
    </span>
  );
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

export function ProgressBar({ value }: { value: number | null }) {
  const pct = Math.round(Math.min(1, Math.max(0, value ?? 0)) * 100);
  return (
    <div className="progress" aria-valuenow={pct} role="progressbar">
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}
