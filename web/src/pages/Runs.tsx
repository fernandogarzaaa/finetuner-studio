import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Run } from "../api";
import { ProgressBar, StatusBadge, VerdictBadge } from "../components/Badges";

export function timeAgo(iso: string | null): string {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .runs()
        .then((r) => alive && setRuns(r))
        .catch((e) => alive && setError(String(e.message ?? e)));
    load();
    const t = setInterval(load, 3000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (!runs) return <p className="muted">Loading…</p>;

  if (runs.length === 0) {
    return (
      <div className="card empty">
        <h2>No runs yet</h2>
        <p className="muted">
          Write a spec, pick a base model and dataset, and Fine-Tuner Studio will train, compare against
          the base, and tell you whether to ship.
        </p>
        <Link to="/new" className="button primary">
          Start your first run
        </Link>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-head">
        <h2>Runs</h2>
        <Link to="/new" className="button primary">
          New run
        </Link>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Base model</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Verdict</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>
                <Link to={`/runs/${r.id}`}>{r.name}</Link>
                {r.simulate && <span className="tag">sim</span>}
              </td>
              <td className="mono">{r.base_model}</td>
              <td>
                <StatusBadge status={r.status} />
              </td>
              <td style={{ width: 140 }}>
                <ProgressBar value={r.status === "succeeded" ? 1 : r.progress} />
              </td>
              <td>
                <VerdictBadge decision={r.decision} />
              </td>
              <td className="muted">{timeAgo(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
