import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  api,
  downloadUrl,
  streamUrl,
  type Report,
  type Run,
  type RunEvent,
  type Sample,
} from "../api";
import { ProgressBar, StatusBadge, VerdictBadge } from "../components/Badges";
import LossChart from "../components/LossChart";

const STAGES = [
  ["load_data", "Data"],
  ["train", "Train"],
  ["evaluate", "Evaluate"],
  ["report", "Verdict"],
] as const;

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

function useRunEvents(id: string, onTerminal: () => void) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const lastId = useRef(0);

  useEffect(() => {
    let es: EventSource | null = null;
    let closed = false;
    lastId.current = 0;
    setEvents([]);

    const connect = () => {
      es = new EventSource(streamUrl(id, lastId.current));
      es.addEventListener("run_event", (msg) => {
        const ev = JSON.parse((msg as MessageEvent).data) as RunEvent;
        if (ev.id <= lastId.current) return;
        lastId.current = ev.id;
        setEvents((prev) => [...prev, ev]);
      });
      es.addEventListener("run_status", () => {
        es?.close();
        onTerminal();
      });
      es.onerror = () => {
        es?.close();
        if (!closed) window.setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      es?.close();
    };
  }, [id]);

  return events;
}

export default function RunDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const [run, setRun] = useState<Run | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [tab, setTab] = useState<"overview" | "samples" | "logs" | "spec">("overview");

  const refresh = () => api.run(id).then(setRun).catch(() => {});
  const events = useRunEvents(id, refresh);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [id]);

  useEffect(() => {
    if (run?.status === "succeeded") {
      api.report(id).then(setReport).catch(() => {});
      api.samples(id).then(setSamples).catch(() => setSamples([]));
    }
  }, [id, run?.status]);

  const loss = useMemo(
    () =>
      events
        .filter((e) => e.type === "train_metrics" && typeof e.data.loss === "number")
        .map((e) => ({ step: e.data.step as number, loss: e.data.loss as number })),
    [events],
  );
  const stageState = useMemo(() => {
    const s: Record<string, string> = {};
    for (const e of events) if (e.type === "stage") s[e.data.stage] = e.data.status;
    return s;
  }, [events]);
  const logs = events.filter((e) => e.type === "log" || e.type === "error");

  if (!run) return <p className="muted">Loading…</p>;
  const live = !TERMINAL.has(run.status);

  return (
    <div className="stack">
      <div className="card">
        <div className="card-head">
          <div>
            <h2>
              {run.name} {run.simulate && <span className="tag">simulated</span>}
            </h2>
            <p className="muted mono">{run.base_model}</p>
          </div>
          <div className="actions">
            <StatusBadge status={run.status} />
            {live && (
              <button onClick={() => api.cancel(id).then(refresh)} disabled={run.cancel_requested}>
                {run.cancel_requested ? "Cancelling…" : "Cancel"}
              </button>
            )}
            {!live && (
              <button
                className="danger"
                onClick={async () => {
                  if (confirm("Delete this run and its artifacts?")) {
                    await api.remove(id);
                    nav("/");
                  }
                }}
              >
                Delete
              </button>
            )}
          </div>
        </div>

        <ol className="stages">
          {STAGES.map(([key, label]) => {
            const st = stageState[key];
            const cls = st === "end" ? "done" : st === "failed" ? "failed" : st === "start" ? "active" : "";
            return (
              <li key={key} className={cls}>
                {label}
              </li>
            );
          })}
        </ol>
        <ProgressBar value={run.status === "succeeded" ? 1 : run.progress} />
        {run.error && <p className="error">{run.error}</p>}
      </div>

      {run.decision && (
        <div className={`card verdict-card verdict-${run.decision.toLowerCase()}`}>
          <VerdictBadge decision={run.decision} large />
          <p className="summary">{run.summary}</p>
          {report && (
            <>
              <table className="table compact">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Base</th>
                    <th>Tuned</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {report.verdict.deltas.map((d) => (
                    <tr key={d.metric}>
                      <td>
                        {d.metric} <span className="muted">{d.higher_is_better ? "↑" : "↓"}</span>
                      </td>
                      <td className="mono">{d.base.toFixed(4)}</td>
                      <td className="mono">{d.tuned.toFixed(4)}</td>
                      <td className={`mono ${d.relative >= 0 ? "pos" : "neg"}`}>
                        {(d.relative * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <ul className="reasons">
                {report.verdict.reasons.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
              <div className="actions">
                <a className="button" href={downloadUrl(id, "report.md")} target="_blank" rel="noreferrer">
                  Report
                </a>
                {!run.simulate && (
                  <a className="button" href={downloadUrl(id, "adapter.zip")}>
                    Download adapter
                  </a>
                )}
              </div>
            </>
          )}
        </div>
      )}

      <div className="card">
        <div className="tabs">
          {(["overview", "samples", "logs", "spec"] as const).map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>

        {tab === "overview" && <LossChart points={loss} />}

        {tab === "samples" &&
          (samples.length === 0 ? (
            <p className="muted">Side-by-side generations appear after evaluation.</p>
          ) : (
            <div className="samples">
              {samples.map((s, i) => (
                <div className="sample" key={i}>
                  <div className="prompt">{s.prompt[s.prompt.length - 1]?.content}</div>
                  <div className="cols">
                    <div>
                      <h4>Reference</h4>
                      <p>{s.reference}</p>
                    </div>
                    <div>
                      <h4>Base</h4>
                      <p>{s.base ?? "—"}</p>
                    </div>
                    <div>
                      <h4>Tuned</h4>
                      <p>{s.tuned ?? "—"}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ))}

        {tab === "logs" && (
          <pre className="logs">
            {logs.length === 0
              ? "No log lines yet."
              : logs
                  .map((e) =>
                    e.type === "error"
                      ? `ERROR ${e.data.message}\n${e.data.traceback ?? ""}`
                      : String(e.data.message),
                  )
                  .join("\n")}
          </pre>
        )}

        {tab === "spec" && <pre className="logs">{run.spec_yaml}</pre>}
      </div>
    </div>
  );
}
