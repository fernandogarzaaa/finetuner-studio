import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import { api, ApiError, getToken, setToken } from "./api";
import NewRun from "./pages/NewRun";
import RunDetail from "./pages/RunDetail";
import Runs from "./pages/Runs";

function HardwarePill() {
  const [label, setLabel] = useState("…");
  useEffect(() => {
    api
      .hardware()
      .then((h) => {
        if (h.force_simulate) setLabel("Simulation mode");
        else if (h.cuda) setLabel(h.gpus.map((g) => `${g.name} · ${g.total_gb} GB`).join(", "));
        else setLabel("No GPU detected");
      })
      .catch(() => setLabel("offline"));
  }, []);
  return <span className="pill">{label}</span>;
}

function TokenGate({ children }: { children: React.ReactNode }) {
  const [needsToken, setNeedsToken] = useState<boolean | null>(null);
  const [value, setValue] = useState(getToken());
  const [error, setError] = useState("");

  const check = () =>
    api
      .runs()
      .then(() => setNeedsToken(false))
      .catch((e) => setNeedsToken(e instanceof ApiError && e.status === 401));

  useEffect(() => {
    check();
  }, []);

  if (needsToken === null) return null;
  if (!needsToken) return <>{children}</>;
  return (
    <div className="card narrow">
      <h2>API token required</h2>
      <p className="muted">This server is protected. Paste the value of FTS_API_TOKEN.</p>
      <input
        type="password"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="token"
      />
      {error && <p className="error">{error}</p>}
      <button
        className="primary"
        onClick={async () => {
          setToken(value.trim());
          await check();
          setError("That token was rejected.");
        }}
      >
        Continue
      </button>
    </div>
  );
}

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="logo">◆</span> Fine-Tuner Studio
        </Link>
        <nav>
          <NavLink to="/" end>
            Runs
          </NavLink>
          <NavLink to="/new">New run</NavLink>
        </nav>
        <HardwarePill />
      </header>
      <main>
        <TokenGate>
          <Routes>
            <Route path="/" element={<Runs />} />
            <Route path="/new" element={<NewRun />} />
            <Route path="/runs/:id" element={<RunDetail />} />
          </Routes>
        </TokenGate>
      </main>
    </div>
  );
}
