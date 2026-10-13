export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type Decision = "SHIP" | "DONT_SHIP" | "INCONCLUSIVE";

export interface Run {
  id: string;
  name: string;
  status: RunStatus;
  base_model: string;
  simulate: boolean;
  stage: string | null;
  progress: number | null;
  decision: Decision | null;
  summary: string | null;
  error: string | null;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  spec_yaml?: string;
}

export interface RunEvent {
  id: number;
  ts: number;
  type: string;
  data: Record<string, any>;
}

export interface FieldError {
  loc: string;
  msg: string;
}

export interface ValidateResult {
  valid: boolean;
  errors: FieldError[];
  effective_batch_size?: number;
}

export interface MetricDelta {
  metric: string;
  base: number;
  tuned: number;
  relative: number;
  higher_is_better: boolean;
}

export interface Report {
  run: string;
  base_model: string;
  dataset: string;
  simulated?: boolean;
  verdict: {
    decision: Decision;
    summary: string;
    reasons: string[];
    deltas: MetricDelta[];
    eval_samples: number;
  };
}

export interface Sample {
  prompt: { role: string; content: string }[];
  reference: string;
  base: string | null;
  tuned: string | null;
}

const TOKEN_KEY = "fts_token";
export const getToken = () => localStorage.getItem(TOKEN_KEY) ?? "";
export const setToken = (t: string) =>
  t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY);

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? body);
  return body as T;
}

export const api = {
  health: () => req<{ status: string; version: string; auth: boolean }>("/api/health"),
  hardware: () =>
    req<{ cuda: boolean; gpus: { name: string; total_gb: number }[]; force_simulate: boolean }>(
      "/api/hardware",
    ),
  examples: () => req<{ name: string; spec_yaml: string }[]>("/api/spec/examples"),
  validate: (spec_yaml: string) =>
    req<ValidateResult>("/api/spec/validate", { method: "POST", body: JSON.stringify({ spec_yaml }) }),
  createRun: (spec_yaml: string, simulate: boolean) =>
    req<Run>("/api/runs", { method: "POST", body: JSON.stringify({ spec_yaml, simulate }) }),
  runs: () => req<Run[]>("/api/runs"),
  run: (id: string) => req<Run>(`/api/runs/${id}`),
  cancel: (id: string) => req<Run>(`/api/runs/${id}/cancel`, { method: "POST" }),
  remove: (id: string) => req<void>(`/api/runs/${id}`, { method: "DELETE" }),
  events: (id: string, after = 0) => req<RunEvent[]>(`/api/runs/${id}/events?after=${after}`),
  report: (id: string) => req<Report>(`/api/runs/${id}/report`),
  samples: (id: string) => req<Sample[]>(`/api/runs/${id}/samples?limit=20`),
};

export function streamUrl(id: string, after: number): string {
  const token = getToken();
  const q = new URLSearchParams({ after: String(after) });
  if (token) q.set("token", token);
  return `/api/runs/${id}/stream?${q}`;
}

export function downloadUrl(id: string, file: string): string {
  const token = getToken();
  return `/api/runs/${id}/${file}${token ? `?token=${encodeURIComponent(token)}` : ""}`;
}
