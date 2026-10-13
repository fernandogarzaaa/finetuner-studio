interface Point {
  step: number;
  loss: number;
}

/** Dependency-free SVG line chart for training loss. */
export default function LossChart({ points }: { points: Point[] }) {
  if (points.length < 2) {
    return <div className="chart-empty muted">Loss appears here once training logs its first steps.</div>;
  }
  const W = 640;
  const H = 220;
  const pad = { l: 44, r: 12, t: 12, b: 28 };
  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.loss);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const ySpan = yMax - yMin || 1;
  const x = (v: number) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - yMin) / ySpan) * (H - pad.t - pad.b);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join("");
  const ticks = [yMin, yMin + ySpan / 2, yMax];
  const last = points[points.length - 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Training loss">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="grid" />
          <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="axis">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      <text x={pad.l} y={H - 6} className="axis">
        step {xMin}
      </text>
      <text x={W - pad.r} y={H - 6} textAnchor="end" className="axis">
        step {xMax}
      </text>
      <path d={d} className="line" />
      <circle cx={x(last.step)} cy={y(last.loss)} r={4} className="dot" />
    </svg>
  );
}
