"use client";

import { useMemo } from "react";

const WIDTH = 320;
const HEIGHT = 190;
const PAD_X = 10;
const PAD_TOP = 14;
const PAD_BOTTOM = 24;
const STEEPNESS = 9; // logistic slope of the sorted risk-score curve

const innerW = WIDTH - PAD_X * 2;
const innerH = HEIGHT - PAD_TOP - PAD_BOTTOM;
const toX = (x: number) => PAD_X + x * innerW;
const toY = (y: number) => PAD_TOP + (1 - y) * innerH;

// model risk score for the patient at population fraction x (sorted low -> high risk)
function score(x: number) {
  return 1 / (1 + Math.exp(-STEEPNESS * (x - 0.5)));
}

// mock labelled patients positioned along the sorted population axis
const SAMPLES: { x: number; actual: "pos" | "neg" }[] = [
  { x: 0.13, actual: "neg" },
  { x: 0.24, actual: "neg" },
  { x: 0.33, actual: "neg" },
  { x: 0.41, actual: "pos" },
  { x: 0.47, actual: "neg" },
  { x: 0.54, actual: "pos" },
  { x: 0.59, actual: "neg" },
  { x: 0.64, actual: "pos" },
  { x: 0.71, actual: "pos" },
  { x: 0.79, actual: "neg" },
  { x: 0.86, actual: "pos" },
  { x: 0.93, actual: "pos" }
];

function line(samples: { x: number; y: number }[]) {
  return samples
    .map((s, i) => `${i === 0 ? "M" : "L"} ${toX(s.x).toFixed(1)} ${toY(s.y).toFixed(1)}`)
    .join(" ");
}

export default function PredictionCurve({ threshold }: { threshold: number }) {
  const t = Math.min(0.995, Math.max(0.005, threshold / 100));

  const { curveNeg, curvePos, crossX, posFill, negFill } = useMemo(() => {
    const steps = 80;
    const all: { x: number; y: number }[] = [];
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      all.push({ x, y: score(x) });
    }

    // where the sorted curve crosses the threshold => the classification boundary
    let cross = 0.5 + Math.log(t / (1 - t)) / STEEPNESS;
    cross = Math.min(1, Math.max(0, cross));

    const neg = all.filter((s) => s.x <= cross);
    const pos = all.filter((s) => s.x >= cross);
    // make the two segments meet exactly on the threshold line
    if (neg.length) neg.push({ x: cross, y: t });
    if (pos.length) pos.unshift({ x: cross, y: t });

    // predicted-positive area: between curve and threshold line, right of crossing
    let posArea = "";
    if (pos.length > 1) {
      posArea =
        line(pos) +
        ` L ${toX(1).toFixed(1)} ${toY(t).toFixed(1)} L ${toX(cross).toFixed(1)} ${toY(t).toFixed(1)} Z`;
    }
    // predicted-negative area: between threshold line and curve, left of crossing
    let negArea = "";
    if (neg.length > 1) {
      negArea =
        `M ${toX(0).toFixed(1)} ${toY(t).toFixed(1)} L ${toX(cross).toFixed(1)} ${toY(t).toFixed(1)} ` +
        neg
          .slice()
          .reverse()
          .map((s) => `L ${toX(s.x).toFixed(1)} ${toY(s.y).toFixed(1)}`)
          .join(" ") +
        " Z";
    }

    return {
      curveNeg: line(neg),
      curvePos: line(pos),
      crossX: toX(cross),
      posFill: posArea,
      negFill: negArea
    };
  }, [t]);

  // false positives / negatives shift as the threshold (and crossing point) moves
  const { fps, fns } = useMemo(() => {
    let cross = 0.5 + Math.log(t / (1 - t)) / STEEPNESS;
    cross = Math.min(1, Math.max(0, cross));
    const fp: { cx: number; cy: number }[] = [];
    const fn: { cx: number; cy: number }[] = [];
    for (const s of SAMPLES) {
      const predictedPos = s.x >= cross;
      const cx = toX(s.x);
      const cy = toY(score(s.x));
      if (predictedPos && s.actual === "neg") fp.push({ cx, cy });
      else if (!predictedPos && s.actual === "pos") fn.push({ cx, cy });
    }
    return { fps: fp, fns: fn };
  }, [t]);

  const yT = toY(t);

  return (
    <div className="curve-wrap">
      <svg
        className="curve-svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Sorted model risk score with current decision threshold"
      >
        <defs>
          <linearGradient id="posFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(82, 211, 156, 0.45)" />
            <stop offset="100%" stopColor="rgba(82, 211, 156, 0.06)" />
          </linearGradient>
          <linearGradient id="negFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(105, 150, 200, 0.06)" />
            <stop offset="100%" stopColor="rgba(105, 150, 200, 0.32)" />
          </linearGradient>
        </defs>

        {/* classification regions split by the threshold line */}
        <path className="curve-shade-neg" d={negFill} fill="url(#negFill)" />
        <path className="curve-shade-pos" d={posFill} fill="url(#posFill)" />

        {/* the single sorted risk-score curve, recolored by side of threshold */}
        <path className="curve-line-neg" d={curveNeg} fill="none" stroke="rgba(120, 160, 210, 0.85)" strokeWidth={2} />
        <path className="curve-line-pos" d={curvePos} fill="none" stroke="rgba(82, 211, 156, 0.95)" strokeWidth={2.4} />

        {/* horizontal decision threshold */}
        <line
          className="curve-threshold"
          x1={PAD_X}
          y1={yT}
          x2={WIDTH - PAD_X}
          y2={yT}
          stroke="var(--accent)"
          strokeWidth={2}
          strokeDasharray="4 3"
        />
        {/* boundary marker where the curve crosses the threshold */}
        <line
          className="curve-cross"
          x1={crossX}
          y1={PAD_TOP - 2}
          x2={crossX}
          y2={HEIGHT - PAD_BOTTOM}
          stroke="rgba(255,255,255,0.18)"
          strokeWidth={1}
          strokeDasharray="2 3"
        />
        <circle cx={crossX} cy={yT} r={3.5} fill="var(--accent)" />

        {/* false positives: actual negative, predicted positive */}
        {fps.map((m, i) => (
          <g key={`fp-${i}`} className="curve-marker fp" transform={`translate(${m.cx.toFixed(1)} ${m.cy.toFixed(1)})`}>
            <line x1={-3} y1={-3} x2={3} y2={3} stroke="rgba(255, 110, 110, 0.95)" strokeWidth={1.8} strokeLinecap="round" />
            <line x1={-3} y1={3} x2={3} y2={-3} stroke="rgba(255, 110, 110, 0.95)" strokeWidth={1.8} strokeLinecap="round" />
          </g>
        ))}
        {/* false negatives: actual positive, predicted negative */}
        {fns.map((m, i) => (
          <circle
            key={`fn-${i}`}
            className="curve-marker fn"
            cx={m.cx.toFixed(1)}
            cy={m.cy.toFixed(1)}
            r={3.4}
            fill="none"
            stroke="rgba(167, 139, 250, 0.95)"
            strokeWidth={1.8}
          />
        ))}
      </svg>
      <div className="curve-legend">
        <span><i className="dot pos" /> Predicted positive</span>
        <span><i className="dot neg" /> Predicted negative</span>
        <span><i className="dot fp" /> False positive</span>
        <span><i className="dot fn" /> False negative</span>
      </div>
    </div>
  );
}
