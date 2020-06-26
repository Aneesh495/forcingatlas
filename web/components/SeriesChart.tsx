"use client";

import type { SeriesRecord } from "@/lib/bundle";

const COLORS: Record<string, string> = {
  "hadcrut5-analysis-global": "#0f6e6b",
  "berkeley-air-rebased": "#1d4e89",
  "pages2k-median": "#5c4d7a",
  "ebm-historical": "#a15c38",
  "ebm-natural": "#6b7c4a",
  "ebm-stabilize": "#8a5a2a",
  "ebm-decline": "#3d6b8a",
  "ssp245-CanESM5": "#8d3b3b",
  "ssp245-MIROC6": "#355c7d",
  "ssp245-MPI-ESM1-2-LR": "#2f6f4e",
  "ssp245-HadGEM3-GC31-LL": "#6b4c7a",
};

interface SeriesChartProps {
  records: SeriesRecord[];
  active: string[];
  year: number;
  onYear: (year: number) => void;
}

export function SeriesChart({ records, active, year, onYear }: SeriesChartProps) {
  const shown = records.filter((record) => active.includes(record.id));
  const domainRecords = shown.some((record) => record.semantics !== "proxy_reconstruction")
    ? shown.filter((record) => record.semantics !== "proxy_reconstruction")
    : shown;
  const years = domainRecords.flatMap((record) => record.years);
  const values = shown.flatMap((record) =>
    record.years.flatMap((item, index) => {
      const value = record.values[index];
      if (value == null) return [];
      if (years.length && (item < Math.min(...years) || item > Math.max(...years))) return [];
      return [value];
    }),
  );
  if (years.length === 0 || values.length === 0) {
    return <p className="p-4 text-sm text-ink/70">Choose a series to draw.</p>;
  }
  const minYear = Math.min(...years);
  const maxYear = Math.max(...years);
  const minValue = Math.min(-0.6, ...values);
  const maxValue = Math.max(1.4, ...values);
  const width = 760;
  const height = 220;
  const pad = 28;

  function x(value: number) {
    return pad + ((value - minYear) / (maxYear - minYear)) * (width - pad * 2);
  }
  function y(value: number) {
    return height - pad - ((value - minValue) / (maxValue - minValue)) * (height - pad * 2);
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full" role="img" aria-label="Global anomaly series">
      <line x1={pad} x2={width - pad} y1={y(0)} y2={y(0)} stroke="#d9d1c3" />
      {shown.map((record) => {
        const points = record.years
          .map((item, index) => {
            const value = record.values[index];
            if (value == null || item < minYear || item > maxYear) return null;
            return `${x(item)},${y(value)}`;
          })
          .filter((point): point is string => point != null)
          .join(" ");
        return (
          <polyline
            key={record.id}
            points={points}
            fill="none"
            stroke={COLORS[record.id] ?? "#1c2430"}
            strokeWidth={record.semantics === "proxy_reconstruction" ? 1.2 : 1.8}
            strokeDasharray={record.semantics === "emulation" ? "4 3" : undefined}
          />
        );
      })}
      <line x1={x(year)} x2={x(year)} y1={pad} y2={height - pad} stroke="#1c2430" strokeWidth={1} />
      <text x={pad} y={16} className="fill-ink text-[11px]">
        {maxValue.toFixed(1)} deg C
      </text>
      <text x={pad} y={height - 8} className="fill-ink text-[11px]">
        {minYear}
      </text>
      <text x={width - pad - 28} y={height - 8} className="fill-ink text-[11px]">
        {maxYear}
      </text>
      {shown[0] && (
        <YearHits minYear={minYear} maxYear={maxYear} x={x} y={y} height={height} pad={pad} onYear={onYear} />
      )}
    </svg>
  );
}

function YearHits({
  minYear,
  maxYear,
  x,
  height,
  pad,
  onYear,
}: {
  minYear: number;
  maxYear: number;
  x: (year: number) => number;
  y: (value: number) => number;
  height: number;
  pad: number;
  onYear: (year: number) => void;
}) {
  const hits = [];
  for (let year = minYear; year <= maxYear; year += 1) {
    hits.push(
      <rect
        key={year}
        x={x(year) - 1.5}
        y={pad}
        width={3}
        height={height - pad * 2}
        fill="transparent"
        onClick={() => onYear(year)}
      >
        <title>{year}</title>
      </rect>,
    );
  }
  return <g>{hits}</g>;
}
