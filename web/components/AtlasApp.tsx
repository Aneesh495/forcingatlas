"use client";

import { useEffect, useMemo, useState } from "react";

import { MapCanvas } from "@/components/MapCanvas";
import { SeriesChart } from "@/components/SeriesChart";
import { Button } from "@/components/ui/button";
import {
  SEMANTICS_LABEL,
  loadBundle,
  valueAt,
  type AtlasBundle,
  type Semantics,
  type SeriesRecord,
} from "@/lib/bundle";
import { SCALE_MAX, SCALE_MIN, anomalyColor } from "@/lib/color";

const DEFAULT_ON = [
  "hadcrut5-analysis-global",
  "berkeley-air-rebased",
  "pages2k-median",
  "ebm-historical",
  "ebm-natural",
];

export function AtlasApp() {
  const [bundle, setBundle] = useState<AtlasBundle | null>(null);
  const [field, setField] = useState<Float32Array | null>(null);
  const [coverage, setCoverage] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [year, setYear] = useState(2023);
  const [active, setActive] = useState<string[]>(DEFAULT_ON);
  const [globe, setGlobe] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [cell, setCell] = useState<{ lat: number; lon: number; value: number | null; coverage: number | null } | null>(
    null,
  );

  useEffect(() => {
    loadBundle()
      .then((loaded) => {
        setBundle(loaded.bundle);
        setField(loaded.field);
        setCoverage(loaded.coverage);
        setYear(loaded.bundle.grid.years[loaded.bundle.grid.years.length - 1] ?? 2023);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "The atlas bundle could not be loaded.");
      });
  }, []);

  useEffect(() => {
    if (!playing || !bundle) return;
    const timer = window.setInterval(() => {
      setYear((current) => {
        const next = current < 1850 ? 1850 : current + 1;
        return next > 2100 ? 1850 : next;
      });
    }, 180);
    return () => window.clearInterval(timer);
  }, [playing, bundle]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "ArrowRight") setYear((current) => Math.min(2100, current + 1));
      if (event.key === "ArrowLeft") setYear((current) => Math.max(1, current - 1));
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((current) => !current);
      }
      if (event.key === "g") setGlobe((current) => !current);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const gridYear = bundle?.grid.years.includes(year) ?? false;

  if (error) {
    return (
      <main className="mx-auto max-w-3xl p-8">
        <h1 className="font-serif text-4xl">Forcing Atlas</h1>
        <p className="mt-4 text-ink/80">{error}</p>
      </main>
    );
  }
  if (!bundle || !field || !coverage) {
    return (
      <main className="p-8">
        <p>Loading the frozen bundle.</p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen flex-col">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-line px-4 py-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-tide">Observed change and conditional futures</p>
          <h1 className="font-serif text-3xl leading-none">Forcing Atlas</h1>
        </div>
        <p className="max-w-xl text-sm text-ink/75">
          Anomalies versus {bundle.baseline}. A pathway is conditional on its forcing. Empty map cells were not observed.
        </p>
      </header>
      <div className="grid flex-1 grid-cols-1 lg:grid-cols-[280px_1fr_300px]">
        <aside className="border-b border-line p-4 lg:border-b-0 lg:border-r">
          <h2 className="font-serif text-xl">Layers</h2>
          <ul className="mt-3 space-y-2">
            {bundle.series.map((record) => (
              <li key={record.id}>
                <label className="flex cursor-pointer items-start gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={active.includes(record.id)}
                    onChange={() =>
                      setActive((current) =>
                        current.includes(record.id) ? current.filter((item) => item !== record.id) : [...current, record.id],
                      )
                    }
                  />
                  <span>
                    <span className="block font-medium">{record.label}</span>
                    <Badge semantics={record.semantics} />
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </aside>
        <section className="flex min-h-[70vh] flex-col p-3">
          <div className="relative min-h-[320px] flex-1">
            <MapCanvas
              bundle={bundle}
              field={field}
              coverage={coverage}
              year={year}
              globe={globe}
              onSelect={setCell}
            />
            {!gridYear && (
              <div className="pointer-events-none absolute inset-x-4 bottom-4 rounded-md bg-paper/90 p-3 text-sm">
                Regional layer unavailable after {bundle.grid.years[bundle.grid.years.length - 1]}. The series below are
                global emulations, not extended maps.
              </div>
            )}
          </div>
          <Scale />
          <div className="mt-3 h-56 rounded-md border border-line bg-white/40">
            <SeriesChart records={bundle.series} active={active} year={year} onYear={setYear} />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button variant="outline" onClick={() => setPlaying((current) => !current)}>
              {playing ? "Pause" : "Play"}
            </Button>
            <Button variant="outline" onClick={() => setGlobe((current) => !current)}>
              {globe ? "Flat map" : "Globe"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setYear((current) => Math.max(1, current - 1))} aria-label="Previous year">
              Previous
            </Button>
            <Button variant="outline" size="sm" onClick={() => setYear((current) => Math.min(2100, current + 1))} aria-label="Next year">
              Next
            </Button>
            <label className="flex items-center gap-2 text-sm">
              <span className="sr-only">Year</span>
              <input
                type="number"
                min={1}
                max={2100}
                value={year}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  if (Number.isFinite(next)) setYear(Math.min(2100, Math.max(1, Math.round(next))));
                }}
                className="w-20 rounded border border-line bg-paper px-2 py-1"
                aria-label="Year"
              />
              <input
                type="range"
                min={1850}
                max={2100}
                step={1}
                value={Math.min(2100, Math.max(1850, year))}
                onChange={(event) => setYear(Number(event.target.value))}
                aria-label="Year from 1850 to 2100"
                className="w-56"
              />
            </label>
          </div>
        </section>
        <aside className="border-t border-line p-4 lg:border-l lg:border-t-0">
          <Inspector bundle={bundle} year={year} active={active} cell={cell} gridYear={gridYear} />
        </aside>
      </div>
    </main>
  );
}

function Badge({ semantics }: { semantics: Semantics }) {
  const tone =
    semantics === "proxy_reconstruction"
      ? "bg-proxy/15 text-proxy"
      : semantics === "emulation"
        ? "bg-clay/15 text-clay"
        : semantics === "observation"
          ? "bg-tide/15 text-tide"
          : "bg-ink/10 text-ink";
  return <span className={`mt-1 inline-block rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${tone}`}>{SEMANTICS_LABEL[semantics]}</span>;
}

function Scale() {
  const stops = [-2, -1, 0, 1, 2];
  return (
    <div className="mt-2 flex items-center gap-3 text-xs">
      <span>deg C anomaly</span>
      <div className="flex h-3 flex-1 overflow-hidden rounded">
        {Array.from({ length: 24 }, (_, index) => {
          const value = SCALE_MIN + ((SCALE_MAX - SCALE_MIN) * index) / 23;
          return <span key={index} className="h-full flex-1" style={{ background: anomalyColor(value) }} />;
        })}
      </div>
      <span>
        {stops[0]} to {stops[stops.length - 1]}
      </span>
    </div>
  );
}

function Inspector({
  bundle,
  year,
  active,
  cell,
  gridYear,
}: {
  bundle: AtlasBundle;
  year: number;
  active: string[];
  cell: { lat: number; lon: number; value: number | null; coverage: number | null } | null;
  gridYear: boolean;
}) {
  const rows = useMemo(
    () => bundle.series.filter((record) => active.includes(record.id)).map((record) => ({ record, value: valueAt(record, year) })),
    [bundle.series, active, year],
  );
  const fit = bundle.fit;
  return (
    <div className="space-y-4 text-sm">
      <div>
        <h2 className="font-serif text-xl">{year}</h2>
        <p className="text-ink/70">{gridYear ? "Gridded observations exist for this year." : "Map layer unavailable."}</p>
      </div>
      <ul className="space-y-2">
        {rows.map(({ record, value }) => (
          <li key={record.id}>
            <div className="flex justify-between gap-2">
              <span>{record.label}</span>
              <span className="tabular-nums">{value == null ? "absent" : `${value.toFixed(2)} deg C`}</span>
            </div>
            <p className="text-xs text-ink/65">{record.detail}</p>
          </li>
        ))}
      </ul>
      <div>
        <h3 className="font-medium">Cell</h3>
        {cell ? (
          <p>
            {cell.lat.toFixed(1)} deg, {cell.lon.toFixed(1)} deg.{" "}
            {cell.value == null ? "No observation in this cell." : `${cell.value.toFixed(2)} deg C anomaly.`} Coverage{" "}
            {cell.coverage == null ? "unknown" : `${Math.round(cell.coverage * 100)}% of the year`}.
          </p>
        ) : (
          <p className="text-ink/70">Click the flat map to read a 5 degree cell. Missing cells stay empty.</p>
        )}
      </div>
      <div>
        <h3 className="font-medium">Hindcast, 1995 onward</h3>
        <p>Two-box holdout RMSE {fit.holdout_rmse_k.toFixed(3)} K.</p>
        <p>Persistence {fit.persistence_holdout_rmse_k.toFixed(3)} K. Linear trend {fit.trend_holdout_rmse_k.toFixed(3)} K.</p>
        <p>
          Feedback {fit.lambda_w_m2_k.toFixed(2)} W m-2 K-1, on the prior boundary used for the search. The holdout was
          not used to move it.
        </p>
      </div>
      <p className="text-xs text-ink/60">{bundle.grid.source}</p>
      <p className="text-xs text-ink/60">{bundle.grid.licence}</p>
    </div>
  );
}
