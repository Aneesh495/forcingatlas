export type Semantics =
  | "observation"
  | "observational_analysis"
  | "proxy_reconstruction"
  | "emulation"
  | "source_simulation";

export interface SeriesRecord {
  id: string;
  semantics: Semantics;
  years: number[];
  values: Array<number | null>;
  lower?: Array<number | null>;
  upper?: Array<number | null>;
  label: string;
  detail: string;
  future_start?: number;
}

export interface AtlasBundle {
  format: string;
  unit: string;
  baseline: string;
  notes: string[];
  fit: {
    lambda_w_m2_k: number;
    gamma_w_m2_k: number;
    c_surface: number;
    c_deep: number;
    train_rmse_k: number;
    holdout_rmse_k: number;
    persistence_holdout_rmse_k: number;
    trend_holdout_rmse_k: number;
    mean_holdout_rmse_k: number;
    lambda_mean: number;
    lambda_p05: number;
    lambda_p95: number;
    train_end: number;
    holdout_start: number;
  };
  grid: {
    id: string;
    semantics: Semantics;
    variable: string;
    baseline: string;
    years: number[];
    nlat: number;
    nlon: number;
    lat: number[];
    lon: number[];
    data: string;
    coverage: string;
    dtype: string;
    layout: string;
    missing: string;
    licence: string;
    source: string;
  };
  series: SeriesRecord[];
  studies?: StudyRecord[];
}

export interface StudyRecord {
  id: string;
  title: string;
  claim: string;
  result: string;
  limit: string;
  sources: string[];
}

export const SEMANTICS_LABEL: Record<Semantics, string> = {
  observation: "Observation",
  observational_analysis: "Analysis",
  proxy_reconstruction: "Proxy",
  emulation: "Emulation",
  source_simulation: "Simulation",
};

export function valueAt(record: SeriesRecord, year: number): number | null {
  const index = record.years.indexOf(year);
  if (index < 0) return null;
  const value = record.values[index];
  return value == null || !Number.isFinite(value) ? null : value;
}

export async function loadBundle(): Promise<{ bundle: AtlasBundle; field: Float32Array; coverage: Float32Array }> {
  const response = await fetch("/bundle/atlas.json");
  if (!response.ok) {
    throw new Error("The atlas bundle is not on this server. Generate it with python -m forcing_atlas.cli export-atlas.");
  }
  const bundle = (await response.json()) as AtlasBundle;
  if (bundle.format !== "forcing-atlas-bundle/1") {
    throw new Error(`Unexpected bundle format ${bundle.format}`);
  }
  const [fieldBuffer, coverageBuffer] = await Promise.all([
    fetch(`/bundle/${bundle.grid.data}`).then((item) => {
      if (!item.ok) throw new Error("Missing gridded field");
      return item.arrayBuffer();
    }),
    fetch(`/bundle/${bundle.grid.coverage}`).then((item) => {
      if (!item.ok) throw new Error("Missing coverage field");
      return item.arrayBuffer();
    }),
  ]);
  const field = new Float32Array(fieldBuffer);
  const coverage = new Float32Array(coverageBuffer);
  const expected = bundle.grid.years.length * bundle.grid.nlat * bundle.grid.nlon;
  if (field.length !== expected || coverage.length !== expected) {
    throw new Error("Gridded file length does not match the bundle header.");
  }
  return { bundle, field, coverage };
}
