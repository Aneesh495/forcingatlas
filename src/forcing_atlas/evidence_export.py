"""Assemble the small evidence file the independent verifier reads.

The file stores series and the campaign split. It does not store a pass bit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from forcing_atlas.historical import load_context_series, load_historical_bundle


def _clean(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values).tolist()]


def write_evidence(root: Path, campaign: dict, destination: Path, *, source_commit: str) -> Path:
    bundle = load_historical_bundle(root)
    pages = load_context_series(root)["pages"]["proxy"]
    seeds = [float(row["holdout_rmse_k"]) for row in campaign["seeds"]]
    volcanic_year = 1992
    volcanic_index = int(np.where(bundle.years == volcanic_year)[0][0])
    document = {
        "format": "forcing-atlas-evidence/1",
        "source_commit": source_commit,
        "anomaly_records": [
            {
                "id": "hadcrut5-analysis-global",
                "baseline_start": 1961,
                "baseline_end": 1990,
                "years": [int(year) for year in bundle.years.tolist()],
                "values": _clean(bundle.hadcrut_anomaly_k),
            }
        ],
        "proxy_records": [
            {
                "id": "pages2k-2019",
                "semantics": "proxy_reconstruction",
                "used_as_thermometer": False,
                "years": [int(year) for year in pages.years.tolist()],
                "values": _clean(pages.values),
            }
        ],
        "campaign": {
            "train_years": campaign["train_years"],
            "holdout_years": campaign["holdout_years"],
            "seed_holdout_rmse_k": seeds,
            "mean_seed_holdout_rmse_k": float(sum(seeds) / len(seeds)),
            "uniform_trend_holdout_rmse_k": campaign["uniform_trend_holdout_rmse_k"],
            "pattern_trend_holdout_rmse_k": campaign["pattern_trend_holdout_rmse_k"],
            "beats_uniform_trend": campaign["beats_uniform_trend"],
        },
        "volcanic": {
            "year": volcanic_year,
            "years": [int(year) for year in bundle.years.tolist()],
            "values": _clean(bundle.forcing_volcanic_w_m2),
            "value_at_check_year": float(bundle.forcing_volcanic_w_m2[volcanic_index]),
        },
        "scenarios": [
            {
                "id": "stabilize-global",
                "unit": "degC_anomaly",
                "values": [0.2, 0.4, 0.6],
            }
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document))
    return destination


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return completed.stdout.strip()
