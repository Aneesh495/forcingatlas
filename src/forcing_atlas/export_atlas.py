"""Write the frozen atlas bundle.

The browser reads this bundle. It does not rerun the solver. Regenerating
the bundle is a command-line action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forcing_atlas.cmip_means import load_ssp245_means
from forcing_atlas.grid_campaign import emulator_global_mean
from forcing_atlas.grids_io import assert_ogl_licence, read_hadcrut_grid
from forcing_atlas.historical import load_historical_bundle
from forcing_atlas.product import build_global_product, conditional_extension


def _series(identifier: str, semantics: str, years: np.ndarray, values: np.ndarray, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": identifier,
        "semantics": semantics,
        "years": [int(year) for year in np.asarray(years).tolist()],
        "values": [None if not np.isfinite(value) else float(value) for value in np.asarray(values).tolist()],
    }
    payload.update(extra)
    return payload


def export_bundle(root: Path, destination: Path, *, particles: int = 24) -> Path:
    product = build_global_product(root, particles=particles)
    bundle_hist = load_historical_bundle(root)
    mask = bundle_hist.years <= int(product.years[-1])
    history_years = np.concatenate([bundle_hist.spinup_years, bundle_hist.years[mask]])
    history_forcing = np.concatenate([bundle_hist.spinup_forcing_w_m2, bundle_hist.forcing_w_m2[mask]])
    extension = conditional_extension(product.parameters, history_years, history_forcing, end_year=2100)
    grid = read_hadcrut_grid(root / "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc", name="noninfilled")
    assert_ogl_licence(grid.attributes)
    # Keep years that the annual field actually contains and that overlap the product.
    year_lookup = {int(year): index for index, year in enumerate(grid.years.tolist())}
    keep = [year_lookup[int(year)] for year in product.years.tolist() if int(year) in year_lookup]
    field = grid.annual[keep].astype(np.float32)
    coverage = grid.annual_coverage[keep].astype(np.float32)
    destination.mkdir(parents=True, exist_ok=True)
    field_path = destination / "noninfilled.f32"
    coverage_path = destination / "noninfilled-coverage.f32"
    field.tofile(field_path)
    coverage.tofile(coverage_path)
    parameters = product.parameters
    document = {
        "format": "forcing-atlas-bundle/1",
        "unit": "degC_anomaly",
        "baseline": "1961-1990",
        "notes": list(product.notes),
        "fit": {
            "lambda_w_m2_k": parameters.lambda_w_m2_k,
            "gamma_w_m2_k": parameters.gamma_w_m2_k,
            "c_surface": parameters.c_surface_w_yr_m2_k,
            "c_deep": parameters.c_deep_w_yr_m2_k,
            "train_rmse_k": product.train_rmse,
            "holdout_rmse_k": product.holdout_rmse,
            "persistence_holdout_rmse_k": product.persistence_holdout_rmse,
            "trend_holdout_rmse_k": product.trend_holdout_rmse,
            "mean_holdout_rmse_k": product.mean_holdout_rmse,
            "lambda_mean": product.lambda_interval[0],
            "lambda_p05": product.lambda_interval[1],
            "lambda_p95": product.lambda_interval[2],
            "train_end": 1994,
            "holdout_start": 1995,
        },
        "grid": {
            "id": "hadcrut5-noninfilled-annual",
            "semantics": "observation",
            "variable": "tas_anomaly",
            "baseline": "1961-1990",
            "years": [int(year) for year in product.years.tolist() if int(year) in year_lookup],
            "nlat": grid.grid.nlat,
            "nlon": grid.grid.nlon,
            "lat": grid.grid.lat.tolist(),
            "lon": grid.grid.lon.tolist(),
            "data": field_path.name,
            "coverage": coverage_path.name,
            "dtype": "float32",
            "layout": "year,lat,lon",
            "missing": "nan",
            "licence": grid.attributes.get("licence", ""),
            "source": "HadCRUT.5.1.0.0 non-infilled ensemble mean, annualized with Gregorian month lengths",
        },
        "series": [
            _series(
                "hadcrut5-analysis-global",
                "observational_analysis",
                product.years,
                product.hadcrut,
                lower=[None if not np.isfinite(v) else float(v) for v in product.hadcrut_lower],
                upper=[None if not np.isfinite(v) else float(v) for v in product.hadcrut_upper],
                label="HadCRUT5 analysis global",
                detail="Hemisphere-average anomaly versus 1961-1990. Summary limits are 2.5 and 97.5 percent.",
            ),
            _series(
                "berkeley-air-rebased",
                "observational_analysis",
                product.berkeley_years,
                product.berkeley_rebased,
                label="Berkeley Earth, air above sea ice",
                detail="Rebased from 1951-1980 onto 1961-1990. The ocean uses HadSST, so it is not fully independent of HadCRUT.",
            ),
            _series(
                "pages2k-median",
                "proxy_reconstruction",
                product.pages_years,
                product.pages_median,
                lower=[None if not np.isfinite(v) else float(v) for v in product.pages_low],
                upper=[None if not np.isfinite(v) else float(v) for v in product.pages_high],
                label="PAGES2k ensemble median",
                detail="Proxy reconstruction versus 1961-1990. Not a thermometer. The embedded instrumental target is a separate series.",
            ),
            _series(
                "ebm-historical",
                "emulation",
                product.years,
                product.ebm_historical,
                label="Two-box historical forcing",
                detail="Conditional on the IGCC effective radiative forcing. Fit on 1850-1994 after a 1750 spinup. Not an observation.",
            ),
            _series(
                "ebm-natural",
                "emulation",
                product.natural_years,
                product.ebm_natural,
                label="Two-box solar and volcanic only",
                detail="Same parameters, solar plus volcanic forcing only. Not a hist-nat simulation and not an observed world without greenhouse gases.",
            ),
            _series(
                "ebm-stabilize",
                "emulation",
                extension["years"],
                extension["stabilize_k"],
                label="Forcing held after the last assessed year",
                detail="Global path only. The map does not extend. This is not an SSP and not a policy forecast.",
                future_start=int(extension["future_start"][0]),
            ),
            _series(
                "ebm-decline",
                "emulation",
                extension["years"],
                extension["decline_k"],
                label="Forcing declines toward its 1990 value by 2100",
                detail="A declared linear forcing path. It is not ScenarioMIP and it has no regional layer after the assessed years.",
                future_start=int(extension["future_start"][0]),
            ),
        ],
    }
    checkpoint_dir = Path("results/checkpoints")
    if all((checkpoint_dir / f"emulator-seed-{seed}.pt").is_file() for seed in (11, 29, 47)):
        emulator_years, emulator_values = emulator_global_mean(root, checkpoint_dir)
        document["series"].append(
            _series(
                "emulator-ensemble",
                "emulation",
                emulator_years,
                emulator_values,
                label="Spatial emulator, three-seed mean",
                detail=(
                    "Area-weighted mean of the non-infilled grid emulator. "
                    "Trained through 1994. Years outside the training forcing radius are absent, including 2023."
                ),
            )
        )
    if (root / "cmip6").is_dir():
        for row in load_ssp245_means(root / "cmip6"):
            model = str(row["file"]).split("_")[2]
            document["series"].append(
                _series(
                    f"ssp245-{model}",
                    "source_simulation",
                    row["years"],
                    row["anomaly_k"],
                    label=f"{model} ssp245",
                    detail=(
                        f"Source simulation, annual tas, calendar {row['calendar']}. "
                        "Anomaly versus 2015-2019 in this file, not versus 1961-1990. "
                        "One member of one pathway. Not a HadCRUT observation and not a multi-model mean."
                    ),
                )
            )
    studies_path = Path("results/studies.json")
    if studies_path.is_file():
        document["studies"] = json.loads(studies_path.read_text())
    (destination / "atlas.json").write_text(json.dumps(document))
    return destination / "atlas.json"
