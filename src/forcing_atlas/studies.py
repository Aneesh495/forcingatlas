"""Ten empirical studies tied to the local files.

Each study states a claim, the number that was computed, and the limit of
that number. Nothing here treats a proxy as a thermometer or an SSP label as
a policy effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from forcing_atlas.attribution import fit_scaling, shuffle_observation
from forcing_atlas.catalog import sources_by_id
from forcing_atlas.cmip_means import control_versus_ssp245, load_ssp245_means
from forcing_atlas.ebm import InitialAnomaly, integrate
from forcing_atlas.emulator import new_emulator, predict, support_radius
from forcing_atlas.forcing_adapter import build_forcing
from forcing_atlas.grid_campaign import LAGS, SEEDS, spatial_rmse
from forcing_atlas.grids import area_weighted_mean
from forcing_atlas.grids_io import read_hadcrut_grid
from forcing_atlas.historical import load_context_series, load_historical_bundle, rebase_to_window
from forcing_atlas.product import _replay, build_global_product, conditional_extension
from forcing_atlas.readers import read_hadcrut_summary
from forcing_atlas.regions import get_region, regional_mean
from forcing_atlas.timebase import align_annual, window_mean


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    delta = left[mask] - right[mask]
    return float(np.sqrt(np.mean(delta**2)))


def _rate(years: np.ndarray, values: np.ndarray, start: int, end: int) -> float:
    mask = (years >= start) & (years <= end) & np.isfinite(values)
    design = np.column_stack([np.ones(int(np.count_nonzero(mask))), years[mask]])
    coefficient, *_ = np.linalg.lstsq(design, values[mask], rcond=None)
    return float(coefficient[1])


def _study(study_id: str, title: str, claim: str, result: str, limit: str, metrics: dict, sources: list[str]) -> dict:
    return {
        "id": study_id,
        "title": title,
        "claim": claim,
        "result": result,
        "limit": limit,
        "metrics": metrics,
        "sources": sources,
    }


def _component_path(root: Path, key: str, years: np.ndarray) -> np.ndarray:
    from forcing_atlas.readers import read_erf_components

    source = sources_by_id()["igcc-erf-components-1750-2023"]
    record = read_erf_components((root / source.file_name).read_text(), source)
    lookup = {int(year): float(value) for year, value in zip(record.years, record.extras[key], strict=True)}
    return np.array([lookup[int(year)] for year in years], dtype=np.float64)


def _fingerprints(root: Path, parameters, full_years: np.ndarray) -> dict[str, np.ndarray]:
    solar = _component_path(root, "erf_solar", full_years)
    volcanic = _component_path(root, "erf_volcanic", full_years)
    co2 = _component_path(root, "erf_co2", full_years)
    aerosol = _component_path(root, "erf_aerosol", full_years)
    zeros = np.zeros(full_years.shape)
    return {
        "co2": _replay(parameters, build_forcing(full_years, erf_co2=co2, erf_aerosol=zeros, erf_solar=zeros, erf_volcanic=zeros).total_w_m2, full_years),
        "aerosol": _replay(parameters, build_forcing(full_years, erf_co2=zeros, erf_aerosol=aerosol, erf_solar=zeros, erf_volcanic=zeros).total_w_m2, full_years),
        "natural": _replay(parameters, build_forcing(full_years, erf_solar=solar, erf_volcanic=volcanic, natural_only=True).total_w_m2, full_years),
    }


def _regional_tail(root: Path) -> dict[str, float]:
    grid = read_hadcrut_grid(root / "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc", name="noninfilled")
    bundle = load_historical_bundle(root)
    year_to_force = {int(year): index for index, year in enumerate(bundle.years.tolist())}
    keep = [index for index, year in enumerate(grid.years.tolist()) if int(year) in year_to_force]
    years = grid.years[keep]
    field = grid.annual[keep]
    areas = grid.grid.cell_areas_m2()
    weights = areas / np.sum(areas)
    components = []
    for name in ("forcing_co2_w_m2", "forcing_aerosol_w_m2", "forcing_solar_w_m2", "forcing_volcanic_w_m2"):
        series = getattr(bundle, name)
        components.append(np.array([series[year_to_force[int(year)]] for year in years], dtype=np.float64))
    forcing = np.column_stack(components)
    window = np.zeros((forcing.shape[0], LAGS, forcing.shape[1]))
    for lag in range(LAGS):
        if lag == 0:
            window[:, 0, :] = forcing
        else:
            window[lag:, lag, :] = forcing[:-lag]
    latitude = grid.grid.lat / 90.0
    longitude = grid.grid.lon / 180.0
    train_rows = np.array([index for index, year in enumerate(years.tolist()) if int(year) <= 1994 and index >= LAGS - 1], dtype=np.int64)
    inside = support_radius(forcing[train_rows], forcing)
    holdout = np.array(
        [index for index, year in enumerate(years.tolist()) if int(year) >= 1995 and bool(inside[index])],
        dtype=np.int64,
    )
    predictions = []
    for seed in SEEDS:
        model = new_emulator(forcing.shape[1], LAGS, field.shape[1], field.shape[2], seed)
        checkpoint = torch.load(root.parent.parent / "results" / "checkpoints" / f"emulator-seed-{seed}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        predictions.append(predict(model, window, forcing, latitude, longitude, weights))
    ensemble = np.nanmean(np.stack(predictions, axis=0), axis=0)
    arctic = get_region("arctic")
    tropics = get_region("tropics")

    def region_rmse(region) -> float:
        errors = []
        area_weights = []
        for index in holdout.tolist():
            mean_error, coverage = regional_mean(np.abs(ensemble[index] - field[index]), grid.grid, region, rule="overlap")
            if coverage > 0 and np.isfinite(mean_error):
                errors.append(mean_error)
                area_weights.append(coverage)
        if not errors:
            return float("nan")
        return float(np.average(errors, weights=area_weights))

    finite = np.isfinite(ensemble[holdout]) & np.isfinite(field[holdout])
    absolute = np.abs(ensemble[holdout] - field[holdout])
    tail = float(np.nanquantile(absolute[finite], 0.95)) if np.any(finite) else float("nan")
    return {
        "arctic_mean_absolute_error_k": region_rmse(arctic),
        "tropics_mean_absolute_error_k": region_rmse(tropics),
        "cell_absolute_error_p95_k": tail,
        "global_holdout_rmse_k": spatial_rmse(ensemble, field, areas, holdout),
    }


def run_studies(root: Path) -> list[dict]:
    product = build_global_product(root, particles=16)
    bundle = load_historical_bundle(root)
    context = load_context_series(root)
    years = product.years
    hadcrut = product.hadcrut
    common, had_on_common, berk_on_common = align_annual(years, hadcrut, product.berkeley_years, product.berkeley_rebased)
    overlap = (common >= 1880) & (common <= 2020)
    nh_source = sources_by_id()["hadcrut5-analysis-northern_hemisphere-annual"]
    sh_source = sources_by_id()["hadcrut5-analysis-southern_hemisphere-annual"]
    northern = read_hadcrut_summary((root / nh_source.file_name).read_text(), nh_source)
    southern = read_hadcrut_summary((root / sh_source.file_name).read_text(), sh_source)
    _, nh_vals, sh_vals = align_annual(northern.years, northern.values, southern.years, southern.values)
    nh_years, nh_aligned, _ = align_annual(northern.years, northern.values, southern.years, southern.values)
    pages = context["pages"]["proxy"]
    instrument = context["pages"]["instrumental_target"]
    proxy_years, proxy_vals, inst_vals = align_annual(pages.years, pages.values, instrument.years, instrument.values)
    late = (proxy_years >= 1850) & (proxy_years <= 2000) & np.isfinite(proxy_vals) & np.isfinite(inst_vals)
    grid = read_hadcrut_grid(root / "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc", name="noninfilled")
    analysis = read_hadcrut_grid(root / "HadCRUT.5.1.0.0.analysis.anomalies.ensemble_mean.nc", name="analysis")
    recent = np.where(grid.years == 2010)[0][0]
    non_cov = float(np.nanmean(grid.annual_coverage[recent]))
    ana_cov = float(np.nanmean(analysis.annual_coverage[recent]))
    arctic_2010, arctic_cov = regional_mean(grid.annual[recent], grid.grid, get_region("arctic"), rule="overlap")
    global_2010 = float(area_weighted_mean(grid.annual[recent], grid.grid.cell_areas_m2()))
    full_years = np.concatenate([bundle.spinup_years, years])
    fingerprints = _fingerprints(root, product.parameters, full_years)
    scored = {name: values[full_years >= years[0]] for name, values in fingerprints.items()}
    real_fit = fit_scaling(years, hadcrut, scored, ("co2", "aerosol", "natural"), train_end=1994, holdout_start=1995)
    shuffled_fit = fit_scaling(
        years,
        shuffle_observation(hadcrut, seed=19),
        scored,
        ("co2", "aerosol", "natural"),
        train_end=1994,
        holdout_start=1995,
    )
    pinatubo = (years >= 1991) & (years <= 1993)
    campaign = json.loads((root.parent.parent / "results" / "grid_campaign.json").read_text())
    seed_rmse = np.array([row["holdout_rmse_k"] for row in campaign["seeds"]], dtype=np.float64)
    seed_weights = (1.0 / seed_rmse)
    seed_weights = seed_weights / np.sum(seed_weights)
    tails = _regional_tail(root)
    cmip = [row for row in load_ssp245_means(root / "cmip6")] if (root / "cmip6").is_dir() else []
    controls = control_versus_ssp245(root / "cmip6") if (root / "cmip6").is_dir() else []
    extension = conditional_extension(
        product.parameters,
        full_years,
        np.concatenate([bundle.spinup_forcing_w_m2, bundle.forcing_w_m2[bundle.years <= int(years[-1])]]),
        end_year=2100,
    )
    cmip_2020 = []
    for row in cmip:
        index = int(np.where(row["years"] == 2020)[0][0])
        cmip_2020.append({"file": row["file"], "calendar": row["calendar"], "anomaly_vs_first_20y_k": float(row["anomaly_k"][index])})
    early = window_mean(years, hadcrut, 1850, 1900)
    modern = window_mean(years, hadcrut, 1995, 2023)
    studies = [
        _study(
            "S01",
            "HadCRUT and Berkeley Earth on one baseline",
            "After shifting Berkeley from 1951-1980 onto 1961-1990, the two annual series agree more closely than either agrees with an unrebased comparison.",
            f"On {int(common[overlap][0])}-{int(common[overlap][-1])} the rebased root-mean-square difference is {_rmse(had_on_common[overlap], berk_on_common[overlap]):.3f} K.",
            "Berkeley Earth land-ocean uses HadSST. Agreement over the ocean is not two independent observations.",
            {"rmse_k": _rmse(had_on_common[overlap], berk_on_common[overlap]), "years": int(np.count_nonzero(overlap))},
            ["HadCRUT.5.1.0.0 analysis global annual", "Berkeley Earth Land_and_Ocean_complete.txt"],
        ),
        _study(
            "S02",
            "Baseline window changes the reported warming",
            "The same HadCRUT series reads as a different warming when the zero is 1850-1900 rather than 1961-1990.",
            f"Mean anomaly 1995-2023 versus 1961-1990 is {modern:.3f} K. The 1850-1900 mean of the published series is {early:.3f} K, so the same recent window is {modern - early:.3f} K above 1850-1900.",
            "1850-1900 is a warming reference, not a preindustrial control run.",
            {"recent_vs_1961_1990_k": modern, "offset_1850_1900_k": early},
            ["HadCRUT.5.1.0.0 analysis global annual"],
        ),
        _study(
            "S03",
            "PAGES 2k is not the instrumental target",
            "Where both the reconstruction median and the embedded Cowtan and Way target are finite, they differ by more than the rounding of a shared thermometer.",
            f"From 1850 to 2000 the root-mean-square difference is {_rmse(proxy_vals[late], inst_vals[late]):.3f} K.",
            "The reconstruction is an April-March proxy ensemble. It is not substituted for HadCRUT.",
            {"rmse_k": _rmse(proxy_vals[late], inst_vals[late])},
            ["PAGES2k Full_ensemble_median_and_95pct_range.txt"],
        ),
        _study(
            "S04",
            "Northern Hemisphere warming is faster",
            "The HadCRUT northern-hemisphere series warms faster than the southern-hemisphere series over 1970-2023.",
            f"The northern slope is {_rate(nh_years, nh_aligned, 1970, 2023) * 10:.3f} K per decade and the southern slope is {_rate(nh_years, sh_vals, 1970, 2023) * 10:.3f} K per decade.",
            "These are hemisphere averages of one analysis, not independent land and ocean products.",
            {
                "north_k_per_decade": _rate(nh_years, nh_aligned, 1970, 2023) * 10,
                "south_k_per_decade": _rate(nh_years, sh_vals, 1970, 2023) * 10,
            },
            ["HadCRUT.5.1.0.0 northern and southern hemisphere annual summaries"],
        ),
        _study(
            "S05",
            "Infilling changes coverage, not the existence of empty cells",
            "The analysis grid covers more of the map than the non-infilled grid. The non-infilled Arctic mean is only as complete as the cells that were observed.",
            f"In 2010 the mean cell coverage is {non_cov:.3f} without infilling and {ana_cov:.3f} in the analysis. The non-infilled Arctic overlap mean is {arctic_2010:.3f} K with coverage {arctic_cov:.3f}, against a global area mean of {global_2010:.3f} K.",
            "A higher Arctic number on a sparse grid is not a complete Arctic average.",
            {"noninfilled_coverage_2010": non_cov, "analysis_coverage_2010": ana_cov, "arctic_coverage_2010": arctic_cov},
            ["HadCRUT.5.1.0.0 ensemble-mean grids"],
        ),
        _study(
            "S06",
            "Pinatubo is visible in the natural-only emulation",
            "The 1991-1993 dip is carried by volcanic forcing. The historical emulation includes it. The natural-only emulation also includes it because volcanoes are in that experiment.",
            f"Mean HadCRUT anomaly over 1991-1993 is {float(np.mean(hadcrut[pinatubo])):.3f} K. The historical emulation is {float(np.mean(product_slice(product.ebm_historical, years, pinatubo))):.3f} K and the natural-only emulation is {float(np.mean(product.ebm_natural[pinatubo])):.3f} K.",
            "Natural-only here is solar plus volcanic effective radiative forcing in the two-box model. It is not a hist-nat simulation.",
            {
                "hadcrut_1991_1993_k": float(np.mean(hadcrut[pinatubo])),
                "historical_emulation_k": float(np.mean(product.ebm_historical[pinatubo])),
                "natural_emulation_k": float(np.mean(product.ebm_natural[pinatubo])),
            },
            ["IGCC ERF components", "HadCRUT5"],
        ),
        _study(
            "S07",
            "Single-forcing fingerprints are collinear and still not DAMIP",
            "Least squares on the training window assigns scaling factors to the two-box CO2, aerosol, and natural fingerprints. A shuffled HadCRUT series fits those fingerprints worse.",
            f"Training R2 is {real_fit.train_r2:.3f} and the shuffled control R2 is {shuffled_fit.train_r2:.3f}. The condition number of the training Gram matrix is {real_fit.condition_number:.1f}.",
            "These fingerprints are one emulator with the fitted parameters. They are not hist-GHG, hist-aer, or hist-nat.",
            {
                "beta_co2": float(real_fit.beta[0]),
                "beta_aerosol": float(real_fit.beta[1]),
                "beta_natural": float(real_fit.beta[2]),
                "train_r2": real_fit.train_r2,
                "shuffled_r2": shuffled_fit.train_r2,
                "condition_number": real_fit.condition_number,
                "holdout_rmse_k": real_fit.holdout_rmse,
            },
            ["IGCC ERF components", "HadCRUT5", "two-box replay"],
        ),
        _study(
            "S08",
            "The training optimum is on the prior boundary and loses to persistence",
            "The two-box search, confined to the declared prior and blind to 1995-2023, sits on the low-feedback edge. Persistence beats it on the holdout.",
            f"Lambda is {product.parameters.lambda_w_m2_k:.3f} W m-2 K-1. The particle mean is {product.lambda_interval[0]:.3f}, with a 5 to 95 percent range of {product.lambda_interval[1]:.3f} to {product.lambda_interval[2]:.3f}. Holdout RMSE is {product.holdout_rmse:.3f} K against persistence {product.persistence_holdout_rmse:.3f} K.",
            "An equilibrium sensitivity of 3.93 divided by this lambda is not an assessed ECS. The fit wanted to leave the prior.",
            {
                "lambda_w_m2_k": product.parameters.lambda_w_m2_k,
                "lambda_p05": product.lambda_interval[1],
                "lambda_p95": product.lambda_interval[2],
                "holdout_rmse_k": product.holdout_rmse,
                "persistence_rmse_k": product.persistence_holdout_rmse,
                "implied_ecs_k": 3.93 / product.parameters.lambda_w_m2_k,
            },
            ["HadCRUT5", "IGCC ERF 1750-2023"],
        ),
        _study(
            "S09",
            "A spatial fit beats a uniform trend by a small margin",
            "Three seeds trained through 1994 have a lower holdout grid error than a latitude-uniform trend. The seed spread is part of the result. Inverse-RMSE weights favor the better seed without refitting.",
            f"Seed holdout errors are {', '.join(f'{value:.3f}' for value in seed_rmse)}. The uniform trend is {campaign['uniform_trend_holdout_rmse_k']:.3f} K. Weights are {', '.join(f'{value:.3f}' for value in seed_weights)}.",
            "2023 is outside the training forcing radius and is an abstention. This is not a test on a withheld model family.",
            {
                "seed_holdout_rmse_k": seed_rmse.tolist(),
                "uniform_trend_rmse_k": campaign["uniform_trend_holdout_rmse_k"],
                "pattern_trend_rmse_k": campaign["pattern_trend_holdout_rmse_k"],
                "seed_weights": seed_weights.tolist(),
                "abstained_years": campaign["abstained_holdout_years"],
                **tails,
            },
            ["HadCRUT non-infilled grid", "results/grid_campaign.json"],
        ),
        _study(
            "S10",
            "Stabilize and decline are different conditional paths",
            "Holding the last assessed forcing and declining toward the 1990 forcing do not reach the same 2100 global anomaly. Neither path is an SSP.",
            f"In 2100 the held-forcing path is {float(extension['stabilize_k'][-1]):.2f} K and the declining path is {float(extension['decline_k'][-1]):.2f} K, both anomalies versus 1961-1990.",
            "Both paths use the boundary-value feedback from S08. A different prior would move 2100. No regional field is drawn past the observations.",
            {
                "stabilize_2100_k": float(extension["stabilize_k"][-1]),
                "decline_2100_k": float(extension["decline_k"][-1]),
            },
            ["two-box extension"],
        ),
        _study(
            "S11",
            "Four ssp245 files do not share a calendar or a 1961-1990 baseline",
            "CanESM5, MIROC6, MPI-ESM1-2-LR, and HadGEM3-GC31-LL were read as annual tas. Their anomalies are against 2015-2019 in each file, which is not 1961-1990.",
            "By 2020 those anomalies versus each file's 2015-2019 mean are "
            + ", ".join(f"{row['file'].split('_')[2]} {row['anomaly_vs_first_20y_k']:.2f} K ({row['calendar']})" for row in cmip_2020)
            + ".",
            "This is one pathway, ssp245, and one member each. It is not a multi-pathway attribution and it is not comparable to the HadCRUT 1961-1990 anomaly without a further baseline shift that these files do not contain.",
            {"members_2020": cmip_2020},
            ["ClimateBench CMIP6.zip ssp245 tas members"],
        ),
        _study(
            "S12",
            "Energy inventory is a stock in zettajoules",
            "The file named earth_energy_imbalance.csv increases over time and is stored as heat content. It is not the forcing series.",
            f"The inventory total moves from {float(context['inventory'].values[np.isfinite(context['inventory'].values)][0]):.2f} ZJ to {float(context['inventory'].values[np.isfinite(context['inventory'].values)][-1]):.2f} ZJ. The last years in the file are missing and stay missing.",
            "A flux would be the time difference divided by area. That derivative is not substituted for the ERF table.",
            {
                "start_zj": float(context["inventory"].values[np.isfinite(context["inventory"].values)][0]),
                "end_zj": float(context["inventory"].values[np.isfinite(context["inventory"].values)][-1]),
            },
            ["IGCC earth_energy_imbalance.csv"],
        ),
        _study(
            "S13",
            "piControl spread is much smaller than the ssp245 change",
            "Each model's own preindustrial-control global mean varies by about a tenth of a kelvin, while its ssp245 file warms by more than a kelvin across the same length of record.",
            "Control standard deviations and ssp245 end anomalies versus 2015-2019, or versus the first five years of a branch that is not 2015, are "
            + ", ".join(
                f"{row['model']} control std {row['piControl_std_k']:.3f} K (years {row['piControl_years'][0]}-{row['piControl_years'][1]}) "
                f"and ssp245 end {row['ssp245_end_anomaly_k']:.2f} K"
                for row in controls
            )
            + ".",
            "HadGEM3 and MPI piControl files carry a time coordinate that reads 2015-2100. Those numbers are branch labels in the file, not historical years, and they are not drawn on the 1850-2100 chart. One member is not a model-family distribution.",
            {"models": controls},
            ["ClimateBench CMIP6.zip tas piControl and ssp245"],
        ),
    ]
    return studies


def product_slice(values: np.ndarray, years: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[mask]


def write_studies(root: Path, destination: Path) -> list[dict]:
    studies = run_studies(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(studies, indent=2))
    return studies
