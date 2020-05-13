"""Command line for acquisition, hindcast, export, and the walkthrough.

Commands that are not implemented print the reason and exit 2. A diagnostic
training run is not the full campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from forcing_atlas.acquire import stream_download
from forcing_atlas.catalog import PINNED
from forcing_atlas.export_atlas import export_bundle
from forcing_atlas.historical import load_historical_bundle
from forcing_atlas.product import build_global_product, conditional_extension


def _root() -> Path:
    return Path("data/raw")


def _acquire() -> int:
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    for source in PINNED:
        if source.group == "simulation-archive":
            print(f"skip {source.dataset_id}: archive is fetched by member, not stored whole")
            continue
        dest = root / source.file_name
        cap = (source.byte_size or 2_000_000) + 1_000_000
        result = stream_download(
            source.url,
            dest,
            max_bytes=cap,
            expected_md5=source.checksum_md5,
            expected_sha256=source.checksum_sha256,
            timeout_s=180,
        )
        print(f"{source.dataset_id} {result.byte_count} {result.sha256}")
    return 0


def _summary(particles: int) -> dict[str, object]:
    product = build_global_product(_root(), particles=particles)
    return {
        "train_rmse_k": product.train_rmse,
        "holdout_rmse_k": product.holdout_rmse,
        "persistence_holdout_rmse_k": product.persistence_holdout_rmse,
        "trend_holdout_rmse_k": product.trend_holdout_rmse,
        "lambda_w_m2_k": product.parameters.lambda_w_m2_k,
        "lambda_interval": product.lambda_interval,
        "years": [int(product.years[0]), int(product.years[-1])],
        "notes": list(product.notes),
    }


def _walkthrough() -> int:
    bundle = load_historical_bundle(_root())
    product = build_global_product(_root(), particles=16)
    berk_years = product.berkeley_years
    common = [int(year) for year in product.years.tolist() if int(year) in set(int(y) for y in berk_years.tolist())]
    print("Walkthrough")
    print(f"HadCRUT5 and Berkeley Earth overlap on {common[0]}-{common[-1]} after Berkeley is rebased to 1961-1990.")
    print("Berkeley's ocean uses HadSST, so the two products are not independent over the ocean.")
    print("PAGES2k is loaded as a proxy reconstruction. It is not treated as a thermometer.")
    print(
        f"Two-box holdout RMSE {product.holdout_rmse:.3f} K. "
        f"Persistence {product.persistence_holdout_rmse:.3f} K. "
        f"Trend {product.trend_holdout_rmse:.3f} K."
    )
    print("Natural-only is solar plus volcanic forcing with the same parameters. It is not hist-nat.")
    history_years = list(bundle.spinup_years) + [int(year) for year in bundle.years.tolist() if int(year) <= int(product.years[-1])]
    print(f"Forcing spinup starts in {history_years[0]}. Assessed overlap ends in {int(product.years[-1])}.")
    print("Regional grids stop at the last non-infilled year in the export. Global paths may be extended to 2100.")
    print("Rerun: python -m forcing_atlas.cli walkthrough")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forcing-atlas")
    parser.add_argument(
        "command",
        choices=["acquire", "calibrate", "hindcast", "scenario", "export-atlas", "walkthrough", "train", "cmip-sample"],
    )
    parser.add_argument("--particles", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "acquire":
        return _acquire()
    if args.command in {"calibrate", "hindcast"}:
        json.dump(_summary(args.particles), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.command == "scenario":
        product = build_global_product(_root(), particles=args.particles)
        bundle = load_historical_bundle(_root())
        mask = bundle.years <= int(product.years[-1])
        extension = conditional_extension(
            product.parameters,
            np.concatenate([bundle.spinup_years, bundle.years[mask]]),
            np.concatenate([bundle.spinup_forcing_w_m2, bundle.forcing_w_m2[mask]]),
        )
        print(
            json.dumps(
                {
                    "future_start": int(extension["future_start"][0]),
                    "stabilize_2100_k": float(extension["stabilize_k"][-1]),
                    "decline_2100_k": float(extension["decline_k"][-1]),
                    "regional_after_assessed": "unavailable",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "export-atlas":
        path = export_bundle(_root(), Path("web/public/bundle"), particles=args.particles)
        print(path)
        return 0
    if args.command == "walkthrough":
        return _walkthrough()
    if args.command == "train":
        from forcing_atlas.evidence_export import git_head, write_evidence
        from forcing_atlas.grid_campaign import run_grid_campaign

        campaign_path = Path("results/grid_campaign.json")
        campaign = run_grid_campaign(_root(), campaign_path, epochs=30)
        evidence = write_evidence(_root(), campaign, Path("results/evidence.json"), source_commit=git_head())
        print(json.dumps({"campaign": str(campaign_path), "evidence": str(evidence), "beats_uniform_trend": campaign["beats_uniform_trend"], "seeds": campaign["seeds"]}, indent=2))
        return 0
    if args.command == "cmip-sample":
        from forcing_atlas.catalog import CMIP6_TAS_MEMBERS, cmip6_member_name
        from forcing_atlas.cmip_fetch import describe_calendars, fetch_tas_members

        names = [cmip6_member_name(entry) for entry in CMIP6_TAS_MEMBERS if entry["experiment"] == "ssp245" and entry["source_id"] in {"CanESM5", "MIROC6", "MPI-ESM1-2-LR", "HadGEM3-GC31-LL"}]
        paths = fetch_tas_members(Path("data/raw/cmip6"), names)
        print(json.dumps(describe_calendars(paths), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
