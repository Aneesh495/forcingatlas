"""Recompute evidence relationships. Do not trust a stored pass bit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


class Rejected(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _mean(values: list[float]) -> float:
    if not values:
        raise Rejected("baseline window has no finite values")
    return sum(values) / len(values)


def _check_baseline(record: dict) -> None:
    years = record["years"]
    values = record["values"]
    start = int(record["baseline_start"])
    end = int(record["baseline_end"])
    window = []
    for year, value in zip(years, values, strict=True):
        if start <= int(year) <= end and value is not None and math.isfinite(float(value)):
            window.append(float(value))
    center = _mean(window)
    # An anomaly versus the claimed window has a mean near zero on that window.
    if abs(center) > 0.05:
        raise Rejected(
            f"wrong baseline: {record['id']} claims {start}-{end} but the mean on that window is {center:.3f}"
        )


def _check_proxy(record: dict) -> None:
    if record.get("semantics") != "proxy_reconstruction":
        raise Rejected(
            f"fake proxy observation: {record['id']} is {record.get('semantics')}, not a proxy reconstruction"
        )
    if record.get("used_as_thermometer") is True:
        raise Rejected(f"fake proxy observation: {record['id']} is marked as a thermometer")


def _check_split(campaign: dict) -> None:
    train = set(int(year) for year in campaign["train_years"])
    holdout = set(int(year) for year in campaign["holdout_years"])
    leaked = sorted(train & holdout)
    if leaked:
        raise Rejected(f"member leakage: holdout years appear in training: {leaked[:8]}")
    if not holdout or not train:
        raise Rejected("member leakage: train or holdout year list is empty")


def _check_volcanic_sign(record: dict) -> None:
    year = int(record["year"])
    value = None
    for item_year, item_value in zip(record["years"], record["values"], strict=True):
        if int(item_year) == year:
            value = None if item_value is None else float(item_value)
            break
    if value is None:
        raise Rejected(f"wrong forcing sign: volcanic series has no finite value in {year}")
    # IGCC volcanic ERF is negative during large eruptions. A flipped series is positive.
    if value >= 0.0:
        raise Rejected(f"wrong forcing sign: volcanic ERF in {year} is {value:.3f}, expected negative")


def _check_scenario_units(record: dict) -> None:
    unit = record["unit"]
    values = [float(value) for value in record["values"] if value is not None and math.isfinite(float(value))]
    if not values:
        raise Rejected("modified scenario units: scenario has no finite values")
    scale = _mean([abs(value) for value in values])
    absolute_names = {"K", "degC"}
    anomaly_names = {"K_anomaly", "degC_anomaly"}
    if unit in absolute_names and scale < 40.0:
        raise Rejected(
            f"modified scenario units: {record['id']} is labeled {unit} but the values are anomaly-scale ({scale:.2f})"
        )
    if unit in anomaly_names and scale > 40.0:
        raise Rejected(
            f"modified scenario units: {record['id']} is labeled {unit} but the values are absolute-scale ({scale:.2f})"
        )
    if unit not in absolute_names and unit not in anomaly_names:
        raise Rejected(f"modified scenario units: {record['id']} has unrecognized unit {unit}")


def verify_document(document: dict, *, expected_commit: str | None = None) -> None:
    if document.get("format") != "forcing-atlas-evidence/1":
        raise Rejected(f"source mismatch: format is {document.get('format')}")
    if expected_commit is not None and document.get("source_commit") != expected_commit:
        raise Rejected(
            f"source mismatch: evidence commit {document.get('source_commit')} != {expected_commit}"
        )
    for record in document["anomaly_records"]:
        _check_baseline(record)
    for record in document["proxy_records"]:
        _check_proxy(record)
    _check_split(document["campaign"])
    _check_volcanic_sign(document["volcanic"])
    for record in document["scenarios"]:
        _check_scenario_units(record)
    # Recompute the reported ensemble RMSE from the stored per-seed scores.
    # A stored pass boolean is ignored.
    seeds = document["campaign"]["seed_holdout_rmse_k"]
    if len(seeds) < 3:
        raise Rejected("campaign stored fewer than three seed scores")
    recomputed = sum(float(value) for value in seeds) / len(seeds)
    reported = float(document["campaign"]["mean_seed_holdout_rmse_k"])
    if abs(recomputed - reported) > 1.0e-6:
        raise Rejected(
            f"campaign mean RMSE {reported} does not match the recomputed seed mean {recomputed}"
        )


def verify_path(path: Path, *, expected_commit: str | None = None) -> None:
    document = json.loads(path.read_text())
    verify_document(document, expected_commit=expected_commit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forcing-atlas-verify")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expect-commit", default=None)
    args = parser.parse_args(argv)
    try:
        verify_path(Path(args.evidence), expected_commit=args.expect_commit)
    except Rejected as exc:
        print(exc.reason)
        return 1
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"corrupt evidence: {exc}")
        return 1
    print("verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
