"""The verifier rejects semantic damage even when the file is otherwise well formed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _document() -> dict:
    years = list(range(1961, 1991))
    return {
        "format": "forcing-atlas-evidence/1",
        "source_commit": "abc",
        "anomaly_records": [
            {
                "id": "hadcrut",
                "baseline_start": 1961,
                "baseline_end": 1990,
                "years": years,
                "values": [0.0 for _ in years],
            }
        ],
        "proxy_records": [
            {
                "id": "pages",
                "semantics": "proxy_reconstruction",
                "used_as_thermometer": False,
                "years": [1, 2],
                "values": [-0.2, -0.1],
            }
        ],
        "campaign": {
            "train_years": [1850, 1994],
            "holdout_years": [1995, 2023],
            "seed_holdout_rmse_k": [0.4, 0.5, 0.6],
            "mean_seed_holdout_rmse_k": 0.5,
        },
        "volcanic": {"year": 1992, "years": [1991, 1992], "values": [-0.4, -1.5]},
        "scenarios": [{"id": "stabilize", "unit": "degC_anomaly", "values": [0.2, 0.8, 1.1]}],
    }


def _run(path: Path, expect: str | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "forcing_atlas_verify", "--evidence", str(path)]
    if expect is not None:
        command.extend(["--expect-commit", expect])
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_valid_evidence_passes(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_document()))
    completed = _run(path, "abc")
    assert completed.returncode == 0
    assert "verified" in completed.stdout


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        ("baseline", "wrong baseline"),
        ("proxy", "fake proxy"),
        ("leak", "member leakage"),
        ("sign", "wrong forcing sign"),
        ("units", "modified scenario units"),
        ("commit", "source mismatch"),
    ],
)
def test_semantic_controls_are_rejected(tmp_path: Path, mutate: str, fragment: str) -> None:
    document = _document()
    expect = "abc"
    if mutate == "baseline":
        document["anomaly_records"][0]["values"] = [1.0 for _ in document["anomaly_records"][0]["years"]]
    elif mutate == "proxy":
        document["proxy_records"][0]["semantics"] = "observation"
    elif mutate == "leak":
        document["campaign"]["train_years"].append(1995)
    elif mutate == "sign":
        document["volcanic"]["values"] = [0.4, 1.5]
    elif mutate == "units":
        document["scenarios"][0]["unit"] = "K"
    elif mutate == "commit":
        expect = "def"
    path = tmp_path / f"{mutate}.json"
    path.write_text(json.dumps(document))
    completed = _run(path, expect)
    assert completed.returncode != 0
    assert fragment in completed.stdout


def test_corrupt_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{")
    completed = _run(path)
    assert completed.returncode != 0
    assert "corrupt" in completed.stdout
