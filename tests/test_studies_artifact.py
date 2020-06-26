"""The committed study file has a limit on every claim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_studies_name_their_limits() -> None:
    path = Path("results/studies.json")
    if not path.is_file():
        pytest.skip("results/studies.json is produced by forcing-atlas studies")
    studies = json.loads(path.read_text())
    assert len(studies) >= 10
    ids = [study["id"] for study in studies]
    assert "S03" in ids and "S07" in ids and "S11" in ids
    for study in studies:
        assert study["result"]
        assert study["limit"]
        assert study["sources"]
    proxy = next(study for study in studies if study["id"] == "S03")
    assert "thermometer" in proxy["limit"] or "not substituted" in proxy["limit"]
    attribution = next(study for study in studies if study["id"] == "S07")
    assert attribution["metrics"]["shuffled_r2"] < attribution["metrics"]["train_r2"]
