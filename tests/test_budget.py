"""Disk cap behaviour on a temporary tree, not on the live checkout size."""

from __future__ import annotations

from pathlib import Path

import pytest

from forcing_atlas.budget import assert_room, assert_stage_budget, measure
from forcing_atlas.errors import BudgetError


def test_measure_counts_regular_files_and_skips_symlinks(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "series.csv").write_bytes(b"abc")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"0123456789")
    (data / "linked.bin").symlink_to(outside)
    report = measure(tmp_path)
    assert report.by_directory["data"] == 3


def test_assert_room_rejects_a_request_past_the_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forcing_atlas.budget.PROJECT_DISK_CAP_BYTES", 4)
    (tmp_path / "note.txt").write_bytes(b"abcd")
    with pytest.raises(BudgetError):
        assert_room(1, tmp_path)
    assert_room(0, tmp_path)


def test_stage_cap_is_tighter() -> None:
    assert_stage_budget(10, 20, "acquire")
    with pytest.raises(BudgetError):
        assert_stage_budget(21, 20, "acquire")
