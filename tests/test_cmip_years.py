"""CMIP years come from the time coordinate, including numpy datetimes."""

from __future__ import annotations

import numpy as np

from forcing_atlas.cmip_means import years_from_time


def test_numpy_datetime64_years() -> None:
    stamps = np.array(["2015-07-01", "2100-07-01"], dtype="datetime64[ns]")
    assert years_from_time(stamps).tolist() == [2015, 2100]


class _Stamp:
    def __init__(self, year: int) -> None:
        self.year = year


def test_cftime_like_years() -> None:
    assert years_from_time(np.array([_Stamp(5366), _Stamp(5451)], dtype=object)).tolist() == [5366, 5451]
