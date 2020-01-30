"""Calendar engine against an independent Gregorian oracle and closed forms."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from forcing_atlas.calendars import (
    CalendarKind,
    CivilDate,
    add_months,
    date_to_day_number,
    day_number_to_date,
    days_in_year,
    month_length,
    parse_calendar,
    parse_iso_date,
)
from forcing_atlas.errors import CalendarError, MissingDataError
from forcing_atlas.timebase import (
    align_annual,
    annualize_monthly,
    monthly_axis,
    parse_time_unit,
    window_mean,
)


def _python_days(year: int, month: int, day: int, epoch: dt.date) -> int:
    return (dt.date(year, month, day) - epoch).days


def test_parse_aliases() -> None:
    assert parse_calendar("360_day") is CalendarKind.DAY_360
    assert parse_calendar("365_day") is CalendarKind.NOLEAP
    assert parse_calendar("standard") is CalendarKind.PROLEPTIC_GREGORIAN
    with pytest.raises(CalendarError):
        parse_calendar("julian")


def test_gregorian_matches_datetime_over_a_span() -> None:
    epoch = CivilDate(1850, 1, 1)
    py_epoch = dt.date(1850, 1, 1)
    probe = [
        (1850, 1, 1),
        (1852, 2, 29),
        (1900, 2, 28),
        (1900, 3, 1),
        (2000, 2, 29),
        (2000, 3, 1),
        (2024, 12, 31),
    ]
    for year, month, day in probe:
        got = date_to_day_number(CivilDate(year, month, day), epoch, CalendarKind.PROLEPTIC_GREGORIAN)
        expect = _python_days(year, month, day, py_epoch)
        assert got == expect
        back = day_number_to_date(got, epoch, CalendarKind.PROLEPTIC_GREGORIAN)
        assert (back.year, back.month, back.day) == (year, month, day)


def test_century_and_year_zero_leap_rules() -> None:
    assert month_length(1900, 2, CalendarKind.PROLEPTIC_GREGORIAN) == 28
    assert month_length(2000, 2, CalendarKind.PROLEPTIC_GREGORIAN) == 29
    assert month_length(0, 2, CalendarKind.PROLEPTIC_GREGORIAN) == 29
    assert days_in_year(1900, CalendarKind.PROLEPTIC_GREGORIAN) == 365
    assert days_in_year(2000, CalendarKind.PROLEPTIC_GREGORIAN) == 366


def test_noleap_closed_form() -> None:
    epoch = CivilDate(1850, 1, 1)
    date = CivilDate(1852, 3, 1)
    # 1850 and 1851 are full 365-day years, then January and February of 1852.
    expected = 365 + 365 + 31 + 28
    assert date_to_day_number(date, epoch, CalendarKind.NOLEAP) == expected
    with pytest.raises(CalendarError):
        CivilDate(1852, 2, 29).validate(CalendarKind.NOLEAP)


def test_360_day_closed_form() -> None:
    epoch = CivilDate(1, 1, 1)
    date = CivilDate(3, 2, 10)
    expected = 2 * 360 + 30 + 9
    assert date_to_day_number(date, epoch, CalendarKind.DAY_360) == expected
    with pytest.raises(CalendarError):
        CivilDate(3, 1, 31).validate(CalendarKind.DAY_360)
    assert days_in_year(3, CalendarKind.DAY_360) == 360
    assert days_in_year(3, CalendarKind.ALL_LEAP) == 366


def test_negative_day_number_round_trip() -> None:
    epoch = CivilDate(1850, 1, 1)
    earlier = CivilDate(1849, 12, 31)
    number = date_to_day_number(earlier, epoch, CalendarKind.PROLEPTIC_GREGORIAN)
    assert number == -1
    assert day_number_to_date(number, epoch, CalendarKind.PROLEPTIC_GREGORIAN) == earlier


def test_add_months_clamps_february() -> None:
    moved = add_months(CivilDate(1900, 1, 31), 1, CalendarKind.PROLEPTIC_GREGORIAN)
    assert (moved.year, moved.month, moved.day) == (1900, 2, 28)
    moved_leap = add_months(CivilDate(2000, 1, 31), 1, CalendarKind.PROLEPTIC_GREGORIAN)
    assert (moved_leap.month, moved_leap.day) == (2, 29)


def test_months_since_is_rejected() -> None:
    with pytest.raises(CalendarError):
        parse_time_unit("months since 1850-01-01")
    spec = parse_time_unit("days since 1850-01-01 00:00:00")
    assert spec.epoch == CivilDate(1850, 1, 1)
    assert spec.days_per_unit == 1.0


def test_monthly_axis_bounds_include_leap_day() -> None:
    axis = monthly_axis(CivilDate(2000, 1, 1), 3, CalendarKind.PROLEPTIC_GREGORIAN)
    widths = axis.bounds[:, 1] - axis.bounds[:, 0]
    assert widths.tolist() == [31.0, 29.0, 31.0]
    noleap = monthly_axis(CivilDate(2000, 1, 1), 3, CalendarKind.NOLEAP)
    assert (noleap.bounds[:, 1] - noleap.bounds[:, 0]).tolist() == [31.0, 28.0, 31.0]


def test_annualize_uses_month_lengths_and_preserves_gaps() -> None:
    years = np.repeat(np.array([1900]), 12)
    months = np.arange(1, 13)
    values = np.arange(1, 13, dtype=np.float64)
    out_years, out_values, coverage = annualize_monthly(
        values, years, months, CalendarKind.PROLEPTIC_GREGORIAN
    )
    weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
    expect = float(np.sum(values * weights) / np.sum(weights))
    assert int(out_years[0]) == 1900
    assert out_values[0] == pytest.approx(expect)
    assert coverage[0] == pytest.approx(1.0)

    equal = annualize_monthly(values, years, months, CalendarKind.DAY_360)
    assert equal[1][0] == pytest.approx(float(np.mean(values)))

    gappy = values.copy()
    gappy[1] = np.nan
    _, gappy_values, gappy_cov = annualize_monthly(
        gappy, years, months, CalendarKind.PROLEPTIC_GREGORIAN, min_coverage=1.0
    )
    assert np.isnan(gappy_values[0])
    assert gappy_cov[0] == pytest.approx((365 - 28) / 365)


def test_missing_month_is_not_zero() -> None:
    years = np.array([1850, 1850])
    months = np.array([1, 2])
    values = np.array([10.0, np.nan])
    _, annual, _ = annualize_monthly(
        values, years, months, CalendarKind.NOLEAP, min_coverage=0.01
    )
    # If February were treated as zero, the mean would be pulled down.
    assert annual[0] == pytest.approx(10.0)


def test_align_is_an_inner_join() -> None:
    years, left, right = align_annual(
        np.array([1850, 1851, 1853]),
        np.array([1.0, np.nan, 3.0]),
        np.array([1851, 1852, 1853]),
        np.array([4.0, 5.0, 6.0]),
    )
    assert years.tolist() == [1851, 1853]
    assert left.tolist() == [np.nan, 3.0] or (np.isnan(left[0]) and left[1] == 3.0)
    assert right.tolist() == pytest.approx([4.0, 6.0])


def test_window_mean_requires_coverage() -> None:
    years = np.arange(1850, 1901)
    values = np.ones(years.shape)
    values[:40] = np.nan
    with pytest.raises(MissingDataError):
        window_mean(years, values, 1850, 1900, min_fraction=0.8)
    # 1850-1900 has 51 years. 40 missing leaves 11 finite, which fails 0.8.
    values = np.ones(years.shape)
    values[:5] = np.nan
    assert window_mean(years, values, 1850, 1900, min_fraction=0.8) == pytest.approx(1.0)


def test_iso_negative_year() -> None:
    date = parse_iso_date("-0001-12-31")
    assert date.year == -1
    assert date.month == 12
