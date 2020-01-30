"""Time axes, bounds, and calendar-aware annualization.

A time value is a count of days or hours since an epoch on a named calendar.
Missing samples stay NaN. Annualization weights by the length of each month
in that calendar and refuses to treat a gap as zero. A minimum coverage
fraction decides whether the year is reported or left missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.calendars import (
    CalendarKind,
    CivilDate,
    date_to_day_number,
    day_number_to_date,
    days_in_year,
    month_end_exclusive,
    month_length,
    parse_calendar,
    parse_iso_date,
)
from forcing_atlas.errors import CalendarError, MissingDataError


_UNIT_TO_DAYS = {
    "days": 1.0,
    "day": 1.0,
    "hours": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "minutes": 1.0 / (24.0 * 60.0),
    "minute": 1.0 / (24.0 * 60.0),
    "seconds": 1.0 / 86400.0,
    "second": 1.0 / 86400.0,
}


@dataclass(frozen=True, slots=True)
class TimeUnitSpec:
    """Parsed form of strings such as `days since 1850-01-01 00:00:00`."""

    unit: str
    days_per_unit: float
    epoch: CivilDate
    raw: str


def parse_time_unit(text: str) -> TimeUnitSpec:
    raw = " ".join(text.strip().split())
    lowered = raw.lower()
    if " since " not in lowered:
        raise CalendarError(f"time unit {text!r} has no 'since' clause")
    unit_name, rest = lowered.split(" since ", 1)
    unit_name = unit_name.strip()
    if unit_name in {"months", "month", "years", "year"}:
        raise CalendarError(
            f"refusing unbounded '{unit_name} since' encoding in {text!r}. "
            "Provide day-based units or explicit bounds."
        )
    if unit_name not in _UNIT_TO_DAYS:
        raise CalendarError(f"unsupported time unit {unit_name!r} in {text!r}")
    date_token = rest.split()[0]
    # Hours and minutes after the date are accepted and required to be zero.
    # A nonzero clock offset would change the day boundary and is rejected
    # until a caller has a real need for it.
    clock = rest.split()[1:]
    if clock:
        clock_text = clock[0]
        parts = clock_text.split(":")
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise CalendarError(f"illegal clock in time unit {text!r}") from exc
        if any(number != 0.0 for number in numbers):
            raise CalendarError(
                f"nonzero clock offset in {text!r} is not supported; use a midnight epoch"
            )
    epoch = parse_iso_date(date_token)
    return TimeUnitSpec(unit_name, _UNIT_TO_DAYS[unit_name], epoch, raw)


@dataclass(slots=True)
class TimeAxis:
    """Numeric time coordinates on one calendar.

    `values` are in `spec` units since the epoch. Optional `bounds` has shape
    (n, 2) in the same units and is interpreted as [start, end).
    """

    values: np.ndarray
    spec: TimeUnitSpec
    calendar: CalendarKind
    bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1:
            raise CalendarError("time values must be one-dimensional")
        if np.any(~np.isfinite(self.values)):
            raise CalendarError("time coordinates must be finite; missing times are not a time axis")
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds, dtype=np.float64)
            if self.bounds.shape != (self.values.size, 2):
                raise CalendarError(
                    f"bounds shape {self.bounds.shape} does not match values {self.values.size}"
                )
            if np.any(~np.isfinite(self.bounds)):
                raise CalendarError("time bounds must be finite")
            if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
                raise CalendarError("each time bound must satisfy end > start")
        self.spec.epoch.validate(self.calendar)

    def __len__(self) -> int:
        return int(self.values.size)

    def day_numbers(self) -> np.ndarray:
        return self.values * self.spec.days_per_unit

    def bound_day_numbers(self) -> np.ndarray | None:
        if self.bounds is None:
            return None
        return self.bounds * self.spec.days_per_unit

    def to_dates(self) -> list[CivilDate]:
        epoch = self.spec.epoch
        numbers = self.day_numbers()
        dates: list[CivilDate] = []
        for number in numbers:
            if not float(number).is_integer():
                raise CalendarError(
                    "refusing to decode a fractional day as a civil date; use bounds for sub-day axes"
                )
            dates.append(day_number_to_date(int(number), epoch, self.calendar))
        return dates

    def years_if_annual(self) -> np.ndarray:
        """Return integer years when each sample sits on 1 January or carries a one-year bound.

        Model branch years are still calendar years of that model. They are not
        relabeled onto the historical era.
        """

        dates = self.to_dates()
        if self.bounds is None:
            years = []
            for date in dates:
                if date.month != 1 or date.day != 1:
                    raise CalendarError(
                        "annual year labels require January 1 stamps or explicit year bounds"
                    )
                years.append(date.year)
            return np.asarray(years, dtype=np.int64)
        bound_days = self.bound_day_numbers()
        assert bound_days is not None
        years = []
        for index, date in enumerate(dates):
            start = day_number_to_date(int(round(bound_days[index, 0])), self.spec.epoch, self.calendar)
            end = day_number_to_date(int(round(bound_days[index, 1])), self.spec.epoch, self.calendar)
            width = date_to_day_number(end, start, self.calendar)
            if width != days_in_year(start.year, self.calendar):
                raise CalendarError(
                    f"bound width {width} is not one {self.calendar.value} year starting {start.year}"
                )
            years.append(start.year)
        return np.asarray(years, dtype=np.int64)


def monthly_axis(start: CivilDate, count: int, calendar: CalendarKind) -> TimeAxis:
    """Month-start stamps with [month_start, next_month_start) bounds."""

    if count < 1:
        raise CalendarError("monthly axis needs at least one month")
    start.validate(calendar)
    values = np.empty(count, dtype=np.float64)
    bounds = np.empty((count, 2), dtype=np.float64)
    cursor = start
    for index in range(count):
        if cursor.day != 1:
            raise CalendarError("monthly_axis start must be the first day of a month")
        month_end = month_end_exclusive(cursor.year, cursor.month, calendar)
        values[index] = float(date_to_day_number(cursor, start, calendar))
        end_number = date_to_day_number(month_end, start, calendar)
        bounds[index, 0] = values[index]
        bounds[index, 1] = float(end_number)
        cursor = month_end
    spec = TimeUnitSpec("days", 1.0, start, f"days since {start.year:04d}-{start.month:02d}-{start.day:02d}")
    return TimeAxis(values, spec, calendar, bounds)


def annualize_monthly(
    values: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    calendar: CalendarKind,
    *,
    min_coverage: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-free annual means of a monthly series.

    Weights are month lengths on `calendar`. A missing month does not enter
    the sum. If the available weight is below `min_coverage` of the year's
    length, the annual value is NaN. Coverage is the fraction of the year's
    days that were present.
    """

    if not 0.0 < min_coverage <= 1.0:
        raise CalendarError("min_coverage must be in (0, 1]")
    series = np.asarray(values, dtype=np.float64)
    year_arr = np.asarray(years, dtype=np.int64)
    month_arr = np.asarray(months, dtype=np.int64)
    if series.shape != year_arr.shape or series.shape != month_arr.shape:
        raise CalendarError("values, years, and months must share a shape")
    if series.ndim != 1:
        raise CalendarError("annualize_monthly expects a flat series; reduce space first")
    if year_arr.size == 0:
        raise MissingDataError("cannot annualize an empty monthly series")

    order = np.lexsort((month_arr, year_arr))
    unique_years = np.unique(year_arr)
    out_years = unique_years.astype(np.int64)
    out_values = np.full(out_years.shape, np.nan, dtype=np.float64)
    out_coverage = np.full(out_years.shape, np.nan, dtype=np.float64)
    grouped: dict[int, list[tuple[int, float]]] = {int(year): [] for year in out_years}
    for index in order:
        grouped[int(year_arr[index])].append((int(month_arr[index]), float(series[index])))
    for slot, year in enumerate(out_years.tolist()):
        samples = grouped[int(year)]
        seen: dict[int, float] = {}
        for month, sample in samples:
            if month < 1 or month > 12:
                raise CalendarError(f"month {month} is illegal")
            if month in seen:
                raise CalendarError(f"duplicate month {year}-{month:02d}")
            seen[month] = sample
        year_days = days_in_year(int(year), calendar)
        weighted = 0.0
        weight = 0
        for month, sample in seen.items():
            if not np.isfinite(sample):
                continue
            length = month_length(int(year), month, calendar)
            weighted += sample * length
            weight += length
        coverage = weight / year_days
        out_coverage[slot] = coverage
        if weight > 0 and coverage + 1.0e-12 >= min_coverage:
            out_values[slot] = weighted / weight
    return out_years, out_values, out_coverage


def align_annual(
    left_years: np.ndarray,
    left_values: np.ndarray,
    right_years: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner join on integer year. No interpolation and no zero fill."""

    left_y = np.asarray(left_years, dtype=np.int64)
    right_y = np.asarray(right_years, dtype=np.int64)
    left_v = np.asarray(left_values, dtype=np.float64)
    right_v = np.asarray(right_values, dtype=np.float64)
    if left_y.shape != left_v.shape or right_y.shape != right_v.shape:
        raise CalendarError("years and values must pair")
    if left_y.size == 0 or right_y.size == 0:
        raise MissingDataError("cannot align an empty annual series")
    right_map = {int(year): float(value) for year, value in zip(right_y.tolist(), right_v.tolist(), strict=True)}
    years: list[int] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    for year, value in zip(left_y.tolist(), left_v.tolist(), strict=True):
        if int(year) not in right_map:
            continue
        years.append(int(year))
        a_vals.append(float(value))
        b_vals.append(right_map[int(year)])
    if not years:
        raise MissingDataError("annual series have no years in common")
    return (
        np.asarray(years, dtype=np.int64),
        np.asarray(a_vals, dtype=np.float64),
        np.asarray(b_vals, dtype=np.float64),
    )


def window_mean(years: np.ndarray, values: np.ndarray, start_year: int, end_year: int, *, min_fraction: float = 0.8) -> float:
    """Mean of finite annual values inside an inclusive year window.

    The window must contain at least `min_fraction` of its years as finite
    samples. Otherwise the baseline is unsupported and the caller must not
    invent a climatology.
    """

    if end_year < start_year:
        raise CalendarError("window ends before it starts")
    if not 0.0 < min_fraction <= 1.0:
        raise CalendarError("min_fraction must be in (0, 1]")
    year_arr = np.asarray(years, dtype=np.int64)
    value_arr = np.asarray(values, dtype=np.float64)
    expected = end_year - start_year + 1
    mask = (year_arr >= start_year) & (year_arr <= end_year) & np.isfinite(value_arr)
    count = int(np.count_nonzero(mask))
    if count / expected < min_fraction:
        raise MissingDataError(
            f"baseline window {start_year}-{end_year} has {count} of {expected} finite years, "
            f"below the required fraction {min_fraction}"
        )
    return float(np.mean(value_arr[mask]))


def decode_calendar_name(name: str) -> CalendarKind:
    return parse_calendar(name)


def month_weight_vector(year: int, months: np.ndarray, calendar: CalendarKind) -> np.ndarray:
    """Day weights for an explicit list of months in one year."""

    out = np.empty(len(months), dtype=np.float64)
    for index, month in enumerate(np.asarray(months, dtype=np.int64).tolist()):
        out[index] = float(month_length(year, int(month), calendar))
    return out


def span_days(start: CivilDate, end: CivilDate, calendar: CalendarKind) -> int:
    """Days in the half-open interval [start, end)."""

    width = date_to_day_number(end, start, calendar)
    if width <= 0:
        raise CalendarError("span end must be after start")
    return width


def demonstrate_leap_effect(year: int) -> dict[str, int]:
    """Day counts a caller can show when explaining why calendars disagree."""

    from forcing_atlas.calendars import CalendarKind as Kind

    return {
        kind.value: days_in_year(year, kind)
        for kind in (Kind.PROLEPTIC_GREGORIAN, Kind.NOLEAP, Kind.ALL_LEAP, Kind.DAY_360)
    }
