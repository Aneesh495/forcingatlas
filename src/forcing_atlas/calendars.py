"""Calendar arithmetic for climate time axes.

Supported calendars:

- proleptic_gregorian, also accepted under the aliases gregorian and standard
- noleap, also 365_day
- all_leap, also 366_day
- 360_day

The civil date uses astronomical year numbering, so year 0 exists and is a
Gregorian leap year. There is no time zone. A day number is an integer count
of midnights relative to an epoch date, and the epoch date itself is day 0.

Month-based time units are rejected by the time-axis parser unless the file
provides bounds. Month length is not a constant on the Gregorian calendar, and
a "months since" encoding hides that choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from forcing_atlas.errors import CalendarError


class CalendarKind(Enum):
    PROLEPTIC_GREGORIAN = "proleptic_gregorian"
    NOLEAP = "noleap"
    ALL_LEAP = "all_leap"
    DAY_360 = "360_day"


_ALIASES: dict[str, CalendarKind] = {
    "proleptic_gregorian": CalendarKind.PROLEPTIC_GREGORIAN,
    "gregorian": CalendarKind.PROLEPTIC_GREGORIAN,
    "standard": CalendarKind.PROLEPTIC_GREGORIAN,
    "noleap": CalendarKind.NOLEAP,
    "365_day": CalendarKind.NOLEAP,
    "365": CalendarKind.NOLEAP,
    "all_leap": CalendarKind.ALL_LEAP,
    "366_day": CalendarKind.ALL_LEAP,
    "366": CalendarKind.ALL_LEAP,
    "360_day": CalendarKind.DAY_360,
    "360": CalendarKind.DAY_360,
}

_COMMON_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def parse_calendar(name: str) -> CalendarKind:
    key = name.strip().lower().replace(" ", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise CalendarError(
            f"unsupported calendar {name!r}. "
            "Supported names: proleptic_gregorian, noleap, all_leap, 360_day."
        ) from exc


def is_gregorian_leap(year: int) -> bool:
    """Proleptic Gregorian leap rule, including year 0."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_length(year: int, month: int, calendar: CalendarKind) -> int:
    if month < 1 or month > 12:
        raise CalendarError(f"month {month} is outside 1..12 on {calendar.value}")
    if calendar is CalendarKind.DAY_360:
        return 30
    length = _COMMON_LENGTHS[month - 1]
    if month == 2:
        if calendar is CalendarKind.ALL_LEAP:
            return 29
        if calendar is CalendarKind.PROLEPTIC_GREGORIAN and is_gregorian_leap(year):
            return 29
        return 28
    return length


def days_in_year(year: int, calendar: CalendarKind) -> int:
    if calendar is CalendarKind.DAY_360:
        return 360
    if calendar is CalendarKind.NOLEAP:
        return 365
    if calendar is CalendarKind.ALL_LEAP:
        return 366
    return 366 if is_gregorian_leap(year) else 365


@dataclass(frozen=True, slots=True)
class CivilDate:
    year: int
    month: int
    day: int

    def key(self) -> tuple[int, int, int]:
        return (self.year, self.month, self.day)

    def validate(self, calendar: CalendarKind) -> None:
        length = month_length(self.year, self.month, calendar)
        if self.day < 1 or self.day > length:
            raise CalendarError(
                f"{self.year:04d}-{self.month:02d}-{self.day:02d} is not a valid "
                f"{calendar.value} date (month length {length})"
            )


def _ordinal_from_year_zero(date: CivilDate, calendar: CalendarKind) -> int:
    """Days from 0000-01-01 to `date`. The origin date has ordinal 0."""

    date.validate(calendar)
    year = date.year
    if year >= 0:
        days = 0
        for cursor in range(0, year):
            days += days_in_year(cursor, calendar)
    else:
        days = 0
        for cursor in range(year, 0):
            days -= days_in_year(cursor, calendar)
    for month in range(1, date.month):
        days += month_length(year, month, calendar)
    days += date.day - 1
    return days


def _date_from_ordinal(ordinal: int, calendar: CalendarKind) -> CivilDate:
    remaining = ordinal
    year = 0
    if remaining >= 0:
        while True:
            width = days_in_year(year, calendar)
            if remaining < width:
                break
            remaining -= width
            year += 1
    else:
        while remaining < 0:
            year -= 1
            remaining += days_in_year(year, calendar)
    month = 1
    while True:
        width = month_length(year, month, calendar)
        if remaining < width:
            break
        remaining -= width
        month += 1
        if month > 12:
            raise CalendarError("ordinal decode walked past December")
    return CivilDate(year, month, remaining + 1)


def date_to_day_number(date: CivilDate, epoch: CivilDate, calendar: CalendarKind) -> int:
    """Integer days from `epoch` to `date` on `calendar`. Negative if `date` is earlier."""

    return _ordinal_from_year_zero(date, calendar) - _ordinal_from_year_zero(epoch, calendar)


def day_number_to_date(day_number: int, epoch: CivilDate, calendar: CalendarKind) -> CivilDate:
    origin = _ordinal_from_year_zero(epoch, calendar)
    return _date_from_ordinal(origin + int(day_number), calendar)


def add_days(date: CivilDate, days: int, calendar: CalendarKind) -> CivilDate:
    number = date_to_day_number(date, CivilDate(0, 1, 1), calendar)
    return day_number_to_date(number + int(days), CivilDate(0, 1, 1), calendar)


def add_months(date: CivilDate, months: int, calendar: CalendarKind) -> CivilDate:
    """Move by calendar months, clamping to the destination month length.

    Clamping is explicit: 31 January plus one month is 28 or 29 February, not
    a spill into March. Callers that need an end-exclusive bound should use
    `month_end_exclusive` instead of adding a month to a day-of-month.
    """

    date.validate(calendar)
    index = date.year * 12 + (date.month - 1) + int(months)
    year = index // 12
    month = index % 12 + 1
    # Python floor division already handles negative indexes for year 0 and below
    # only if index is computed carefully. For negative months, correct it.
    if month < 1:
        # unreachable with the formula above for Python's modulo sign, but kept
        # as a guard if a future change alters the arithmetic.
        raise CalendarError("month arithmetic produced an illegal month")
    length = month_length(year, month, calendar)
    day = min(date.day, length)
    return CivilDate(year, month, day)


def month_start(year: int, month: int, calendar: CalendarKind) -> CivilDate:
    start = CivilDate(year, month, 1)
    start.validate(calendar)
    return start


def month_end_exclusive(year: int, month: int, calendar: CalendarKind) -> CivilDate:
    """First day of the next month. The month interval is [start, end)."""

    return add_months(month_start(year, month, calendar), 1, calendar)


def day_of_year(date: CivilDate, calendar: CalendarKind) -> int:
    """1-based day of year."""

    date.validate(calendar)
    ordinal = 1
    for month in range(1, date.month):
        ordinal += month_length(date.year, month, calendar)
    return ordinal + date.day - 1


def year_month_day_from_day_of_year(year: int, day_of_year_number: int, calendar: CalendarKind) -> CivilDate:
    if day_of_year_number < 1 or day_of_year_number > days_in_year(year, calendar):
        raise CalendarError(
            f"day of year {day_of_year_number} is outside {calendar.value} year {year}"
        )
    remaining = day_of_year_number
    for month in range(1, 13):
        width = month_length(year, month, calendar)
        if remaining <= width:
            return CivilDate(year, month, remaining)
        remaining -= width
    raise CalendarError("day of year decode failed")


def parse_iso_date(text: str) -> CivilDate:
    """Parse YYYY-MM-DD or Y-M-D. The year may be negative, written as -YYYY-MM-DD."""

    raw = text.strip()
    if not raw:
        raise CalendarError("empty date")
    negative = raw.startswith("-")
    body = raw[1:] if negative else raw
    parts = body.split("-")
    if len(parts) != 3:
        raise CalendarError(f"date {text!r} is not YYYY-MM-DD")
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    except ValueError as exc:
        raise CalendarError(f"date {text!r} is not numeric") from exc
    if negative:
        year = -year
    return CivilDate(year, month, day)


def format_iso_date(date: CivilDate) -> str:
    sign = "-" if date.year < 0 else ""
    return f"{sign}{abs(date.year):04d}-{date.month:02d}-{date.day:02d}"


def leap_day_present(year: int, calendar: CalendarKind) -> bool:
    return month_length(year, 2, calendar) == 29


def annual_weight_days(year: int, calendar: CalendarKind) -> tuple[int, ...]:
    """Month lengths for `year`. Annual means of monthly data use these weights.

    Equal month weights are correct only on the 360-day calendar. On every
    other calendar, February and the 31-day months must keep their lengths,
    including leap days when the calendar has them.
    """

    return tuple(month_length(year, month, calendar) for month in range(1, 13))
