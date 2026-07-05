"""The German economic calendar.

Payment behaviour is strongly periodic, and the periods are *institutional*
rather than arithmetic: salaries land on the last banking day, rent on the
first, and TARGET2 closes on days that depend on when Easter falls. Getting this
right is what makes the monthly fan-out of an employer's payroll look like
payroll — and that matters, because payroll fan-out is the confusable twin of
mule fan-out, and a model that has never seen a real one will flag every
employer in Germany on the 30th.

Under SCT Inst settlement is 24/7/365, so payments *clear* on holidays. What the
calendar governs is when people and businesses *initiate* them.
"""

from __future__ import annotations

import datetime as _dt
import functools
from typing import Final

# German federal holidays, defined either by fixed date or by an offset from
# Easter Sunday. The Easter-relative ones are why a computus is needed at all.
_FIXED_HOLIDAYS: Final[tuple[tuple[int, int, str], ...]] = (
    (1, 1, "Neujahr"),
    (5, 1, "Tag der Arbeit"),
    (10, 3, "Tag der Deutschen Einheit"),
    (12, 25, "1. Weihnachtsfeiertag"),
    (12, 26, "2. Weihnachtsfeiertag"),
)

_EASTER_OFFSETS: Final[tuple[tuple[int, str], ...]] = (
    (-2, "Karfreitag"),
    (1, "Ostermontag"),
    (39, "Christi Himmelfahrt"),
    (50, "Pfingstmontag"),
)


@functools.lru_cache(maxsize=64)
def easter_sunday(year: int) -> _dt.date:
    """Easter Sunday for *year* in the Gregorian calendar.

    Anonymous Gregorian computus (Meeus/Jones/Butcher). Written out rather than
    pulled from a holiday library because it is fifteen lines, has no edge cases
    we care about, and adding a dependency for it would be worse.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * m + 114, 31)
    return _dt.date(year, month, day + 1)


@functools.lru_cache(maxsize=64)
def federal_holidays(year: int) -> dict[_dt.date, str]:
    """Every nationwide German public holiday in *year*, as ``{date: name}``.

    Nationwide only. Fronleichnam, Allerheiligen, Reformationstag and the rest
    are Länder-specific, and modelling sixteen regional calendars would add
    noise without adding signal — the payroll and rent periodicities we care
    about are driven by the federal set.
    """
    holidays = {_dt.date(year, month, day): name for month, day, name in _FIXED_HOLIDAYS}
    easter = easter_sunday(year)
    for offset, name in _EASTER_OFFSETS:
        holidays[easter + _dt.timedelta(days=offset)] = name
    return holidays


def is_holiday(day: _dt.date) -> bool:
    """True iff *day* is a nationwide German public holiday."""
    return day in federal_holidays(day.year)


def is_weekend(day: _dt.date) -> bool:
    return day.weekday() >= 5


def is_business_day(day: _dt.date) -> bool:
    """A day on which businesses initiate payments and payroll can land."""
    return not is_weekend(day) and not is_holiday(day)


def previous_business_day(day: _dt.date) -> _dt.date:
    """The latest business day at or before *day*."""
    current = day
    while not is_business_day(current):
        current -= _dt.timedelta(days=1)
    return current


def next_business_day(day: _dt.date) -> _dt.date:
    """The earliest business day at or after *day*."""
    current = day
    while not is_business_day(current):
        current += _dt.timedelta(days=1)
    return current


def last_business_day_of_month(year: int, month: int) -> _dt.date:
    """When German payroll lands.

    Not simply "the 30th": if the month ends on a Saturday, salaries land on the
    Friday, and the whole population's income arrives a day early. That shift is
    visible in the data and a model that assumes a fixed day-of-month will
    mis-handle it every few months.
    """
    first_of_next = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return previous_business_day(first_of_next - _dt.timedelta(days=1))


def business_days_between(start: _dt.date, end: _dt.date) -> int:
    """Count business days in ``[start, end)``."""
    count, current = 0, start
    while current < end:
        if is_business_day(current):
            count += 1
        current += _dt.timedelta(days=1)
    return count


def date_range(start: _dt.date, days: int) -> list[_dt.date]:
    """``days`` consecutive dates beginning at *start*."""
    return [start + _dt.timedelta(days=offset) for offset in range(days)]


def month_key(day: _dt.date) -> tuple[int, int]:
    """``(year, month)`` — used to fire monthly obligations exactly once."""
    return (day.year, day.month)


# --------------------------------------------------------------- seasonality


def seasonal_spending_multiplier(day: _dt.date) -> float:
    """Consumer spending intensity relative to an average day.

    Two effects the data would look wrong without: the December retail peak,
    and the mid-January trough after it. Both are large enough that a drift
    monitor would fire on them, which makes them a useful honest test of whether
    the drift monitoring can distinguish seasonality from genuine change.
    """
    if day.month == 12:
        if day.day <= 23:
            # Ramp into Christmas rather than a step change.
            return 1.15 + 0.55 * (day.day / 23.0)
        return 0.55  # the days between Christmas and New Year are dead
    if day.month == 1 and day.day <= 15:
        return 0.82  # post-Christmas trough
    if day.month in (7, 8):
        return 1.06  # holiday season: travel and dining up, groceries down
    return 1.0
