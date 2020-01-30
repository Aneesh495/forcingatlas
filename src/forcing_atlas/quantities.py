"""Array quantities that carry a unit, a species, and an optional baseline.

Arithmetic refuses to add different baselines, to add a GWP flux to a carbon
mass flux, or to turn an anomaly into an absolute temperature by adding 273.15.
Missing values stay NaN through every operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.errors import BaselineError, UnitError
from forcing_atlas.units import Accounting, Unit, convert_numeric, get_unit
from forcing_atlas.conversions import anomaly_unit_for_absolute


def _as_float(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.dtype != np.float64:
        array = array.astype(np.float64, copy=False)
    return array


@dataclass(frozen=True, slots=True)
class Quantity:
    values: np.ndarray
    unit: Unit
    baseline_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _as_float(self.values))
        if self.unit.accounting is Accounting.TEMPERATURE_ANOMALY:
            return
        if self.baseline_id is not None:
            raise BaselineError(
                f"baseline_id is only meaningful on a temperature anomaly, not {self.unit.name}"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def to(self, unit: Unit | str) -> Quantity:
        target = unit if isinstance(unit, Unit) else get_unit(unit)
        converted = convert_numeric(self.values, self.unit, target)
        return Quantity(converted, target, baseline_id=self.baseline_id)

    def _aligned(self, other: Quantity) -> Quantity:
        if self.unit == other.unit and self.baseline_id == other.baseline_id:
            return other
        converted = other.to(self.unit)
        if self.unit.accounting is Accounting.TEMPERATURE_ANOMALY:
            if self.baseline_id != other.baseline_id:
                raise BaselineError(
                    f"refusing to combine anomalies on baselines "
                    f"{self.baseline_id!r} and {other.baseline_id!r} without an explicit rebase"
                )
        return Quantity(converted.values, self.unit, baseline_id=self.baseline_id)

    def __add__(self, other: Quantity) -> Quantity:
        return _add(self, other, sign=1.0)

    def __sub__(self, other: Quantity) -> Quantity:
        return _add(self, other, sign=-1.0)

    def __neg__(self) -> Quantity:
        return Quantity(-self.values, self.unit, baseline_id=self.baseline_id)

    def scale(self, factor: float) -> Quantity:
        if not np.isfinite(factor):
            raise UnitError("scale factor must be finite")
        return Quantity(self.values * factor, self.unit, baseline_id=self.baseline_id)

    def mean_finite(self) -> float:
        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            raise UnitError(f"no finite values in {self.unit.name} quantity")
        return float(np.mean(finite))


def _add(left: Quantity, right: Quantity, sign: float) -> Quantity:
    if left.unit.accounting is Accounting.ABSOLUTE_TEMPERATURE and (
        right.unit.accounting is Accounting.ABSOLUTE_TEMPERATURE
    ):
        if sign > 0:
            raise UnitError("refusing to add two absolute temperatures")
        # Difference of absolutes is an anomaly. Baselines are not involved:
        # the result is a temperature difference, not an anomaly versus a window.
        left_k = left.to("K")
        right_k = right.to("K")
        delta = left_k.values + sign * right_k.values
        return Quantity(delta, get_unit("K_anomaly"), baseline_id=None)

    if (
        left.unit.accounting is Accounting.ABSOLUTE_TEMPERATURE
        and right.unit.accounting is Accounting.TEMPERATURE_ANOMALY
    ):
        anomaly_unit = anomaly_unit_for_absolute(left.unit)
        right_same = right.to(anomaly_unit)
        return Quantity(left.values + sign * right_same.values, left.unit, baseline_id=None)

    if (
        left.unit.accounting is Accounting.TEMPERATURE_ANOMALY
        and right.unit.accounting is Accounting.ABSOLUTE_TEMPERATURE
    ):
        raise UnitError(
            "add the anomaly to the absolute temperature, not the absolute to the anomaly, "
            "so the result kind is unambiguous"
        )

    aligned = left._aligned(right)
    return Quantity(left.values + sign * aligned.values, left.unit, baseline_id=left.baseline_id)


def anomaly_from_absolute(absolute: Quantity, climatology: Quantity, baseline_id: str) -> Quantity:
    """Subtract a climatology of the same absolute unit and tag the baseline.

    The climatology must already be the mean over `baseline_id`. This function
    does not recompute the window.
    """

    if not baseline_id:
        raise BaselineError("baseline_id is required")
    difference = absolute - climatology
    # absolute - absolute yields K_anomaly. Convert to the caller's anomaly twin if needed.
    if absolute.unit.name == "degC":
        difference = Quantity(difference.values, get_unit("degC_anomaly"))
    return Quantity(difference.values, difference.unit, baseline_id=baseline_id)


def rebase_anomaly(series: Quantity, window_mean: float, new_baseline_id: str) -> Quantity:
    """Shift an anomaly so the declared window has mean zero.

    `window_mean` is the mean of `series` over the new window, in the series
    unit. The caller computes it. Passing a mean from a different baseline
    without saying so is the caller's contract violation to avoid, and the new
    id is stored so later arithmetic can see the change.
    """

    if series.unit.accounting is not Accounting.TEMPERATURE_ANOMALY:
        raise UnitError("rebase applies to temperature anomalies")
    if not new_baseline_id:
        raise BaselineError("new baseline id is required")
    if not np.isfinite(window_mean):
        raise BaselineError("window mean must be finite")
    return Quantity(series.values - window_mean, series.unit, baseline_id=new_baseline_id)
