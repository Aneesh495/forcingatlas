"""Four-pool carbon impulse response.

The coefficients are the Joos et al. (2013) multi-model mean, declared in
`docs/NUMERICAL_CONTRACTS.md`. The integration is the exact solution of the
linear pools for a piecewise-constant emission rate. It is not FaIR and it
does not add a temperature feedback onto the sinks. Turning that feedback on
would be a different model and is not implied here.

Pool 0 is the permanent fraction. Pools 1 to 3 decay. The state is excess
carbon in GtC relative to whatever reference the caller started from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forcing_atlas.constants import GTC_PER_PPM_CO2
from forcing_atlas.errors import ContractError


# Joos et al. (2013), multi-model mean airborne fraction.
AIRBORNE_PERMANENT = 0.2173
AIRBORNE_DECAY = (0.2240, 0.2824, 0.2763)
DECAY_YEARS = (394.4, 36.54, 4.304)


def coefficient_sum() -> float:
    return AIRBORNE_PERMANENT + float(sum(AIRBORNE_DECAY))


@dataclass(slots=True)
class CarbonState:
    """Excess carbon in GtC. Index 0 is the permanent pool."""

    pools_gtc: np.ndarray

    def __post_init__(self) -> None:
        self.pools_gtc = np.asarray(self.pools_gtc, dtype=np.float64).reshape(4)
        if self.pools_gtc.shape != (4,):
            raise ContractError("carbon state needs four pools")

    @property
    def excess_gtc(self) -> float:
        return float(np.sum(self.pools_gtc))

    def copy(self) -> CarbonState:
        return CarbonState(self.pools_gtc.copy())


def zero_state() -> CarbonState:
    return CarbonState(np.zeros(4, dtype=np.float64))


def apply_pulse(state: CarbonState, mass_gtc: float) -> CarbonState:
    """Instantaneous addition of `mass_gtc` split by the airborne fractions.

    A negative mass is a removal. The pools may become negative if removals
    exceed the excess carbon. That is reported, not clipped, because clipping
    would hide an unphysical scenario.
    """

    if not np.isfinite(mass_gtc):
        raise ContractError("carbon pulse mass must be finite")
    pools = state.pools_gtc.copy()
    pools[0] += AIRBORNE_PERMANENT * mass_gtc
    for index, fraction in enumerate(AIRBORNE_DECAY, start=1):
        pools[index] += fraction * mass_gtc
    return CarbonState(pools)


def step_constant_emission(state: CarbonState, emission_gtc_per_year: float, dt_years: float) -> CarbonState:
    """Exact pool update for an emission rate held constant over `dt_years`.

    dC0/dt = a0 E
    dCi/dt = ai E - Ci / tau_i
    """

    if dt_years < 0.0 or not np.isfinite(dt_years):
        raise ContractError("carbon time step must be nonnegative and finite")
    if not np.isfinite(emission_gtc_per_year):
        raise ContractError("emission rate must be finite")
    if dt_years == 0.0:
        return state.copy()
    pools = state.pools_gtc.copy()
    pools[0] = pools[0] + AIRBORNE_PERMANENT * emission_gtc_per_year * dt_years
    for index, (fraction, tau) in enumerate(zip(AIRBORNE_DECAY, DECAY_YEARS, strict=True), start=1):
        decay = np.exp(-dt_years / tau)
        equilibrium = fraction * emission_gtc_per_year * tau
        pools[index] = pools[index] * decay + equilibrium * (1.0 - decay)
    return CarbonState(pools)


def integrate_emissions(
    years: np.ndarray,
    emissions_gtc_per_year: np.ndarray,
    *,
    state: CarbonState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate annual emission rates.

    `years` must be strictly increasing. The rate at index i is held from
    years[i] to years[i+1]. The returned excess is reported at each year,
    including the initial year, so it has the same length as `years`. The last
    rate has no following interval and does not move the state. Callers that
    need a final step should append one extra year.
    """

    time = np.asarray(years, dtype=np.float64)
    rate = np.asarray(emissions_gtc_per_year, dtype=np.float64)
    if time.shape != rate.shape or time.ndim != 1 or time.size < 2:
        raise ContractError("carbon integration needs at least two paired years")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ContractError("carbon years must be finite and strictly increasing")
    if np.any(~np.isfinite(rate)):
        raise ContractError("carbon emissions contain a non-finite value. Missing emissions are not zero.")
    current = zero_state() if state is None else state.copy()
    excess = np.empty(time.shape, dtype=np.float64)
    excess[0] = current.excess_gtc
    for index in range(time.size - 1):
        current = step_constant_emission(current, float(rate[index]), float(time[index + 1] - time[index]))
        excess[index + 1] = current.excess_gtc
    return time, excess


def excess_gtc_to_ppm(excess_gtc: np.ndarray, reference_ppm: float, *, gtc_per_ppm: float = GTC_PER_PPM_CO2) -> np.ndarray:
    if reference_ppm <= 0.0 or gtc_per_ppm <= 0.0:
        raise ContractError("CO2 reference and GtC per ppm must be positive")
    return reference_ppm + np.asarray(excess_gtc, dtype=np.float64) / gtc_per_ppm


def impulse_response(time_years: np.ndarray) -> np.ndarray:
    """Airborne fraction remaining after a unit pulse. G(inf) = a0."""

    time = np.asarray(time_years, dtype=np.float64)
    if np.any(time < 0.0):
        raise ContractError("impulse response time cannot be negative")
    remaining = np.full(time.shape, AIRBORNE_PERMANENT, dtype=np.float64)
    for fraction, tau in zip(AIRBORNE_DECAY, DECAY_YEARS, strict=True):
        remaining = remaining + fraction * np.exp(-time / tau)
    return remaining
