# Specification

This document is the contract the code is written against. A behavior that is not implemented is described as a contract, not as a feature that already runs. See `docs/BUILD_STATUS.md` for the current line between the two.

## Purpose

Forcing Atlas reads instrumental temperature products, a paleoclimate reconstruction, assessed forcing and heat-inventory tables, and a bounded set of source-model fields. It fits a two-reservoir energy-balance model, trains a small spatial emulator, and shows conditional pathways. The atlas and the command line expose the same exported fields.

A pathway is conditional on its forcing or emissions assumption. It is not a prediction of a policy, and it is not labeled as a world without climate policy. High forcing, lower forcing, stabilization, decline, and natural-only experiments answer different questions and are captioned that way.

## Semantic layers

Every dataset carries one of these semantics:

| Semantics | Meaning |
| --- | --- |
| observation | Instrumental gridded or station-derived field without statistical infilling |
| observational_analysis | Instrumental product that may infill unobserved cells. The infill is part of the product, not a measurement |
| proxy_reconstruction | A statistical reconstruction from paleoclimate proxies. Not a thermometer |
| source_simulation | Output of a climate model experiment, with institution, experiment, member, grid, and calendar |
| emulation | Output of the original emulator or energy-balance model in this repository |
| forcing_assessment | An assessed radiative forcing table |
| emission_inventory | An emissions table, including GWP accounts |
| energy_inventory | Heat content by component |

Direct map overlays refuse to pair a proxy or a simulation with an observation. A study may place them in one figure only with an explicit context sentence and distinct badges.

HadCRUT5 and Berkeley Earth land-ocean both use HadSST for the ocean. Cross-product agreement over the ocean is not two independent observations. The land processing differs. The audit states that shared ancestor.

Ensemble members of one observational analysis are an uncertainty ensemble. Initial-condition members of one model are not separate model families. Counts of independent models use the source id.

## Variables

These kinds are not interchangeable:

- emission, including a GWP100 carbon-dioxide-equivalent rate
- concentration
- radiative forcing
- temperature, absolute or anomaly
- heat content
- energy uptake rate
- rainfall
- sea level
- coverage fraction

The IGCC file `earth_energy_imbalance.csv` is heat content in zettajoules. The filename says imbalance. The unit wins. A flux in W m-2 is computed only by differencing the inventory and dividing by area and time, and the result is stored as `energy_uptake_rate`.

GWP100 emissions cannot enter the carbon-cycle mass balance. SO2 emissions are not aerosol forcing. CO2 concentration is not CO2 forcing. Supplying both a CO2 concentration and a CO2 forcing component to one scenario call is double counting and is rejected. Supplying a total forcing and also a component that is already inside that total is rejected.

Temperature anomalies require a baseline id. Declared windows are `1961-1990` (HadCRUT5 and the PAGES 2k files used here), `1951-1980` (Berkeley Earth), and `1850-1900` (a warming reference, not a control run). Rebasing subtracts the mean of the finite samples in the new window. It does not add 273.15.

## Time

Calendars: proleptic Gregorian, noleap, all-leap, 360-day. Aliases `gregorian` and `standard` mean proleptic Gregorian. There is no silent Julian reform.

Time units are days, hours, minutes, or seconds since a midnight epoch. `months since` and `years since` are rejected unless the file provides bounds, because those units do not have a unique length.

Annual means of monthly data use month lengths on the file calendar. Leap days count on Gregorian and all-leap calendars, never on noleap, and never on the 360-day calendar. A missing month stays missing. If the present days fall below the requested coverage fraction, the annual value is missing. Unknown values are never written as zero.

Model branch years, including a preindustrial control whose time axis sits in model year 5000, stay on that axis. They are not relabeled as 1850-2100 because a filename contains those digits.

## Space

Cell area on a regular latitude-longitude grid uses the spherical formula

```
area = R^2 * (sin lat_north - sin lat_south) * delta_longitude
```

with `R = 6_371_000 m` and longitude in radians. A cell that crosses the branch cut is split before the width is computed. The global mean is the area-weighted mean of finite cells. The unweighted mean is reported beside it as a diagnostic, not as the default. Empty coverage stays empty. A map does not paint a missing cell with the global mean or with zero.

Conservative remapping of an intensive field or an extensive density uses overlap area on the same sphere. A target cell that is only partly covered stores the overlap-weighted value and a coverage fraction. It does not divide by the full target area, because that would treat the uncovered part as zero.

Interpolation is reserved for intensive variables and is labeled. Extensive quantities use the overlap weights so the area integral of the covered domain is unchanged.

## Forcing and carbon

The default CO2 forcing is

```
F = (F_2x / ln 2) * ln(C / C_ref)
```

with `F_2x = 3.93 W m-2`, the AR6 central assessed effective radiative forcing for a doubling. The Myhre coefficient 5.35 is available as a named comparator and is not the default.

The carbon module, when a caller works in emissions space, integrates a four-term impulse response with the Joos et al. (2013) multi-model coefficients:

```
a = (0.2173, 0.2240, 0.2824, 0.2763)
tau = (infinite, 394.4, 36.54, 4.304) years
```

The airborne fraction sums to one. The conversion from ppm to GtC uses 2.124 GtC ppm-1 unless the caller overrides it. The module does not also apply a forcing time series for the same CO2.

## Energy balance

The two-reservoir model, in anomaly form, is

```
Cs dTs/dt = F - lambda Ts - gamma (Ts - Td)
Cd dTd/dt = gamma (Ts - Td)
```

`Ts` and `Td` are surface and deep temperature anomalies in kelvin. `F` is effective radiative forcing in W m-2. `lambda` and `gamma` are in W m-2 K-1. `Cs` and `Cd` are in W yr m-2 K-1 on the Julian year of 365.25 days. Time in the ordinary differential equation is in those years.

With constant forcing and positive coefficients the equilibrium is `Ts = Td = F / lambda`. The net energy uptake `F - lambda Ts` equals the heat storage rate `Cs dTs/dt + Cd dTd/dt`. The exchange term cancels and is not an energy source.

Initial state defaults to zero anomaly at the start of the forcing window. Starting in 1850 from zero, after forcing has already accumulated since 1750, is a different experiment from starting in 1750. Both are allowed only when the caller names the initial state.

A forward Euler step larger than the stability bound implied by the fast eigenvalue is rejected. Backward Euler and the eigen-expansion of the linear system remain available. Parameters that produce a non-negative real part, or a nonpositive capacity, are rejected.

## Pathways

Named source experiments keep their source ids. An SSP label is the ScenarioMIP experiment name when the field comes from that experiment. It is not a no-policy baseline.

Natural-only forcing in the energy-balance model means the solar and volcanic components of an assessed forcing table. It is not a `hist-nat` simulation unless a `hist-nat` file with a source id is actually loaded. If that file is absent, the comparison is reported as unsupported.

Maps and regional fields stop at the last year present in the source. They are not extended by repeating the endpoint. A global energy-balance integral past the end of a source forcing exists only when the caller supplies an explicit continuation and the export marks it as emulation.

## Learning

The spatial emulator has a global response and a regional residual. The training loss may include an aggregation term that penalizes disagreement between the area mean of the field and the global response. Preprocessing statistics and early stopping use the training and validation identities only. Held-out model families and held-out years stay out of fitting. Three training seeds are retained. Correlated members of one model do not count as independent validation families.

The emulator abstains when the forcing window lies outside the training support defined in the experiment plan. Abstention is a result, not a filled field.

## Storage

Streaming acquisition uses capped chunks, records a checksum, and can resume a partial file. The project cap is 5 GB. A stage declares a tighter cap before it starts. Symlinked escape hatches are not counted as a way around the cap, and they are not followed.

## Verification

The verifier is a separate package. It recomputes decisive scalars from raw records and replays a frozen probe. It does not retrain and it does not trust a stored pass bit. Semantic negative controls mutate a baseline, a proxy badge, a member grouping, a forcing sign, or a scenario unit, rebuild ordinary hashes, and must be rejected for the semantic reason.
