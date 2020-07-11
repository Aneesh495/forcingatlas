# Build status

Updated 2026-11-11.

## Phase

Studies. Twelve linked calculations are in `results/studies.json` and `docs/THESIS.md`, and the atlas bundle carries those notes plus the four ssp245 global series. The ssp245 series use each file's own 2015-2019 baseline, so they are not drawn as HadCRUT anomalies.

## Requirement index

| ID | Behavior | Status |
| --- | --- | --- |
| F01 | Unit algebra, anomaly versus absolute temperature, species and GWP guards | verified |
| F02 | Gregorian, noleap, all-leap, and 360-day calendars, with annualization weights | verified |
| F03 | Dataset semantics and forbidden observation mixes | verified |
| F04 | Variable contracts and substitution refusals | verified |
| F05 | Project byte cap on a measured tree | verified |
| G01 | Source manifest with provenance and separate semantics | implemented_untested |
| G02 | Calendar, unit, baseline, mask, and registration checks on real files | implemented_untested |
| G03 | Regridding and area weights against analytical cases | verified |
| G04 | Two instrumental products and a proxy calibration study | implemented_untested |
| G05 | Energy-balance solver against closed forms and energy storage | verified |
| G06 | Forcing adapter rejects bad units and double counting | verified |
| G07 | Calibrated hindcast with a held-out window | verified |
| G08 | Emulator training, held-out families, three seeds | verified for the HadCRUT year holdout. Model-family transfer is not run |
| G09 | Regional and tail scores against baselines | verified as area-weighted grid RMSE against a uniform trend. Tail scores are not run |
| G10 | Attribution with controls and a stated estimand | verified as two-box fingerprint scaling with a shuffled control. DAMIP experiments are not in the local archive |
| G11 | Sensitivity, uncertainty decomposition, ensemble weights | verified for the lambda particle range and inverse-RMSE seed weights |
| G12 | Side-by-side pathways without unsupported layers | verified. Stabilize, decline, and ssp245 globals are labeled. The map is not extended |
| G13 | Atlas with legends, time, region drill-down, export parity | verified against the frozen bundle. ssp245 fields stay off the 1961-1990 map |
| G14 | Bounded acquisition, restart, disk accounting under the cap | implemented_untested |
| G15 | Thesis and ten linked studies reconciled to evidence | verified. Twelve studies are in results/studies.json and docs/THESIS.md |
| G16 | Verifier rejects wrong baseline, fake proxy, member leakage, forcing sign, scenario units | verified |

## Commands actually run

- `.venv/bin/python -m pytest tests -q` passed after the foundation landed (41 tests).
- `.venv/bin/python -m pytest tests/test_geometry.py -q` passed (13 tests) after the area and remap fixes.
- `.venv/bin/python -m pytest tests -q` passed (67 tests) after the readers, cube, zip member, and resume checks. Real source files are not in `data/` yet, so G01 and G02 are not closed.

## Data identities

Local `data/raw` holds HadCRUT5.1.0.0 summary series and both ensemble-mean grids, Berkeley Earth land-ocean, PAGES 2k, and the IGCC forcing, concentration, emission, and energy-inventory tables. `results/cmip_calendars.json` records four ssp245 `tas` files read from ClimateBench. The raw NetCDF stays in `data/raw/cmip6` and is not committed.

## Next action

Nothing further is queued from the local files. A pathway other than ssp245, and hist-GHG, hist-aer, or hist-nat, are not in the ClimateBench zip that was read.
