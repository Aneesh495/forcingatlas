# Build status

Updated 2026-11-11.

## Phase

Grid campaign. Three seeds are trained on the non-infilled HadCRUT grid through 1994 and scored on 1995-2023. The verifier rejects tampered evidence. Four ssp245 source files have been read for their calendars. Attribution, the thesis, and drawing those model grids in the atlas are not done.

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
| G10 | Attribution with controls and a stated estimand | missing |
| G11 | Sensitivity, uncertainty decomposition, ensemble weights | missing |
| G12 | Side-by-side pathways without unsupported layers | missing |
| G13 | Atlas with legends, time, region drill-down, export parity | implemented_untested |
| G14 | Bounded acquisition, restart, disk accounting under the cap | implemented_untested |
| G15 | Thesis and ten linked studies reconciled to evidence | missing |
| G16 | Verifier rejects wrong baseline, fake proxy, member leakage, forcing sign, scenario units | verified |

## Commands actually run

- `.venv/bin/python -m pytest tests -q` passed after the foundation landed (41 tests).
- `.venv/bin/python -m pytest tests/test_geometry.py -q` passed (13 tests) after the area and remap fixes.
- `.venv/bin/python -m pytest tests -q` passed (67 tests) after the readers, cube, zip member, and resume checks. Real source files are not in `data/` yet, so G01 and G02 are not closed.

## Data identities

Local `data/raw` holds HadCRUT5.1.0.0 summary series and both ensemble-mean grids, Berkeley Earth land-ocean, PAGES 2k, and the IGCC forcing, concentration, emission, and energy-inventory tables. `results/cmip_calendars.json` records four ssp245 `tas` files read from ClimateBench. The raw NetCDF stays in `data/raw/cmip6` and is not committed.

## Next action

Draw the emulator holdout next to the uniform trend on the atlas, and keep model-family transfer blocked until a pathway other than ssp245 is actually in hand.
