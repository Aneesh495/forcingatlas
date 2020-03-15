# Build status

Updated 2026-11-11.

## Phase

Energy balance. The two-reservoir solver, carbon pools, and forcing adapter match their closed forms in tests. Calibration against real series, the emulator, the atlas, and the campaign are not implemented yet.

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
| G04 | Two instrumental products and a proxy calibration study | missing |
| G05 | Energy-balance solver against closed forms and energy storage | verified |
| G06 | Forcing adapter rejects bad units and double counting | verified |
| G07 | Calibrated hindcast with a held-out window | missing |
| G08 | Emulator training, held-out families, three seeds | missing |
| G09 | Regional and tail scores against baselines | missing |
| G10 | Attribution with controls and a stated estimand | missing |
| G11 | Sensitivity, uncertainty decomposition, ensemble weights | missing |
| G12 | Side-by-side pathways without unsupported layers | missing |
| G13 | Atlas with legends, time, region drill-down, export parity | missing |
| G14 | Bounded acquisition, restart, disk accounting under the cap | implemented_untested |
| G15 | Thesis and ten linked studies reconciled to evidence | missing |
| G16 | Verifier rejects wrong baseline, fake proxy, member leakage, forcing sign, scenario units | missing |

## Commands actually run

- `.venv/bin/python -m pytest tests -q` passed after the foundation landed (41 tests).
- `.venv/bin/python -m pytest tests/test_geometry.py -q` passed (13 tests) after the area and remap fixes.
- `.venv/bin/python -m pytest tests -q` passed (67 tests) after the readers, cube, zip member, and resume checks. Real source files are not in `data/` yet, so G01 and G02 are not closed.

## Data identities

No source file has been acquired into `data/` yet. Pinned URLs are recorded in the specification and will be copied into a manifest when acquisition exists.

## Next action

Fit the two-box parameters on a declared training window with a holdout, and add the persistence, trend, pattern-scaling, and linear-response baselines.
