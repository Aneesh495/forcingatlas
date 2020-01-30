# Build status

Updated 2026-11-11.

## Phase

Foundation. Units, calendars, semantic identities, variable contracts, and the disk accountant are implemented and tested. Acquisition, the energy-balance solver, the emulator, the atlas, and the campaign are not implemented yet.

## Requirement index

| ID | Behavior | Status |
| --- | --- | --- |
| F01 | Unit algebra, anomaly versus absolute temperature, species and GWP guards | verified |
| F02 | Gregorian, noleap, all-leap, and 360-day calendars, with annualization weights | verified |
| F03 | Dataset semantics and forbidden observation mixes | verified |
| F04 | Variable contracts and substitution refusals | verified |
| F05 | Project byte cap on a measured tree | verified |
| G01 | Source manifest with provenance and separate semantics | missing |
| G02 | Calendar, unit, baseline, mask, and registration checks on real files | missing |
| G03 | Regridding and area weights against analytical cases | missing |
| G04 | Two instrumental products and a proxy calibration study | missing |
| G05 | Energy-balance solver against closed forms and energy storage | missing |
| G06 | Forcing adapter rejects bad units and double counting | missing |
| G07 | Calibrated hindcast with a held-out window | missing |
| G08 | Emulator training, held-out families, three seeds | missing |
| G09 | Regional and tail scores against baselines | missing |
| G10 | Attribution with controls and a stated estimand | missing |
| G11 | Sensitivity, uncertainty decomposition, ensemble weights | missing |
| G12 | Side-by-side pathways without unsupported layers | missing |
| G13 | Atlas with legends, time, region drill-down, export parity | missing |
| G14 | Bounded acquisition, restart, disk accounting under the cap | missing |
| G15 | Thesis and ten linked studies reconciled to evidence | missing |
| G16 | Verifier rejects wrong baseline, fake proxy, member leakage, forcing sign, scenario units | missing |

## Commands actually run

- `.venv/bin/python -m pytest tests -q` passed after the foundation landed (41 tests).

## Data identities

No source file has been acquired into `data/` yet. Pinned URLs are recorded in the specification and will be copied into a manifest when acquisition exists.

## Next action

Implement spherical cell areas, longitude conventions, and conservative regridding, with analytical tests that do not call the regridder to build their expected integrals.
