# Forcing Atlas

Forcing Atlas reads instrumental temperature products, a paleoclimate reconstruction, and an assessed forcing table, then compares them with a two-reservoir energy-balance model. A future pathway in this project is conditional on the forcing you supply. It is not a forecast of policy and it is not an observation.

The atlas in `web/` reads a frozen bundle. It does not rerun the solver when you move the year. Regenerate the bundle from the command line.

## Run the bounded demo

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
make test
python -m forcing_atlas.cli acquire
python -m forcing_atlas.cli export-atlas
cd web && npm install && npm run dev
```

The dev server listens on port 43117. `make doctor` prints the machine, the numerical stack, and the disk accountant. It does not mark a full campaign as passed.

`python -m forcing_atlas.cli walkthrough` reprints the instrumental comparison, the proxy badge, and the holdout scores from the files in `data/raw`.

## What the current bundle shows

HadCRUT5.1.0.0 and Berkeley Earth are both anomalies. Berkeley is shifted from its native 1951-1980 baseline onto 1961-1990 before it is drawn next to HadCRUT. The ocean part of Berkeley Earth uses HadSST, so agreement over the ocean is not two independent observations.

PAGES 2k is drawn as a proxy reconstruction. The file also contains a Cowtan and Way instrumental target. That column stays on its own record and is not used as a proxy value.

The non-infilled HadCRUT grid is annualized with Gregorian month lengths. Cells without a full year of months stay empty. The map does not paint them with zero or with the global mean.

The two-box model is spun up from 1750 with the IGCC effective radiative forcing and scored on HadCRUT from 1850. Parameters are fit on 1850-1994 inside a declared prior. 1995-2023 is reported and not used to move the parameters. On this series the training search sits on the low-feedback edge of the prior and the holdout is worse than a one-year persistence forecast. A separate check, with feedback fixed rather than fit, shows that a lower feedback can win on the training window and then run warm.

Solar-plus-volcanic forcing uses the same parameters and is labeled as an emulation. It is not a `hist-nat` simulation. Two global paths continue to 2100: one holds the last assessed forcing, and one declines linearly toward the 1990 forcing. Neither is an SSP. The map stops when the gridded observations stop.

## Layout

- `src/forcing_atlas` units, calendars, grids, readers, the two-box solver, calibration, and the atlas export
- `web` the map and the linked series
- `tests` closed-form checks and the real-series hindcast when `data/raw` is present
- `docs` contracts and build status

## Data licenses

Code in this repository is MIT. HadCRUT5.1.0.0 states the Open Government Licence v3.0 in the NetCDF attribute, and the download page footer also states Crown Copyright. Berkeley Earth land-ocean is CC BY-NC 4.0. The IGCC forcing, concentration, emission, and energy-inventory tables are CC BY 4.0. PAGES 2k is distributed by NOAA Paleoclimatology and should be cited as Neukom et al. 2019. Raw files stay in `data/raw` and are not committed.

The IGCC concentration metadata yaml records an md5 and a size that do not match the file at the pinned URL. The catalog records the sha256 of the bytes that were read.

## Grid fit

`python -m forcing_atlas.cli train` fits the spatial emulator on the non-infilled HadCRUT grid for seeds 11, 29, and 47. Years through 1994 are the training rows. 1995-2023 is scored and is not in the loss. On that holdout the three area-weighted root-mean-square errors are 0.78 K, 0.74 K, and 0.70 K. A latitude-uniform trend fit on the same training years scores 0.80 K. The best seed beats that trend. The gain is small, and 2023 is outside the training forcing radius so the emulator abstains there. Checkpoints stay local. `results/grid_campaign.json` and `results/evidence.json` record the split and the scores.

`python -m forcing_atlas_verify --evidence results/evidence.json --expect-commit <source-commit>` recomputes the mean of the three seed scores. It rejects a baseline whose window mean is not near zero, a proxy record labeled as an observation, a holdout year present in the training list, a positive volcanic forcing in 1992, and an absolute temperature unit on anomaly-sized scenario values.

## Source-model sample

`python -m forcing_atlas.cli cmip-sample` reads four ssp245 annual `tas` files out of the ClimateBench zip without storing the zip. Their calendars are 365-day (CanESM5), Gregorian (MIROC6), proleptic Gregorian (MPI-ESM1-2-LR), and 360-day (HadGEM3-GC31-LL). That archive does not contain the other future pathways for those models. They are not filled in.

## Studies

`python -m forcing_atlas.cli studies` writes `results/studies.json`. `docs/THESIS.md` walks through the same twelve calculations. The atlas lists them beside the map. Regenerate the bundle with `python -m forcing_atlas.cli export-atlas` after that command so the page matches the file.

The four ssp245 global series are on the chart as source simulations. Each is an anomaly versus 2015-2019 in its own file. They are not placed on the HadCRUT map, because that map is a 1961-1990 anomaly. The spatial emulator's three-seed global mean is on the chart as well. 2023 is absent because it is outside the training forcing radius.

The same four models' piControl segments vary by about 0.11 to 0.13 K, while their ssp245 files warm by 1.5 to 2.7 K across an equal-length record. HadGEM3 and MPI write 2015-2100 on the control time axis. Those branch labels are not drawn as historical years. Checksums for the local files are in `results/source_manifest.json`.

## Limits

The two-box holdout is still worse than persistence, and the fitted feedback is on the edge of the prior. The spatial model beats a uniform trend by a small margin, with much larger Arctic errors than tropical errors. Fingerprint scaling is one emulator, not a DAMIP attribution. The energy inventory file is heat content in zettajoules even though the filename says imbalance. It is not used as a flux inside the fit.
