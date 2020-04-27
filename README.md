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

## Limits

ClimateBench members and a three-seed spatial campaign on the HadCRUT grid are not in the default export. The emulator module trains on caller-supplied rows and abstains outside a training radius. Its unit tests cover that behavior. A full multi-model campaign is not claimed.

The energy inventory file is heat content in zettajoules even though the filename says imbalance. It is not used as a flux inside the fit.
