# Forcing Atlas

Forcing Atlas is a local-first workbench for reading observed and proxy climate records beside source-model output and a small original emulator. A future pathway in this project is conditional on the forcing you put in. It is not a forecast of policy and it is not an observation.

The science code lives in `src/forcing_atlas`. The atlas interface is added on top of frozen exports so the browser does not recompute a scenario on every request. The command line is the path that runs the numerical core.

## What runs now

Units, calendars, dataset semantics, and the variable contracts are in place and covered by tests. Later sections of this file will name the acquisition, energy-balance, emulator, and atlas commands as those paths become runnable. They are not runnable yet.

Supported calendars are proleptic Gregorian (aliases `gregorian` and `standard`), `noleap`, `all_leap`, and `360_day`. Temperature anomalies are a different kind from absolute temperature. Emissions, concentrations, radiative forcing, heat content, rainfall, and sea level are different variables and the contracts refuse to relabel one as another.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
make test
make doctor
```

`make doctor` prints an environment report. It does not mark a full campaign as passed.

The project accountant uses a 5 GB local cap covering the checkout, git objects, the virtualenv, `data/`, `evidence/`, checkpoints, and the project-local temp directory. Raw source files are downloaded into `data/` and are not part of the git history.

## Layout

- `src/forcing_atlas` numerical core and, as it lands, the command line
- `src/forcing_atlas_verify` independent verifier (added with the evidence bundle)
- `tests` checks whose expected values come from closed forms or a second implementation
- `docs` contracts, numerical definitions, and build status

## License

The code in this repository is MIT. Climate datasets keep the licenses of their publishers. HadCRUT is Crown Copyright of the Met Office Hadley Centre. Berkeley Earth land-ocean is CC BY-NC 4.0. The Indicators of Global Climate Change tables used for forcing, concentrations, and the energy inventory are CC BY 4.0. ClimateBench is CC BY 4.0. PAGES 2k reconstructions are used with the NOAA Paleoclimatology citation. None of those source files are vendored here.
