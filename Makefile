PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUN_ID ?=
PROFILE ?= diagnostic
EVIDENCE ?=

export PIP_CACHE_DIR := $(CURDIR)/.agent-local/pip-cache
export TMPDIR := $(CURDIR)/.agent-local/tmp
export PYTHONPATH := $(CURDIR)/src

.PHONY: bootstrap doctor build test data demo bench stress docs acceptance verify release \
	climate-data calibrate train hindcast scenario atlas

bootstrap:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu

doctor:
	$(PYTHON) -m forcing_atlas.doctor

build:
	$(PYTHON) -m compileall -q src tests

test:
	$(PYTHON) -m pytest

data: climate-data

climate-data:
	$(PYTHON) -m forcing_atlas.cli acquire

calibrate:
	$(PYTHON) -m forcing_atlas.cli calibrate

train:
	$(PYTHON) -m forcing_atlas.cli train

hindcast:
	$(PYTHON) -m forcing_atlas.cli hindcast

scenario:
	$(PYTHON) -m forcing_atlas.cli scenario

atlas:
	$(PYTHON) -m forcing_atlas.cli export-atlas

demo:
	$(PYTHON) -m forcing_atlas.cli walkthrough

bench:
	$(PYTHON) -m forcing_atlas.cli bench --profile diagnostic

stress:
	$(PYTHON) -m forcing_atlas.cli stress --profile diagnostic

docs:
	$(PYTHON) -m forcing_atlas.docs_build

acceptance:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required" >&2; exit 2)
	$(PYTHON) -m forcing_atlas.cli acceptance --run-id "$(RUN_ID)" --profile "$(PROFILE)"

verify:
	@test -n "$(EVIDENCE)" || (echo "EVIDENCE is required" >&2; exit 2)
	$(PYTHON) -m forcing_atlas_verify --evidence "$(EVIDENCE)"

release:
	$(PYTHON) -m forcing_atlas.cli release
