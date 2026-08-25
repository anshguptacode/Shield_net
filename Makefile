# ShieldNet - the handful of commands worth remembering.
# ============================================================================
# Everything in this file is a thin wrapper. `make train` is `shieldnet train` with the
# flags you would otherwise retype; no target contains logic of its own, and none of
# them is required - `shieldnet --help` is the real interface. The point of the file is
# that the flags which matter are written down somewhere they cannot drift.
#
# Two tiers:
#   offline (help, check, notebook, clean) need only numpy and pandas
#   real    (download, prepare, train, serve, predict) need `make install`
#
# Commands are run as `$(PYTHON) -m shieldnet` rather than as the `shieldnet` console
# script, so they work in a checkout that has not been installed and in a virtualenv
# whose bin/ is not on PATH.
# ============================================================================

PYTHON ?= python
CONFIG ?= config/default.yaml
ROOT   ?= .
PORT   ?= 8501
OUTPUT ?= verdicts.csv
SHIELDNET = $(PYTHON) -m shieldnet

.DEFAULT_GOAL := help
.PHONY: help install install-min doctor check check-fast wiring smoke prepare download \
        train info serve predict notebook notebook-check clean clean-runs distclean

help:  ## list these targets
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  variables: PYTHON=$(PYTHON) CONFIG=$(CONFIG) ROOT=$(ROOT) PORT=$(PORT)"

# -- setting up ---------------------------------------------------------------

install:  ## editable install with every optional dependency
	$(PYTHON) -m pip install -e ".[all]"

install-min:  ## editable install with numpy and pandas only (enough for `doctor`)
	$(PYTHON) -m pip install -e .

doctor:  ## what works in this environment, and what is missing
	$(SHIELDNET) doctor

# -- checking -----------------------------------------------------------------
# These need no ML libraries at all: the model backends and Streamlit are replaced by
# doubles under tests/_stubs. They check that the stages agree with each other, which
# is the part that breaks. Whether XGBoost detects intrusions is the notebook's job.

check:  ## run every offline check (a minute or two; six of the seven fit real models)
	$(PYTHON) scripts/run_checks.py

check-fast:  ## the quick ones, for a tight edit loop
	$(PYTHON) scripts/run_checks.py wiring inference explain narrate

wiring:  ## static checks only: signatures, config fields, exports, documented flags
	$(PYTHON) scripts/check_wiring.py

smoke:  ## prove a real install can train: one model, synthetic data, no tuning
	$(SHIELDNET) train --synthetic 20000 --models logistic_regression \
		--features 12 --no-tune --no-explain --root $(ROOT)

# -- the pipeline -------------------------------------------------------------

download:  ## fetch CICIDS2017 from Kaggle into ROOT/data/raw (about 1 GB)
	$(SHIELDNET) download --config $(CONFIG) --root $(ROOT)

prepare:  ## clean, split and cache; also writes a labelled sample for the app
	$(SHIELDNET) prepare --config $(CONFIG) --root $(ROOT) \
		--sample $(ROOT)/data/samples/cicids2017_sample.csv --sample-rows 5000

train:  ## the full run, ending in a saved artifact (CONFIG=config/colab.yaml for the long one)
	$(SHIELDNET) train --config $(CONFIG) --root $(ROOT) \
		--log-file $(ROOT)/reports/train.log

info:  ## describe the saved artifact: features, classes, metrics, provenance
	$(SHIELDNET) info --root $(ROOT)

predict:  ## score a capture: make predict INPUT=capture.csv [OUTPUT=verdicts.csv]
	@test -n "$(INPUT)" || { \
		  echo "usage: make predict INPUT=capture.csv [OUTPUT=verdicts.csv]"; \
		  exit 2; }
	$(SHIELDNET) predict --input "$(INPUT)" --output "$(OUTPUT)" --top-k 3 --root $(ROOT)

serve:  ## launch the Streamlit app (override with PORT=8502)
	$(SHIELDNET) serve --port $(PORT) --root $(ROOT)

# -- the notebook -------------------------------------------------------------
# notebooks/ShieldNet_Colab.ipynb is generated. Edit the generator, not the notebook -
# and if you did edit it in Colab, `make notebook-check` is what tells you.

notebook:  ## regenerate notebooks/ShieldNet_Colab.ipynb
	$(PYTHON) scripts/build_colab_notebook.py

notebook-check:  ## fail if the notebook no longer matches its generator
	$(PYTHON) scripts/build_colab_notebook.py --check

# -- tidying ------------------------------------------------------------------

clean:  ## remove caches and compiled files (touches nothing you made)
	 find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	 find . -name "*.py[co]" -delete 2>/dev/null || true
	 rm -rf .pytest_cache build dist *.egg-info src/*.egg-info

clean-runs:  ## remove the /tmp directories the checks build
	 rm -rf /tmp/shieldnet_run /tmp/shieldnet_infer /tmp/shieldnet_explain \
	        /tmp/shieldnet_cli /tmp/shieldnet_app
	@echo "removed the checks' scratch directories"

distclean:  ## also delete artifacts, reports and cached data - needs CONFIRM=1
	@test "$(CONFIRM)" = "1" || { \
		  echo "This deletes $(ROOT)/artifacts, $(ROOT)/reports and $(ROOT)/data/interim."; \
		  echo "A trained artifact is 35-60 minutes of Colab. Re-run with CONFIRM=1 if you"; \
		  echo "are sure - and note that data/raw is never touched, so you will not have"; \
		  echo "to download the dataset again."; \
		  exit 2; }
	 rm -rf $(ROOT)/artifacts $(ROOT)/reports $(ROOT)/data/interim $(ROOT)/data/processed
	@echo "removed artifacts, reports and cached data (data/raw kept)"
