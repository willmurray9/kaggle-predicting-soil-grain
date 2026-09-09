PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src

.PHONY: download data labels baselines eda image-model experiments validate test

download:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli download

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli data

labels:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli labels

baselines:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli baselines

eda:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli eda

image-model:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli image-model

experiments:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli experiments

validate:
	@if [ -z "$(SUBMISSION)" ]; then echo "Usage: make validate SUBMISSION=path/to/submission.csv"; exit 2; fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli validate --submission "$(SUBMISSION)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
