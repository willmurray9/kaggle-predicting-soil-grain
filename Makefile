PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src

.PHONY: download data labels baselines eda image-model experiments multicrop audit camera-balance validate test

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

multicrop:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli multicrop

audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli audit

camera-balance:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli camera-balance

validate:
	@if [ -z "$(SUBMISSION)" ]; then echo "Usage: make validate SUBMISSION=path/to/submission.csv"; exit 2; fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli validate --submission "$(SUBMISSION)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
