PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src

.PHONY: download data labels baselines eda image-model experiments multicrop audit camera-balance kernel-ridge frozen-model nested-ridge model-blend nested-neighbors spatial-coverage physical-texture validate test

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

kernel-ridge:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli kernel-ridge

frozen-model:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli frozen-model

nested-ridge:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli nested-ridge

model-blend:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli model-blend

nested-neighbors:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli nested-neighbors

spatial-coverage:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli spatial-coverage

physical-texture:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli physical-texture

validate:
	@if [ -z "$(SUBMISSION)" ]; then echo "Usage: make validate SUBMISSION=path/to/submission.csv"; exit 2; fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli validate --submission "$(SUBMISSION)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
