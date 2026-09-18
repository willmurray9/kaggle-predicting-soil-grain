PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH := src

.PHONY: download data labels baselines eda image-model experiments multicrop audit camera-balance kernel-ridge frozen-model nested-ridge model-blend nested-neighbors spatial-coverage physical-texture pca-ridge official-preprocessing texture-kernel linear-svr shallow-trees mobilenet-pca physical-photo spectral-photo dino-pca geometry-transport patch-mixture particle-audit autonomous-search camera-transfer weibull visual-language validate test

camera-transfer:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli camera-transfer

weibull:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli weibull

visual-language:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli visual-language

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

pca-ridge:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli pca-ridge

official-preprocessing:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli official-preprocessing

texture-kernel:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli texture-kernel

linear-svr:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli linear-svr

shallow-trees:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli shallow-trees

mobilenet-pca:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli mobilenet-pca

physical-photo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli physical-photo

spectral-photo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli spectral-photo

dino-pca:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli dino-pca

geometry-transport:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli geometry-transport

patch-mixture:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli patch-mixture

particle-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli particle-audit

autonomous-search:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli autonomous-search

validate:
	@if [ -z "$(SUBMISSION)" ]; then echo "Usage: make validate SUBMISSION=path/to/submission.csv"; exit 2; fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m soilgrain.cli validate --submission "$(SUBMISSION)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
