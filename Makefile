# Shell to use
SHELL = /bin/bash

# Directory structure
SRC_DIR = src
SCRIPTS_DIR = scripts

# Python files and virtual environment
VENV = ~/.virtualenvs/tshred/bin/activate
PYTHONS = $(wildcard $(SRC_DIR)/*.py) $(wildcard $(SCRIPTS_DIR)/*.py)

all: format

clean:
	rm -rf bash logs pickles results slurms checkpoints
	rm -rf transformer_sindy_shred.egg-info

# Format Python files with black
format:
	source $(VENV) && black $(PYTHONS)

# Phony targets (not actual files)
.PHONY: all format help

# Help target
help:
	@echo "Available targets:"
	@echo "  format               		 - Format Python files with black"
