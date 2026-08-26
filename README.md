
# CHB
Cahn-Hilliard-Biot Simulations

## Installation

### Prerequisites
This package requires FEniCSx v0.9. We recommend using conda to install all dependencies.

### Conda Installation (Recommended)
Clone the repository from github and enter the chb folder. Then create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate chb
```

This will install FEniCSx v0.9, all Python dependencies, development tools, and the chb package in editable mode.

### Alternative: pip Installation
If you already have FEniCSx v0.9 installed, you can install just the chb package:

```bash
pip install .
```
To install chb as editable (recommended), along with the tools to develop and run tests:
```bash
pip install -e .[dev]
```

## Developing CHB

Use black and flake8 formatting.

## Citing
https://doi.org/10.5281/zenodo.18244133
