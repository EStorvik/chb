
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

## Reproducing figures from paper XXX
- For reproduction of figure 1 - snapshots from simulation output run the file:
    chb/computations/chb/semi_imp_paper/generate_solution_figure.py
    The outputs will then be stored in
    ../output (folder will automatically be created)
    and can be opened with paraview to manually create the figures that are in the paper.
- For reproducing the numbers used to create figure 2 and 3, run the script
    chb/computations/chb/semi_imp_paper/run_gamma_study.py
    and
    chb/computations/chb/semi_imp_paper/run_swelling_study.py
    respectively.
    CSV-files will then be created that contain the data used to create the figures with ticks and pgfplots in latex.
- For reproducing figure 4 run the script (after running the gamma and swelling     studies):
    chb/computations/chb/semi_imp_paper/plot_energies.py


## Citing
https://doi.org/10.5281/zenodo.18244133
