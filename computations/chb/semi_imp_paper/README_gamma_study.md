# CHB Gamma Parameter Study Scripts

This directory contains scripts to automatically run parametric studies over gamma values for all CHB (Cahn-Hilliard-Biot) simulations in the `semi_imp_paper` folder.

## Scripts Created

### 1. `run_gamma_study.py` (Recommended)
A comprehensive Python script with full error handling, progress reporting, and automatic file management.

**Features:**
- Modifies gamma parameter in each simulation file
- Runs all simulations for each gamma value
- Automatically renames output files with gamma values
- Comprehensive error handling and progress reporting
- Timeout protection (1 hour per simulation)
- Detailed summary reports

**Usage:**
```bash
cd /Users/erlend/src/fenicsx/chb/computations/chb/semi_imp_paper
python run_gamma_study.py
```

### 2. `run_gamma_study.sh` (Alternative)
A bash script version for users who prefer shell scripts.

**Usage:**
```bash
cd /Users/erlend/src/fenicsx/chb/computations/chb/semi_imp_paper
chmod +x run_gamma_study.sh
./run_gamma_study.sh
```

## Configuration

### Gamma Values
The scripts will run simulations for the following gamma values:
- 0.25, 0.5, 1, 2, 4, 8, 16

### Simulation Scripts
The following scripts will be executed for each gamma value:
- `chb_monolithic_semi_imp.py`
- `chb_monolithic_imp.py`
- `chb_splitting_ch_biot_semi_imp.py`
- `chb_splitting_ch_biot_imp.py`
- `chb_splitting_ch_fixed_stress_semi_imp.py`
- `chb_splitting_ch_fixed_stress_imp.py`

**Total simulations:** 7 gamma values × 6 scripts = 42 simulations

## Output Files

The scripts will automatically rename output log files to include the gamma value:

**Original:** `chb_monolithic_semi_imp.xlsx`
**Renamed:** `chb_monolithic_semi_imp_gamma_0.25.xlsx`

All output files are saved to: `../output/log/`

## How It Works

1. **Parameter Modification:** The script finds lines like `gamma = 5` and replaces them with the current gamma value
2. **Simulation Execution:** Runs each Python script using the specified Python environment
3. **File Management:** Renames output files to prevent overwriting between gamma values
4. **Progress Reporting:** Shows detailed progress and error information

## Error Handling

- **Missing Scripts:** Checks that all simulation scripts exist before starting
- **Timeout Protection:** Each simulation has a 1-hour timeout
- **Graceful Failures:** If one simulation fails, others continue
- **File Management:** Safely handles file renaming and directory creation

## Time Estimates

Depending on simulation complexity:
- **Per simulation:** 5-30 minutes
- **Total runtime:** 3-21 hours for all 42 simulations

## Prerequisites

1. **Python Environment:** The scripts use the specified conda environment:
   ```
   /Users/erlend/miniforge3/envs/fenicsx-env/bin/python
   ```

2. **Required Packages:** All simulation dependencies should be installed in the environment

3. **Output Directory:** The `../output/log/` directory will be created automatically

## Monitoring Progress

The Python script provides detailed progress information:
- Current gamma value being processed
- Current simulation running
- Time elapsed per simulation
- Success/failure status
- Final summary with statistics

## Troubleshooting

1. **Permission Errors:** Make sure the bash script is executable:
   ```bash
   chmod +x run_gamma_study.sh
   ```

2. **Python Path Issues:** Verify the Python executable path in the script matches your environment

3. **Missing Dependencies:** Ensure all required packages are installed in the conda environment

4. **Disk Space:** Make sure you have sufficient disk space for all output files

## Customization

To modify the gamma values or add/remove scripts, edit the configuration section at the top of either script:

**Python version:**
```python
GAMMA_VALUES = [0.25, 0.5, 1, 2, 4, 8, 16]
SIMULATION_SCRIPTS = [
    "chb_monolithic_semi_imp.py",
    # ... add more scripts here
]
```

**Bash version:**
```bash
GAMMA_VALUES=(0.25 0.5 1 2 4 8 16)
SCRIPTS=(
    "chb_monolithic_semi_imp.py"
    # ... add more scripts here
)
```