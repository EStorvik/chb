#!/usr/bin/env python3
"""
Automated script for gamma parameter study.

Run parametric study over gamma values for all semi_implicit CHB simulations.

This script will:
1. Modify gamma parameter in each simulation file
2. Enforce fixed swelling parameter = 0.5 in each simulation file
3. Run the simulation
4. Rename output files to include gamma value
5. Repeat for all gamma values in the study

Author: Auto-generated script for CHB gamma study
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Configuration
GAMMA_VALUES = [0.25, 0.5, 1, 2, 4]
SWELLING_PARAMETER = 0.5  # Fixed swelling parameter for all simulations
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "../output/log"
PYTHON_EXECUTABLE = sys.executable  # Use the same Python that's running this script

# List of simulation scripts to run
SIMULATION_SCRIPTS = [
    "chb_monolithic_semi_imp.py",
    "chb_monolithic_imp.py",
    "chb_splitting_ch_biot_semi_imp.py",
    "chb_splitting_ch_biot_imp.py",
    "chb_splitting_ch_fixed_stress_semi_imp.py",
    "chb_splitting_ch_fixed_stress_imp.py",
]


def modify_gamma_parameter(script_path: Path, gamma_value: float) -> bool:
    """
    Modify the gamma parameter in a simulation script.

    Args:
        script_path: Path to the simulation script
        gamma_value: New gamma value to set

    Returns:
        True if modification was successful, False otherwise
    """
    try:
        # Read the file
        with open(script_path, "r") as f:
            content = f.read()

        # Find and replace gamma parameter
        # Pattern matches: gamma = <number>
        pattern = r"gamma\s*=\s*[0-9]+\.?[0-9]*"
        new_gamma = f"gamma = {gamma_value}"

        if re.search(pattern, content):
            content = re.sub(pattern, new_gamma, content)

            # Write back to file
            with open(script_path, "w") as f:
                f.write(content)

            print(f"  ✓ Updated gamma = {gamma_value} in {script_path.name}")
            return True
        else:
            print(f"  ✗ Could not find gamma parameter in {script_path.name}")
            return False

    except Exception as e:
        print(f"  ✗ Error modifying {script_path.name}: {e}")
        return False


def modify_swelling_parameter(script_path: Path, swelling_value: float) -> bool:
    """
    Modify the swelling parameter in a simulation script.

    Args:
        script_path: Path to the simulation script
        swelling_value: New swelling parameter value to set

    Returns:
        True if modification was successful, False otherwise
    """
    try:
        # Read the file
        with open(script_path, "r") as f:
            content = f.read()

        # Find and replace swelling parameter
        # Pattern matches: swelling_parameter=<number> (in function call)
        pattern = r"swelling_parameter\s*=\s*[0-9]+\.?[0-9]*"
        new_swelling = f"swelling_parameter={swelling_value}"

        if re.search(pattern, content):
            content = re.sub(pattern, new_swelling, content)

            # Write back to file
            with open(script_path, "w") as f:
                f.write(content)

            print(f"  ✓ Updated swelling_parameter = {swelling_value} in {script_path.name}")
            return True
        else:
            print(f"  ⚠ Could not find swelling_parameter in {script_path.name}")
            return False

    except Exception as e:
        print(f"  ✗ Error modifying swelling in {script_path.name}: {e}")
        return False


def run_simulation(script_path: Path) -> bool:
    """
    Run a simulation script.

    Args:
        script_path: Path to the simulation script

    Returns:
        True if simulation completed successfully, False otherwise
    """
    try:
        print(f"    Running {script_path.name}...")
        start_time = time.time()

        # Run the script
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"    ✓ Completed in {elapsed_time:.1f}s")
            return True
        else:
            print(f"    ✗ Failed with return code {result.returncode}")
            print(f"    Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after 15 minutes")
        return False
    except Exception as e:
        print(f"    ✗ Error running simulation: {e}")
        return False


def rename_output_files(gamma_value: float) -> None:
    """
    Rename output files to include gamma value.

    Args:
        gamma_value: Current gamma value being processed
    """
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Expected output files based on the script patterns
        output_files = [
            "chb_monolithic_semi_imp.csv",
            "chb_monolithic_imp.csv",
            "chb_splitting_ch_biot_semi_imp.csv",
            "chb_splitting_ch_biot_imp.csv",
            "chb_splitting_ch_fixedstress_semi_imp.csv",
            "chb_splitting_ch_fixedstress_imp.csv",
        ]

        for filename in output_files:
            old_path = OUTPUT_DIR / filename
            if old_path.exists():
                # Create new filename with gamma value
                name_stem = old_path.stem
                new_filename = f"{name_stem}_gamma_{gamma_value}.csv"
                new_path = OUTPUT_DIR / new_filename

                # Rename/move the file
                shutil.move(str(old_path), str(new_path))
                print(f"    Renamed {filename} → {new_filename}")

    except Exception as e:
        print(f"    ✗ Error renaming output files: {e}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("CHB Gamma Parameter Study - Automated Simulation Runner")
    print("=" * 70)
    print(f"Gamma values: {GAMMA_VALUES}")
    print(f"Fixed swelling parameter: {SWELLING_PARAMETER}")
    print(f"Scripts to run: {len(SIMULATION_SCRIPTS)}")
    total_sims = len(GAMMA_VALUES) * len(SIMULATION_SCRIPTS)
    print(
        f"Total simulations: {len(GAMMA_VALUES)} × "
        f"{len(SIMULATION_SCRIPTS)} = {total_sims}"
    )
    print("=" * 70)

    # Check if all scripts exist
    missing_scripts = []
    for script_name in SIMULATION_SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            missing_scripts.append(script_name)

    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        return

    print("✓ All simulation scripts found")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

    # Record start time
    total_start_time = time.time()
    successful_runs = 0
    total_runs = len(GAMMA_VALUES) * len(SIMULATION_SCRIPTS)

    # Main loop: iterate over gamma values
    for i, gamma in enumerate(GAMMA_VALUES, 1):
        print(f"\n[{i}/{len(GAMMA_VALUES)}] Processing gamma = {gamma}")
        print("-" * 50)

        # Track successes for this gamma
        gamma_successes = 0

        # Run all simulations for this gamma value
        for j, script_name in enumerate(SIMULATION_SCRIPTS, 1):
            script_path = SCRIPT_DIR / script_name

            print(f"  [{j}/{len(SIMULATION_SCRIPTS)}] {script_name}")

            # Modify gamma parameter
            gamma_modified = modify_gamma_parameter(script_path, gamma)

            # Enforce fixed swelling parameter
            swelling_modified = modify_swelling_parameter(script_path, SWELLING_PARAMETER)

            if gamma_modified:
                # Run simulation
                if run_simulation(script_path):
                    successful_runs += 1
                    gamma_successes += 1
                else:
                    print(f"    ⚠️  Simulation failed, continuing...")
            else:
                print(f"    ⚠️  Could not modify gamma, skipping...")

        # Rename output files for this gamma
        if gamma_successes > 0:
            print(f"  Renaming output files for gamma = {gamma}")
            rename_output_files(gamma)

        print(
            f"  Summary: {gamma_successes}/{len(SIMULATION_SCRIPTS)} simulations successful"
        )

    # Final summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"Total time elapsed: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    print(f"Successful simulations: {successful_runs} / {total_runs}")
    print(f"Success rate: {100 * successful_runs / total_runs:.1f}%")
    print(f"Output files saved to: {OUTPUT_DIR}")

    if successful_runs == total_runs:
        print("🎉 All simulations completed successfully!")
    elif successful_runs > 0:
        print("⚠️  Some simulations failed - check output above")
    else:
        print("❌ All simulations failed - check configuration")

    print("=" * 70)


if __name__ == "__main__":
    main()
