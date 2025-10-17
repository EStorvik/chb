#!/usr/bin/env python3
"""
Automated script to run parametric study over swelling parameter values for all CHB simulations.

This script will:
1. Fix gamma = 2 in each simulation file
2. Modify swelling_parameter in each simulation file
3. Run the simulation 
4. Rename output files to include swelling_parameter value
5. Repeat for all swelling_parameter values in the study

Author: Auto-generated script for CHB swelling parameter study
"""

import os
import re
import subprocess
import shutil
import time
from pathlib import Path

# Configuration
SWELLING_VALUES = [0.0625, 0.125, 0.25, 0.5, 1]
FIXED_GAMMA = 2  # Fixed gamma value for this study
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "../output/log"
PYTHON_EXECUTABLE = "/Users/erlend/miniforge3/envs/fenicsx-env/bin/python"

# List of simulation scripts to run
SIMULATION_SCRIPTS = [
    "chb_monolithic_semi_imp.py",
    "chb_monolithic_imp.py", 
    "chb_splitting_ch_biot_semi_imp.py",
    "chb_splitting_ch_biot_imp.py",
    "chb_splitting_ch_fixed_stress_semi_imp.py",
    "chb_splitting_ch_fixed_stress_imp.py"
]

def modify_parameters(script_path: Path, swelling_value: float, gamma_value: float) -> bool:
    """
    Modify the swelling_parameter and gamma parameters in a simulation script.
    
    Args:
        script_path: Path to the simulation script
        swelling_value: New swelling_parameter value to set
        gamma_value: Fixed gamma value to set
        
    Returns:
        True if modification was successful, False otherwise
    """
    try:
        # Read the file
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Find and replace gamma parameter
        gamma_pattern = r'gamma\s*=\s*[0-9]+\.?[0-9]*'
        new_gamma = f'gamma = {gamma_value}'
        
        # Find and replace swelling_parameter
        # Pattern matches: swelling_parameter=<number>
        swelling_pattern = r'swelling_parameter\s*=\s*[0-9]+\.?[0-9]*'
        new_swelling = f'swelling_parameter={swelling_value}'
        
        modifications_made = 0
        
        # Update gamma
        if re.search(gamma_pattern, content):
            content = re.sub(gamma_pattern, new_gamma, content)
            modifications_made += 1
            print(f"    ✓ Updated gamma = {gamma_value}")
        else:
            print(f"    ⚠️  Could not find gamma parameter in {script_path.name}")
        
        # Update swelling_parameter
        if re.search(swelling_pattern, content):
            content = re.sub(swelling_pattern, new_swelling, content)
            modifications_made += 1
            print(f"    ✓ Updated swelling_parameter = {swelling_value}")
        else:
            print(f"    ✗ Could not find swelling_parameter in {script_path.name}")
            return False
            
        if modifications_made > 0:
            # Write back to file
            with open(script_path, 'w') as f:
                f.write(content)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"    ✗ Error modifying {script_path.name}: {e}")
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
        
        # Set environment for local serial execution (completely disable networking)
        env = os.environ.copy()
        env.update({
            # Disable OFI completely
            'FI_PROVIDER': 'tcp',
            'MPICH_NETMASK': '0.0.0.0/0',
            
            # Force OpenMPI to use only local communication
            'OMPI_MCA_btl': 'self,vader',
            'OMPI_MCA_oob': 'tcp',
            'OMPI_MCA_pml': 'ob1',  # Force specific point-to-point layer
            
            # Disable problematic network interfaces  
            'OMPI_MCA_btl_tcp_if_exclude': 'lo,docker0,virbr0',
            'OMPI_MCA_oob_tcp_if_exclude': 'lo,docker0,virbr0',
            
            # Disable OFI/libfabric completely
            'MPICH_CH3_NETMOD': 'tcp',
            'I_MPI_FABRICS': 'tcp',
            
            # Single-threaded execution
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            
            # Disable UCX (another transport layer)
            'OMPI_MCA_pml_ucx_priority': '0',
        })
        
        # Run the script
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
            env=env  # Use local-only environment
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"    ✓ Completed in {elapsed_time:.1f}s")
            return True
        elif result.returncode == 130:  # KeyboardInterrupt (Ctrl+C)
            print(f"    ⚠️  Interrupted by user")
            return False
        else:
            # Check if it's just an MPI finalization error but simulation completed
            stderr_output = result.stderr
            if ("MPI_Finalize failed" in stderr_output or 
                "MPIDI_OFI" in stderr_output or 
                "OFI poll failed" in stderr_output):
                # Check if output files were still generated (simulation likely succeeded)
                expected_outputs = ["../output/log/" + script_path.stem + ".xlsx"]
                output_exists = any(Path(script_path.parent / output).exists() 
                                  for output in expected_outputs)
                
                if output_exists:
                    print(f"    ⚠️  MPI finalization error, but output files exist - treating as success")
                    print(f"    ✓ Completed in {elapsed_time:.1f}s (with MPI cleanup warning)")
                    return True
                else:
                    print(f"    ✗ MPI error and no output files found")
                    print(f"    Error output: {stderr_output[:500]}...")  # Truncate long errors
                    return False
            else:
                print(f"    ✗ Failed with return code {result.returncode}")
                print(f"    Error output: {stderr_output[:500]}...")  # Truncate long errors
                return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout after 15 minutes")
        return False
    except Exception as e:
        print(f"    ✗ Error running simulation: {e}")
        return False

def rename_output_files(swelling_value: float) -> None:
    """
    Rename output files to include swelling_parameter value.
    
    Args:
        swelling_value: Current swelling_parameter value being processed
    """
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Expected output files based on the script patterns
        output_files = [
            "chb_monolithic_semi_imp.xlsx",
            "chb_monolithic_imp.xlsx",
            "chb_splitting_ch_biot_semi_imp.xlsx", 
            "chb_splitting_ch_biot_imp.xlsx",
            "chb_splitting_ch_fixedstress_semi_imp.xlsx",
            "chb_splitting_ch_fixedstress_imp.xlsx"  # Correct filename for fixed_stress_imp
        ]
        
        for filename in output_files:
            old_path = OUTPUT_DIR / filename
            if old_path.exists():
                # Create new filename with swelling_parameter value
                name_stem = old_path.stem
                new_filename = f"{name_stem}_swelling_{swelling_value}.xlsx"
                new_path = OUTPUT_DIR / new_filename
                
                # Rename/move the file
                shutil.move(str(old_path), str(new_path))
                print(f"    Renamed {filename} → {new_filename}")
                
                # Also handle CSV files if they exist
                csv_old = old_path.with_suffix('.csv')
                if csv_old.exists():
                    csv_new = new_path.with_suffix('.csv')
                    shutil.move(str(csv_old), str(csv_new))
                    print(f"    Renamed {csv_old.name} → {csv_new.name}")
                    
    except Exception as e:
        print(f"    ✗ Error renaming output files: {e}")

def main():
    """Main execution function."""
    print("=" * 80)
    print("CHB Swelling Parameter Study - Automated Simulation Runner")
    print("=" * 80)
    print(f"Fixed gamma value: {FIXED_GAMMA}")
    print(f"Swelling parameter values: {SWELLING_VALUES}")
    print(f"Scripts to run: {len(SIMULATION_SCRIPTS)}")
    print(f"Total simulations: {len(SWELLING_VALUES)} × {len(SIMULATION_SCRIPTS)} = {len(SWELLING_VALUES) * len(SIMULATION_SCRIPTS)}")
    print("=" * 80)
    
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
    total_runs = len(SWELLING_VALUES) * len(SIMULATION_SCRIPTS)
    
    # Main loop: iterate over swelling_parameter values
    for i, swelling in enumerate(SWELLING_VALUES, 1):
        print(f"\n[{i}/{len(SWELLING_VALUES)}] Processing swelling_parameter = {swelling} (gamma = {FIXED_GAMMA})")
        print("-" * 60)
        
        # Track successes for this swelling_parameter
        swelling_successes = 0
        
        # Run all simulations for this swelling_parameter value
        for j, script_name in enumerate(SIMULATION_SCRIPTS, 1):
            script_path = SCRIPT_DIR / script_name
            
            print(f"  [{j}/{len(SIMULATION_SCRIPTS)}] {script_name}")
            
            # Modify swelling_parameter and gamma parameters
            if modify_parameters(script_path, swelling, FIXED_GAMMA):
                # Run simulation
                if run_simulation(script_path):
                    successful_runs += 1
                    swelling_successes += 1
                else:
                    print(f"    ⚠️  Simulation failed, continuing...")
            else:
                print(f"    ⚠️  Could not modify parameters, skipping...")
        
        # Rename output files for this swelling_parameter
        if swelling_successes > 0:
            print(f"  Renaming output files for swelling_parameter = {swelling}")
            rename_output_files(swelling)
        
        print(f"  Summary: {swelling_successes}/{len(SIMULATION_SCRIPTS)} simulations successful")
    
    # Final summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total time elapsed: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful simulations: {successful_runs}/{total_runs}")
    print(f"Success rate: {100*successful_runs/total_runs:.1f}%")
    print(f"Output files saved to: {OUTPUT_DIR}")
    
    if successful_runs == total_runs:
        print("🎉 All simulations completed successfully!")
    elif successful_runs > 0:
        print("⚠️  Some simulations failed - check output above")
    else:
        print("❌ All simulations failed - check configuration")
    
    print("=" * 80)

if __name__ == "__main__":
    main()