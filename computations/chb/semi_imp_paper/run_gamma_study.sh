#!/bin/bash

# CHB Gamma Parameter Study - Bash Script Version
# This script modifies gamma values and runs all simulations

set -e  # Exit on any error

# Configuration
GAMMA_VALUES=(0.25 0.5 1 2 4 8 16)
PYTHON_EXE="/Users/erlend/miniforge3/envs/fenicsx-env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="../output/log"

# Simulation scripts to run
SCRIPTS=(
    "chb_monolithic_semi_imp.py"
    "chb_monolithic_imp.py"
    "chb_splitting_ch_biot_semi_imp.py"
    "chb_splitting_ch_biot_imp.py"
    "chb_splitting_ch_fixed_stress_semi_imp.py"
    "chb_splitting_ch_fixed_stress_imp.py"
)

# Function to modify gamma parameter in a file
modify_gamma() {
    local file="$1"
    local gamma_val="$2"
    
    if [[ -f "$file" ]]; then
        # Use sed to replace gamma = <number> with new value
        sed -i '' "s/gamma = [0-9]*\.?[0-9]*/gamma = $gamma_val/g" "$file"
        echo "  ✓ Updated gamma = $gamma_val in $(basename "$file")"
        return 0
    else
        echo "  ✗ File not found: $file"
        return 1
    fi
}

# Function to run simulation
run_simulation() {
    local script="$1"
    echo "    Running $script..."
    
    if "$PYTHON_EXE" "$script"; then
        echo "    ✓ Completed successfully"
        return 0
    else
        echo "    ✗ Simulation failed"
        return 1
    fi
}

# Function to rename output files
rename_outputs() {
    local gamma_val="$1"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # List of expected output files
    local outputs=(
        "chb_monolithic_semi_imp.xlsx"
        "chb_monolithic_imp.xlsx"
        "chb_splitting_ch_biot_semi_imp.xlsx"
        "chb_splitting_ch_biot_imp.xlsx"
        "chb_splitting_ch_fixedstress_semi_imp.xlsx"
    )
    
    for output in "${outputs[@]}"; do
        local old_path="$OUTPUT_DIR/$output"
        if [[ -f "$old_path" ]]; then
            local base_name="${output%.xlsx}"
            local new_name="${base_name}_gamma_${gamma_val}.xlsx"
            local new_path="$OUTPUT_DIR/$new_name"
            
            mv "$old_path" "$new_path"
            echo "    Renamed $output → $new_name"
            
            # Also rename CSV if exists
            local csv_old="${old_path%.xlsx}.csv"
            local csv_new="${new_path%.xlsx}.csv"
            if [[ -f "$csv_old" ]]; then
                mv "$csv_old" "$csv_new"
                echo "    Renamed $(basename "$csv_old") → $(basename "$csv_new")"
            fi
        fi
    done
}

# Main execution
main() {
    echo "======================================================================"
    echo "CHB Gamma Parameter Study - Bash Script"
    echo "======================================================================"
    echo "Gamma values: ${GAMMA_VALUES[*]}"
    echo "Scripts to run: ${#SCRIPTS[@]}"
    echo "Total simulations: $((${#GAMMA_VALUES[@]} * ${#SCRIPTS[@]}))"
    echo "======================================================================"
    
    # Check if all scripts exist
    for script in "${SCRIPTS[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            echo "❌ Missing script: $script"
            exit 1
        fi
    done
    echo "✓ All simulation scripts found"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo "✓ Output directory: $OUTPUT_DIR"
    
    local successful_runs=0
    local total_runs=$((${#GAMMA_VALUES[@]} * ${#SCRIPTS[@]}))
    local start_time=$(date +%s)
    
    # Main loop over gamma values
    local gamma_idx=0
    for gamma in "${GAMMA_VALUES[@]}"; do
        gamma_idx=$((gamma_idx + 1))
        echo ""
        echo "[$gamma_idx/${#GAMMA_VALUES[@]}] Processing gamma = $gamma"
        echo "--------------------------------------------------"
        
        local gamma_successes=0
        
        # Run all scripts for this gamma
        local script_idx=0
        for script in "${SCRIPTS[@]}"; do
            script_idx=$((script_idx + 1))
            local script_path="$SCRIPT_DIR/$script"
            
            echo "  [$script_idx/${#SCRIPTS[@]}] $script"
            
            # Modify gamma parameter
            if modify_gamma "$script_path" "$gamma"; then
                # Run simulation
                if run_simulation "$script_path"; then
                    successful_runs=$((successful_runs + 1))
                    gamma_successes=$((gamma_successes + 1))
                else
                    echo "    ⚠️  Simulation failed, continuing..."
                fi
            else
                echo "    ⚠️  Could not modify gamma, skipping..."
            fi
        done
        
        # Rename output files for this gamma
        if [[ $gamma_successes -gt 0 ]]; then
            echo "  Renaming output files for gamma = $gamma"
            rename_outputs "$gamma"
        fi
        
        echo "  Summary: $gamma_successes/${#SCRIPTS[@]} simulations successful"
    done
    
    # Final summary
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    echo ""
    echo "======================================================================"
    echo "FINAL SUMMARY"
    echo "======================================================================"
    echo "Total time elapsed: ${total_time} seconds ($((total_time / 60)) minutes)"
    echo "Successful simulations: $successful_runs/$total_runs"
    echo "Success rate: $(( (successful_runs * 100) / total_runs ))%"
    echo "Output files saved to: $OUTPUT_DIR"
    
    if [[ $successful_runs -eq $total_runs ]]; then
        echo "🎉 All simulations completed successfully!"
    elif [[ $successful_runs -gt 0 ]]; then
        echo "⚠️  Some simulations failed - check output above"
    else
        echo "❌ All simulations failed - check configuration"
    fi
    
    echo "======================================================================"
}

# Run main function
main "$@"