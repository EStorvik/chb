#!/usr/bin/env python3
"""
Swelling Parameter Iterations Analysis Script

This script analyzes all swelling parameter study Excel files and creates
a summary table showing:
- Total number of iterations per swelling parameter
- Total time spent per swelling parameter
- Organized by simulation method

Author: Analysis script for CHB swelling parameter study
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Configuration
OUTPUT_DIR = Path(__file__).parent / "../output/log"

# Simulation method names (from the file naming pattern)
METHODS = [
    "chb_monolithic_semi_imp",
    "chb_monolithic_imp",
    "chb_splitting_ch_biot_semi_imp",
    "chb_splitting_ch_biot_imp",
    "chb_splitting_ch_fixedstress_semi_imp",
    "chb_splitting_ch_fixedstress_imp",
]


def find_parameter_files(param_name: str) -> List[Path]:
    """
    Find all Excel files containing a given parameter in the filename.

    Args:
        param_name: The parameter name to search for (e.g., 'swelling', 'gamma')

    Returns:
        List of Path objects for parameter study files
    """
    param_files = []

    if not OUTPUT_DIR.exists():
        print(f"⚠️  Output directory does not exist: {OUTPUT_DIR}")
        return param_files

    # Look for files with pattern: *_{param_name}_*.xlsx
    for file in OUTPUT_DIR.glob(f"*_{param_name}_*.xlsx"):
        param_files.append(file)

    return sorted(param_files)


def extract_parameter_value(filename: str, param_name: str) -> float:
    """
    Extract parameter value from filename.

    Args:
        filename: Name of the file
        param_name: Parameter name (e.g., 'swelling', 'gamma')

    Returns:
        Parameter value as float, or None if not found
    """
    import re

    # Pattern: _param_<number> (handles integers and decimals)
    match = re.search(rf"_{param_name}_([\d]+(?:\.[\d]+)?)", filename)
    if match:
        value_str = match.group(1)
        try:
            return float(value_str)
        except ValueError as e:
            print(f"  ⚠️  Could not parse '{value_str}' as float from: {filename}")
            print(f"      Error: {e}")
            return None
    print(f"  ⚠️  No {param_name} pattern found in: {filename}")
    return None


def extract_method_name(filename: str) -> str:
    """
    Extract simulation method name from filename.

    Args:
        filename: Name of the file

    Returns:
        Method name string
    """
    for method in METHODS:
        if method in filename:
            return method
    return "unknown"


def analyze_file(file_path: Path, param_name: str) -> Tuple[float, int, float]:
    """
    Analyze a single Excel file to extract parameter value, iterations, and time.

    Args:
        file_path: Path to the Excel file
        param_name: Parameter name (e.g., 'swelling', 'gamma')

    Returns:
        Tuple of (parameter_value, total_iterations, total_time)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=0)

        # Extract parameter value from filename
        param_value = extract_parameter_value(file_path.name, param_name)

        if param_value is None:
            print(f"  ⚠️  Could not extract {param_name} value from {file_path.name}")
            return None, 0, 0

        # Calculate total iterations (column should be 'Iterations')
        if "Iterations" in df.columns:
            total_iterations = int(df["Iterations"].sum())
        else:
            print(f"  ⚠️  No 'Iterations' column found in {file_path.name}")
            print(f"      Available columns: {list(df.columns)}")
            total_iterations = 0

        # Calculate total time (third column, index 2)
        if len(df.columns) > 2:
            # Assume third column is 'Time'
            time_col = df.columns[2]
            total_time = float(df[time_col].sum())
        else:
            print(f"  ⚠️  Not enough columns in {file_path.name}")
            print(f"      Available columns: {list(df.columns)}")
            total_time = 0.0

        return param_value, total_iterations, total_time

    except Exception as e:
        print(f"  ✗ Error reading {file_path.name}: {e}")
        import traceback

        traceback.print_exc()
        return None, 0, 0


def create_summary_tables(
    results: Dict[str, List[Tuple[float, int, float]]], param_label: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create summary DataFrames from results.

    Args:
        results: Dictionary mapping method names to list of
                 (param_value, iterations, time) tuples
        param_label: Label for the parameter (e.g., 'Swelling_Parameter', 'Gamma_Parameter')

    Returns:
        Tuple of (iterations_df, time_df, totals_df)
    """
    # Get all unique parameter values
    all_param_values = set()
    for method_results in results.values():
        for param_value, _, _ in method_results:
            if param_value is not None:
                all_param_values.add(param_value)

    param_values = sorted(all_param_values)

    # Create dictionaries for iterations and time
    iterations_data = {param_label: param_values}
    time_data = {param_label: param_values}

    # Fill in data for each method
    for method in METHODS:
        iterations_col = []
        time_col = []

        if method in results:
            # Create lookup dictionary
            method_dict = {
                param_value: (iters, time)
                for param_value, iters, time in results[method]
                if param_value is not None
            }

            for param_value in param_values:
                if param_value in method_dict:
                    iters, time = method_dict[param_value]
                    iterations_col.append(iters)
                    time_col.append(time)
                else:
                    iterations_col.append(None)
                    time_col.append(None)
        else:
            iterations_col = [None] * len(param_values)
            time_col = [None] * len(param_values)

        # Shorten method name for column header
        short_name = method.replace("chb_", "").replace("_", " ").title()
        iterations_data[short_name] = iterations_col
        time_data[short_name] = time_col

    iterations_df = pd.DataFrame(iterations_data)
    time_df = pd.DataFrame(time_data)

    # Create totals DataFrame (sum across all methods for each parameter value)
    totals_data = {param_label: param_values}

    total_iterations = []
    total_time = []

    for param_value in param_values:
        param_iters = 0
        param_time = 0.0

        for method_results in results.values():
            for s, iters, time in method_results:
                if s == param_value:
                    param_iters += iters
                    param_time += time

        total_iterations.append(param_iters)
        total_time.append(param_time)

    totals_data["Total_Iterations"] = total_iterations
    totals_data["Total_Time_Seconds"] = total_time
    totals_data["Total_Time_Minutes"] = [t / 60 for t in total_time]

    totals_df = pd.DataFrame(totals_data)

    return iterations_df, time_df, totals_df


def analyze_parameter(param_name: str, param_label: str, summary_file_prefix: str):
    print("=" * 80)
    print(f"CHB {param_label.replace('_', ' ')} Study - Iterations & Time Analysis")
    print("=" * 80)

    # Find all parameter files
    param_files = find_parameter_files(param_name)

    if not param_files:
        print(f"❌ No {param_name} parameter files found in output directory")
        print(f"   Looking in: {OUTPUT_DIR}")
        return

    print(f"✓ Found {len(param_files)} {param_name} parameter files")

    # Debug: show all found files
    print("\nFiles found:")
    for f in param_files:
        param_val = extract_parameter_value(f.name, param_name)
        method = extract_method_name(f.name)
        print(f"  - {f.name}")
        print(f"    → {param_label.replace('_', ' ')}: {param_val}, Method: {method}")
    print()

    # Analyze each file and organize by method
    results = {}

    for file_path in param_files:
        print(f"Analyzing: {file_path.name}")
        method = extract_method_name(file_path.name)
        param_value, iterations, time_spent = analyze_file(file_path, param_name)

        if method not in results:
            results[method] = []

        results[method].append((param_value, iterations, time_spent))

    print()
    print("=" * 80)
    print("Creating Summary Tables")
    print("=" * 80)

    # Create summary tables
    iterations_df, time_df, totals_df = create_summary_tables(results, param_label)

    # Save to Excel with multiple sheets
    output_file = OUTPUT_DIR / f"{summary_file_prefix}_iterations_time_summary.xlsx"

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            # Write totals summary (main table)
            totals_df.to_excel(
                writer, sheet_name=f"Totals_By_{param_label}", index=False
            )

            # Write iterations summary by method
            iterations_df.to_excel(
                writer, sheet_name=f"Iterations_By_Method", index=False
            )

            # Write time summary by method
            time_df.to_excel(writer, sheet_name=f"Time_By_Method", index=False)

            # Create a combined summary with both metrics
            combined_rows = []
            for idx, param_value in enumerate(iterations_df[param_label]):
                row_data = {param_label: param_value, "Metric": "Iterations"}
                for col in iterations_df.columns[1:]:
                    row_data[col] = iterations_df.loc[idx, col]
                combined_rows.append(row_data)

                row_data = {param_label: param_value, "Metric": "Time (s)"}
                for col in time_df.columns[1:]:
                    row_data[col] = time_df.loc[idx, col]
                combined_rows.append(row_data)

            combined_df = pd.DataFrame(combined_rows)
            combined_df.to_excel(writer, sheet_name="Combined_By_Method", index=False)

        print(f"✓ Summary file created: {output_file}")
        print()
        print(f"Summary contains 4 sheets:")
        print(
            f"  1. Totals_By_{param_label} - TOTAL iterations and time per {param_name}"
        )
        print(f"  2. Iterations_By_Method - Iterations breakdown by method")
        print(f"  3. Time_By_Method - Time breakdown by method")
        print(f"  4. Combined_By_Method - Both metrics by method")
        print()

        # Display preview of totals
        print(f"Preview - Totals by {param_label.replace('_', ' ')}:")
        print(totals_df.to_string(index=False))
        print()
        print("Preview - Total Iterations by Method:")
        print(iterations_df.to_string(index=False))
        print()
        print("Preview - Total Time by Method (seconds):")
        print(time_df.to_string(index=False))

    except ImportError:
        # Fallback to CSV
        csv_totals = output_file.with_name(f"{summary_file_prefix}_totals_summary.csv")
        csv_iterations = output_file.with_name(
            f"{summary_file_prefix}_iterations_summary.csv"
        )
        csv_time = output_file.with_name(f"{summary_file_prefix}_time_summary.csv")

        totals_df.to_csv(csv_totals, index=False)
        iterations_df.to_csv(csv_iterations, index=False)
        time_df.to_csv(csv_time, index=False)

        print(f"✓ CSV files created:")
        print(f"  - {csv_totals}")
        print(f"  - {csv_iterations}")
        print(f"  - {csv_time}")
        print("  (Install openpyxl for Excel format: pip install openpyxl)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    # Analyze swelling parameter files
    analyze_parameter("swelling", "Swelling_Parameter", "swelling")
    # Analyze gamma parameter files
    analyze_parameter("gamma", "Gamma_Parameter", "gamma")
