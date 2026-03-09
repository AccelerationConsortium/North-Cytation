#!/usr/bin/env python3
"""
V2 Calibration Format Converter
===============================

Converts calibration files from v2 format to original pipetting wizard format.
Takes files from calibration_modular_v2/optimized_parameters/ and converts them
to be compatible with pipetting_data/pipetting_wizard.py

V2 Format → Original Format Mapping:
- hardware_parameters_aspirate_speed → aspirate_speed  
- hardware_parameters_dispense_speed → dispense_speed
- hardware_parameters_aspirate_wait_time → aspirate_wait_time
- hardware_parameters_retract_speed → retract_speed
- hardware_parameters_blowout_vol → blowout_vol
- hardware_parameters_post_asp_air_vol → post_asp_air_vol
- hardware_parameters_pre_asp_air_vol → pre_asp_air_vol
- calibration_overaspirate_vol → overaspirate_vol
- volume_target_ml → volume_target (converted to uL)

Missing parameters added with reasonable defaults:
- dispense_wait_time = 0.0
- post_retract_wait_time = 0.0

Usage:
    python convert_to_pipetting_wizard_format.py                    # Convert default file (edit DEFAULT_INPUT_FILE)
    python convert_to_pipetting_wizard_format.py input.csv          # Convert specific file
    python convert_to_pipetting_wizard_format.py input.csv output.csv  # Convert with custom output
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import sys

# Default file path - edit this to point to your target file
DEFAULT_INPUT_FILE = r"C:\Users\Imaging Controller\Desktop\utoronto_demo\calibration_modular_v2\output\run_1772849500_large_toluene\optimal_conditions_toluene_large.csv"

# Column mapping from v2 format to original format
COLUMN_MAPPING = {
    'hardware_parameters_aspirate_speed': 'aspirate_speed',
    'hardware_parameters_dispense_speed': 'dispense_speed', 
    'hardware_parameters_aspirate_wait_time': 'aspirate_wait_time',
    'hardware_parameters_retract_speed': 'retract_speed',
    'hardware_parameters_blowout_vol': 'blowout_vol',
    'hardware_parameters_post_asp_air_vol': 'post_asp_air_vol',
    'hardware_parameters_pre_asp_air_vol': 'pre_asp_air_vol',
    'hardware_parameters_dispense_wait_time': 'dispense_wait_time',
    'hardware_parameters_asp_disp_cycles': 'asp_disp_cycles',
    'calibration_overaspirate_vol': 'overaspirate_vol',
    'volume_target_ml': 'volume_target',  # Will convert mL to uL
}

# Additional column mappings (with multiple possible source names)
ADDITIONAL_COLUMN_MAPPING = {
    'volume_measured_ml': 'volume_measured',  # Will convert mL to uL
    'volume_measured_ul': 'volume_measured', 
    'volume_measured': 'volume_measured',
    'volume_target_ul': 'volume_target_ul',  # Keep original uL target for reference
    'duration_s': 'time',
    'time': 'time', 
    'deviation_pct': 'average_deviation',
    'average_deviation': 'average_deviation',
    'precision_cv_pct': 'variability',
    'variability': 'variability',
    'trials_count': 'trials_count',
    'status': 'status',
    'measurement_count': 'measurement_count',
}

# Parameters that should be added if missing (with default values)
DEFAULT_PARAMETERS = {
    'post_retract_wait_time': 0.0,
    'asp_disp_cycles': 0,
    'retract_speed': 5.0,
}

def setup_logging():
    """Configure logging for the converter."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def convert_file(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single v2 calibration file to original pipetting wizard format.
    
    Args:
        input_path: Path to v2 format CSV file
        output_path: Path where converted file should be saved
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Load the v2 format file
        logging.info(f"Loading v2 calibration file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Create new DataFrame for converted format
        converted_df = pd.DataFrame()
        
        # Map columns from v2 to original format
        for v2_col, original_col in COLUMN_MAPPING.items():
            if v2_col in df.columns:
                if original_col == 'volume_target' and v2_col == 'volume_target_ml':
                    # Convert mL to uL for volume_target
                    converted_df[original_col] = df[v2_col] * 1000
                    logging.info(f"Converted {v2_col} (mL) → {original_col} (uL)")
                else:
                    converted_df[original_col] = df[v2_col]
                    logging.info(f"Mapped {v2_col} → {original_col}")
            else:
                logging.warning(f"Column {v2_col} not found in input file")
        
        # Add default parameters if missing
        for param, default_value in DEFAULT_PARAMETERS.items():
            if param not in converted_df.columns:
                converted_df[param] = default_value
                logging.info(f"Added missing parameter {param} = {default_value}")
        
        # Copy over any additional useful columns using the mapping
        for v2_col, original_col in ADDITIONAL_COLUMN_MAPPING.items():
            if v2_col in df.columns and original_col not in converted_df.columns:
                if v2_col == 'volume_measured_ml':
                    # Convert mL to uL
                    converted_df[original_col] = df[v2_col] * 1000
                    logging.info(f"Converted {v2_col} (mL) → {original_col} (uL)")
                else:
                    converted_df[original_col] = df[v2_col]
                    logging.info(f"Mapped {v2_col} → {original_col}")
        
        # Sort by volume_target for consistency with original wizard
        converted_df = converted_df.sort_values('volume_target').reset_index(drop=True)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save converted file
        converted_df.to_csv(output_path, index=False)
        logging.info(f"Saved converted file: {output_path}")
        logging.info(f"Converted {len(converted_df)} rows with {len(converted_df.columns)} columns")
        
        return True
        
    except Exception as e:
        logging.error(f"Error converting file {input_path}: {e}")
        return False

def convert_all_files(input_dir: Path, output_dir: Path):
    """
    Convert all optimal_conditions_*.csv files from input directory to output directory.
    
    Args:
        input_dir: Directory containing v2 calibration files
        output_dir: Directory where converted files should be saved
    """
    # Find all optimal_conditions_*.csv files
    pattern = "optimal_conditions_*.csv"
    input_files = list(input_dir.glob(pattern))
    
    if not input_files:
        logging.warning(f"No files matching {pattern} found in {input_dir}")
        return
    
    logging.info(f"Found {len(input_files)} files to convert")
    
    converted_count = 0
    for input_file in input_files:
        # Generate output filename
        output_file = output_dir / input_file.name
        
        # Check if output file already exists
        if output_file.exists():
            logging.warning(f"Output file {output_file} already exists, skipping")
            continue
        
        # Convert the file
        if convert_file(input_file, output_file):
            converted_count += 1
    
    logging.info(f"Successfully converted {converted_count}/{len(input_files)} files")

def main():
    """Main function with command line interface."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Convert v2 calibration files to pipetting wizard format")
    parser.add_argument("input", nargs="?", help="Input v2 calibration file (if not specified, converts all files)")
    parser.add_argument("output", nargs="?", help="Output file path (if not specified, saves to pipetting_data/)")
    parser.add_argument("--single", "-s", action="store_true", help="Process single file mode (requires input file)")
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    if args.input:
        # Convert single file
        input_path = Path(args.input)
        
        if not input_path.exists():
            logging.error(f"Input file {input_path} does not exist")
            return 1
        
        # Generate output path if not provided
        if args.output:
            output_path = Path(args.output)
        else:
            # Save to pipetting_data with same filename
            output_dir = script_dir.parent / "pipetting_data"
            output_path = output_dir / input_path.name
        
        logging.info(f"Converting single file: {input_path} -> {output_path}")
        success = convert_file(input_path, output_path)
        return 0 if success else 1
    
    else:
        # Use default file path
        input_path = Path(DEFAULT_INPUT_FILE)
        
        if not input_path.exists():
            logging.error(f"Default file {input_path} does not exist")
            logging.error("Update DEFAULT_INPUT_FILE constant at top of script or provide input file as argument")
            return 1
        
        # Save to pipetting_data with same filename
        output_dir = script_dir.parent / "pipetting_data"
        output_path = output_dir / input_path.name
        
        logging.info(f"Converting default file: {input_path} -> {output_path}")
        success = convert_file(input_path, output_path)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())