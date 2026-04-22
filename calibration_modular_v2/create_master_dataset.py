"""
Master Dataset Creator for Calibration Analysis
==============================================

This script consolidates ALL calibration and validation data from the entire
experimental database into a single master dataset for comprehensive analysis.

Key Features:
- Automatically detects calibration vs validation experiments
- Skips simulated data (checks config files)
- Combines all raw measurements with proper labeling
- Handles different column formats between calibration and validation
- Creates comprehensive master measurement list
- Exports analysis-ready datasets

Output Files (in master_data/ folder):
- master_measurements_{timestamp}.csv - All individual measurements
- master_trials_{timestamp}.csv - Trial-level summaries  
- master_optimal_conditions_{timestamp}.csv - Best parameters from each experiment
- data_compilation_report_{timestamp}.txt - Processing summary

Usage:
    python create_master_dataset.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_dataset_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterDatasetCreator:
    """Creates master dataset from all calibration and validation experiments."""
    
    def __init__(self, base_dir: str = "."):
        # Use the directory where this script is located (calibration_modular_v2)
        self.base_dir = Path(__file__).parent.absolute()
        self.output_dir = self.base_dir / "output"
        self.validation_dir = self.base_dir / "validation"
        
        # Master data storage
        self.master_measurements = []
        self.master_trials = []
        self.master_optimal_conditions = []
        
        # Processing statistics
        self.stats = {
            'total_folders_found': 0,
            'calibration_folders': 0,
            'validation_folders': 0,
            'simulated_folders_skipped': 0,
            'empty_folders_skipped': 0,
            'successfully_processed': 0,
            'errors': 0,
            'total_measurements': 0,
            'total_trials': 0
        }
        
        logger.info("Master Dataset Creator initialized")
        logger.info(f"Working directory: {self.base_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Validation directory: {self.validation_dir}")
    
    def is_simulation_experiment(self, folder_path: Path) -> Tuple[bool, Optional[Dict]]:
        """
        Check if experiment is simulated by examining config file.
        
        Returns:
            (is_simulation, config_dict)
        """
        config_files = [
            "experiment_config_used.yaml",
            "experiment_config.yaml", 
            "config.yaml"
        ]
        
        for config_file in config_files:
            config_path = folder_path / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Check simulation flag in experiment section
                    if 'experiment' in config:
                        simulate = config['experiment'].get('simulate', False)
                        if simulate:
                            logger.info(f"SKIPPING SIMULATION: {folder_path.name} (simulate: true in {config_file})")
                        return simulate, config
                    
                    # Fallback: check for simulation protocol
                    protocol = config.get('experiment', {}).get('hardware_protocol', '')
                    if 'simulated' in protocol.lower():
                        logger.info(f"SKIPPING SIMULATION: {folder_path.name} (simulated protocol in {config_file})")
                        return True, config
                        
                except Exception as e:
                    logger.warning(f"Error reading config from {config_path}: {e}")
                    continue
        
        # No config found - assume real data
        return False, None
    
    def extract_metadata_from_folder(self, folder_path: Path, config: Optional[Dict]) -> Dict:
        """Extract experiment metadata from folder and config."""
        metadata = {
            'experiment_folder': folder_path.name,
            'timestamp': None,
            'liquid': 'unknown'
        }
        
        # Try to get liquid from config file first
        if config:
            # Check various possible locations for liquid info
            liquid_sources = [
                config.get('experiment', {}).get('liquid'),
                config.get('liquid'),
                config.get('experiment', {}).get('liquid_name'),
                config.get('liquid_name'),
                config.get('experiment', {}).get('target_liquid')
            ]
            
            for liquid in liquid_sources:
                if liquid and isinstance(liquid, str):
                    metadata['liquid'] = liquid.strip()
                    break
        
        # If no liquid in config, try to get from raw_measurements.csv
        if metadata['liquid'] == 'unknown':
            raw_files = [
                folder_path / 'raw_measurements.csv',
                folder_path / 'output' / 'raw_measurements.csv'
            ]
            
            for raw_file in raw_files:
                if raw_file.exists():
                    try:
                        # Read just the first few rows to get liquid column
                        df_sample = pd.read_csv(raw_file, nrows=5)
                        if 'liquid' in df_sample.columns and not df_sample['liquid'].empty:
                            liquid_val = df_sample['liquid'].iloc[0]
                            if pd.notna(liquid_val):
                                metadata['liquid'] = str(liquid_val).strip()
                                break
                    except Exception as e:
                        logger.debug(f"Could not read liquid from {raw_file}: {e}")
                        continue
        
        # Try to extract timestamp from folder name
        folder_name = folder_path.name
        if 'run_' in folder_name:
            try:
                # Extract timestamp from run_XXXXXXXXXX format
                timestamp_str = folder_name.split('_')[1]
                if timestamp_str.isdigit():
                    metadata['timestamp'] = int(timestamp_str)
            except (IndexError, ValueError):
                pass
        
        return metadata
        
    def remove_duplicate_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Remove duplicate columns, keeping non-metadata versions when duplicates exist."""
        logger.info(f"Cleaning duplicate columns from {data_type}...")
        original_cols = len(df.columns)
        
        # Group columns by identical data
        duplicate_groups = []
        processed_columns = set()
        
        for col1 in df.columns:
            if col1 in processed_columns:
                continue
                
            similar_columns = [col1]
            for col2 in df.columns:
                if col2 == col1 or col2 in processed_columns:
                    continue
                    
                # Check if columns are identical
                if self._are_columns_identical(df[col1], df[col2]):
                    similar_columns.append(col2)
            
            if len(similar_columns) > 1:
                duplicate_groups.append(similar_columns)
                processed_columns.update(similar_columns)
        
        # Remove duplicate columns, preferring clean names over prefixed ones
        columns_to_drop = []
        for group in duplicate_groups:
            # Preference order: clean names > metadata_ > hardware_parameters_ > calibration_
            clean_names = [col for col in group if not any(col.startswith(prefix) for prefix in ['metadata_', 'hardware_parameters_', 'calibration_'])]
            metadata_cols = [col for col in group if col.startswith('metadata_')]
            hardware_cols = [col for col in group if col.startswith('hardware_parameters_')]
            calibration_cols = [col for col in group if col.startswith('calibration_')]
            other_cols = [col for col in group if col not in clean_names + metadata_cols + hardware_cols + calibration_cols]
            
            if clean_names:
                # Keep first clean name, drop everything else
                keep_col = clean_names[0]
                drop_cols = clean_names[1:] + metadata_cols + hardware_cols + calibration_cols + other_cols
            elif metadata_cols:
                # Keep first metadata column, drop the rest
                keep_col = metadata_cols[0]
                drop_cols = metadata_cols[1:] + hardware_cols + calibration_cols + other_cols
            else:
                # Keep first remaining column
                keep_col = group[0]
                drop_cols = group[1:]
            
            columns_to_drop.extend(drop_cols)
            if drop_cols:  # Only log if there are columns to drop
                logger.info(f"  Duplicate group: KEEP '{keep_col}', DROP {drop_cols}")
        
        # Drop duplicate columns
        if columns_to_drop:
            df_cleaned = df.drop(columns=columns_to_drop)
            logger.info(f"Removed {len(columns_to_drop)} duplicate columns from {data_type}")
            logger.info(f"  Columns: {original_cols} -> {len(df_cleaned.columns)}")
        else:
            df_cleaned = df
            logger.info(f"No duplicate columns found in {data_type}")
            
        return df_cleaned
    
    def _are_columns_identical(self, col1: pd.Series, col2: pd.Series, tolerance=0.001) -> bool:
        """Check if two columns contain identical data."""
        # Handle completely missing data
        if col1.isna().all() and col2.isna().all():
            return True
        
        # Get overlapping non-null positions
        mask = col1.notna() & col2.notna()
        if mask.sum() == 0:
            return col1.isna().equals(col2.isna())  # Same NaN pattern
        
        clean_col1 = col1[mask]
        clean_col2 = col2[mask]
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(clean_col1) and pd.api.types.is_numeric_dtype(clean_col2):
            try:
                return np.allclose(clean_col1, clean_col2, rtol=tolerance, atol=tolerance, equal_nan=True)
            except:
                return False
        
        # For other columns
        try:
            return clean_col1.equals(clean_col2)
        except:
            return False
    
    def determine_experiment_type(self, folder_path: Path) -> str:
        """Determine if folder contains calibration or validation data."""
        if 'validation' in str(folder_path):
            return 'validation'
        else:
            return 'calibration'
            
    def has_meaningful_data(self, folder_path: Path) -> bool:
        """Check if folder has actual measurement data (not just emergency files)."""
        # Check for main measurement files (calibration and validation)
        measurement_files = [
            folder_path / 'raw_measurements.csv',
            folder_path / 'raw_validation_data.csv',
            folder_path / 'output' / 'raw_measurements.csv'
        ]
        
        for file_path in measurement_files:
            if file_path.exists():
                try:
                    # Check file size and content
                    file_size = file_path.stat().st_size
                    logger.debug(f"Found measurement file {file_path} with size {file_size} bytes")
                    if file_size > 200:  # More than just headers
                        df = pd.read_csv(file_path, nrows=2)  # Check if has data rows
                        logger.debug(f"File has {len(df)} data rows")
                        if len(df) > 0:
                            return True
                except Exception as e:
                    logger.debug(f"Error checking {file_path}: {e}")
                    continue
                    
        logger.debug(f"No meaningful data found in {folder_path}")
        return False
    
    def extract_experiment_metadata(self, folder_path: Path, config: Dict) -> Dict:
        """Extract key metadata from experiment folder and config."""
        folder_name = folder_path.name
        
        # Parse timestamp from folder name
        timestamp_str = None
        if folder_name.startswith('run_'):
            timestamp_part = folder_name.split('_')[1]
            try:
                timestamp = datetime.fromtimestamp(int(timestamp_part))
                timestamp_str = timestamp.isoformat()
            except:
                pass
        elif folder_name.startswith('validation_run_'):
            # Format: validation_run_20260309_122906_water_small
            date_part = folder_name.split('_')[2:4]  # ['20260309', '122906']
            try:
                timestamp = datetime.strptime('_'.join(date_part), '%Y%m%d_%H%M%S')
                timestamp_str = timestamp.isoformat()
            except:
                pass
        
        # Extract liquid type from config or folder name
        liquid = 'unknown'
        if config and 'experiment' in config:
            liquid = config['experiment'].get('liquid', 'unknown')
        
        # Fallback: extract from folder name
        if liquid == 'unknown':
            name_parts = folder_name.lower().split('_')
            common_liquids = ['water', 'dmso', 'ethanol', 'tfa', 'hcl', 'sds', 'glycerol', 
                            'heptane', 'toluene', '2methf', 'pva', 'citric', 'h2so4', 'h3po4']
            for part in name_parts:
                if part in common_liquids:
                    liquid = part
                    break
        
        # Extract volume information
        volumes = []
        if config and 'experiment' in config:
            volumes = config['experiment'].get('volume_targets_ml', [])
        
        return {
            'experiment_folder': folder_name,
            'timestamp': timestamp_str,
            'liquid': liquid,
            'target_volumes_ml': volumes,
            'config_data': config
        }
    
    def process_raw_measurements(self, folder_path: Path, experiment_type: str, metadata: Dict) -> int:
        """Process raw measurements from a single experiment folder."""
        measurement_files = [
            folder_path / "raw_measurements.csv",
            folder_path / "raw_validation_data.csv",
            folder_path / "output" / "raw_measurements.csv"
        ]
        
        for measurements_file in measurement_files:
            if measurements_file.exists():
                logger.debug(f"Attempting to process {measurements_file}")
                break
        else:
            logger.warning(f"No measurements file (raw_measurements.csv or raw_validation_data.csv) found in {folder_path}")
            return 0
        
        try:
            df = pd.read_csv(measurements_file)
            logger.info(f"Loaded {len(df)} measurements from {measurements_file}")
            
            # Handle validation data format - parse JSON parameters
            if 'parameters_used' in df.columns and experiment_type == 'validation':
                df = self.parse_validation_parameters(df)
                logger.info(f"Parsed validation JSON parameters into {len(df.columns)} columns")
            
            # Add experiment metadata to each measurement
            df['experiment_type'] = experiment_type
            df['experiment_folder'] = metadata['experiment_folder'] 
            df['experiment_timestamp'] = metadata['timestamp']
            df['liquid_name'] = metadata['liquid']
            
            # Add to master list
            self.master_measurements.append(df)
            
            return len(df)
            
        except Exception as e:
            logger.error(f"Error processing measurements from {folder_path}: {e}")
            return 0
    
    def parse_validation_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse JSON parameters from validation data into separate columns with standardized names."""
        import json
        
        # Define mapping from validation JSON keys to clean standardized column names
        hardware_param_mapping = {
            # Clean hardware parameter names (no ugly prefixes)
            'aspirate_speed': 'aspirate_speed',
            'dispense_speed': 'dispense_speed',
            'overaspirate_vol': 'overaspirate_vol',
            'aspirate_wait_time': 'aspirate_wait_time', 
            'dispense_wait_time': 'dispense_wait_time',
            'pre_asp_air_vol': 'pre_asp_air_vol',
            'post_asp_air_vol': 'post_asp_air_vol',
            'blowout_vol': 'blowout_vol',
            'asp_disp_cycles': 'asp_disp_cycles',
            'retract_speed': 'retract_speed',
            'post_retract_wait_time': 'post_retract_wait_time',
            'volume_ml': 'target_volume_from_params_ml'  # Different from measured volume
        }
        
        protocol_param_mapping = {
            # Clean protocol result names - avoid duplicates
            'replicate': 'replicate_id',
            'start_time': 'start_time', 
            'end_time': 'end_time',
            'elapsed_s': 'duration_from_protocol_s',  # Already have duration_s column
            'target_volume_mL': 'target_volume_from_protocol_ml',  # Already have volume_target_ml
            'measured_volume_uL': 'measured_volume_from_protocol_ul',  # Already have volume_measured_ul
            'target_volume_uL': 'target_volume_from_protocol_ul'  # Already have volume_target_ul
            # Skip hardware params from protocol_result since they duplicate parameters_used
        }
        
        # Parse parameters_used JSON column
        if 'parameters_used' in df.columns:
            logger.debug("Parsing parameters_used JSON column...")
            param_records = []
            
            for idx, params_str in df['parameters_used'].items():
                try:
                    if pd.isna(params_str):
                        param_records.append({})
                    else:
                        # Parse JSON string
                        params_dict = json.loads(params_str.replace("'", '"'))  # Handle single quotes
                        # Map to standardized column names
                        mapped_params = {hardware_param_mapping.get(k, k): v  # Use original name if not mapped
                                       for k, v in params_dict.items()}
                        param_records.append(mapped_params)
                except Exception as e:
                    logger.warning(f"Error parsing parameters at row {idx}: {e}")
                    param_records.append({})
            
            # Convert to DataFrame and join with original
            params_df = pd.DataFrame(param_records)
            if not params_df.empty:
                df = pd.concat([df, params_df], axis=1)
                logger.info(f"Added {len(params_df.columns)} mapped hardware parameter columns")
        
        # Parse protocol_result JSON column - avoid hardware param duplicates
        if 'protocol_result' in df.columns:
            logger.debug("Parsing protocol_result JSON column...")
            result_records = []
            
            for idx, result_str in df['protocol_result'].items():
                try:
                    if pd.isna(result_str):
                        result_records.append({})
                    else:
                        result_dict = json.loads(result_str.replace("'", '"'))
                        # Only map non-hardware parameters to avoid duplicates
                        mapped_results = {}
                        for k, v in result_dict.items():
                            if k in protocol_param_mapping:
                                mapped_results[protocol_param_mapping[k]] = v
                            elif k not in hardware_param_mapping:  # Skip hardware params
                                mapped_results[k] = v  # Use clean original name
                        result_records.append(mapped_results)
                except Exception as e:
                    logger.warning(f"Error parsing protocol_result at row {idx}: {e}")
                    result_records.append({})
            
            result_df = pd.DataFrame(result_records)
            if not result_df.empty:
                df = pd.concat([df, result_df], axis=1)
                logger.info(f"Added {len(result_df.columns)} mapped protocol result columns")
        
        # Drop original JSON columns to save space
        json_cols = ['parameters_used', 'protocol_result']
        df = df.drop(columns=[col for col in json_cols if col in df.columns])
        
        return df
    
    def standardize_all_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename ugly column names to clean standardized names (after duplicates removed)."""
        
        # Create mapping only for columns that actually exist (no collisions)
        column_mapping = {}
        
        # Hardware parameters - remove ugly prefixes
        hw_mappings = {
            'hardware_parameters_aspirate_speed': 'aspirate_speed',
            'hardware_parameters_dispense_speed': 'dispense_speed', 
            'hardware_parameters_overaspirate_vol': 'overaspirate_vol',
            'hardware_parameters_aspirate_wait_time': 'aspirate_wait_time',
            'hardware_parameters_dispense_wait_time': 'dispense_wait_time',
            'hardware_parameters_pre_asp_air_vol': 'pre_asp_air_vol',
            'hardware_parameters_post_asp_air_vol': 'post_asp_air_vol',
            'hardware_parameters_blowout_vol': 'blowout_vol',
            'hardware_parameters_asp_disp_cycles': 'asp_disp_cycles',
            'hardware_parameters_retract_speed': 'retract_speed',
            'hardware_parameters_post_retract_wait_time': 'post_retract_wait_time',
        }
        
        # Metadata parameters - remove metadata_ prefix 
        meta_mappings = {
            'metadata_aspirate_speed': 'aspirate_speed',
            'metadata_dispense_speed': 'dispense_speed',
            'metadata_overaspirate_vol': 'overaspirate_vol', 
            'metadata_aspirate_wait_time': 'aspirate_wait_time',
            'metadata_dispense_wait_time': 'dispense_wait_time',
            'metadata_pre_asp_air_vol': 'pre_asp_air_vol',
            'metadata_post_asp_air_vol': 'post_asp_air_vol',
            'metadata_blowout_vol': 'blowout_vol',
            'metadata_asp_disp_cycles': 'asp_disp_cycles',
            'metadata_retract_speed': 'retract_speed',
            'metadata_post_retract_wait_time': 'post_retract_wait_time',
            'metadata_start_time': 'start_time',
            'metadata_end_time': 'end_time',
            'metadata_replicate': 'replicate_id',
            'metadata_source': 'source_info',
        }
        
        # Other mappings
        other_mappings = {
            'calibration_overaspirate_vol': 'overaspirate_vol',
            'replicate': 'replicate_id'
        }
        
        # Only add mappings for columns that actually exist (no collisions after dedup)
        for old_col, new_col in {**hw_mappings, **meta_mappings, **other_mappings}.items():
            if old_col in df.columns:
                column_mapping[old_col] = new_col
        
        # Apply the mapping - rename columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Standardized {len(column_mapping)} column names: {sorted(column_mapping.keys())}")
        else:
            logger.info("No column names needed standardization")
        
        return df
    
    def merge_duplicate_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge columns that have identical names (created by parsing + renaming)."""
        
        # Find columns with duplicate names
        col_counts = pd.Series(df.columns).value_counts()
        duplicate_names = col_counts[col_counts > 1].index.tolist()
        
        if not duplicate_names:
            logger.info("No duplicate column names found")
            return df
        
        logger.info(f"Found {len(duplicate_names)} duplicate column names: {duplicate_names}")
        
        # Process each duplicate name
        for col_name in duplicate_names:
            # Get all columns with this name
            duplicate_cols = [i for i, name in enumerate(df.columns) if name == col_name]
            logger.info(f"Merging {len(duplicate_cols)} columns named '{col_name}'")
            
            # Merge by filling NaN values from left to right
            merged_series = df.iloc[:, duplicate_cols[0]].copy()
            for col_idx in duplicate_cols[1:]:
                merged_series = merged_series.fillna(df.iloc[:, col_idx])
            
            # Keep only the first column, drop the rest
            cols_to_drop = duplicate_cols[1:]
            df = df.drop(df.columns[cols_to_drop], axis=1)
            df.iloc[:, duplicate_cols[0]] = merged_series
            
            logger.info(f"  Merged '{col_name}': kept 1, dropped {len(cols_to_drop)} duplicates")
        
        return df
    
    def filter_useful_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only useful columns, drop redundant metadata garbage."""
        
        # Always keep these core columns
        core_columns = [
            'measurement_id', 'trial_id', 'strategy', 'liquid', 'liquid_name',
            'volume_target_ml', 'volume_target_ul', 'volume_measured_ml', 'volume_measured_ul', 
            'duration_s', 'timestamp', 'replicate_id', 'deviation_individual_pct',
            'experiment_type', 'experiment_folder', 'experiment_timestamp'
        ]
        
        # For hardware parameters, keep the version with most data
        hardware_param_groups = {
            'aspirate_speed': ['hardware_parameters_aspirate_speed', 'metadata_aspirate_speed', 'aspirate_speed'],
            'dispense_speed': ['hardware_parameters_dispense_speed', 'metadata_dispense_speed', 'dispense_speed'], 
            'overaspirate_vol': ['calibration_overaspirate_vol', 'hardware_parameters_overaspirate_vol', 'metadata_overaspirate_vol', 'overaspirate_vol'],
            'aspirate_wait_time': ['hardware_parameters_aspirate_wait_time', 'metadata_aspirate_wait_time', 'aspirate_wait_time'],
            'dispense_wait_time': ['hardware_parameters_dispense_wait_time', 'metadata_dispense_wait_time', 'dispense_wait_time'],
            'pre_asp_air_vol': ['hardware_parameters_pre_asp_air_vol', 'metadata_pre_asp_air_vol', 'pre_asp_air_vol'],
            'post_asp_air_vol': ['hardware_parameters_post_asp_air_vol', 'metadata_post_asp_air_vol', 'post_asp_air_vol'],
            'blowout_vol': ['hardware_parameters_blowout_vol', 'metadata_blowout_vol', 'blowout_vol'],
            'retract_speed': ['hardware_parameters_retract_speed', 'metadata_retract_speed', 'retract_speed'],
            'asp_disp_cycles': ['hardware_parameters_asp_disp_cycles', 'metadata_asp_disp_cycles', 'asp_disp_cycles'],
            'post_retract_wait_time': ['hardware_parameters_post_retract_wait_time', 'metadata_post_retract_wait_time', 'post_retract_wait_time']
        }
        
        # Keep other potentially useful columns (excluding unwanted ones)
        other_useful = []
        
        # Build final column list
        keep_columns = []
        
        # Add core columns that exist
        for col in core_columns:
            if col in df.columns:
                keep_columns.append(col)
        
        # Add best hardware parameter from each group (COMBINE the data)
        for param_name, candidates in hardware_param_groups.items():
            # Combine all versions of this parameter into one column
            combined_data = None
            sources_found = []
            
            for candidate in candidates:
                if candidate in df.columns:
                    if combined_data is None:
                        combined_data = df[candidate].copy()
                    else:
                        # Fill missing values with data from this source
                        combined_data = combined_data.fillna(df[candidate])
                    sources_found.append(candidate)
            
            if combined_data is not None:
                # Add the combined parameter with clean name
                keep_columns.append(param_name)  # Use clean name like 'aspirate_speed'
                df[param_name] = combined_data   # Add combined data under clean name
                logger.info(f"Parameter '{param_name}': combined data from {sources_found}")
        
        # Add other useful columns that exist
        for col in other_useful:
            if col in df.columns and col not in keep_columns:
                keep_columns.append(col)
        
        # Filter dataframe
        original_cols = len(df.columns)
        df_filtered = df[keep_columns]
        dropped_cols = original_cols - len(keep_columns)
        
        logger.info(f"Column filtering: {original_cols} -> {len(keep_columns)} columns (dropped {dropped_cols} redundant)")
        
        return df_filtered
    
    def extract_all_configurations(self) -> pd.DataFrame:
        """Extract configuration data from all experiments and validations."""
        all_configs = []
        
        logger.info("Extracting configurations from all experiments...")
        
        # Process calibration experiments
        if self.output_dir.exists():
            for folder in self.output_dir.iterdir():
                if folder.is_dir() and (folder.name.startswith('run_') or 
                                      folder.name not in ['hardware', 'incomplete', 'simulation', 'sample_output', 'test_analysis', 'test_run_debug']):
                    config_data = self.extract_experiment_config(folder, source_type='calibration')
                    if config_data:
                        all_configs.append(config_data)
        
        # Process validation experiments  
        if self.validation_dir.exists():
            for folder in self.validation_dir.iterdir():
                if folder.is_dir() and folder.name.startswith('validation_run_'):
                    config_data = self.extract_experiment_config(folder, source_type='validation')
                    if config_data:
                        all_configs.append(config_data)
        
        if not all_configs:
            logger.warning("No configuration data found!")
            return pd.DataFrame()
            
        # Convert to DataFrame and handle schema differences
        configs_df = pd.DataFrame(all_configs)
        logger.info(f"Extracted {len(configs_df)} configuration records")
        
        return configs_df
    
    def extract_experiment_config(self, folder_path: Path, source_type: str) -> Optional[Dict]:
        """Extract configuration from a single experiment folder."""
        try:
            config_data = {
                'experiment_folder': folder_path.name,
                'experiment_timestamp': folder_path.name,
                'source_type': source_type
            }
            
            # Look for config files
            config_files = [
                'experiment_config_used.yaml', 'experiment_config.yaml', 'config.yaml', 'workflow_config.yaml',
                'parameters.yaml', 'settings.yaml', 'experiment_parameters.json', 'experiment_summary.json',
                'experiment_insights.json', 'optimal_conditions.json'
            ]
            
            for config_file in config_files:
                config_path = folder_path / config_file
                if config_path.exists():
                    config_content = self.load_config_file(config_path)
                    if config_content:
                        # Flatten nested config with prefixes
                        flattened = self.flatten_config(config_content, prefix=config_file.split('.')[0])
                        config_data.update(flattened)
                        break
            
            # For validation experiments, also extract from validation summary
            if source_type == 'validation':
                summary_file = folder_path / 'validation_summary.csv'
                if summary_file.exists():
                    validation_config = self.extract_validation_config(summary_file)
                    config_data.update(validation_config)
            
            return config_data if len(config_data) > 3 else None  # Only return if we found actual config
            
        except Exception as e:
            logger.error(f"Error extracting config from {folder_path}: {e}")
            return None
    
    def load_config_file(self, config_path: Path) -> Optional[Dict]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    import yaml
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    import json
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return None
    
    def flatten_config(self, config_dict: Dict, prefix: str = '') -> Dict:
        """Flatten nested configuration dictionary with prefixes."""
        flattened = {}
        
        for key, value in config_dict.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                nested = self.flatten_config(value, prefix=new_key)
                flattened.update(nested)
            elif isinstance(value, (list, tuple)):
                # Convert lists to strings
                flattened[new_key] = str(value)
            else:
                # Keep simple values as-is
                flattened[new_key] = value
        
        return flattened
    
    def extract_validation_config(self, summary_file: Path) -> Dict:
        """Extract config from validation summary parameters."""
        try:
            df = pd.read_csv(summary_file)
            config_data = {}
            
            if 'parameters_used' in df.columns:
                # Parse JSON parameters like we do in parse_validation_parameters
                for _, row in df.iterrows():
                    if pd.notna(row.get('parameters_used')):
                        try:
                            params = json.loads(row['parameters_used'])
                            for key, value in params.items():
                                config_key = f"validation_{key}"
                                if config_key not in config_data:  # Keep first occurrence
                                    config_data[config_key] = value
                        except json.JSONDecodeError:
                            continue
            
            return config_data
            
        except Exception as e:
            logger.error(f"Error extracting validation config from {summary_file}: {e}")
            return {}
    
    def process_trial_results(self, folder_path: Path, experiment_type: str, metadata: Dict) -> int:
        """Process trial results from a single experiment folder."""
        trial_files = [
            folder_path / "trial_results.csv",
            folder_path / "validation_summary.csv",
            folder_path / "output" / "trial_results.csv"
        ]
        
        for trials_file in trial_files:
            if trials_file.exists():
                logger.debug(f"Found trial results at {trials_file}")
                break
        else:
            logger.warning(f"No trial file (trial_results.csv or validation_summary.csv) found in {folder_path}")
            return 0
        
        try:
            df = pd.read_csv(trials_file)
            logger.info(f"Loaded {len(df)} trials from {trials_file}")
            
            # Add experiment metadata
            df['experiment_type'] = experiment_type
            df['experiment_folder'] = metadata['experiment_folder']
            df['experiment_timestamp'] = metadata['timestamp']
            df['liquid_name'] = metadata['liquid']
            
            # Add to master list
            self.master_trials.append(df)
            
            return len(df)
            
        except Exception as e:
            logger.error(f"Error processing trials from {folder_path}: {e}")
            return 0
    
    def process_optimal_conditions(self, folder_path: Path, experiment_type: str, metadata: Dict) -> int:
        """Process optimal conditions from a single experiment folder."""
        # Try multiple possible filenames and locations for optimal conditions
        optimal_files = [
            folder_path / f"optimal_conditions_{metadata['liquid']}.csv",
            folder_path / "optimal_conditions.csv",
            folder_path / "output" / f"optimal_conditions_{metadata['liquid']}.csv",
            folder_path / "output" / "optimal_conditions.csv"
        ]
        
        optimal_df = None
        for optimal_file in optimal_files:
            if optimal_file.exists():
                try:
                    optimal_df = pd.read_csv(optimal_file)
                    logger.info(f"Loaded {len(optimal_df)} optimal conditions from {optimal_file}")
                    break
                except Exception as e:
                    logger.warning(f"Error reading {optimal_file}: {e}")
                    continue
        
        if optimal_df is None:
            logger.warning(f"No optimal conditions file found in {folder_path}")
            return 0
        
        # Add experiment metadata
        optimal_df['experiment_type'] = experiment_type
        optimal_df['experiment_folder'] = metadata['experiment_folder']
        optimal_df['experiment_timestamp'] = metadata['timestamp']
        optimal_df['liquid_name'] = metadata['liquid']
        
        # Add to master list
        self.master_optimal_conditions.append(optimal_df)
        
        return len(optimal_df)
    
    def process_single_experiment(self, folder_path: Path) -> bool:
        """Process a single experiment folder."""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {folder_path.name}")
            
            # Check if simulation
            is_simulation, config = self.is_simulation_experiment(folder_path)
            if is_simulation:
                logger.info(f"SKIPPING simulation experiment: {folder_path.name}")
                self.stats['simulated_folders_skipped'] += 1
                return True
            
            # Determine experiment type
            experiment_type = self.determine_experiment_type(folder_path)
            logger.info(f"Experiment type: {experiment_type}")
            
            if experiment_type == 'calibration':
                self.stats['calibration_folders'] += 1
            else:
                self.stats['validation_folders'] += 1
            
            # Extract metadata using config and measurements
            metadata = self.extract_metadata_from_folder(folder_path, config)
            logger.info(f"Liquid: {metadata['liquid']}")
            logger.info(f"Timestamp: {metadata['timestamp']}")
            
            # Process measurements, trials, and optimal conditions
            measurements_count = self.process_raw_measurements(folder_path, experiment_type, metadata)
            trials_count = self.process_trial_results(folder_path, experiment_type, metadata)
            optimal_count = self.process_optimal_conditions(folder_path, experiment_type, metadata)
            
            self.stats['total_measurements'] += measurements_count
            self.stats['total_trials'] += trials_count
            self.stats['successfully_processed'] += 1
            
            logger.info(f"PROCESSED: {measurements_count} measurements, {trials_count} trials, {optimal_count} optimal conditions")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR processing {folder_path}: {e}")
            logger.error(traceback.format_exc())
            self.stats['errors'] += 1
            return False
    
    def find_all_experiment_folders(self) -> List[Path]:
        """Find all experiment folders in output and validation directories."""
        folders = []
        calibration_count = 0
        validation_count = 0
        
        # Calibration experiments in output/
        if self.output_dir.exists():
            logger.info(f"Scanning calibration folder: {self.output_dir}")
            for folder in self.output_dir.iterdir():
                if folder.is_dir() and (folder.name.startswith('run_') or 
                                      folder.name not in ['hardware', 'incomplete', 'simulation', 'sample_output', 'test_analysis', 'test_run_debug']):
                    folders.append(folder)
                    calibration_count += 1
        else:
            logger.warning(f"Calibration directory does not exist: {self.output_dir}")
        
        # Validation experiments in validation/
        if self.validation_dir.exists():
            logger.info(f"Scanning validation folder: {self.validation_dir}")
            for folder in self.validation_dir.iterdir():
                if folder.is_dir() and folder.name.startswith('validation_run_'):
                    folders.append(folder)
                    validation_count += 1
                    logger.info(f"  Found validation folder: {folder.name}")
        else:
            logger.warning(f"Validation directory does not exist: {self.validation_dir}")
        
        logger.info(f"Found {calibration_count} calibration + {validation_count} validation = {len(folders)} experiment folders total")
        return sorted(folders)
    
    def combine_and_export_data(self):
        """Combine all collected data and export to master files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create master_data folder in calibration_modular_v2 directory
        master_data_dir = self.base_dir / "master_data"
        master_data_dir.mkdir(exist_ok=True)
        logger.info(f"Saving files to: {master_data_dir.absolute()}")
        
        try:
            # Combine measurements
            if self.master_measurements:
                master_measurements_df = pd.concat(self.master_measurements, ignore_index=True)
                
                # Filter out redundant garbage columns
                master_measurements_df = self.filter_useful_columns(master_measurements_df)
                
                measurements_file = master_data_dir / f"master_measurements_{timestamp}.csv"
                master_measurements_df.to_csv(measurements_file, index=False)
                logger.info(f"SAVED -> {measurements_file.absolute()}")
                logger.info(f"         ({len(master_measurements_df)} measurements, {len(master_measurements_df.columns)} columns)")
            else:
                logger.warning("No measurements data to export")
            
            # Combine trials  
            if self.master_trials:
                master_trials_df = pd.concat(self.master_trials, ignore_index=True)
                
                # Just save it as-is. No fancy processing that breaks everything.
                trials_file = master_data_dir / f"master_trials_{timestamp}.csv"
                master_trials_df.to_csv(trials_file, index=False)
                logger.info(f"SAVED -> {trials_file.absolute()}")
                logger.info(f"         ({len(master_trials_df)} trials, {len(master_trials_df.columns)} columns)")
            else:
                logger.warning("No trials data to export")
            
            # Combine optimal conditions
            if self.master_optimal_conditions:
                master_optimal_df = pd.concat(self.master_optimal_conditions, ignore_index=True)
                optimal_file = master_data_dir / f"master_optimal_conditions_{timestamp}.csv"
                master_optimal_df.to_csv(optimal_file, index=False)
                logger.info(f"SAVED -> {optimal_file.absolute()}")
                logger.info(f"         ({len(master_optimal_df)} optimal conditions)")
            else:
                logger.warning("No optimal conditions data to export")
            
            # Extract and save configurations
            logger.info("Creating master configurations dataset...")
            master_configs_df = self.extract_all_configurations()
            if not master_configs_df.empty:
                configs_file = master_data_dir / f"master_configurations_{timestamp}.csv"
                master_configs_df.to_csv(configs_file, index=False)
                logger.info(f"SAVED -> {configs_file.absolute()}")
                logger.info(f"         ({len(master_configs_df)} configuration records)")
            else:
                logger.warning("No configuration data found to export")
                
        except Exception as e:
            logger.error(f"Error exporting combined data: {e}")
            logger.error(traceback.format_exc())
    
    def generate_compilation_report(self):
        """Generate a comprehensive report of the data compilation process."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create master_data folder if it doesn't exist
        master_data_dir = self.base_dir / "master_data"
        master_data_dir.mkdir(exist_ok=True)
        
        report_file = master_data_dir / f"data_compilation_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("MASTER DATASET COMPILATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total folders found: {self.stats['total_folders_found']}\n")
            f.write(f"Calibration folders: {self.stats['calibration_folders']}\n")
            f.write(f"Validation folders: {self.stats['validation_folders']}\n")
            f.write(f"Simulated folders skipped: {self.stats['simulated_folders_skipped']}\n")
            f.write(f"Empty folders skipped: {self.stats.get('empty_folders_skipped', 0)}\n")
            f.write(f"Successfully processed: {self.stats['successfully_processed']}\n")
            f.write(f"Errors encountered: {self.stats['errors']}\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total measurements: {self.stats['total_measurements']}\n")
            f.write(f"Total trials: {self.stats['total_trials']}\n\n")
            
            success_rate = (self.stats['successfully_processed'] / max(1, self.stats['total_folders_found'])) * 100
            f.write(f"Processing success rate: {success_rate:.1f}%\n\n")
            
            if self.master_measurements:
                total_measurements = sum(len(df) for df in self.master_measurements)
                f.write(f"Master measurements dataset: {total_measurements:,} rows\n")
            
            if self.master_trials:
                total_trials = sum(len(df) for df in self.master_trials)
                f.write(f"Master trials dataset: {total_trials:,} rows\n")
            
            if self.master_optimal_conditions:
                total_optimal = sum(len(df) for df in self.master_optimal_conditions)
                f.write(f"Master optimal conditions dataset: {total_optimal:,} rows\n")
        
        logger.info(f"REPORT -> {report_file.absolute()}")
    
    def run(self):
        """Execute the complete master dataset creation process."""
        logger.info("Starting master dataset creation...")
        logger.info(f"Debug logging enabled: {logger.level <= 10}")
        logger.info("=" * 60)
        
        # Find all experiment folders
        experiment_folders = self.find_all_experiment_folders()
        self.stats['total_folders_found'] = len(experiment_folders)
        
        if not experiment_folders:
            logger.error("No experiment folders found!")
            return
        
        logger.info(f"Processing {len(experiment_folders)} experiment folders...")
        
        # Process each folder
        for folder in experiment_folders:
            # Check if simulation
            is_simulation, config = self.is_simulation_experiment(folder)
            if is_simulation:
                self.stats['simulated_folders_skipped'] += 1
                continue
                
            # Check if has meaningful data
            if not self.has_meaningful_data(folder):
                logger.info(f"SKIPPING EMPTY: {folder.name} (no measurement data)")
                self.stats['empty_folders_skipped'] = self.stats.get('empty_folders_skipped', 0) + 1
                continue
                
            self.process_single_experiment(folder)
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPILATION PHASE")
        logger.info("=" * 60)
        
        # Combine and export all data
        self.combine_and_export_data()
        
        # Generate summary report
        self.generate_compilation_report()
        
        logger.info("\nMASTER DATASET CREATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Processed: {self.stats['successfully_processed']}/{self.stats['total_folders_found']} folders")
        logger.info(f"Total measurements: {self.stats['total_measurements']:,}")
        logger.info(f"Total trials: {self.stats['total_trials']:,}")
        logger.info(f"Simulated experiments skipped: {self.stats['simulated_folders_skipped']}")
        logger.info(f"Empty folders skipped: {self.stats.get('empty_folders_skipped', 0)}")
        
        if self.stats['errors'] > 0:
            logger.warning(f"WARNING: {self.stats['errors']} folders had processing errors")


if __name__ == "__main__":
    print("Starting Master Dataset Creator...")
    print(f"Working from: {Path(__file__).parent.absolute()}")
    
    # Enable debug logging
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    creator = MasterDatasetCreator()
    creator.run()
