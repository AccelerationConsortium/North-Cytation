# reservoir_wizard.py
"""
Reservoir Parameter Wizard

Provides intelligent parameter lookup and interpolation based on calibration data.
Searches for liquid-specific reservoir calibration files and interpolates parameters for target volumes.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import logging

# Standard reservoir parameters to extract/interpolate
RESERVOIR_PARAMETERS = [
    'aspirate_speed',
    'dispense_speed', 
    'aspirate_wait_time',
    'dispense_wait_time',
    'valve_switch_delay',
    'overaspirate_vol'
]

# Required columns in calibration files  
REQUIRED_COLUMNS = ['volume_target'] + RESERVOIR_PARAMETERS

class ReservoirWizard:
    """
    Intelligent reservoir parameter provider based on calibration data.
    """
    
    def __init__(self, search_directory: str = None):
        """
        Initialize the reservoir wizard.
        
        Args:
            search_directory: Directory to search for calibration files. 
                            If None, searches same directory as this script.
        """
        if search_directory:
            self.search_directory = Path(search_directory)
        else:
            # Default to same directory as this script
            self.search_directory = Path(__file__).parent
        self.cache = {}  # Cache loaded calibration data
        
    def find_calibration_files(self, liquid: str) -> List[Path]:
        """
        Find reservoir calibration files for a specific liquid.
        
        Args:
            liquid: Liquid name to search for (e.g., 'water', 'glycerol', 'DMSO')
            
        Returns:
            List of Path objects for matching calibration files
        """
        # Search patterns - look for reservoir_conditions files with liquid name (case-insensitive)
        patterns = [
            f"reservoir_conditions*{liquid.lower()}*.csv",
            f"reservoir_conditions*{liquid.upper()}*.csv", 
            f"reservoir_conditions*{liquid.title()}*.csv",
            # Also accept optimal_conditions files if they have reservoir data
            f"optimal_conditions*{liquid.lower()}*reservoir*.csv",
            f"optimal_conditions*{liquid.upper()}*reservoir*.csv"
        ]
        
        found_files = []
        
        # Search in the same directory as the wizard (no recursive search needed)
        for pattern in patterns:
            matching_files = glob.glob(str(self.search_directory / pattern))
            found_files.extend([Path(f) for f in matching_files])
        
        # Remove duplicates and sort
        unique_files = list(set(found_files))
        unique_files.sort()
        
        return unique_files
    
    def load_calibration_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate reservoir calibration data from a CSV file.
        
        Args:
            file_path: Path to calibration CSV file
            
        Returns:
            DataFrame with calibration data, or None if invalid
        """
        try:
            df = pd.read_csv(file_path)
            
            # Check for volume_target column (essential)
            if 'volume_target' not in df.columns:
                logging.warning(f"File {file_path} missing required 'volume_target' column")
                return None
            
            # Check which parameters are available
            available_params = [p for p in RESERVOIR_PARAMETERS if p in df.columns]
            if not available_params:
                logging.warning(f"File {file_path} has no reservoir parameters")
                return None
            
            # Ensure we have volume data
            if df.empty:
                logging.warning(f"File {file_path} is empty")
                return None
            
            # Sort by volume for easier interpolation
            df = df.sort_values('volume_target').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading reservoir calibration file {file_path}: {e}")
            return None
    
    def find_best_calibration_file(self, liquid: str, target_volume_ml: float) -> Optional[Tuple[Path, pd.DataFrame]]:
        """
        Find the best reservoir calibration file for interpolation.
        Prefers files where target volume can be interpolated rather than extrapolated.
        
        Args:
            liquid: Liquid name
            target_volume_ml: Target volume in mL
            
        Returns:
            Tuple of (file_path, dataframe) for best file, or None if no suitable file found
        """
        calibration_files = self.find_calibration_files(liquid)
        
        if not calibration_files:
            logging.error(f"No reservoir calibration files found for liquid '{liquid}' in directory {self.search_directory}")
            return None
        
        best_file = None
        best_df = None
        best_score = float('inf')  # Lower is better
        
        for file_path in calibration_files:
            # Use cache if available
            cache_key = str(file_path)
            if cache_key in self.cache:
                df = self.cache[cache_key]
            else:
                df = self.load_calibration_data(file_path)
                if df is None:
                    continue
                self.cache[cache_key] = df
            
            volumes = df['volume_target'].values  # These are in mL (different from pipetting which uses uL)
            min_vol, max_vol = volumes.min(), volumes.max()
            
            # Target is already in mL for reservoirs (different from pipetting)
            target_volume_ml_check = target_volume_ml
            
            # Scoring: prefer interpolation over extrapolation
            if min_vol <= target_volume_ml_check <= max_vol:
                # Can interpolate - excellent
                score = 0
            else:
                # Must extrapolate - score based on distance
                if target_volume_ml_check < min_vol:
                    score = min_vol - target_volume_ml_check  # Distance below range
                else:
                    score = target_volume_ml_check - max_vol  # Distance above range
            
            # Prefer files with more data points (tie breaker)
            score += 1.0 / len(volumes)
            
            if score < best_score:
                best_score = score
                best_file = file_path
                best_df = df
        
        if best_file is None:
            logging.error(f"No valid reservoir calibration files found for liquid '{liquid}'")
            return None
            
        return best_file, best_df
    
    def interpolate_parameters(self, df: pd.DataFrame, target_volume_ml: float) -> Dict[str, float]:
        """
        Interpolate reservoir parameters for target volume.
        
        Args:
            df: Calibration dataframe
            target_volume_ml: Target volume in mL
            
        Returns:
            Dictionary of interpolated parameters
        """
        # Target is already in mL for reservoirs (different from pipetting)
        volumes = df['volume_target'].values  # These are in mL
        min_vol, max_vol = volumes.min(), volumes.max()
        
        # Check if we need to extrapolate and warn user
        if target_volume_ml < min_vol:
            logging.warning(f"Target volume {target_volume_ml}mL is below available range ({min_vol}-{max_vol}mL). Extrapolating...")
        elif target_volume_ml > max_vol:
            logging.warning(f"Target volume {target_volume_ml}mL is above available range ({min_vol}-{max_vol}mL). Extrapolating...")
        
        # If exact match exists, return it
        exact_match = df[df['volume_target'] == target_volume_ml]
        if not exact_match.empty:
            result = {}
            for param in RESERVOIR_PARAMETERS:
                if param in exact_match.columns:
                    value = float(exact_match.iloc[0][param])
                    # Convert speed parameters to integers to avoid conversion warnings
                    if param in ['aspirate_speed', 'dispense_speed']:
                        value = int(round(value))
                    result[param] = value
            result['volume_ml'] = target_volume_ml
            return result
        
        # Interpolate between closest points
        result = {'volume_ml': target_volume_ml}
        
        for param in RESERVOIR_PARAMETERS:
            if param in df.columns:
                param_values = df[param].values
                
                # Use numpy interp for linear interpolation (volumes in mL)
                interpolated_value = np.interp(target_volume_ml, volumes, param_values)
                value = float(interpolated_value)
                
                # Convert speed parameters to integers to avoid conversion warnings
                if param in ['aspirate_speed', 'dispense_speed']:
                    value = int(round(value))
                
                result[param] = value
        
        return result
    
    def apply_overvolume_compensation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust overaspirate_vol based on measured vs target volume accuracy for reservoir operations.
        
        Args:
            df: Calibration dataframe with volume_target, volume_measured, and overaspirate_vol columns
            
        Returns:
            DataFrame with adjusted overaspirate_vol values
        """
        if 'volume_measured' not in df.columns or 'overaspirate_vol' not in df.columns:
            logging.warning("Cannot apply overvolume compensation - missing volume_measured or overaspirate_vol columns")
            return df
        
        # Ensure we have volume_target column
        if 'volume_target' not in df.columns:
            logging.warning("Cannot apply overvolume compensation - missing volume_target column")
            return df
        
        compensated_count = 0
        adjustment_details = []
        
        for idx, row in df.iterrows():
            volume_target = row['volume_target']  # mL
            volume_measured = row['volume_measured']  # mL  
            current_overasp = row['overaspirate_vol']  # mL
            
            # Calculate volume error in mL (different from pipetting which uses uL)
            volume_error = volume_measured - volume_target  # Positive = over-target, Negative = under-target
            
            # Adjust overaspirate: if over-target, decrease overasp; if under-target, increase overasp
            # Apply full compensation for the measured error
            adjustment_factor = 1.0  # Apply 100% of the measured error
            adjustment = -volume_error * adjustment_factor  # Negative error (under) → positive adjustment (increase overasp)
            
            new_overasp = current_overasp + adjustment
            
            # Apply reasonable bounds: -50% to +50% of target volume (more conservative than pipetting)
            min_overasp = -0.5 * volume_target  # -50% of target volume
            max_overasp = 0.5 * volume_target   # +50% of target volume
            new_overasp = max(min_overasp, min(new_overasp, max_overasp))
            
            # Calculate the actual adjustment that would be applied
            actual_adjustment_ml = new_overasp - current_overasp
            
            # Store details for reporting
            adjustment_details.append({
                'volume': volume_target,
                'error': volume_error, 
                'adjustment_ml': actual_adjustment_ml,
                'old_overasp': current_overasp,
                'new_overasp': new_overasp
            })
            
            # Apply compensation if there's any volume error >0.001mL (1uL equivalent)
            if abs(volume_error) > 0.001:  # Only skip truly negligible errors
                df.at[idx, 'overaspirate_vol'] = new_overasp
                compensated_count += 1
                
                logging.debug(f"  {volume_target}mL: error {volume_error:+.3f}mL → overasp {current_overasp:.4f}→{new_overasp:.4f}mL "
                      f"(Δ{actual_adjustment_ml:+.3f}mL)")
            else:
                logging.debug(f"  {volume_target}mL: error {volume_error:+.3f}mL → no adjustment needed (negligible)")
        
        if compensated_count > 0:
            logging.info(f"Applied reservoir overvolume compensation to {compensated_count}/{len(df)} parameter sets")
        else:
            logging.debug("No reservoir overvolume compensation applied - all volume errors were negligible (<0.001mL)")
            
        return df
    
    def get_reservoir_parameters(self, liquid: str, volume_ml: float, compensate_overvolume: bool = True) -> Optional[Dict[str, float]]:
        """
        Get reservoir parameters for a specific liquid and volume.
        
        Args:
            liquid: Liquid name (e.g., 'water', 'glycerol')
            volume_ml: Target volume in mL
            compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
            
        Returns:
            Dictionary with reservoir parameters, or None if not available
        """
        # Find best calibration file
        file_result = self.find_best_calibration_file(liquid, volume_ml)
        if file_result is None:
            return None
        
        file_path, df = file_result
        
        # Apply overvolume compensation before interpolation if requested
        if compensate_overvolume:
            df = self.apply_overvolume_compensation(df.copy())
        
        # Interpolate parameters
        parameters = self.interpolate_parameters(df, volume_ml)
        
        # Add metadata
        parameters['_source_file'] = str(file_path)
        parameters['_liquid'] = liquid
        parameters['_compensated'] = compensate_overvolume
        
        processing_note = " (with compensation)" if compensate_overvolume else ""
        logging.debug(f"Reservoir parameters for {liquid} {volume_ml}mL from {file_path.name}{processing_note}:")
        for key, value in parameters.items():
            if not key.startswith('_'):
                logging.debug(f"  {key}: {value:.6f}")
        
        return parameters


def create_wizard(search_directory: str = None) -> ReservoirWizard:
    """
    Convenience function to create a ReservoirWizard instance.
    
    Args:
        search_directory: Directory to search for calibration files
        
    Returns:
        ReservoirWizard instance
    """
    return ReservoirWizard(search_directory)


def get_reservoir_parameters(liquid: str, volume_ml: float, search_directory: str = None, compensate_overvolume: bool = True) -> Optional[Dict[str, float]]:
    """
    Convenience function to get reservoir parameters without creating a wizard instance.
    
    Args:
        liquid: Liquid name (e.g., 'water', 'glycerol', 'DMSO')
        volume_ml: Target volume in mL
        search_directory: Directory to search for calibration files
        compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
        
    Returns:
        Dictionary with reservoir parameters, or None if not available
    """
    wizard = ReservoirWizard(search_directory)
    return wizard.get_reservoir_parameters(liquid, volume_ml, compensate_overvolume)


# Example usage
if __name__ == "__main__":
    # Create wizard instance
    wizard = ReservoirWizard()
    
    # Test 1: Get parameters for water at 2.0 mL without compensation
    print("=== Test 1: Water 2.0mL (raw) ===")
    params1 = wizard.get_reservoir_parameters("water", 2.0, compensate_overvolume=False)
    
    # Test 2: With compensation
    print("\n=== Test 2: Water 2.0mL (compensated) ===")
    params2 = wizard.get_reservoir_parameters("water", 2.0, compensate_overvolume=True)
    
    # Compare overvolume values
    if all([params1, params2]):
        print("\nOvervolume comparison for 2.0mL:")
        print(f"  Raw:         {params1.get('overaspirate_vol', 'N/A'):.6f}mL")
        print(f"  Compensated: {params2.get('overaspirate_vol', 'N/A'):.6f}mL")