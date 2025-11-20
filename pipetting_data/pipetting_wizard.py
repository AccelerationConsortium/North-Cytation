# pipetting_wizard.py
"""
Pipetting Parameter Wizard

Provides intelligent parameter lookup and interpolation based on calibration data.
Searches for liquid-specific calibration files and interpolates parameters for target volumes.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import logging

# Standard pipetting parameters to extract/interpolate
PIPETTING_PARAMETERS = [
    'aspirate_speed',
    'dispense_speed', 
    'aspirate_wait_time',
    'dispense_wait_time',
    'retract_speed',
    'blowout_vol',
    'post_asp_air_vol',
    'overaspirate_vol'
]

# Required columns in calibration files  
REQUIRED_COLUMNS = ['volume_target'] + PIPETTING_PARAMETERS

class PipettingWizard:
    """
    Intelligent pipetting parameter provider based on calibration data.
    """
    
    def __init__(self, search_directory: str = None):
        """
        Initialize the pipetting wizard.
        
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
        Find calibration files for a specific liquid.
        
        Args:
            liquid: Liquid name to search for (e.g., 'water', 'glycerol', 'DMSO')
            
        Returns:
            List of Path objects for matching calibration files
        """
        # Search patterns - look for optimal_conditions files with liquid name (case-insensitive)
        patterns = [
            f"optimal_conditions*{liquid.lower()}*.csv",
            f"optimal_conditions*{liquid.upper()}*.csv", 
            f"optimal_conditions*{liquid.title()}*.csv"
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
        Load and validate calibration data from a CSV file.
        
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
            available_params = [p for p in PIPETTING_PARAMETERS if p in df.columns]
            if not available_params:
                logging.warning(f"File {file_path} has no pipetting parameters")
                return None
            
            # Ensure we have volume data
            if df.empty:
                logging.warning(f"File {file_path} is empty")
                return None
            
            # Sort by volume for easier interpolation
            df = df.sort_values('volume_target').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading calibration file {file_path}: {e}")
            return None
    
    def find_best_calibration_file(self, liquid: str, target_volume_ml: float) -> Optional[Tuple[Path, pd.DataFrame]]:
        """
        Find the best calibration file for interpolation.
        Prefers files where target volume can be interpolated rather than extrapolated.
        
        Args:
            liquid: Liquid name
            target_volume_ml: Target volume in mL
            
        Returns:
            Tuple of (file_path, dataframe) for best file, or None if no suitable file found
        """
        calibration_files = self.find_calibration_files(liquid)
        
        if not calibration_files:
            logging.error(f"No calibration files found for liquid '{liquid}' in directory {self.search_directory}")
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
            
            volumes = df['volume_target'].values  # These are in μL
            min_vol, max_vol = volumes.min(), volumes.max()
            
            # Convert target from mL to μL for comparison
            target_volume_ul = target_volume_ml * 1000
            
            # Scoring: prefer interpolation over extrapolation
            if min_vol <= target_volume_ul <= max_vol:
                # Can interpolate - excellent
                score = 0
            else:
                # Must extrapolate - score based on distance
                if target_volume_ul < min_vol:
                    score = min_vol - target_volume_ul  # Distance below range
                else:
                    score = target_volume_ul - max_vol  # Distance above range
            
            # Prefer files with more data points (tie breaker)
            score += 1.0 / len(volumes)
            
            if score < best_score:
                best_score = score
                best_file = file_path
                best_df = df
        
        if best_file is None:
            logging.error(f"No valid calibration files found for liquid '{liquid}'")
            return None
            
        return best_file, best_df
    
    def interpolate_parameters(self, df: pd.DataFrame, target_volume_ml: float) -> Dict[str, float]:
        """
        Interpolate pipetting parameters for target volume.
        
        Args:
            df: Calibration dataframe
            target_volume_ml: Target volume in mL
            
        Returns:
            Dictionary of interpolated parameters
        """
        # Convert target from mL to μL to match volume_target column
        target_volume_ul = target_volume_ml * 1000
        
        volumes = df['volume_target'].values  # These are in μL
        min_vol, max_vol = volumes.min(), volumes.max()
        
        # Check if we need to extrapolate and warn user
        if target_volume_ul < min_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}μL) is below available range ({min_vol}-{max_vol}μL). Extrapolating...")
        elif target_volume_ul > max_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}μL) is above available range ({min_vol}-{max_vol}μL). Extrapolating...")
        
        # If exact match exists, return it
        exact_match = df[df['volume_target'] == target_volume_ul]
        if not exact_match.empty:
            result = {}
            for param in PIPETTING_PARAMETERS:
                if param in exact_match.columns:
                    value = float(exact_match.iloc[0][param])
                    # Convert speed parameters to integers to avoid conversion warnings
                    if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
                        value = int(round(value))
                    result[param] = value
            result['volume_ml'] = target_volume_ml
            return result
        
        # Interpolate between closest points
        result = {'volume_ml': target_volume_ml}
        
        for param in PIPETTING_PARAMETERS:
            if param in df.columns:
                param_values = df[param].values
                
                # Use numpy interp for linear interpolation (volumes in μL)
                interpolated_value = np.interp(target_volume_ul, volumes, param_values)
                value = float(interpolated_value)
                
                # Convert speed parameters to integers to avoid conversion warnings
                if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
                    value = int(round(value))
                
                result[param] = value
        
        return result
    
    def apply_overvolume_compensation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust overaspirate_vol based on measured vs target volume accuracy.
        
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
            volume_target = row['volume_target']  # μL
            volume_measured = row['volume_measured']  # μL  
            current_overasp = row['overaspirate_vol']  # mL
            
            # Calculate volume error in μL
            volume_error = volume_measured - volume_target  # Positive = over-target, Negative = under-target
            
            # Convert error to mL to match overaspirate units
            volume_error_ml = volume_error / 1000  # Convert μL to mL
            
            # Adjust overaspirate: if over-target, decrease overasp; if under-target, increase overasp
            # Apply full compensation for the measured error
            adjustment_factor = 1.0  # Apply 100% of the measured error
            adjustment = -volume_error_ml * adjustment_factor  # Negative error (under) → positive adjustment (increase overasp)
            
            new_overasp = current_overasp + adjustment
            
            # Apply reasonable bounds: -100% to +100% of target volume
            target_volume_ml = volume_target / 1000  # Convert μL to mL
            min_overasp = -target_volume_ml  # -100% of target volume
            max_overasp = target_volume_ml   # +100% of target volume
            new_overasp = max(min_overasp, min(new_overasp, max_overasp))
            
            # Calculate the actual adjustment that would be applied
            actual_adjustment_ml = new_overasp - current_overasp
            actual_adjustment_ul = actual_adjustment_ml * 1000
            
            # Store details for reporting
            adjustment_details.append({
                'volume': volume_target,
                'error': volume_error, 
                'adjustment_ul': actual_adjustment_ul,
                'old_overasp': current_overasp,
                'new_overasp': new_overasp
            })
            
            # Always apply compensation if there's any volume error >0.01μL
            if abs(volume_error) > 0.01:  # Only skip truly negligible errors
                df.at[idx, 'overaspirate_vol'] = new_overasp
                compensated_count += 1
                
                logging.debug(f"  {volume_target}μL: error {volume_error:+.2f}μL → overasp {current_overasp:.4f}→{new_overasp:.4f}mL "
                      f"(Δ{actual_adjustment_ul:+.2f}μL)")
            else:
                logging.debug(f"  {volume_target}μL: error {volume_error:+.2f}μL → no adjustment needed (negligible)")
        
        if compensated_count > 0:
            logging.info(f"Applied overvolume compensation to {compensated_count}/{len(df)} parameter sets")
        else:
            logging.debug("No overvolume compensation applied - all volume errors were negligible (<0.01μL)")
            
        return df
    
    def get_pipetting_parameters(self, liquid: str, volume_ml: float, compensate_overvolume: bool = True) -> Optional[Dict[str, float]]:
        """
        Get pipetting parameters for a specific liquid and volume.
        
        Args:
            liquid: Liquid name (e.g., 'water', 'glycerol')
            volume_ml: Target volume in mL
            compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
            
        Returns:
            Dictionary with pipetting parameters, or None if not available
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
        
        compensation_note = " (with overvolume compensation)" if compensate_overvolume else ""
        logging.debug(f"Parameters for {liquid} {volume_ml}mL from {file_path.name}{compensation_note}:")
        for key, value in parameters.items():
            if not key.startswith('_'):
                logging.debug(f"  {key}: {value:.6f}")
        
        return parameters


def create_wizard(search_directory: str = None) -> PipettingWizard:
    """
    Convenience function to create a PipettingWizard instance.
    
    Args:
        search_directory: Directory to search for calibration files
        
    Returns:
        PipettingWizard instance
    """
    return PipettingWizard(search_directory)


def get_pipetting_parameters(liquid: str, volume_ml: float, search_directory: str = None, compensate_overvolume: bool = False) -> Optional[Dict[str, float]]:
    """
    Convenience function to get pipetting parameters without creating a wizard instance.
    
    Args:
        liquid: Liquid name (e.g., 'water', 'glycerol', 'DMSO')
        volume_ml: Target volume in mL
        search_directory: Directory to search for calibration files
        compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
        
    Returns:
        Dictionary with pipetting parameters, or None if not available
    """
    wizard = PipettingWizard(search_directory)
    return wizard.get_pipetting_parameters(liquid, volume_ml, compensate_overvolume)


# Example usage
if __name__ == "__main__":
    # Create wizard instance
    wizard = PipettingWizard()
    
    # Test 1: Get parameters for glycerol at 0.035 mL (should interpolate)
    print("=== Test 1: Glycerol 0.035 mL (interpolation) ===")
    params = wizard.get_pipetting_parameters("glycerol", 0.035)
    
    # Test 2: Get parameters for glycerol at 0.15 mL (should extrapolate)  
    print("\n=== Test 2: Glycerol 0.15 mL (extrapolation) ===")
    params2 = wizard.get_pipetting_parameters("glycerol", 0.15)
    
    # Test 3: Try a liquid that doesn't exist
    print("\n=== Test 3: Water (should fail) ===")
    params3 = wizard.get_pipetting_parameters("water", 0.05)
