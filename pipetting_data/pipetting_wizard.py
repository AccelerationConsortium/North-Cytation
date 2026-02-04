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
    'overaspirate_vol',
    'post_retract_wait_time',
    'pre_asp_air_vol'
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
            
            volumes = df['volume_target'].values  # These are in uL
            min_vol, max_vol = volumes.min(), volumes.max()
            
            # Convert target from mL to uL for comparison
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
        # Convert target from mL to uL to match volume_target column
        target_volume_ul = target_volume_ml * 1000
        
        volumes = df['volume_target'].values  # These are in uL
        min_vol, max_vol = volumes.min(), volumes.max()
        
        # Check if we need to extrapolate and warn user
        if target_volume_ul < min_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is below available range ({min_vol}-{max_vol}uL). Extrapolating...")
        elif target_volume_ul > max_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is above available range ({min_vol}-{max_vol}uL). Extrapolating...")
        
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
                
                # Use numpy interp for linear interpolation (volumes in uL)
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
            volume_target = row['volume_target']  # uL
            volume_measured = row['volume_measured']  # uL  
            current_overasp = row['overaspirate_vol']  # mL
            
            # Calculate volume error in uL
            volume_error = volume_measured - volume_target  # Positive = over-target, Negative = under-target
            
            # Convert error to mL to match overaspirate units
            volume_error_ml = volume_error / 1000  # Convert uL to mL
            
            # Adjust overaspirate: if over-target, decrease overasp; if under-target, increase overasp
            # Apply full compensation for the measured error
            adjustment_factor = 1.0  # Apply 100% of the measured error
            adjustment = -volume_error_ml * adjustment_factor  # Negative error (under) → positive adjustment (increase overasp)
            
            new_overasp = current_overasp + adjustment
            
            # Apply reasonable bounds: -100% to +100% of target volume
            target_volume_ml = volume_target / 1000  # Convert uL to mL
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
            
            # Always apply compensation if there's any volume error >0.01uL
            if abs(volume_error) > 0.01:  # Only skip truly negligible errors
                df.at[idx, 'overaspirate_vol'] = new_overasp
                compensated_count += 1
                
                logging.debug(f"  {volume_target}uL: error {volume_error:+.2f}uL → overasp {current_overasp:.4f}→{new_overasp:.4f}mL "
                      f"(Δ{actual_adjustment_ul:+.2f}uL)")
            else:
                logging.debug(f"  {volume_target}uL: error {volume_error:+.2f}uL → no adjustment needed (negligible)")
        
        if compensated_count > 0:
            logging.info(f"Applied overvolume compensation to {compensated_count}/{len(df)} parameter sets")
        else:
            logging.debug("No overvolume compensation applied - all volume errors were negligible (<0.01uL)")
            
        return df
    
    def apply_local_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply local smoothing to overvolume values within volume groups.
        Smooths outliers that create non-physical "bumps" in the overvolume function.
        
        Args:
            df: Calibration dataframe with volume_target and overaspirate_vol columns
            
        Returns:
            DataFrame with smoothed overaspirate_vol values
        """
        if 'overaspirate_vol' not in df.columns:
            logging.warning("Cannot apply local smoothing - missing overaspirate_vol column")
            return df
        
        df_smoothed = df.copy()
        df_smoothed['overaspirate_vol_original'] = df['overaspirate_vol']  # Preserve original
        
        # Define volume groups based on parameter regimes
        group1_mask = (df['volume_target'] >= 1) & (df['volume_target'] <= 150)
        group2_mask = df['volume_target'] >= 200
        
        smoothed_count = 0
        
        # Group 1: Small volumes (1-150uL) - use quadratic smoothing
        if group1_mask.sum() >= 3:  # Need at least 3 points for quadratic fit
            group1 = df[group1_mask]
            volumes = group1['volume_target'].values
            overvolumes = group1['overaspirate_vol'].values
            
            # Fit quadratic polynomial
            poly_coeffs = np.polyfit(volumes, overvolumes, 2)
            smoothed_overvolumes = np.polyval(poly_coeffs, volumes)
            
            # Update the dataframe
            for i, (idx, row) in enumerate(group1.iterrows()):
                original_val = row['overaspirate_vol']
                smoothed_val = smoothed_overvolumes[i]
                df_smoothed.at[idx, 'overaspirate_vol'] = smoothed_val
                
                # Log significant changes
                change = smoothed_val - original_val
                if abs(change) > 0.0005:  # >0.5uL equivalent change
                    logging.debug(f"Smoothed {row['volume_target']}uL overvolume: "
                                f"{original_val:.6f} -> {smoothed_val:.6f} (change: {change:+.6f})")
                    smoothed_count += 1
        
        # Group 2: Large volumes (200+uL) - use linear smoothing
        if group2_mask.sum() >= 2:  # Need at least 2 points for linear fit
            group2 = df[group2_mask]
            volumes = group2['volume_target'].values
            overvolumes = group2['overaspirate_vol'].values
            
            # Fit linear polynomial
            poly_coeffs = np.polyfit(volumes, overvolumes, 1)
            smoothed_overvolumes = np.polyval(poly_coeffs, volumes)
            
            # Update the dataframe
            for i, (idx, row) in enumerate(group2.iterrows()):
                original_val = row['overaspirate_vol']
                smoothed_val = smoothed_overvolumes[i]
                df_smoothed.at[idx, 'overaspirate_vol'] = smoothed_val
                
                # Log significant changes
                change = smoothed_val - original_val
                if abs(change) > 0.0005:
                    logging.debug(f"Smoothed {row['volume_target']}uL overvolume: "
                                f"{original_val:.6f} -> {smoothed_val:.6f} (change: {change:+.6f})")
                    smoothed_count += 1
        
        if smoothed_count > 0:
            logging.info(f"Applied local smoothing to {smoothed_count} overvolume values")
        else:
            logging.debug("Local smoothing applied - no significant changes needed")
        
        return df_smoothed
    
    def get_pipetting_parameters(self, liquid: str, volume_ml: float, compensate_overvolume: bool = True, smooth_overvolume: bool = False) -> Optional[Dict[str, float]]:
        """
        Get pipetting parameters for a specific liquid and volume.
        
        Args:
            liquid: Liquid name (e.g., 'water', 'glycerol')
            volume_ml: Target volume in mL
            compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
            smooth_overvolume: If True, apply local smoothing to remove overvolume outliers
            
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
        
        # Apply local smoothing if requested
        if smooth_overvolume:
            df = self.apply_local_smoothing(df.copy())
        
        # Interpolate parameters
        parameters = self.interpolate_parameters(df, volume_ml)
        
        # Add metadata
        parameters['_source_file'] = str(file_path)
        parameters['_liquid'] = liquid
        parameters['_compensated'] = compensate_overvolume
        parameters['_smoothed'] = smooth_overvolume
        
        processing_notes = []
        if compensate_overvolume:
            processing_notes.append("compensation")
        if smooth_overvolume:
            processing_notes.append("smoothing")
        processing_note = f" (with {', '.join(processing_notes)})" if processing_notes else ""
        logging.debug(f"Parameters for {liquid} {volume_ml}mL from {file_path.name}{processing_note}:")
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


def get_pipetting_parameters(liquid: str, volume_ml: float, search_directory: str = None, compensate_overvolume: bool = False, smooth_overvolume: bool = False) -> Optional[Dict[str, float]]:
    """
    Convenience function to get pipetting parameters without creating a wizard instance.
    
    Args:
        liquid: Liquid name (e.g., 'water', 'glycerol', 'DMSO')
        volume_ml: Target volume in mL
        search_directory: Directory to search for calibration files
        compensate_overvolume: If True, adjust overaspirate_vol based on measured accuracy
        smooth_overvolume: If True, apply local smoothing to remove overvolume outliers
        
    Returns:
        Dictionary with pipetting parameters, or None if not available
    """
    wizard = PipettingWizard(search_directory)
    return wizard.get_pipetting_parameters(liquid, volume_ml, compensate_overvolume, smooth_overvolume)


# Example usage
if __name__ == "__main__":
    # Create wizard instance
    wizard = PipettingWizard()
    
    # Test 1: Get parameters for water at 0.020 mL with different options
    print("=== Test 1: Water 20uL (raw) ===")
    params1 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=False, smooth_overvolume=False)
    
    # Test 2: With compensation only
    print("\n=== Test 2: Water 20uL (compensated) ===")
    params2 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=True, smooth_overvolume=False)
    
    # Test 3: With smoothing only
    print("\n=== Test 3: Water 20uL (smoothed) ===")
    params3 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=False, smooth_overvolume=True)
    
    # Test 4: With both compensation and smoothing
    print("\n=== Test 4: Water 20uL (compensated + smoothed) ===")
    params4 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=True, smooth_overvolume=True)
    
    # Compare overvolume values
    if all([params1, params2, params3, params4]):
        print("\nOvervolume comparison for 20uL:")
        print(f"  Raw:                    {params1.get('overaspirate_vol', 'N/A'):.6f}")
        print(f"  Compensated only:       {params2.get('overaspirate_vol', 'N/A'):.6f}")
        print(f"  Smoothed only:          {params3.get('overaspirate_vol', 'N/A'):.6f}")
        print(f"  Compensated + Smoothed: {params4.get('overaspirate_vol', 'N/A'):.6f}")
