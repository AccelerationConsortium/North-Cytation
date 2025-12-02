# pipetting_wizard_v2.py
"""
Hardware-Agnostic Pipetting Parameter Wizard V2

Provides intelligent parameter lookup and interpolation based on calibration_v2 data.
Works with the new format that uses hardware_parameters_ prefixes and calibration_overaspirate_vol.
Dynamically discovers available hardware parameters without hardcoded defaults.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import logging

class PipettingWizardV2:
    """
    Hardware-agnostic pipetting parameter provider for calibration_v2 data format.
    """
    
    def __init__(self, search_directory: str = None):
        """
        Initialize the V2 pipetting wizard.
        
        Args:
            search_directory: Directory to search for optimal_conditions.csv files. 
                            If None, searches same directory as this script.
        """
        if search_directory:
            self.search_directory = Path(search_directory)
        else:
            # Default to same directory as this script
            self.search_directory = Path(__file__).parent
        self.cache = {}  # Cache loaded calibration data
        
    def find_calibration_files(self, liquid: str = None) -> List[Path]:
        """
        Find calibration files in V2 format.
        
        Args:
            liquid: Liquid name to search for (optional - V2 format may not have liquid in filename)
            
        Returns:
            List of Path objects for matching calibration files
        """
        # Search patterns for V2 format - look for optimal_conditions.csv files
        patterns = ["optimal_conditions.csv"]
        
        # If liquid specified, also try liquid-specific patterns
        if liquid:
            patterns.extend([
                f"optimal_conditions*{liquid.lower()}*.csv",
                f"optimal_conditions*{liquid.upper()}*.csv", 
                f"optimal_conditions*{liquid.title()}*.csv"
            ])
        
        found_files = []
        
        # Search recursively in subdirectories (V2 data often in run folders)
        for pattern in patterns:
            # Search current directory
            matching_files = glob.glob(str(self.search_directory / pattern))
            found_files.extend([Path(f) for f in matching_files])
            
            # Search subdirectories
            matching_files = glob.glob(str(self.search_directory / "**" / pattern), recursive=True)
            found_files.extend([Path(f) for f in matching_files])
        
        # Remove duplicates and sort by modification time (newest first)
        unique_files = list(set(found_files))
        unique_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return unique_files
    
    def discover_hardware_parameters(self, df: pd.DataFrame) -> List[str]:
        """
        Dynamically discover available hardware parameters in the dataframe.
        
        Args:
            df: Calibration dataframe
            
        Returns:
            List of hardware parameter names (without hardware_parameters_ prefix)
        """
        hardware_params = []
        
        for col in df.columns:
            if col.startswith('hardware_parameters_'):
                param_name = col.replace('hardware_parameters_', '')
                hardware_params.append(param_name)
        
        return sorted(hardware_params)
    
    def load_calibration_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate V2 calibration data from a CSV file.
        
        Args:
            file_path: Path to calibration CSV file
            
        Returns:
            DataFrame with calibration data, or None if invalid
        """
        try:
            df = pd.read_csv(file_path)
            
            # Check for required V2 columns
            required_v2_cols = ['volume_target_ul', 'volume_measured_ml', 'calibration_overaspirate_vol']
            missing_cols = [col for col in required_v2_cols if col not in df.columns]
            
            if missing_cols:
                logging.warning(f"File {file_path} missing V2 format columns: {missing_cols}")
                return None
            
            # Check for at least one hardware parameter
            hardware_params = self.discover_hardware_parameters(df)
            if not hardware_params:
                logging.warning(f"File {file_path} has no hardware_parameters_ columns")
                return None
            
            # Ensure we have volume data
            if df.empty:
                logging.warning(f"File {file_path} is empty")
                return None
            
            # Sort by volume for easier interpolation
            df = df.sort_values('volume_target_ul').reset_index(drop=True)
            
            logging.info(f"Loaded V2 calibration data from {file_path.name}")
            logging.info(f"  Found {len(hardware_params)} hardware parameters: {', '.join(hardware_params)}")
            logging.info(f"  Volume range: {df['volume_target_ul'].min()}-{df['volume_target_ul'].max()} uL")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading V2 calibration file {file_path}: {e}")
            return None
    
    def find_best_calibration_file(self, target_volume_ml: float, liquid: str = None) -> Optional[Tuple[Path, pd.DataFrame]]:
        """
        Find the best V2 calibration file for interpolation.
        Prefers files where target volume can be interpolated rather than extrapolated.
        
        Args:
            target_volume_ml: Target volume in mL
            liquid: Optional liquid name (for future filtering)
            
        Returns:
            Tuple of (file_path, dataframe) for best file, or None if no suitable file found
        """
        calibration_files = self.find_calibration_files(liquid)
        
        if not calibration_files:
            logging.error(f"No V2 calibration files found in directory {self.search_directory}")
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
            
            volumes = df['volume_target_ul'].values  # Already in uL
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
            logging.error(f"No valid V2 calibration files found")
            return None
            
        return best_file, best_df
    
    def interpolate_parameters(self, df: pd.DataFrame, target_volume_ml: float) -> Dict[str, float]:
        """
        Interpolate pipetting parameters for target volume using V2 format.
        
        Args:
            df: V2 calibration dataframe
            target_volume_ml: Target volume in mL
            
        Returns:
            Dictionary of interpolated parameters
        """
        # Convert target from mL to uL to match volume_target_ul column
        target_volume_ul = target_volume_ml * 1000
        
        volumes = df['volume_target_ul'].values  # Already in uL
        min_vol, max_vol = volumes.min(), volumes.max()
        
        # Check if we need to extrapolate and warn user
        if target_volume_ul < min_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is below available range ({min_vol}-{max_vol}uL). Extrapolating...")
        elif target_volume_ul > max_vol:
            logging.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is above available range ({min_vol}-{max_vol}uL). Extrapolating...")
        
        # If exact match exists, return it
        exact_match = df[df['volume_target_ul'] == target_volume_ul]
        if not exact_match.empty:
            result = {'volume_ml': target_volume_ml}
            
            # Extract overaspirate_vol (special case)
            result['overaspirate_vol'] = float(exact_match.iloc[0]['calibration_overaspirate_vol'])
            
            # Extract all hardware parameters dynamically
            hardware_params = self.discover_hardware_parameters(df)
            for param in hardware_params:
                col_name = f'hardware_parameters_{param}'
                if col_name in exact_match.columns:
                    value = float(exact_match.iloc[0][col_name])
                    # Convert speed parameters to integers to avoid conversion warnings
                    if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
                        value = int(round(value))
                    result[param] = value
            
            return result
        
        # Interpolate between closest points
        result = {'volume_ml': target_volume_ml}
        
        # Interpolate overaspirate_vol (special case)
        overasp_values = df['calibration_overaspirate_vol'].values
        interpolated_overasp = np.interp(target_volume_ul, volumes, overasp_values)
        result['overaspirate_vol'] = float(interpolated_overasp)
        
        # Interpolate all hardware parameters dynamically
        hardware_params = self.discover_hardware_parameters(df)
        for param in hardware_params:
            col_name = f'hardware_parameters_{param}'
            if col_name in df.columns:
                param_values = df[col_name].values
                
                # Use numpy interp for linear interpolation
                interpolated_value = np.interp(target_volume_ul, volumes, param_values)
                value = float(interpolated_value)
                
                # Convert speed parameters to integers to avoid conversion warnings
                if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
                    value = int(round(value))
                
                result[param] = value
        
        return result
    
    def apply_overvolume_compensation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust calibration_overaspirate_vol based on measured vs target volume accuracy.
        Uses volume_measured_ml (average) for compensation calculations.
        
        Args:
            df: V2 calibration dataframe with volume_target_ul, volume_measured_ml, and calibration_overaspirate_vol columns
            
        Returns:
            DataFrame with adjusted calibration_overaspirate_vol values
        """
        if 'volume_measured_ml' not in df.columns or 'calibration_overaspirate_vol' not in df.columns:
            logging.warning("Cannot apply overvolume compensation - missing volume_measured_ml or calibration_overaspirate_vol columns")
            return df
        
        # Ensure we have volume_target_ul column
        if 'volume_target_ul' not in df.columns:
            logging.warning("Cannot apply overvolume compensation - missing volume_target_ul column")
            return df
        
        compensated_count = 0
        adjustment_details = []
        
        for idx, row in df.iterrows():
            volume_target_ul = row['volume_target_ul']  # uL
            volume_measured_ml = row['volume_measured_ml']  # mL - this is the average
            current_overasp = row['calibration_overaspirate_vol']  # mL (assuming same units as measured)
            
            # Convert target to mL for comparison
            volume_target_ml = volume_target_ul / 1000  # Convert uL to mL
            
            # Calculate volume error in mL
            volume_error_ml = volume_measured_ml - volume_target_ml  # Positive = over-target, Negative = under-target
            
            # Adjust overaspirate: if over-target, decrease overasp; if under-target, increase overasp
            # Apply full compensation for the measured error
            adjustment_factor = 1.0  # Apply 100% of the measured error
            adjustment = -volume_error_ml * adjustment_factor  # Negative error (under) → positive adjustment (increase overasp)
            
            new_overasp = current_overasp + adjustment
            
            # Apply reasonable bounds: -100% to +100% of target volume
            min_overasp = -volume_target_ml  # -100% of target volume
            max_overasp = volume_target_ml   # +100% of target volume
            new_overasp = max(min_overasp, min(new_overasp, max_overasp))
            
            # Calculate the actual adjustment that would be applied
            actual_adjustment_ml = new_overasp - current_overasp
            actual_adjustment_ul = actual_adjustment_ml * 1000
            
            # Store details for reporting
            adjustment_details.append({
                'volume': volume_target_ul,
                'error_ml': volume_error_ml, 
                'adjustment_ul': actual_adjustment_ul,
                'old_overasp': current_overasp,
                'new_overasp': new_overasp
            })
            
            # Always apply compensation if there's any volume error >0.00001 mL (0.01 uL)
            if abs(volume_error_ml) > 0.00001:  # Only skip truly negligible errors
                df.at[idx, 'calibration_overaspirate_vol'] = new_overasp
                compensated_count += 1
                
                logging.debug(f"  {volume_target_ul}uL: error {volume_error_ml*1000:+.2f}uL → overasp {current_overasp:.4f}→{new_overasp:.4f}mL "
                      f"(Δ{actual_adjustment_ul:+.2f}uL)")
            else:
                logging.debug(f"  {volume_target_ul}uL: error {volume_error_ml*1000:+.2f}uL → no adjustment needed (negligible)")
        
        if compensated_count > 0:
            logging.info(f"Applied overvolume compensation to {compensated_count}/{len(df)} parameter sets")
        else:
            logging.debug("No overvolume compensation applied - all volume errors were negligible (<0.01uL)")
            
        return df
    
    def get_pipetting_parameters(self, volume_ml: float, liquid: str = None, compensate_overvolume: bool = True) -> Optional[Dict[str, float]]:
        """
        Get pipetting parameters for a specific volume using V2 format data.
        
        Args:
            volume_ml: Target volume in mL
            liquid: Optional liquid name (for future filtering)
            compensate_overvolume: If True, adjust calibration_overaspirate_vol based on measured accuracy
            
        Returns:
            Dictionary with pipetting parameters, or None if not available
        """
        # Find best calibration file
        file_result = self.find_best_calibration_file(volume_ml, liquid)
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
        parameters['_compensated'] = compensate_overvolume
        if liquid:
            parameters['_liquid'] = liquid
        
        # Discover and log available parameters
        hardware_params = self.discover_hardware_parameters(df)
        compensation_note = " (with overvolume compensation)" if compensate_overvolume else ""
        logging.debug(f"V2 Parameters for {volume_ml}mL from {file_path.name}{compensation_note}:")
        logging.debug(f"  Available hardware parameters: {', '.join(hardware_params)}")
        for key, value in parameters.items():
            if not key.startswith('_'):
                logging.debug(f"  {key}: {value:.6f}")
        
        return parameters


def create_wizard_v2(search_directory: str = None) -> PipettingWizardV2:
    """
    Convenience function to create a PipettingWizardV2 instance.
    
    Args:
        search_directory: Directory to search for calibration files
        
    Returns:
        PipettingWizardV2 instance
    """
    return PipettingWizardV2(search_directory)


def get_pipetting_parameters_v2(volume_ml: float, search_directory: str = None, liquid: str = None, compensate_overvolume: bool = True) -> Optional[Dict[str, float]]:
    """
    Convenience function to get V2 pipetting parameters without creating a wizard instance.
    
    Args:
        volume_ml: Target volume in mL
        search_directory: Directory to search for calibration files
        liquid: Optional liquid name
        compensate_overvolume: If True, adjust calibration_overaspirate_vol based on measured accuracy
        
    Returns:
        Dictionary with pipetting parameters, or None if not available
    """
    wizard = PipettingWizardV2(search_directory)
    return wizard.get_pipetting_parameters(volume_ml, liquid, compensate_overvolume)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create V2 wizard instance
    wizard = PipettingWizardV2()
    
    # Test: Get parameters for 0.025 mL (should match your example data)
    print("=== Test: 0.025 mL V2 Parameters ===")
    params = wizard.get_pipetting_parameters(0.025)
    
    if params:
        print("Found parameters:")
        for key, value in params.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        print(f"Source: {params.get('_source_file', 'Unknown')}")
    else:
        print("No parameters found")
        
    # Test: Get parameters for 0.01 mL 
    print("\n=== Test: 0.01 mL V2 Parameters ===")
    params2 = wizard.get_pipetting_parameters(0.01)
    
    if params2:
        print("Found parameters:")
        for key, value in params2.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")