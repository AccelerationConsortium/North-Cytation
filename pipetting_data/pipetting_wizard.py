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

# Beyond this fraction outside a calibration file's min/max volume, extrapolated
# parameters are unreliable (e.g. a 2-50uL curve used for a 5400uL transfer) -
# fall back to system default parameters instead of extrapolating that far.
MAX_EXTRAPOLATION_FRACTION = 0.25

class PipettingWizard:
    """
    Intelligent pipetting parameter provider based on calibration data.
    """
    
    def __init__(self, search_directory: str = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipetting wizard.
        
        Args:
            search_directory: Directory to search for calibration files. 
                            If None, searches same directory as this script.
            logger: Logger to use (pass the caller's logger, e.g. Lash_E's file-backed
                    logger, so warnings/errors land in the per-run log file instead of
                    only the root logger/console).
        """
        if search_directory:
            self.search_directory = Path(search_directory)
        else:
            # Default to same directory as this script
            self.search_directory = Path(__file__).parent
        self.cache = {}  # Cache loaded calibration data
        self.logger = logger or logging.getLogger(__name__)
        
    def find_calibration_files(self, liquid: str) -> List[Path]:
        """
        Find calibration files for a specific liquid.
        
        Args:
            liquid: Liquid name to search for (e.g., 'water', 'glycerol', 'DMSO')
            
        Returns:
            List of Path objects for matching calibration files
        """
        prefix = 'optimal_conditions_'
        return sorted(path for path in self.search_directory.glob(f'{prefix}*.csv')
                      if path.stem[len(prefix):].casefold() == liquid.casefold())
    
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
                self.logger.warning(f"File {file_path} missing required 'volume_target' column")
                return None
            
            # Check which parameters are available
            available_params = [p for p in PIPETTING_PARAMETERS if p in df.columns]
            if not available_params:
                self.logger.warning(f"File {file_path} has no pipetting parameters")
                return None
            
            # Ensure we have volume data
            if df.empty:
                self.logger.warning(f"File {file_path} is empty")
                return None
            
            # Sort by volume for easier interpolation
            df = df.sort_values('volume_target').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading calibration file {file_path}: {e}")
            return None
    
    def get_max_calibrated_volume(self, liquid: str) -> Optional[float]:
        """Return the largest valid calibrated target volume in mL, if available."""
        max_volume_ul = None
        for file_path in self.find_calibration_files(liquid):
            cache_key = str(file_path)
            df = self.cache.get(cache_key)
            if df is None:
                df = self.load_calibration_data(file_path)
                if df is None:
                    continue
                self.cache[cache_key] = df
            volumes = pd.to_numeric(df['volume_target'], errors='coerce')
            volumes = volumes[np.isfinite(volumes) & (volumes > 0)]
            if not volumes.empty:
                candidate = float(volumes.max())
                max_volume_ul = candidate if max_volume_ul is None else max(max_volume_ul, candidate)
        return None if max_volume_ul is None else max_volume_ul / 1000

    def find_best_calibration_file(self, liquid: str, target_volume_ml: float,
                                   tip_type: str = None) -> Optional[Tuple[Path, pd.DataFrame]]:
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
            self.logger.warning(f"No calibration files found for liquid '{liquid}' in directory {self.search_directory}")
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

            if tip_type is not None:
                if 'tip_type' not in df.columns:
                    continue
                df = df[df['tip_type'] == tip_type]
                if df.empty:
                    continue
            
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
            self.logger.warning(f"No valid calibration files found for liquid '{liquid}' (tip: {tip_type or 'unspecified'})")
            return None
            
        return best_file, best_df
    
    def interpolate_parameters(self, df: pd.DataFrame, target_volume_ml: float,
                               extrapolate: bool = False) -> Dict[str, float]:
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
        matches = np.isclose(volumes, target_volume_ul, rtol=0, atol=1e-9)
        if matches.any():
            target_volume_ul = float(volumes[matches][0])
        min_vol, max_vol = volumes.min(), volumes.max()
        
        # Check if we need to extrapolate and warn user
        if target_volume_ul < min_vol:
            self.logger.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is below available range ({min_vol}-{max_vol}uL). Extrapolating...")
        elif target_volume_ul > max_vol:
            self.logger.warning(f"Target volume {target_volume_ml}mL ({target_volume_ul}uL) is above available range ({min_vol}-{max_vol}uL). Extrapolating...")
        
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
                if extrapolate and len(volumes) >= 2 and target_volume_ul < min_vol:
                    interpolated_value = param_values[0] + (target_volume_ul - volumes[0]) * (
                        param_values[1] - param_values[0]) / (volumes[1] - volumes[0])
                elif extrapolate and len(volumes) >= 2 and target_volume_ul > max_vol:
                    interpolated_value = param_values[-1] + (target_volume_ul - volumes[-1]) * (
                        param_values[-1] - param_values[-2]) / (volumes[-1] - volumes[-2])
                else:
                    interpolated_value = np.interp(target_volume_ul, volumes, param_values)
                value = float(interpolated_value)
                if extrapolate and param != 'overaspirate_vol' and value < 0:
                    self.logger.warning("Extrapolated %s below zero; using zero", param)
                    value = 0.0
                
                # Convert speed parameters to integers to avoid conversion warnings
                if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
                    value = max(1, int(round(value)))
                
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
            self.logger.warning("Cannot apply overvolume compensation - missing volume_measured or overaspirate_vol columns")
            return df
        
        # Ensure we have volume_target column
        if 'volume_target' not in df.columns:
            self.logger.warning("Cannot apply overvolume compensation - missing volume_target column")
            return df
        
        compensated_count = 0
        adjustment_details = []
        detail_lines = []
        
        for idx, row in df.iterrows():
            volume_target = row['volume_target']  # uL
            volume_measured = row['volume_measured']  # uL  
            current_overasp = row['overaspirate_vol']  # mL
            
            # Skip rows with invalid measurement data (but allow fresh optimization data)
            if pd.isna(volume_measured) or volume_measured is None:
                continue
                
            # Calculate volume deviation (measured vs target) in uL
            volume_deviation = volume_measured - volume_target  # Positive = over-target, Negative = under-target
            
            # Convert deviation to mL to match overaspirate units
            volume_deviation_ml = volume_deviation / 1000  # Convert uL to mL
            
            # Adjust overaspirate: if over-target, decrease overasp; if under-target, increase overasp
            # Apply full compensation for the measured deviation
            adjustment_factor = 1.0  # Apply 100% of the measured deviation
            adjustment = -volume_deviation_ml * adjustment_factor  # Negative deviation (under) -> positive adjustment (increase overasp)
            
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
                'deviation': volume_deviation, 
                'adjustment_ul': actual_adjustment_ul,
                'old_overasp': current_overasp,
                'new_overasp': new_overasp
            })
            
            # Always apply compensation if there's any volume deviation >0.01uL
            if abs(volume_deviation) > 0.01:  # Only skip truly negligible deviations
                df.at[idx, 'overaspirate_vol'] = new_overasp
                compensated_count += 1
                
                detail_lines.append(f"{volume_target}uL: deviation {volume_deviation:+.2f}uL -> overasp "
                      f"{current_overasp:.4f}->{new_overasp:.4f}mL (delta{actual_adjustment_ul:+.2f}uL)")
        
        if compensated_count > 0:
            self.logger.info(f"Applied overvolume compensation to {compensated_count}/{len(df)} parameter sets")
            self.logger.debug("Overvolume compensation details: " + "; ".join(detail_lines))
        else:
            self.logger.debug("No overvolume compensation applied - all volume deviations were negligible (<0.01uL)")
            
        return df
    
    def get_pipetting_parameters(self, liquid: str, volume_ml: float, compensate_overvolume: bool = True,
                                 tip_type: str = None) -> Optional[Dict[str, float]]:
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
        file_result = self.find_best_calibration_file(liquid, volume_ml, tip_type=tip_type)
        if file_result is None:
            self.logger.warning(
                "No usable calibration for %s at %g uL (tip: %s); "
                "default pipetting parameters may be used.",
                liquid, volume_ml * 1000, tip_type or 'unspecified',
            )
            return None
        
        file_path, df = file_result
        
        # Refuse to extrapolate too far beyond the calibrated range - beyond
        # MAX_EXTRAPOLATION_FRACTION outside it, parameters are unreliable, so
        # fall back to system defaults (the caller does this when None is returned).
        target_volume_ul = volume_ml * 1000
        volumes = df['volume_target'].values
        min_vol, max_vol = volumes.min(), volumes.max()
        if (target_volume_ul < min_vol * (1 - MAX_EXTRAPOLATION_FRACTION)
                or target_volume_ul > max_vol * (1 + MAX_EXTRAPOLATION_FRACTION)):
            self.logger.warning(
                f"Target volume {volume_ml}mL ({target_volume_ul}uL) is more than "
                f"{MAX_EXTRAPOLATION_FRACTION:.0%} outside the calibrated range "
                f"({min_vol}-{max_vol}uL) for '{liquid}' "
                f"(tip: {tip_type or 'unspecified'}) in {file_path.name}; "
                f"calibration unavailable; default pipetting parameters may be used."
            )
            return None
        
        # Apply overvolume compensation before interpolation if requested
        if compensate_overvolume:
            df = self.apply_overvolume_compensation(df.copy())
        
        # Interpolate parameters
        if tip_type is not None and len(df) < 2 and target_volume_ul not in df['volume_target'].values:
            self.logger.warning(
                "%s %s calibration needs two rows for extrapolation at %g uL; "
                "default pipetting parameters may be used.", liquid, tip_type, target_volume_ul,
            )
            return None
        parameters = self.interpolate_parameters(df, volume_ml, extrapolate=tip_type is not None)
        
        # Add metadata
        parameters['_source_file'] = str(file_path)
        parameters['_liquid'] = liquid
        parameters['_compensated'] = compensate_overvolume
        
        processing_notes = []
        if compensate_overvolume:
            processing_notes.append("compensation")
        processing_note = f" (with {', '.join(processing_notes)})" if processing_notes else ""
        self.logger.debug(f"Parameters for {liquid} {volume_ml}mL from {file_path.name}{processing_note}:")
        for key, value in parameters.items():
            if not key.startswith('_'):
                self.logger.debug(f"  {key}: {value:.6f}")
        
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
    
    # Test 1: Get parameters for water at 0.020 mL with different options
    print("=== Test 1: Water 20uL (raw) ===")
    params1 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=False)
    
    # Test 2: With compensation only
    print("\n=== Test 2: Water 20uL (compensated) ===")
    params2 = wizard.get_pipetting_parameters("water", 0.020, compensate_overvolume=True)
    
    # Compare calibrated and compensated overaspirate values.
    if params1 and params2:
        print("\nOveraspirate comparison for 20uL:")
        print(f"  Raw:         {params1['overaspirate_vol']:.6f}")
        print(f"  Compensated: {params2['overaspirate_vol']:.6f}")
