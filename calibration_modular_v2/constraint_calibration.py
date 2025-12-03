"""
Constraint Calibration Module
============================

This module implements two-point calibration for dynamic constraint calculation.

The two-point method uses actual measurement data from optimized parameters 
to calculate precise overaspirate volume bounds for subsequent volume optimization.

Key Features:
- Data-driven constraint calculation using measurement efficiency
- Tolerance-based bound setting for target volume accuracy
- Type-safe interfaces using proper data structures
- Clear separation of calibration logic from optimization logic

Workflow:
1. Take optimized parameters from Volume N
2. Test at Volume N+1 with two different overaspirate settings
3. Calculate volume delivery efficiency (uL gained per uL overaspirate)
4. Use efficiency + tolerance to set tight bounds for Volume N+1 optimization
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from .data_structures import (
    PipettingParameters, TwoPointCalibrationPoint, 
    TwoPointCalibrationResult, ConstraintBoundsUpdate,
    CalibrationParameters
)

logger = logging.getLogger(__name__)

class ConstraintCalibrator:
    """Handles two-point calibration for dynamic constraint calculation."""
    
    def __init__(self, tolerance_pct: float = 3.0, min_overaspirate_spread_ul: float = 5.0):
        """
        Initialize calibrator with configuration.
        
        Args:
            tolerance_pct: Target volume tolerance percentage (e.g., 3.0 for +/-3%)
            min_overaspirate_spread_ul: Minimum spread between calibration points (uL)
        """
        self.tolerance_pct = tolerance_pct
        self.min_overaspirate_spread_ul = min_overaspirate_spread_ul
    
    def calculate_two_point_bounds(
        self,
        optimized_params: PipettingParameters,
        target_volume_ml: float,
        point_1_overaspirate_ml: float,
        point_1_measured_ml: float,
        point_1_variability_pct: float,
        point_1_measurement_count: int,
        point_2_overaspirate_ml: float,
        point_2_measured_ml: float, 
        point_2_variability_pct: float,
        point_2_measurement_count: int,
        best_trial_variability_ml: Optional[float] = None,
        existing_trials: Optional[List] = None
    ) -> TwoPointCalibrationResult:
        """
        Calculate constraint bounds using two-point calibration data.
        
        Args:
            optimized_params: Best parameters from previous volume optimization
            target_volume_ml: Target volume for next optimization
            point_1_overaspirate_ml: Actual overaspirate used in Point 1
            point_1_measured_ml: Measured volume at Point 1
            point_1_variability_pct: Variability at Point 1
            point_1_measurement_count: Number of measurements at Point 1
            point_2_overaspirate_ml: Actual overaspirate used in Point 2
            point_2_measured_ml: Measured volume at Point 2
            point_2_variability_pct: Variability at Point 2
            point_2_measurement_count: Number of measurements at Point 2
            best_trial_variability_ml: Variability from best trial for bounds adjustment
            
        Returns:
            TwoPointCalibrationResult with constraint bounds
        """
        
        # Create calibration points using ACTUAL tested values
        point_1 = TwoPointCalibrationPoint(
            overaspirate_vol_ml=point_1_overaspirate_ml,
            measured_volume_ml=point_1_measured_ml,
            measurement_count=point_1_measurement_count,
            variability_pct=point_1_variability_pct,
            parameters_used=PipettingParameters(
                calibration=CalibrationParameters(overaspirate_vol=point_1_overaspirate_ml),
                hardware=optimized_params.hardware
            )
        )
        
        point_2 = TwoPointCalibrationPoint(
            overaspirate_vol_ml=point_2_overaspirate_ml,
            measured_volume_ml=point_2_measured_ml,
            measurement_count=point_2_measurement_count,
            variability_pct=point_2_variability_pct,
            parameters_used=PipettingParameters(
                calibration=CalibrationParameters(overaspirate_vol=point_2_overaspirate_ml),
                hardware=optimized_params.hardware
            )
        )
        
        # Simple approach: Higher volume = Point 2, Lower volume = Point 1
        if point_1.measured_volume_ml > point_2.measured_volume_ml:
            # Swap if Point 1 has higher volume
            high_point = point_1
            low_point = point_2
        else:
            # Keep as is if Point 2 has higher volume
            high_point = point_2
            low_point = point_1
        
        # Calculate slope using ordered points
        volume_diff_ul = (high_point.measured_volume_ml - low_point.measured_volume_ml) * 1000
        overaspirate_diff_ul = (high_point.overaspirate_vol_ml - low_point.overaspirate_vol_ml) * 1000
        
        if abs(overaspirate_diff_ul) < 0.1:
            raise ValueError(f"Overaspirate difference too small: {overaspirate_diff_ul:.3f}uL")
        
        # Slope should always be positive now (higher overaspirate = higher volume)
        slope = volume_diff_ul / overaspirate_diff_ul
        
        # PHYSICS-BASED EFFICIENCY CONSTRAINTS
        # Efficiency must be between 0.3-1.5 uL/uL (physical limits)
        # - Lower bound (0.3): Even inefficient pipetting retains some overaspirate
        # - Upper bound (1.5): Can't get more volume than aspirated due to noise
        min_efficiency = 0.3
        max_efficiency = 1.5
        
        if slope < min_efficiency or slope > max_efficiency:
            logger.warning(f"Unrealistic efficiency detected ({slope:.3f}uL/uL) - likely due to measurement noise")
            logger.warning(f"Point 1: {low_point.overaspirate_vol_ml*1000:.1f}uL -> {low_point.measured_volume_ml*1000:.1f}uL")
            logger.warning(f"Point 2: {high_point.overaspirate_vol_ml*1000:.1f}uL -> {high_point.measured_volume_ml*1000:.1f}uL")
            
            # Clamp to physically realistic range
            original_slope = slope
            slope = max(min_efficiency, min(max_efficiency, slope))
            logger.warning(f"Clamping efficiency from {original_slope:.3f} to {slope:.3f}uL/uL")
        
        # Use low point as reference for interpolation
        ref_overaspirate_ul = low_point.overaspirate_vol_ml * 1000
        ref_volume_ul = low_point.measured_volume_ml * 1000
        
        # Linear equation: volume = ref_volume + slope * (overaspirate - ref_overaspirate)
        # Solve for target volume: target_volume = ref_volume + slope * (optimal_overaspirate - ref_overaspirate)
        # Rearrange: optimal_overaspirate = ref_overaspirate + (target_volume - ref_volume) / slope
        
        target_volume_ul = target_volume_ml * 1000
        volume_needed_ul = target_volume_ul - ref_volume_ul
        overaspirate_needed_ul = volume_needed_ul / slope
        optimal_overaspirate_ul = ref_overaspirate_ul + overaspirate_needed_ul
        
        # Store the slope as efficiency for compatibility
        volume_efficiency_ul_per_ul = slope
        
        # Calculate shortfall (how much off from target)
        shortfall_ml = target_volume_ml - point_1_measured_ml
        
        # Calculate tolerance range
        tolerance_range_ml = (target_volume_ml * self.tolerance_pct) / 100
        tolerance_range_ul = tolerance_range_ml * 1000
        
        # Simple bounds calculation around optimal point
        min_target_ul = target_volume_ul - tolerance_range_ul  # Lower volume target
        max_target_ul = target_volume_ul + tolerance_range_ul  # Upper volume target
        
        # Calculate overaspirate needed for min and max target volumes
        # Using same linear relationship: volume = ref_volume + slope * (overaspirate - ref_overaspirate)
        min_volume_needed_ul = min_target_ul - ref_volume_ul
        max_volume_needed_ul = max_target_ul - ref_volume_ul
        
        min_overaspirate_needed_ul = min_volume_needed_ul / slope
        max_overaspirate_needed_ul = max_volume_needed_ul / slope
        
        min_overaspirate_ul = ref_overaspirate_ul + min_overaspirate_needed_ul
        max_overaspirate_ul = ref_overaspirate_ul + max_overaspirate_needed_ul
        
        # Convert back to mL for the rest of the calculation
        min_overaspirate_ml = min_overaspirate_ul / 1000
        max_overaspirate_ml = max_overaspirate_ul / 1000
        
        # Apply simple variability adjustment: min_bound - variability, max_bound + variability
        if best_trial_variability_ml is not None and best_trial_variability_ml > 0:
            # User's requested approach: use min(actual_variability, 5% of target volume) 
            max_variability_cap_ml = target_volume_ml * 0.05  # 5% cap
            capped_variability_ml = min(best_trial_variability_ml, max_variability_cap_ml)
            
            min_overaspirate_ml -= capped_variability_ml
            max_overaspirate_ml += capped_variability_ml
            logger.info(f"Applied capped variability adjustment: +/-{capped_variability_ml*1000:.1f}uL "
                       f"(original: {best_trial_variability_ml*1000:.1f}uL, cap: {max_variability_cap_ml*1000:.1f}uL)")
        else:
            # Fallback to measurement error approach if best trial variability not available
            max_variability_pct = max(point_1_variability_pct, point_2_variability_pct, 1.0)
            error_margin_ml = (max_variability_pct / 100) * target_volume_ml
            min_measurement_error_ul = max(error_margin_ml * 1000, 2.0)
            error_margin_change_ul = min_measurement_error_ul / abs(volume_efficiency_ul_per_ul)
            error_margin_change_ml = error_margin_change_ul / 1000
            
            min_overaspirate_ml -= error_margin_change_ml
            max_overaspirate_ml += error_margin_change_ml
            logger.info(f"Used fallback error margin approach: +/-{error_margin_change_ml*1000:.1f}uL")
        
        # Expand bounds to include existing trial data (critical for first volume with screening data)
        if existing_trials:
            existing_overaspirate_values = [trial.parameters.overaspirate_vol for trial in existing_trials]
            if existing_overaspirate_values:
                existing_min = min(existing_overaspirate_values)
                existing_max = max(existing_overaspirate_values)
                
                # Expand bounds to include existing data
                original_min, original_max = min_overaspirate_ml, max_overaspirate_ml
                min_overaspirate_ml = min(min_overaspirate_ml, existing_min)
                max_overaspirate_ml = max(max_overaspirate_ml, existing_max)
                
                logger.info(f"Expanded bounds to include existing trials:")
                logger.info(f"  Original bounds: [{original_min*1000:.1f}, {original_max*1000:.1f}]uL")
                logger.info(f"  Existing data range: [{existing_min*1000:.1f}, {existing_max*1000:.1f}]uL")
                logger.info(f"  Final bounds: [{min_overaspirate_ml*1000:.1f}, {max_overaspirate_ml*1000:.1f}]uL")
        
        # Ensure bounds are ordered correctly (min < max)
        if min_overaspirate_ml > max_overaspirate_ml:
            min_overaspirate_ml, max_overaspirate_ml = max_overaspirate_ml, min_overaspirate_ml
        
        return TwoPointCalibrationResult(
            target_volume_ml=target_volume_ml,
            point_1=point_1,
            point_2=point_2,
            volume_efficiency_ul_per_ul=volume_efficiency_ul_per_ul,
            shortfall_ml=shortfall_ml,
            optimal_overaspirate_ml=optimal_overaspirate_ul / 1000,  # Convert back to mL
            min_overaspirate_ml=min_overaspirate_ml,
            max_overaspirate_ml=max_overaspirate_ml,
            tolerance_range_ml=tolerance_range_ml
        )
    
    def create_constraint_update(self, calibration_result: TwoPointCalibrationResult) -> ConstraintBoundsUpdate:
        """
        Create constraint bounds update from calibration result.
        
        Args:
            calibration_result: Result from two-point calibration
            
        Returns:
            Constraint update ready to apply to optimization
        """
        # Calculate the max variability for justification
        max_variability_pct = max(calibration_result.point_1.variability_pct, 
                                 calibration_result.point_2.variability_pct, 1.0)
        
        justification = (
            f"Two-point calibration: efficiency={calibration_result.volume_efficiency_ul_per_ul:.3f}uL/uL, "
            f"tolerance=+/-{calibration_result.tolerance_range_ml*1000:.1f}uL, "
            f"error_margin=+/-{max_variability_pct:.1f}% for "
            f"{calibration_result.target_volume_ml*1000:.1f}uL target"
        )
        
        return ConstraintBoundsUpdate(
            parameter_name="overaspirate_vol",
            min_value=float(calibration_result.min_overaspirate_ml),
            max_value=float(calibration_result.max_overaspirate_ml),
            justification=justification,
            source_calibration=calibration_result,
            optimal_overaspirate_ml=float(calibration_result.optimal_overaspirate_ml)
        )


def run_two_point_calibration_measurements(
    lash_e,
    optimized_params: PipettingParameters, 
    target_volume_ml: float,
    measurement_count: int = 5
) -> Tuple[float, float, float, float]:
    """
    Execute the two measurement points for calibration.
    
    Args:
        lash_e: Hardware controller
        optimized_params: Best parameters from previous optimization
        target_volume_ml: Target volume for calibration
        measurement_count: Number of replicate measurements per point
        
    Returns:
        Tuple of (point_1_measured_ml, point_1_variability_pct, 
                 point_2_measured_ml, point_2_variability_pct)
    """
    # This function would coordinate with hardware to run the actual measurements
    # Implementation would be similar to existing measurement functions but focused
    # on the specific two-point protocol
    
    # For now, return placeholder - this needs integration with hardware layer
    raise NotImplementedError("Hardware integration for two-point measurements not yet implemented")