"""
Data-driven constraint updater based on screening trial analysis.

Instead of using magic numbers, this module analyzes actual screening data
to determine realistic overaspirate bounds for optimization.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass

from data_structures import TrialResult
from optimization_structures import OptimizationConstraints, ConstraintBoundsUpdate

logger = logging.getLogger(__name__)


@dataclass
class ScreeningAnalysis:
    """Results from analyzing screening trial performance."""
    target_volume_ml: float
    total_trials: int
    usable_trials: int  # Trials with reasonable precision
    best_trial_overaspirate_ml: float
    best_trial_deviation_pct: float
    best_trial_precision_pct: float
    estimated_shortfall_ml: float
    recommended_max_overaspirate_ml: float
    recommended_min_overaspirate_ml: float
    confidence_level: str  # "high", "medium", "low"


def analyze_screening_for_constraints(
    screening_trials: List[TrialResult], 
    target_volume_ml: float,
    precision_threshold_pct: float = 25.0
) -> Optional[ScreeningAnalysis]:
    """
    Analyze screening trials to determine realistic overaspirate constraints.
    
    This is a data-driven approach that looks at actual trial performance
    rather than using magic number rules.
    
    Args:
        screening_trials: List of completed screening trials
        target_volume_ml: Target volume being calibrated
        precision_threshold_pct: Only use trials with precision below this threshold
        
    Returns:
        ScreeningAnalysis object with constraint recommendations, or None if insufficient data
    """
    if not screening_trials:
        logger.info("No screening trials available for constraint analysis")
        return None
    
    logger.info(f"Analyzing {len(screening_trials)} screening trials for constraint optimization")
    
    # Filter to usable trials (reasonable precision, not penalty values)
    usable_trials = []
    for trial in screening_trials:
        if trial.analysis.cv_volume_pct < precision_threshold_pct:
            usable_trials.append(trial)
    
    if not usable_trials:
        logger.warning(f"No usable screening trials found (all precision ≥ {precision_threshold_pct}%)")
        return None
    
    logger.info(f"Found {len(usable_trials)}/{len(screening_trials)} usable trials for analysis")
    
    # Find best trial by accuracy
    best_trial = min(usable_trials, key=lambda t: t.analysis.absolute_deviation_pct)
    
    # Extract key metrics
    best_overaspirate_ml = best_trial.parameters.calibration.overaspirate_vol
    best_deviation_pct = best_trial.analysis.absolute_deviation_pct
    best_precision_pct = best_trial.analysis.cv_volume_pct
    
    # Calculate actual shortfall from best trial
    target_volume_ul = target_volume_ml * 1000
    measured_volume_ul = best_trial.analysis.mean_volume_ml * 1000
    shortfall_ul = target_volume_ul - measured_volume_ul
    shortfall_ml = shortfall_ul / 1000
    
    logger.info(f"Best screening trial analysis:")
    logger.info(f"  Overaspirate used: {best_overaspirate_ml*1000:.1f}uL")
    logger.info(f"  Deviation achieved: {best_deviation_pct:.1f}%")
    logger.info(f"  Precision achieved: {best_precision_pct:.1f}%")
    logger.info(f"  Target: {target_volume_ul:.1f}uL, Measured: {measured_volume_ul:.1f}uL")
    logger.info(f"  Shortfall: {shortfall_ul:+.1f}uL")
    
    # Calculate recommended bounds based on data
    buffer_ml = 0.005  # 5μL buffer for safety
    
    if shortfall_ul > 0:  # Under-delivery
        # Need more overaspirate to compensate
        estimated_additional_ml = shortfall_ml * 0.8  # Assume 80% overaspirate effectiveness
        recommended_max_ml = best_overaspirate_ml + estimated_additional_ml + buffer_ml
        recommended_min_ml = 0.0
        logger.info(f"  Under-delivery detected: recommending additional {estimated_additional_ml*1000:.1f}uL overaspirate")
    else:  # Over-delivery or accurate
        # Current overaspirate level seems appropriate, allow some range around it
        recommended_max_ml = best_overaspirate_ml + buffer_ml
        recommended_min_ml = max(0.0, best_overaspirate_ml - buffer_ml)
        logger.info(f"  Good delivery achieved: recommending range around current level")
    
    # Determine confidence based on trial quality
    if best_deviation_pct < 5.0 and len(usable_trials) >= 3:
        confidence = "high"
    elif best_deviation_pct < 10.0 and len(usable_trials) >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Apply sanity checks
    max_reasonable_ml = target_volume_ml * 0.5  # Don't exceed 50% of target volume
    recommended_max_ml = min(recommended_max_ml, max_reasonable_ml)
    
    logger.info(f"  Recommended overaspirate bounds: [{recommended_min_ml*1000:.1f}, {recommended_max_ml*1000:.1f}]uL")
    logger.info(f"  Confidence level: {confidence}")
    
    return ScreeningAnalysis(
        target_volume_ml=target_volume_ml,
        total_trials=len(screening_trials),
        usable_trials=len(usable_trials),
        best_trial_overaspirate_ml=best_overaspirate_ml,
        best_trial_deviation_pct=best_deviation_pct,
        best_trial_precision_pct=best_precision_pct,
        estimated_shortfall_ml=shortfall_ml,
        recommended_max_overaspirate_ml=recommended_max_ml,
        recommended_min_overaspirate_ml=recommended_min_ml,
        confidence_level=confidence
    )


def create_constraint_update_from_screening(
    analysis: ScreeningAnalysis,
    apply_update: bool = True
) -> Optional[ConstraintBoundsUpdate]:
    """
    Create constraint bounds update based on screening analysis.
    
    Args:
        analysis: ScreeningAnalysis from analyze_screening_for_constraints()
        apply_update: If False, only creates update for high-confidence recommendations
        
    Returns:
        ConstraintBoundsUpdate for the optimizer, or None if not recommended
    """
    if not analysis:
        return None
    
    # Only apply updates for medium+ confidence unless forced
    if not apply_update and analysis.confidence_level == "low":
        logger.info(f"Skipping constraint update due to low confidence (only {analysis.usable_trials} usable trials)")
        return None
    
    logger.info(f"Creating constraint update based on screening analysis")
    logger.info(f"  Bounds: [{analysis.recommended_min_overaspirate_ml*1000:.1f}, {analysis.recommended_max_overaspirate_ml*1000:.1f}]uL")
    logger.info(f"  Confidence: {analysis.confidence_level}")
    
    return ConstraintBoundsUpdate(
        parameter_name="overaspirate_vol",
        min_value=analysis.recommended_min_overaspirate_ml,
        max_value=analysis.recommended_max_overaspirate_ml,
        source="screening_analysis",
        confidence=analysis.confidence_level,
        metadata={
            "total_screening_trials": analysis.total_trials,
            "usable_trials": analysis.usable_trials,
            "best_trial_deviation_pct": analysis.best_trial_deviation_pct,
            "best_trial_overaspirate_ul": analysis.best_trial_overaspirate_ml * 1000,
            "estimated_shortfall_ul": analysis.estimated_shortfall_ml * 1000
        }
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Screening constraint updater - data-driven constraint optimization")
    print("This module analyzes actual screening trial performance to set realistic bounds")
    print("instead of using magic number rules.")