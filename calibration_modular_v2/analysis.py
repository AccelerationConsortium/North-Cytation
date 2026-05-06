"""
Analysis Engine for Calibration System
======================================

This module handles all measurement analysis, quality evaluation, and result
processing for the calibration system. It provides type-safe analysis of
raw measurements and converts them into actionable results.

Key Features:
- Statistical analysis of measurement replicates
- Quality evaluation against tolerance thresholds
- Multi-objective scoring (accuracy, precision, time)
- Adaptive measurement logic (conditional replicates)
- Volume-dependent tolerance calculation
- Transfer learning compatibility

Analysis Pipeline:
1. Raw measurements -> Replicate analysis
2. Replicate analysis -> Quality evaluation
3. Quality evaluation -> Multi-objective scoring
4. All results -> Trial ranking and recommendations

Example Usage:
    analyzer = CalibrationAnalyzer(config)
    measurements = [measurement1, measurement2, measurement3]
    result = analyzer.analyze_trial(measurements, target_volume_ml=0.05)
    print(f"Trial quality: {result.quality.overall_quality}")
"""

import logging
import statistics
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import replace

from .data_structures import (
    RawMeasurement, AdaptiveMeasurementResult, TrialResult, 
    QualityEvaluation, VolumeTolerances, PipettingParameters
)
from config_manager import ExperimentConfig, ObjectiveWeights

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    Core analysis engine for calibration measurements.
    
    Handles statistical analysis, quality evaluation, and multi-objective
    scoring of calibration trials. Supports adaptive measurement strategies
    and volume-dependent tolerance calculations.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize analyzer with experiment configuration."""
        self.config = config
        self.objective_weights = config.get_objective_weights()
        self.objective_thresholds = config.get_objective_thresholds()
        
    def analyze_trial(self, measurements: List[RawMeasurement], 
                     target_volume_ml: float,
                     strategy: str = "optimization",
                     liquid: str = "water") -> TrialResult:
        """
        Analyze a complete trial (set of replicate measurements).
        
        Args:
            measurements: List of raw measurements for this trial
            target_volume_ml: Target volume for this trial
            
        Returns:
            TrialResult: Complete analysis with quality and scoring
        """
        if not measurements:
            raise ValueError("Cannot analyze trial with no measurements")
        
        # Validate measurements are consistent
        self._validate_measurement_consistency(measurements, target_volume_ml)
        
        # Get parameters (should be same for all measurements in trial)
        parameters = measurements[0].parameters
        
        # Calculate basic statistics
        adaptive_result = self._calculate_adaptive_measurement_result(measurements, target_volume_ml)
        
        # Evaluate quality against tolerances
        tolerances = self.config.calculate_tolerances_for_volume(target_volume_ml)
        quality = self._evaluate_quality(adaptive_result, tolerances, target_volume_ml)
        
        # Calculate multi-objective score
        composite_score = self._calculate_composite_score(adaptive_result, tolerances)
        
        # Determine if additional replicates are needed
        needs_additional_replicates = self._should_run_additional_replicates(
            adaptive_result, measurements, target_volume_ml
        )
        
        trial_result = TrialResult(
            parameters=parameters,
            target_volume_ml=target_volume_ml,
            measurements=measurements,
            analysis=adaptive_result,
            quality=quality,
            composite_score=composite_score,
            tolerances_used=tolerances,
            strategy=strategy,
            liquid=liquid,
            needs_additional_replicates=needs_additional_replicates,
            metadata={
                'analyzer_version': '2.0',
                'analysis_timestamp': measurements[-1].timestamp
            }
        )
        
        logger.debug(f"Trial analysis completed: score={composite_score:.3f}, "
                    f"quality={quality.overall_quality}, "
                    f"replicates={len(measurements)}")
        
        return trial_result
    
    def _validate_measurement_consistency(self, measurements: List[RawMeasurement], 
                                        target_volume_ml: float):
        """Validate that measurements are consistent for a single trial."""
        if not measurements:
            return
        
        reference_params = measurements[0].parameters
        reference_target = measurements[0].target_volume_ml
        
        for i, measurement in enumerate(measurements[1:], 1):
            if measurement.parameters != reference_params:
                raise ValueError(f"Measurement {i} has different parameters than measurement 0")
            if abs(measurement.target_volume_ml - reference_target) > 1e-6:
                raise ValueError(f"Measurement {i} has different target volume than measurement 0")
            if abs(reference_target - target_volume_ml) > 1e-6:
                raise ValueError(f"Measurement target volume {reference_target} != expected {target_volume_ml}")
    
    def _calculate_adaptive_measurement_result(self, measurements: List[RawMeasurement], 
                                             target_volume_ml: float) -> AdaptiveMeasurementResult:
        """Calculate statistical analysis of measurement replicates."""
        volumes_ml = [m.measured_volume_ml for m in measurements]
        durations_s = [m.duration_s for m in measurements]
        
        # Volume statistics
        mean_volume_ml = statistics.mean(volumes_ml)
        if len(volumes_ml) > 1:
            stdev_volume_ml = statistics.stdev(volumes_ml)
            
            # Check for negative or zero mean volume (invalid data)
            if mean_volume_ml <= 0:
                cv_volume_pct = 100.0  # Penalty for invalid volume
                logger.warning(f"Invalid mean volume ({mean_volume_ml:.3f}mL) - assigning penalty precision (100%)")
            else:
                # Choose variability calculation method based on configuration
                if self.config.use_range_based_variability():
                    # Range-based variability for small samples (original method)
                    range_ml = max(volumes_ml) - min(volumes_ml)
                    cv_volume_pct = (range_ml / (2 * mean_volume_ml)) * 100
                else:
                    # Standard coefficient of variation
                    cv_volume_pct = (stdev_volume_ml / mean_volume_ml) * 100
        else:
            stdev_volume_ml = 0.0
            cv_volume_pct = 0.0
        
        # Accuracy calculation - use average absolute deviation across all replicates
        # This properly accounts for individual measurement scatter, not just mean vs target
        if target_volume_ml > 0:
            individual_deviations_pct = [abs(vol - target_volume_ml) / target_volume_ml * 100 for vol in volumes_ml]
            absolute_deviation_pct = statistics.mean(individual_deviations_pct)
            # Keep the mean-based deviation for compatibility/logging
            deviation_ml = mean_volume_ml - target_volume_ml
            deviation_pct = (deviation_ml / target_volume_ml) * 100
        else:
            deviation_ml = 0.0
            deviation_pct = 0.0
            absolute_deviation_pct = 0.0
        
        # Apply precision penalty for high deviation (like calibration_sdl_simplified)
        adaptive_config = self.config.get_adaptive_measurement_config()
        deviation_threshold_pct = adaptive_config.get('deviation_threshold_pct', 10.0)
        penalty_variability_pct = adaptive_config.get('penalty_variability_pct', 100.0)
        
        if absolute_deviation_pct > deviation_threshold_pct:
            # Set precision to penalty value for poor accuracy trials
            cv_volume_pct = penalty_variability_pct
            logger.info(f"High deviation ({absolute_deviation_pct:.1f}% > {deviation_threshold_pct}%) - applying precision penalty: {penalty_variability_pct}%")
        
        # Timing statistics
        mean_duration_s = statistics.mean(durations_s)
        if len(durations_s) > 1:
            stdev_duration_s = statistics.stdev(durations_s)
        else:
            stdev_duration_s = 0.0
        
        return AdaptiveMeasurementResult(
            target_volume_ml=target_volume_ml,
            num_replicates=len(measurements),
            mean_volume_ml=mean_volume_ml,
            stdev_volume_ml=stdev_volume_ml,
            cv_volume_pct=cv_volume_pct,
            deviation_ml=deviation_ml,
            deviation_pct=deviation_pct,
            absolute_deviation_pct=absolute_deviation_pct,
            mean_duration_s=mean_duration_s,
            stdev_duration_s=stdev_duration_s,
            min_volume_ml=min(volumes_ml),
            max_volume_ml=max(volumes_ml),
            median_volume_ml=statistics.median(volumes_ml)
        )
    
    def _evaluate_quality(self, analysis: AdaptiveMeasurementResult, 
                         tolerances: VolumeTolerances,
                         target_volume_ml: float) -> QualityEvaluation:
        """Evaluate measurement quality against tolerance thresholds."""
        
        # Convert deviation to microliters for comparison
        deviation_ul = abs(analysis.deviation_ml * 1000)
        
        # Check individual criteria (time not evaluated for trial success)
        accuracy_good = deviation_ul <= tolerances.accuracy_tolerance_ul
        precision_good = analysis.cv_volume_pct <= tolerances.precision_tolerance_pct
        
        # Determine overall quality (replication worthiness, not final accuracy assessment)
        if accuracy_good and precision_good:
            overall_quality = "within_tolerance"       # Worth replicating, meets accuracy/precision
        elif accuracy_good or precision_good:
            overall_quality = "partial_tolerance"      # Marginal, might be worth replicating
        else:
            overall_quality = "outside_tolerance"      # Not worth replicating
        
        return QualityEvaluation(
            accuracy_good=accuracy_good,
            precision_good=precision_good, 
            overall_quality=overall_quality,
            accuracy_tolerance_ul=tolerances.accuracy_tolerance_ul,
            precision_tolerance_pct=tolerances.precision_tolerance_pct,
            measured_accuracy_ul=deviation_ul,
            measured_precision_pct=analysis.cv_volume_pct,
            measured_time_s=analysis.mean_duration_s
        )
    
    def _calculate_composite_score(self, analysis: AdaptiveMeasurementResult,
                                 tolerances: VolumeTolerances) -> float:
        """Legacy method - real SDL scoring happens in find_best_trials."""
        # Placeholder - SDL scoring requires all trials for proper normalization
        return 0.0  # Actual calculation in _calculate_sdl_composite_score
    
    def _calculate_sdl_composite_score(self, analysis: AdaptiveMeasurementResult, 
                                     all_analyses: List[AdaptiveMeasurementResult]) -> float:
        """Calculate SDL-style composite score using relative standard deviation normalization."""
        import statistics
        
        # Extract metrics from all trials (excluding penalty trials)
        valid_analyses = [a for a in all_analyses if a.cv_volume_pct < 99.9]
        if not valid_analyses:
            valid_analyses = all_analyses  # Fallback if all have penalties
            
        raw_accuracies = [a.absolute_deviation_pct for a in valid_analyses]
        raw_precisions = [a.cv_volume_pct for a in valid_analyses] 
        raw_times = [a.mean_duration_s for a in valid_analyses]
        
        # Calculate standard deviations (SDL method)
        acc_std = max(statistics.stdev(raw_accuracies) if len(raw_accuracies) > 1 else 0.1, 0.1)
        prec_std = max(statistics.stdev(raw_precisions) if len(raw_precisions) > 1 else 0.1, 0.1) 
        time_std = max(statistics.stdev(raw_times) if len(raw_times) > 1 else 1.0, 1.0)
        
        # Normalize scores (SDL method)
        acc_score = analysis.absolute_deviation_pct / acc_std * 100
        prec_score = analysis.cv_volume_pct / prec_std * 100
        time_score = analysis.mean_duration_s / time_std * 100
        
        # Weighted composite score (SDL method)
        composite_score = (
            self.objective_weights.accuracy_weight * acc_score +
            self.objective_weights.precision_weight * prec_score +
            self.objective_weights.time_weight * time_score
        )
        
        logger.debug(f"SDL Score breakdown: accuracy={acc_score:.3f}, "
                    f"precision={prec_score:.3f}, time={time_score:.3f}, "
                    f"composite={composite_score:.3f}")
        
        return composite_score
    
    # NOTE: Old absolute threshold scoring methods removed - now using SDL scoring
    # Tolerances are still used for success/failure criteria, not ranking
    
    def _should_run_additional_replicates(self, analysis: AdaptiveMeasurementResult,
                                        measurements: List[RawMeasurement],
                                        target_volume_ml: float) -> bool:
        """Determine if additional replicates are needed based on adaptive strategy."""
        
        if not self.config.is_adaptive_measurement_enabled():
            return False
        
        adaptive_config = self.config.get_adaptive_measurement_config()
        max_total = self.config.get_max_replicates_per_trial()
        deviation_threshold_pct = adaptive_config.get('deviation_threshold_pct', 10.0)
        
        # Don't add replicates if we already have enough
        if len(measurements) >= max_total:
            return False
        
        # Add replicates if accuracy is good (might be worth further optimization)
        if analysis.absolute_deviation_pct <= deviation_threshold_pct:
            return True
        
        return False
    
    def rank_trials(self, trials: List[TrialResult]) -> List[TrialResult]:
        """
        Rank trials by composite score (best first).
        
        Args:
            trials: List of trial results to rank
            
        Returns:
            List[TrialResult]: Trials sorted by score (best first)
        """
        return sorted(trials, key=lambda t: t.composite_score)
    
    def find_best_trials(self, trials: List[TrialResult], 
                        quality_filter: Optional[str] = None,
                        max_results: int = 10) -> List[TrialResult]:
        """
        Find best trials with optional quality filtering.
        
        CRITICAL FIX: Recalculates all composite scores using the same baseline
        for fair comparison instead of using stale scores calculated at different times.
        
        Args:
            trials: List of trial results
            quality_filter: Optional quality filter ("within_tolerance", "partial_tolerance")
            max_results: Maximum number of results to return
            
        Returns:
            List[TrialResult]: Best trials matching criteria
        """
        logger.info("[SELECTION] Starting best trials selection with SDL scoring...")
        
        filtered_trials = trials
        
        if quality_filter:
            filtered_trials = [t for t in trials if t.quality.overall_quality == quality_filter]
            logger.info(f"   Applied quality filter '{quality_filter}': {len(filtered_trials)}/{len(trials)} trials")
        
        # Filter out trials with fewer than 2 measurements (can't calculate valid precision)
        original_count = len(filtered_trials)
        filtered_trials = [t for t in filtered_trials if len(t.measurements) >= 2]
        logger.info(f"   Filtered measurement count: {len(filtered_trials)}/{original_count} trials (>=2 measurements)")
        
        if not filtered_trials:
            logger.warning("   No valid trials found for selection!")
            return []
        
        # CRITICAL FIX: Recalculate all composite scores using SDL normalization
        # This ensures fair comparison using relative population normalization
        logger.info("    [RECALC] Recalculating composite scores with SDL normalization:")
        
        # Collect all analyses for proper SDL normalization
        all_analyses = [trial.analysis for trial in filtered_trials]
        
        for i, trial in enumerate(filtered_trials):
            old_score = trial.composite_score
            # Recalculate using SDL method with population normalization
            trial.composite_score = self._calculate_sdl_composite_score(trial.analysis, all_analyses)
            trial_number = i + 1  # Use position as trial number since trial_id might not be set
            logger.info(f"      Trial {trial_number}: "
                       f"old_score={old_score:.3f} -> new_score={trial.composite_score:.3f} "
                       f"(acc={trial.analysis.absolute_deviation_pct:.1f}%, "
                       f"prec={trial.analysis.cv_volume_pct:.1f}%, "
                       f"time={trial.analysis.mean_duration_s:.1f}s)")
        
        ranked_trials = self.rank_trials(filtered_trials)
        
        logger.info(f"   [RANKING] Final ranking (best {min(max_results, len(ranked_trials))} trials):")
        for i, trial in enumerate(ranked_trials[:max_results]):
            trial_number = filtered_trials.index(trial) + 1
            logger.info(f"     Rank {i+1}: Trial {trial_number} "
                       f"(score={trial.composite_score:.3f})")
        
        return ranked_trials[:max_results]
    
    def calculate_trial_statistics(self, trials: List[TrialResult]) -> Dict[str, float]:
        """Calculate summary statistics across multiple trials."""
        if not trials:
            return {}
        
        scores = [t.composite_score for t in trials]
        qualities = [t.quality.overall_quality for t in trials]
        
        quality_counts = {}
        for quality in qualities:
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        return {
            'num_trials': len(trials),
            'mean_score': statistics.mean(scores),
            'median_score': statistics.median(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'stdev_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'within_tolerance_count': quality_counts.get('within_tolerance', 0),
            'partial_tolerance_count': quality_counts.get('partial_tolerance', 0),
            'outside_tolerance_count': quality_counts.get('outside_tolerance', 0),
            'success_rate': quality_counts.get('within_tolerance', 0) / len(trials)
        }
    
    def apply_single_measurement_penalty(self, trial_result: TrialResult) -> TrialResult:
        """Apply penalty for single measurements with poor precision estimation."""
        if len(trial_result.measurements) > 1:
            return trial_result  # No penalty needed
        
        if not self.config.is_adaptive_measurement_enabled():
            return trial_result  # Adaptive measurement disabled
        
        adaptive_config = self.config.get_adaptive_measurement_config()
        penalty_variability_pct = adaptive_config.get('penalty_variability_pct', 100.0)
        
        # Apply penalty by artificially inflating the precision score
        original_score = trial_result.composite_score
        precision_weight = self.objective_weights.precision_weight
        
        # Calculate penalty as if we had very poor precision
        penalty_contribution = precision_weight * (penalty_variability_pct / 100.0)
        adjusted_score = original_score + penalty_contribution
        
        # Update the trial result
        adjusted_trial = replace(
            trial_result,
            composite_score=adjusted_score,
            metadata={
                **trial_result.metadata,
                'single_measurement_penalty_applied': True,
                'original_score': original_score,
                'penalty_amount': penalty_contribution
            }
        )
        
        logger.debug(f"Applied single measurement penalty: {original_score:.3f} -> {adjusted_score:.3f}")
        
        return adjusted_trial