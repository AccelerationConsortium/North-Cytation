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
        
        # Accuracy calculation
        deviation_ml = mean_volume_ml - target_volume_ml
        deviation_pct = (deviation_ml / target_volume_ml) * 100 if target_volume_ml > 0 else 0
        absolute_deviation_pct = abs(deviation_pct)
        
        # Apply precision penalty for high deviation (like calibration_sdl_simplified)
        deviation_threshold_pct = getattr(self.config, 'deviation_threshold_pct', 10.0)
        if absolute_deviation_pct > deviation_threshold_pct:
            # Set precision to penalty value for poor accuracy trials
            penalty_variability_pct = getattr(self.config, 'penalty_variability_pct', 100.0)
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
        """Calculate multi-objective composite score."""
        
        # Individual objective scores (lower is better, 0 = perfect)
        accuracy_score = self._calculate_accuracy_score(analysis, tolerances)
        precision_score = self._calculate_precision_score(analysis, tolerances)
        time_score = self._calculate_time_score_for_ranking(analysis)
        
        # Weighted combination (time used for ranking, not success criteria)
        composite_score = (
            self.objective_weights.accuracy_weight * accuracy_score +
            self.objective_weights.precision_weight * precision_score +
            self.objective_weights.time_weight * time_score
        )
        
        logger.debug(f"Score breakdown: accuracy={accuracy_score:.3f}, "
                    f"precision={precision_score:.3f}, time={time_score:.3f}, "
                    f"composite={composite_score:.3f}")
        
        return composite_score
    
    def _calculate_accuracy_score(self, analysis: AdaptiveMeasurementResult,
                                tolerances: VolumeTolerances) -> float:
        """Calculate accuracy objective score (0 = perfect, higher = worse)."""
        deviation_ul = abs(analysis.deviation_ml * 1000)
        tolerance_ul = tolerances.accuracy_tolerance_ul
        threshold_ul = self.objective_thresholds['deviation_threshold_pct'] / 100 * analysis.target_volume_ml * 1000
        
        if deviation_ul <= tolerance_ul:
            # Within tolerance: score between 0 and 1
            return deviation_ul / tolerance_ul
        elif deviation_ul <= threshold_ul:
            # Outside tolerance but not catastrophic: score between 1 and 2
            return 1.0 + (deviation_ul - tolerance_ul) / (threshold_ul - tolerance_ul)
        else:
            # Catastrophic: flat penalty
            return 2.0
    
    def _calculate_precision_score(self, analysis: AdaptiveMeasurementResult,
                                 tolerances: VolumeTolerances) -> float:
        """Calculate precision objective score (0 = perfect, higher = worse)."""
        cv_pct = analysis.cv_volume_pct
        tolerance_pct = tolerances.precision_tolerance_pct
        threshold_pct = self.objective_thresholds['variability_threshold_pct']
        
        if cv_pct <= tolerance_pct:
            # Within tolerance: score between 0 and 1
            return cv_pct / tolerance_pct
        elif cv_pct <= threshold_pct:
            # Outside tolerance but not catastrophic: score between 1 and 2
            return 1.0 + (cv_pct - tolerance_pct) / (threshold_pct - tolerance_pct)
        else:
            # Catastrophic: flat penalty
            return 2.0
    
    def _calculate_time_score_for_ranking(self, analysis: AdaptiveMeasurementResult) -> float:
        """Calculate time score for ranking purposes only (not tolerance-based)."""
        duration_s = analysis.mean_duration_s
        
        # Simple time scoring: normalize around reasonable baseline (30s)
        baseline_time = 30.0
        if duration_s <= baseline_time:
            # Good time: score between 0 and 1
            return duration_s / baseline_time
        else:
            # Slower than baseline: linear penalty 
            return 1.0 + (duration_s - baseline_time) / baseline_time
    
    def _should_run_additional_replicates(self, analysis: AdaptiveMeasurementResult,
                                        measurements: List[RawMeasurement],
                                        target_volume_ml: float) -> bool:
        """Determine if additional replicates are needed based on adaptive strategy."""
        
        if not self.config.is_adaptive_measurement_enabled():
            return False
        
        adaptive_config = self.config.get_adaptive_measurement_config()
        base_replicates = adaptive_config.get('base_replicates', 1)
        max_additional = adaptive_config.get('additional_replicates', 2)
        deviation_threshold_pct = adaptive_config.get('deviation_threshold_pct', 10.0)
        
        # Don't add replicates if we already have enough
        if len(measurements) >= base_replicates + max_additional:
            return False
        
        # Only consider adding replicates if we have at least the base number
        if len(measurements) < base_replicates:
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
        
        Args:
            trials: List of trial results
            quality_filter: Optional quality filter ("within_tolerance", "partial_tolerance")
            max_results: Maximum number of results to return
            
        Returns:
            List[TrialResult]: Best trials matching criteria
        """
        filtered_trials = trials
        
        if quality_filter:
            filtered_trials = [t for t in trials if t.quality.overall_quality == quality_filter]
        
        ranked_trials = self.rank_trials(filtered_trials)
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