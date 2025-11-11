"""
Core Calibration Workflow Engine
================================

This module implements the main calibration experiment workflow, coordinating
all components to execute complete calibration experiments with Bayesian
optimization, adaptive measurement, and transfer learning.

Key Features:
- Multi-volume calibration with parameter inheritance
- Bayesian optimization using qNEHVI and qLogEI
- Adaptive measurement with conditional replicates
- Transfer learning between volumes
- External data integration
- Comprehensive result tracking and export

Workflow Phases:
1. Initialization (protocol setup, configuration validation)
2. Screening (initial parameter exploration)
3. Optimization (Bayesian optimization iterations)
4. Precision testing (final validation with best parameters)
5. Analysis and reporting

Example Usage:
    config = ExperimentConfig.from_yaml("experiment_config.yaml")
    experiment = CalibrationExperiment(config)
    results = experiment.run()
    
    print(f"Best parameters: {results.optimal_conditions}")
    print(f"Success rate: {results.overall_statistics['success_rate']:.1%}")
"""

import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import asdict

from config_manager import ExperimentConfig
from protocol_loader import create_protocol
from analysis import CalibrationAnalyzer
from external_data import ExternalDataLoader
from data_structures import (
    PipettingParameters, RawMeasurement, TrialResult, 
    ExperimentResults, VolumeCalibrationResult,
    CalibrationParameters, HardwareParameters
)

# Import new visualization and export modules
try:
    from visualization import generate_calibration_plots
    from csv_export import export_clean_csvs
    from experiment_analysis import analyze_calibration_experiment
    ENHANCED_OUTPUTS_AVAILABLE = True
except ImportError as e:
    ENHANCED_OUTPUTS_AVAILABLE = False

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class CalibrationExperiment:
    """
    Main calibration experiment coordinator.
    
    Orchestrates the complete calibration workflow including protocol
    initialization, Bayesian optimization, adaptive measurement,
    transfer learning, and result analysis.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration."""
        self.config = config
        self.protocol = None  # Will be ProtocolWrapper from protocol_loader
        self.analyzer = CalibrationAnalyzer(config)
        self.external_data_loader = ExternalDataLoader(config)
        
        # Experiment state
        self.current_volume_index = 0
        self.total_measurements = 0
        self.volume_results: List[VolumeCalibrationResult] = []
        self.all_trials: List[TrialResult] = []
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        self.output_dir = Path(config.get_output_directory()) / f"run_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Calibration experiment initialized")
        logger.info(f"Experiment: {config.get_experiment_name()}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Configure experiment-specific logging."""
        # This would set up file logging, etc.
        pass
    
    def run(self) -> ExperimentResults:
        """
        Execute the complete calibration experiment.
        
        Returns:
            ExperimentResults: Complete experiment results and analysis
        """
        experiment_start_time = time.time()
        
        try:
            logger.info("Starting calibration experiment")
            
            # Initialize protocol
            self._initialize_protocol()
            
            # Process each target volume
            target_volumes = self.config.get_target_volumes_ml()
            for volume_index, target_volume_ml in enumerate(target_volumes):
                self.current_volume_index = volume_index
                
                logger.info(f"Starting calibration for volume {target_volume_ml} mL "
                           f"({volume_index + 1}/{len(target_volumes)})")
                
                volume_result = self._calibrate_volume(target_volume_ml)
                self.volume_results.append(volume_result)
                
                # Check budget constraints
                if self.total_measurements >= self.config.get_max_total_measurements():
                    logger.warning("Reached maximum total measurements, stopping experiment")
                    break
            
            # Finalize protocol
            self._finalize_protocol()
            
            # Generate comprehensive results
            results = self._generate_final_results(experiment_start_time)
            
            # Export results
            self._export_results(results)
            
            # Generate enhanced outputs (visualizations, clean CSVs, analysis)
            self._generate_enhanced_outputs(results)
            
            logger.info(f"Calibration experiment completed successfully")
            logger.info(f"Total measurements: {self.total_measurements}")
            logger.info(f"Total duration: {results.total_duration_s:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            logger.error(traceback.format_exc())
            
            # Try to clean up protocol
            if self.protocol:
                try:
                    self.protocol.wrapup()
                except:
                    pass
            
            raise
    
    def _initialize_protocol(self):
        """Initialize the measurement protocol."""
        self.protocol = create_protocol(self.config)
        
        success = self.protocol.initialize()
        if not success:
            raise RuntimeError("Failed to initialize measurement protocol")
        
        logger.info(f"Protocol initialized: {type(self.protocol).__name__}")
    
    def _finalize_protocol(self):
        """Clean up and finalize the protocol."""
        if self.protocol:
            success = self.protocol.wrapup()
            if not success:
                logger.warning("Protocol cleanup reported issues")
    
    def _calibrate_volume(self, target_volume_ml: float) -> VolumeCalibrationResult:
        """Calibrate parameters for a specific volume."""
        volume_start_time = time.time()
        
        logger.info(f"Calibrating volume: {target_volume_ml} mL")
        
        # Initialize optimizer
        optimizer = self._initialize_optimizer(target_volume_ml)
        
        # Different phases based on volume index
        screening_trials = []
        optimization_trials = []
        overaspirate_trials = []
        
        if self.current_volume_index == 0:
            # First volume: Full optimization (screening + optimization)
            logger.info("First volume: running full optimization with screening")
            screening_trials = self._run_screening_phase(target_volume_ml, optimizer)
            optimization_trials = self._run_optimization_phase(target_volume_ml, optimizer)
        else:
            # Subsequent volumes: Transfer learning only (no screening)
            logger.info(f"Subsequent volume: using transfer learning from first volume")
            optimization_trials = self._run_optimization_phase(target_volume_ml, optimizer)
            
            # Add overaspirate calibration for subsequent volumes
            overaspirate_trials = self._run_overaspirate_calibration_phase(target_volume_ml, optimization_trials)
        
        # Combine all trials for this volume
        all_volume_trials = screening_trials + optimization_trials + overaspirate_trials
        self.all_trials.extend(all_volume_trials)
        
        # Analyze results
        best_trials = self.analyzer.find_best_trials(all_volume_trials, max_results=5)
        volume_statistics = self.analyzer.calculate_trial_statistics(all_volume_trials)
        
        volume_result = VolumeCalibrationResult(
            target_volume_ml=target_volume_ml,
            trials=all_volume_trials,
            best_trials=best_trials,
            optimal_parameters=best_trials[0].parameters if best_trials else None,
            statistics=volume_statistics,
            duration_s=time.time() - volume_start_time,
            measurement_count=len([m for t in all_volume_trials for m in t.measurements])
        )
        
        logger.info(f"Volume {target_volume_ml} mL completed: "
                   f"{len(all_volume_trials)} trials, "
                   f"best score: {best_trials[0].composite_score:.3f}")
        
        return volume_result
    
    def _initialize_optimizer(self, target_volume_ml: float):
        """Initialize Bayesian optimizer for this volume."""
        # This would initialize BayBe/Ax optimizer
        # For now, return a mock optimizer
        logger.info(f"Initialized {self.config.get_optimizer_type()} optimizer")
        return None
    
    def _run_screening_phase(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """Run initial screening phase with exploratory trials."""
        
        # Check if external data is available
        if self.external_data_loader.has_valid_data():
            logger.info("Using external data for screening phase")
            num_trials = self.config.get_screening_trials()
            external_trials = self.external_data_loader.generate_screening_trials(
                target_volume_ml, max_trials=num_trials
            )
            
            if external_trials:
                logger.info(f"Generated {len(external_trials)} trials from external data")
                # Add measurement count for budget tracking
                for trial in external_trials:
                    self.total_measurements += len(trial.measurements)
                return external_trials
            else:
                logger.warning("External data available but no suitable trials generated, falling back to normal screening")
        
        # Fall back to normal screening
        logger.info("Running normal screening phase")
        
        screening_trials = []
        num_trials = self.config.get_screening_trials()
        
        for trial_idx in range(num_trials):
            # Generate screening parameters (would use SOBOL or LLM suggestions)
            parameters = self._generate_screening_parameters(target_volume_ml, trial_idx)
            
            # Execute trial
            trial_result = self._execute_trial(parameters, target_volume_ml, f"screening_{trial_idx}")
            screening_trials.append(trial_result)
            
            # Log meaningful results instead of abstract scores
            avg_volume_ul = trial_result.analysis.mean_volume_ml * 1000
            deviation_pct = abs(trial_result.analysis.deviation_pct)
            avg_time_s = trial_result.analysis.mean_duration_s
            
            logger.info(f"Screening trial {trial_idx + 1}/{num_trials}: "
                       f"{avg_volume_ul:.1f}uL measured ({deviation_pct:.1f}% dev, {avg_time_s:.1f}s)")
        
        return screening_trials
    
    def _generate_screening_parameters(self, target_volume_ml: float, trial_idx: int) -> PipettingParameters:
        """Generate screening parameters using SOBOL sequence or LLM suggestions."""
        
        # Check if LLM screening is enabled
        if self.config.use_llm_for_screening():
            llm_config_path = self.config.get_screening_llm_config_path()
            if llm_config_path:
                try:
                    from llm_recommender import LLMRecommender
                    llm_recommender = LLMRecommender(self.config, llm_config_path, phase="screening")
                    parameters = llm_recommender.suggest_parameters(target_volume_ml, trial_idx)
                    logger.info(f"Using LLM-generated parameters for screening trial {trial_idx}")
                    return self.config.apply_volume_constraints(parameters, target_volume_ml)
                except Exception as e:
                    logger.warning(f"LLM screening failed, falling back to random: {e}")
        
        # Fall back to random/SOBOL screening parameters
        # Generate parameters using config-driven names (hardware-agnostic)
        np.random.seed(self.config.get_random_seed() or 0 + trial_idx)
        
        parameters = self._generate_parameters_from_config(target_volume_ml)
        
        # Apply volume constraints
        return self.config.apply_volume_constraints(parameters, target_volume_ml)
    
    def _generate_parameters_from_config(self, target_volume_ml: float) -> PipettingParameters:
        """Generate parameters using config-driven names - hardware agnostic."""
        # Get mandatory calibration parameters
        cal_bounds = self.config.get_calibration_parameter_bounds()
        cal_params = CalibrationParameters(
            overaspirate_vol=np.random.uniform(*cal_bounds.overaspirate_vol)
        )
        
        # Get optional hardware parameters
        hw_bounds = self.config.get_hardware_parameter_bounds()
        hw_param_dict = {}
        for param_name, bounds in hw_bounds.parameters.items():
            hw_param_dict[param_name] = np.random.uniform(*bounds)
        
        hw_params = HardwareParameters(parameters=hw_param_dict)
        
        return PipettingParameters(
            calibration=cal_params,
            hardware=hw_params
        )
        return self.config.apply_volume_constraints(parameters, target_volume_ml)
    
    def _run_optimization_phase(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """Run Bayesian optimization phase with measurement-based stopping."""
        logger.info("Running optimization phase")
        
        # Use different optimization strategies for first vs subsequent volumes
        if self.current_volume_index == 0:
            return self._run_first_volume_optimization(target_volume_ml, optimizer)
        else:
            return self._run_subsequent_volume_optimization(target_volume_ml, optimizer)
    
    def _run_first_volume_optimization(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """
        Optimize ALL parameters for the first volume.
        This establishes the baseline parameter set for transfer learning.
        """
        logger.info("First volume: optimizing ALL parameters")
        
        optimization_trials = []
        min_good_trials = self.config.get_min_good_trials()
        max_measurements = self.config.get_max_measurements_first_volume()
        
        good_trials_count = 0
        iteration = 0
        
        # Continue until we hit measurement budget or have enough good trials
        while (self.total_measurements < max_measurements and 
               self.total_measurements < self.config.get_max_total_measurements()):
            
            # Generate next parameters using Bayesian optimizer (all parameters)
            parameters = self._generate_optimization_parameters(target_volume_ml, iteration)
            
            # Execute trial
            trial_result = self._execute_trial(parameters, target_volume_ml, f"optimization_{iteration}")
            optimization_trials.append(trial_result)
            
            # Check if this is a good trial
            if trial_result.quality.overall_quality in ['excellent', 'good']:
                good_trials_count += 1
            
            # Log meaningful results instead of abstract scores
            avg_volume_ul = trial_result.analysis.mean_volume_ml * 1000
            deviation_pct = abs(trial_result.analysis.deviation_pct)
            avg_time_s = trial_result.analysis.mean_duration_s
            
            logger.info(f"Optimization trial {iteration + 1}: "
                       f"{avg_volume_ul:.1f}uL measured ({deviation_pct:.1f}% dev, {avg_time_s:.1f}s), "
                       f"measurements={self.total_measurements}/{max_measurements}")
            
            # Check stopping criteria
            if good_trials_count >= min_good_trials:
                logger.info(f"Reached {good_trials_count} good trials, stopping optimization")
                break
                
            iteration += 1
        
        if self.total_measurements >= max_measurements:
            logger.info(f"Reached measurement budget ({max_measurements}), stopping optimization")
        
        return optimization_trials
    
    def _run_subsequent_volume_optimization(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """
        Optimize only VOLUME-DEPENDENT parameters for subsequent volumes.
        Uses optimized parameters from first volume as base, only varies volume-dependent ones.
        """
        logger.info("Subsequent volume: optimizing only volume-dependent parameters")
        
        # Get optimized parameters from first volume
        base_parameters = self._get_transfer_learning_parameters()
        if not base_parameters:
            logger.warning("No base parameters available, falling back to full optimization")
            return self._run_first_volume_optimization(target_volume_ml, optimizer)
        
        volume_dependent_params = self.config.get_volume_dependent_parameters()
        logger.info(f"Volume-dependent parameters: {volume_dependent_params}")
        
        optimization_trials = []
        max_measurements = self.config.get_max_measurements_per_volume()
        min_good_trials = max(2, self.config.get_min_good_trials() // 2)  # Fewer trials needed
        
        good_trials_count = 0
        iteration = 0
        
        while (self.total_measurements < self.config.get_max_total_measurements() and
               len(optimization_trials) < max_measurements):
            
            # Generate parameters with only volume-dependent ones varying
            parameters = self._generate_volume_dependent_parameters(
                base_parameters, target_volume_ml, volume_dependent_params, iteration
            )
            
            # Execute trial
            trial_result = self._execute_trial(parameters, target_volume_ml, f"vol_dependent_{iteration}")
            optimization_trials.append(trial_result)
            
            # Check if this is a good trial
            if trial_result.quality.overall_quality in ['excellent', 'good']:
                good_trials_count += 1
            
            # Log meaningful results instead of abstract scores
            avg_volume_ul = trial_result.analysis.mean_volume_ml * 1000
            deviation_pct = abs(trial_result.analysis.deviation_pct)
            avg_time_s = trial_result.analysis.mean_duration_s
            
            logger.info(f"Volume-dependent trial {iteration + 1}: "
                       f"{avg_volume_ul:.1f}uL measured ({deviation_pct:.1f}% dev, {avg_time_s:.1f}s)")
            
            # Check stopping criteria (fewer trials needed for volume-dependent optimization)
            if good_trials_count >= min_good_trials:
                logger.info(f"Reached {good_trials_count} good trials for volume-dependent optimization")
                break
                
            iteration += 1
        
        return optimization_trials
    
    def _get_transfer_learning_parameters(self) -> Optional[PipettingParameters]:
        """
        Get the best parameters from the first volume for transfer learning.
        
        Returns:
            Best parameters from first volume, or None if not available
        """
        if not self.volume_results:
            logger.warning("No previous volume results available for transfer learning")
            return None
        
        first_volume_result = self.volume_results[0]
        if not first_volume_result.optimal_parameters:
            logger.warning("First volume has no optimal parameters")
            return None
        
        logger.info(f"Using transfer learning from volume {first_volume_result.target_volume_ml} mL")
        return first_volume_result.optimal_parameters
    
    def _generate_volume_dependent_parameters(self, 
                                            base_parameters: PipettingParameters,
                                            target_volume_ml: float,
                                            volume_dependent_params: List[str],
                                            iteration: int) -> PipettingParameters:
        """
        Generate parameters with only volume-dependent ones varying - HARDWARE AGNOSTIC.
        
        Args:
            base_parameters: Optimized parameters from first volume
            target_volume_ml: Current target volume
            volume_dependent_params: List of parameter names to vary
            iteration: Current iteration number
            
        Returns:
            Parameters with volume-dependent parameters varied, others fixed
        """
        # Get bounds for both calibration and hardware parameters
        cal_bounds = self.config.get_calibration_parameter_bounds()
        hw_bounds = self.config.get_hardware_parameter_bounds()
        
        # Start with base calibration parameters
        new_cal_params = CalibrationParameters(
            overaspirate_vol=base_parameters.overaspirate_vol
        )
        
        # Start with base hardware parameters
        new_hw_params = dict(base_parameters.hardware.parameters)
        
        # Vary only volume-dependent parameters
        for param_name in volume_dependent_params:
            if param_name == "overaspirate_vol":
                # Mandatory calibration parameter
                new_cal_params = CalibrationParameters(
                    overaspirate_vol=self._add_parameter_noise(
                        base_parameters.overaspirate_vol, 
                        cal_bounds.overaspirate_vol, 
                        iteration
                    )
                )
            else:
                # Hardware parameter - check if it exists in config
                param_bounds = hw_bounds.get(param_name)
                if param_bounds and param_name in new_hw_params:
                    base_value = base_parameters.get_hardware_param(param_name)
                    new_hw_params[param_name] = self._add_parameter_noise(
                        base_value, param_bounds, iteration
                    )
        
        # Apply volume constraints
        if new_cal_params.overaspirate_vol != base_parameters.overaspirate_vol:
            # Apply dynamic constraint for overaspirate_vol
            max_fraction = self.config.get_overaspirate_max_fraction()
            max_allowed = min(cal_bounds.overaspirate_vol[1], target_volume_ml * max_fraction)
            new_cal_params = CalibrationParameters(
                overaspirate_vol=min(new_cal_params.overaspirate_vol, max_allowed)
            )
        
        return PipettingParameters(
            calibration=new_cal_params,
            hardware=HardwareParameters(parameters=new_hw_params)
        )
    
    def _add_parameter_noise(self, base_value: float, bounds: Tuple[float, float], iteration: int) -> float:
        """
        Add controlled noise to a parameter value for exploration.
        
        Args:
            base_value: Base parameter value
            bounds: (min, max) bounds for parameter
            iteration: Current iteration (affects noise level)
            
        Returns:
            Perturbed parameter value within bounds
        """
        min_val, max_val = bounds
        
        # Reduce noise over iterations (start broad, get more focused)
        noise_factor = max(0.1, 0.5 - iteration * 0.05)
        param_range = max_val - min_val
        noise = np.random.normal(0, param_range * noise_factor)
        
        # Apply noise and clamp to bounds
        new_value = base_value + noise
        return max(min_val, min(max_val, new_value))
    
    def _generate_optimization_parameters(self, target_volume_ml: float, iteration: int) -> PipettingParameters:
        """Generate parameters using Bayesian optimization - HARDWARE AGNOSTIC."""
        # Get bounds for both calibration and hardware parameters
        cal_bounds = self.config.get_calibration_parameter_bounds()
        hw_bounds = self.config.get_hardware_parameter_bounds()
        
        # Use transfer learning if enabled and we have previous volume results
        if (self.config.is_transfer_learning_enabled() and 
            self.volume_results and 
            iteration == 0):
            
            # Start from best parameters from previous volume
            previous_best = self.volume_results[-1].optimal_parameters
            if previous_best:
                logger.info("Using transfer learning from previous volume")
                base_cal = previous_best.calibration
                base_hw = previous_best.hardware
            else:
                base_cal = self.config.get_default_calibration_parameters()
                base_hw = self.config.get_default_hardware_parameters()
        else:
            base_cal = self.config.get_default_calibration_parameters()
            base_hw = self.config.get_default_hardware_parameters()
        
        # Add optimization noise - hardware agnostic
        noise_scale = 0.1  # 10% of parameter range
        np.random.seed(self.config.get_random_seed() or 0 + iteration + 1000)
        
        def add_noise(value: float, bounds: Tuple[float, float]) -> float:
            range_size = bounds[1] - bounds[0]
            noise = np.random.normal(0, noise_scale * range_size)
            return np.clip(value + noise, bounds[0], bounds[1])
        
        # Generate new calibration parameters
        new_cal = CalibrationParameters(
            overaspirate_vol=add_noise(base_cal.overaspirate_vol, cal_bounds.overaspirate_vol)
        )
        
        # Generate new hardware parameters
        new_hw_dict = {}
        for param_name, param_bounds in hw_bounds.parameters.items():
            base_value = base_hw.get(param_name, 
                                   self.config.get_default_hardware_parameters().get(param_name, 0.0))
            new_hw_dict[param_name] = add_noise(base_value, param_bounds)
        
        new_hw = HardwareParameters(parameters=new_hw_dict)
        
        parameters = PipettingParameters(
            calibration=new_cal,
            hardware=new_hw
        )
        
        return self.config.apply_volume_constraints(parameters, target_volume_ml)
    
    def _run_overaspirate_calibration_phase(self, target_volume_ml: float, optimization_trials: List[TrialResult]) -> List[TrialResult]:
        """
        Run post-optimization overaspirate calibration for volume-dependent parameters.
        
        This phase uses the best overall parameters from the optimization phase and
        tests them on the current volume to determine volume-specific overaspirate adjustments.
        Based on calibration_sdl_simplified post-optimization overaspirate calibration.
        
        Args:
            target_volume_ml: Current volume being calibrated
            optimization_trials: Results from optimization phase
            
        Returns:
            List of overaspirate calibration trial results
        """
        logger.info("Starting post-optimization overaspirate calibration phase")
        
        overaspirate_trials = []
        max_measurements = self.config.get_max_total_measurements()
        
        # Get best parameters from optimization (should be from first volume)
        if self.volume_results:
            # Use best parameters from first volume
            first_volume_result = self.volume_results[0]
            if first_volume_result.optimal_parameters:
                best_params = first_volume_result.optimal_parameters
                logger.info(f"Using optimized parameters from first volume ({first_volume_result.target_volume_ml} mL)")
            else:
                logger.warning("No optimal parameters from first volume, using best from current optimization")
                best_params = min(optimization_trials, key=lambda t: t.composite_score).parameters
        else:
            logger.warning("No previous volume results, using best from current optimization")
            best_params = min(optimization_trials, key=lambda t: t.composite_score).parameters
        
        # Test optimized parameters on current volume with multiple replicates
        precision_replicates = 3  # Match calibration_sdl_simplified PRECISION_MEASUREMENTS
        
        logger.info(f"Testing optimized parameters on {target_volume_ml*1000:.0f}uL with {precision_replicates} replicates")
        
        if self.total_measurements + precision_replicates > max_measurements:
            logger.warning(f"Insufficient budget for overaspirate calibration ({precision_replicates} measurements needed)")
            return overaspirate_trials
        
        # Execute precision test with optimized parameters
        baseline_trial = self._execute_trial(
            best_params, 
            target_volume_ml, 
            f"overaspirate_baseline_{target_volume_ml}",
            force_replicates=precision_replicates
        )
        overaspirate_trials.append(baseline_trial)
        
        # Calculate volume-specific shortfall and adjustment
        avg_measured_volume = np.mean([m.actual_volume_ml for m in baseline_trial.measurements])
        shortfall_ml = target_volume_ml - avg_measured_volume
        
        # Only proceed with adjustment if there's significant shortfall
        if abs(shortfall_ml) > target_volume_ml * 0.01:  # 1% threshold
            logger.info(f"Volume shortfall detected: {shortfall_ml*1000:.2f}uL, adjusting overaspirate")
            
            # Calculate adjusted overaspirate (conservative 70% compensation like calibration_sdl_simplified)
            current_overaspirate = best_params.overaspirate_vol
            adjusted_overaspirate = current_overaspirate + (shortfall_ml * 0.7)
            
            # Apply bounds checking using new config structure
            calib_bounds = self.config.get_calibration_parameter_bounds()
            min_over, max_over = calib_bounds.overaspirate_vol
            adjusted_overaspirate = max(min_over, min(max_over, adjusted_overaspirate))
            
            # Create adjusted parameters - HARDWARE AGNOSTIC
            adjusted_cal = CalibrationParameters(overaspirate_vol=adjusted_overaspirate)
            adjusted_hw = best_params.hardware  # Keep all hardware params same
            
            adjusted_params = PipettingParameters(
                calibration=adjusted_cal,
                hardware=adjusted_hw
            )
            
            # Test adjusted parameters
            if self.total_measurements + precision_replicates <= max_measurements:
                logger.info(f"Testing adjusted overaspirate: {adjusted_overaspirate*1000:.2f}uL")
                adjusted_trial = self._execute_trial(
                    adjusted_params,
                    target_volume_ml,
                    f"overaspirate_adjusted_{target_volume_ml}",
                    force_replicates=precision_replicates
                )
                overaspirate_trials.append(adjusted_trial)
                
                # Compare results
                baseline_score = baseline_trial.composite_score
                adjusted_score = adjusted_trial.composite_score
                
                if adjusted_score < baseline_score:
                    logger.info(f"Overaspirate adjustment improved score: {adjusted_score:.3f} vs {baseline_score:.3f}")
                else:
                    logger.info(f"Overaspirate adjustment did not improve score: {adjusted_score:.3f} vs {baseline_score:.3f}")
            else:
                logger.warning("Insufficient budget for adjusted overaspirate test")
        else:
            logger.info(f"No significant shortfall ({shortfall_ml*1000:.2f}uL), skipping adjustment")
        
        logger.info(f"Overaspirate calibration complete: {len(overaspirate_trials)} trials")
        return overaspirate_trials
    
    def _execute_trial(self, parameters: PipettingParameters, 
                      target_volume_ml: float,
                      trial_id: str,
                      force_replicates: Optional[int] = None) -> TrialResult:
        """Execute a complete trial with adaptive measurement."""
        measurements: List[RawMeasurement] = []
        
        # Determine number of initial replicates
        if force_replicates is not None:
            initial_replicates = force_replicates
            use_adaptive = False
        elif self.config.is_adaptive_measurement_enabled():
            adaptive_config = self.config.get_adaptive_measurement_config()
            initial_replicates = adaptive_config.get('base_replicates', 1)
            use_adaptive = True
        else:
            initial_replicates = 1
            use_adaptive = False
        
        # Execute initial replicates
        for replicate_idx in range(initial_replicates):
            measurement = self.protocol.measure(
                parameters, 
                target_volume_ml
            )
            measurements.append(measurement)
            self.total_measurements += 1
        
        # Analyze and determine if more replicates are needed
        while use_adaptive and len(measurements) < force_replicates if force_replicates else 10:
            # Analyze current measurements
            trial_result = self.analyzer.analyze_trial(measurements, target_volume_ml)
            
            if not trial_result.needs_additional_replicates:
                break
            
            # Check budget
            if self.total_measurements >= self.config.get_max_total_measurements():
                logger.warning("Budget exhausted, stopping replicates")
                break
            
            # Add another replicate
            replicate_idx = len(measurements)
            measurement = self.protocol.measure(
                parameters, 
                target_volume_ml
            )
            measurements.append(measurement)
            self.total_measurements += 1
            
            logger.debug(f"Added replicate {replicate_idx + 1} for trial {trial_id}")
        
        # Final analysis
        trial_result = self.analyzer.analyze_trial(measurements, target_volume_ml)
        
        # Apply single measurement penalty if applicable
        if len(measurements) == 1:
            trial_result = self.analyzer.apply_single_measurement_penalty(trial_result)
        
        return trial_result
    
    def _generate_final_results(self, experiment_start_time: float) -> ExperimentResults:
        """Generate comprehensive experiment results."""
        overall_statistics = self.analyzer.calculate_trial_statistics(self.all_trials)
        
        # Find global best parameters
        all_best_trials = self.analyzer.find_best_trials(self.all_trials, max_results=1)
        optimal_conditions = all_best_trials[0].parameters if all_best_trials else None
        
        return ExperimentResults(
            experiment_name=self.config.get_experiment_name(),
            volume_results=self.volume_results,
            optimal_conditions=optimal_conditions,
            total_measurements=self.total_measurements,
            total_duration_s=time.time() - experiment_start_time,
            overall_statistics=overall_statistics,
            config_used=self.config.get_raw_config()
        )
    
    def _export_results(self, results: ExperimentResults):
        """Export results to files."""
        if not self.config.should_export_optimal_conditions():
            return
        
        # Export optimal conditions
        if results.optimal_conditions:
            optimal_df = pd.DataFrame([asdict(results.optimal_conditions)])
            optimal_path = self.output_dir / "optimal_conditions.csv"
            optimal_df.to_csv(optimal_path, index=False)
            logger.info(f"Exported optimal conditions to {optimal_path}")
        
        # Export raw measurements if requested
        if self.config.should_save_raw_measurements():
            all_measurements = []
            for trial in self.all_trials:
                for measurement in trial.measurements:
                    measurement_dict = asdict(measurement)
                    measurement_dict['trial_score'] = trial.composite_score
                    measurement_dict['trial_quality'] = trial.quality.overall_quality
                    all_measurements.append(measurement_dict)
            
            if all_measurements:
                measurements_df = pd.DataFrame(all_measurements)
                measurements_path = self.output_dir / "raw_measurements.csv"
                measurements_df.to_csv(measurements_path, index=False)
                logger.info(f"Exported raw measurements to {measurements_path}")
        
        # Export summary statistics
        summary_path = self.output_dir / "experiment_summary.json"
        import json
        with open(summary_path, 'w') as f:
            summary_data = {
                'experiment_name': results.experiment_name,
                'total_measurements': results.total_measurements,
                'total_duration_s': results.total_duration_s,
                'overall_statistics': results.overall_statistics,
                'volume_count': len(results.volume_results)
            }
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Exported experiment summary to {summary_path}")
    
    def _generate_enhanced_outputs(self, results: ExperimentResults) -> None:
        """Generate enhanced visualization and analysis outputs."""
        if not ENHANCED_OUTPUTS_AVAILABLE:
            logger.info("Enhanced outputs not available (modules not imported)")
            return
            
        try:
            logger.info("Generating enhanced outputs...")
            
            # Prepare data for enhanced outputs
            trial_results = []
            optimal_conditions = []
            raw_measurements = []
            
            # Convert trials to dictionaries
            trial_counter = 0
            for trial in self.all_trials:
                trial_counter += 1
                trial_dict = {
                    'trial_id': f"trial_{trial_counter}",
                    'target_volume_ml': trial.target_volume_ml,
                    'parameters': asdict(trial.parameters),
                    'analysis': asdict(trial.analysis),
                    'quality': asdict(trial.quality),
                    'composite_score': trial.composite_score,
                    'tolerances_used': asdict(trial.tolerances_used) if trial.tolerances_used else {},
                    'measurements': [asdict(m) for m in trial.measurements]
                }
                trial_results.append(trial_dict)
                
                # Add raw measurements
                for measurement in trial.measurements:
                    measurement_dict = asdict(measurement)
                    measurement_dict['trial_id'] = f"trial_{trial_counter}"
                    raw_measurements.append(measurement_dict)
            
            # Convert optimal conditions
            for volume_result in results.volume_results:
                if volume_result.optimal_parameters:
                    # Create optimal condition dictionary from the best trial
                    best_trial = volume_result.best_trials[0] if volume_result.best_trials else None
                    if best_trial:
                        optimal_dict = {
                            'volume_ul': volume_result.target_volume_ml * 1000,
                            'measured_volume_ul': best_trial.analysis.mean_volume_ml * 1000,
                            'deviation_pct': best_trial.analysis.absolute_deviation_pct,
                            'cv_pct': best_trial.analysis.cv_volume_pct,
                            'time_s': best_trial.analysis.mean_duration_s,
                            'trials_used': len(volume_result.trials),
                            'measurements_count': volume_result.measurement_count,
                            'status': 'success' if best_trial.analysis.absolute_deviation_pct <= 10 else 'partial_success',
                            'parameters': asdict(volume_result.optimal_parameters)
                        }
                        optimal_conditions.append(optimal_dict)
            
            # Generate visualizations
            logger.info("Generating calibration plots...")
            generate_calibration_plots(trial_results, optimal_conditions, raw_measurements, str(self.output_dir))
            
            # Export clean CSV files
            logger.info("Exporting clean CSV files...")
            export_clean_csvs(trial_results, optimal_conditions, raw_measurements, str(self.output_dir))
            
            # Generate analysis report
            logger.info("Generating analysis insights...")
            insights = analyze_calibration_experiment(trial_results, optimal_conditions, str(self.output_dir))
            
            # Log summary of enhanced outputs
            plots_dir = self.output_dir / "plots"
            if plots_dir.exists():
                plot_count = len(list(plots_dir.glob("*.png")))
                logger.info(f"[SUCCESS] Generated {plot_count} visualization plots in plots/")
            
            clean_csvs = [f for f in self.output_dir.glob("*_clean.csv")]
            if clean_csvs:
                logger.info(f"[SUCCESS] Generated {len(clean_csvs)} clean CSV files")
            
            if insights:
                logger.info("[SUCCESS] Generated analysis insights and recommendations")
                
        except Exception as e:
            logger.error(f"Error generating enhanced outputs: {e}")
            logger.debug(f"Enhanced outputs error details: {traceback.format_exc()}")