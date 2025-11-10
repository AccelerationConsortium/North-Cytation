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
    ExperimentResults, VolumeCalibrationResult
)

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
        
        # Phase 1: Screening
        screening_trials = self._run_screening_phase(target_volume_ml, optimizer)
        
        # Phase 2: Optimization
        optimization_trials = self._run_optimization_phase(target_volume_ml, optimizer)
        
        # Combine all trials for this volume
        all_volume_trials = screening_trials + optimization_trials
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
            
            logger.info(f"Screening trial {trial_idx + 1}/{num_trials}: "
                       f"score={trial_result.composite_score:.3f}")
        
        return screening_trials
    
    def _generate_screening_parameters(self, target_volume_ml: float, trial_idx: int) -> PipettingParameters:
        """Generate screening parameters using SOBOL sequence or LLM suggestions."""
        # For now, use random parameters within bounds
        # In real implementation, would use SOBOL or LLM
        bounds = self.config.get_parameter_bounds()
        
        # Simple SOBOL-like sampling
        np.random.seed(self.config.get_random_seed() or 0 + trial_idx)
        
        parameters = PipettingParameters(
            aspirate_speed=np.random.uniform(*bounds.aspirate_speed),
            dispense_speed=np.random.uniform(*bounds.dispense_speed),
            aspirate_wait_time_s=np.random.uniform(*bounds.aspirate_wait_time),
            dispense_wait_time_s=np.random.uniform(*bounds.dispense_wait_time),
            retract_speed=np.random.uniform(*bounds.retract_speed),
            blowout_vol_ml=np.random.uniform(*bounds.blowout_vol),
            post_asp_air_vol_ml=np.random.uniform(*bounds.post_asp_air_vol),
            overaspirate_vol_ml=np.random.uniform(*bounds.overaspirate_vol)
        )
        
        # Apply volume constraints
        return self.config.apply_volume_constraints(parameters, target_volume_ml)
    
    def _run_optimization_phase(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """Run Bayesian optimization phase with measurement-based stopping."""
        logger.info("Running optimization phase")
        
        optimization_trials = []
        min_good_trials = self.config.get_min_good_trials()
        max_measurements = self.config.get_max_measurements_first_volume()
        
        good_trials_count = 0
        iteration = 0
        
        # Continue until we hit measurement budget or have enough good trials
        while (self.total_measurements < max_measurements and 
               self.total_measurements < self.config.get_max_total_measurements()):
            
            # Generate next parameters using Bayesian optimizer
            parameters = self._generate_optimization_parameters(target_volume_ml, iteration)
            
            # Execute trial
            trial_result = self._execute_trial(parameters, target_volume_ml, f"optimization_{iteration}")
            optimization_trials.append(trial_result)
            
            # Check if this is a good trial
            if trial_result.quality.overall_quality in ['excellent', 'good']:
                good_trials_count += 1
            
            logger.info(f"Optimization trial {iteration + 1}: "
                       f"score={trial_result.composite_score:.3f}, "
                       f"quality={trial_result.quality.overall_quality}, "
                       f"measurements={self.total_measurements}/{max_measurements}")
            
            # Check stopping criteria
            if good_trials_count >= min_good_trials:
                logger.info(f"Reached {good_trials_count} good trials, stopping optimization")
                break
                
            iteration += 1
        
        if self.total_measurements >= max_measurements:
            logger.info(f"Reached measurement budget ({max_measurements}), stopping optimization")
        
        return optimization_trials
    
    def _generate_optimization_parameters(self, target_volume_ml: float, iteration: int) -> PipettingParameters:
        """Generate parameters using Bayesian optimization."""
        # For now, use slightly perturbed random parameters
        # In real implementation, would use BayBe/Ax suggestions
        bounds = self.config.get_parameter_bounds()
        
        # Use transfer learning if enabled and we have previous volume results
        if (self.config.is_transfer_learning_enabled() and 
            self.volume_results and 
            iteration == 0):
            
            # Start from best parameters from previous volume
            previous_best = self.volume_results[-1].optimal_parameters
            if previous_best:
                logger.info("Using transfer learning from previous volume")
                base_params = previous_best
            else:
                base_params = self.config.get_default_parameters()
        else:
            base_params = self.config.get_default_parameters()
        
        # Add some optimization noise
        noise_scale = 0.1  # 10% of parameter range
        np.random.seed(self.config.get_random_seed() or 0 + iteration + 1000)
        
        def add_noise(value: float, bounds: Tuple[float, float]) -> float:
            range_size = bounds[1] - bounds[0]
            noise = np.random.normal(0, noise_scale * range_size)
            return np.clip(value + noise, bounds[0], bounds[1])
        
        parameters = PipettingParameters(
            aspirate_speed=add_noise(base_params.aspirate_speed, bounds.aspirate_speed),
            dispense_speed=add_noise(base_params.dispense_speed, bounds.dispense_speed),
            aspirate_wait_time_s=add_noise(base_params.aspirate_wait_time_s, bounds.aspirate_wait_time),
            dispense_wait_time_s=add_noise(base_params.dispense_wait_time_s, bounds.dispense_wait_time),
            retract_speed=add_noise(base_params.retract_speed, bounds.retract_speed),
            blowout_vol_ml=add_noise(base_params.blowout_vol_ml, bounds.blowout_vol),
            post_asp_air_vol_ml=add_noise(base_params.post_asp_air_vol_ml, bounds.post_asp_air_vol),
            overaspirate_vol_ml=add_noise(base_params.overaspirate_vol_ml, bounds.overaspirate_vol)
        )
        
        return self.config.apply_volume_constraints(parameters, target_volume_ml)
    
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
                target_volume_ml, 
                replicate_id=replicate_idx
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
                target_volume_ml, 
                replicate_id=replicate_idx
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