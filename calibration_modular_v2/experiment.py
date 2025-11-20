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

from .constraint_calibration import ConstraintCalibrator
from .data_structures import (
    PipettingParameters, TrialResult, VolumeCalibrationResult, 
    ExperimentResults, ConstraintBoundsUpdate, TwoPointCalibrationResult,
    CalibrationParameters
)

from .config_manager import ExperimentConfig
from .protocol_loader import load_hardware_protocol
from .analysis import CalibrationAnalyzer
from .external_data import ExternalDataLoader
from .data_structures import (
    PipettingParameters, RawMeasurement, TrialResult, 
    ExperimentResults, VolumeCalibrationResult,
    CalibrationParameters, HardwareParameters
)

# Import new visualization and export modules
try:
    from .visualization import generate_calibration_plots
    from .csv_export import export_clean_csvs
    from .experiment_analysis import analyze_calibration_experiment
    ENHANCED_OUTPUTS_AVAILABLE = True
except ImportError as e:
    ENHANCED_OUTPUTS_AVAILABLE = False

# Import optimization modules (required - no fallbacks)
from .optimization_structures import OptimizationObjectives, OptimizerType
from .bayesian_recommender import create_optimizer, AxBayesianOptimizer

logger = logging.getLogger(__name__)


class CalibrationExperiment:
    """
    Main calibration experiment coordinator.
    
    Orchestrates the complete calibration workflow including protocol
    initialization, Bayesian optimization, adaptive measurement,
    transfer learning, and result analysis.
    """
    
    def __init__(self, config: ExperimentConfig, protocol=None):
        """Initialize experiment with configuration and optional protocol.
        
        Args:
            config: Experiment configuration
            protocol: Optional protocol instance. If None, will be created from config.
        """
        self._initialize_core_components(config, protocol)
        self._initialize_experiment_state()
        self._setup_logging()
        self._create_output_directory()
        
        logger.info(f"Calibration experiment initialized")
        logger.info(f"Experiment: {config.get_experiment_name()}")
        logger.info(f"Output directory: {self.output_dir}")

    def _initialize_core_components(self, config: ExperimentConfig, protocol=None):
        """Initialize core experiment components."""
        self.config = config
        self.analyzer = CalibrationAnalyzer(config)
        self.external_data_loader = ExternalDataLoader(config)
        
        # Store protocol for direct use
        self.protocol_instance = protocol  # From abstract base class if provided
        self.protocol_module = None  # From function-based protocol if needed
        self.protocol_state = None  # Protocol state
        
    def _initialize_experiment_state(self):
        """Initialize experiment state variables."""
        self.current_volume_index = 0
        self.total_measurements = 0
        self.volume_results: List[VolumeCalibrationResult] = []
        self.all_trials: List[TrialResult] = []
        
    def _create_output_directory(self):
        """Create timestamped output directory."""
        self.output_dir = Path(self.config.get_output_directory()) / f"run_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_optimization_loop(self, target_volume_ml: float, max_measurements: int, 
                              strategy_name: str, param_generation_func) -> List[TrialResult]:
        """Execute optimization loop with specified parameter generation strategy.
        
        Args:
            target_volume_ml: Target volume for optimization
            max_measurements: Maximum measurements for this optimization  
            strategy_name: Description for logging
            param_generation_func: Function to generate parameters for each iteration
            
        Returns:
            List of trial results from optimization
        """
        logger.info(f"{strategy_name}: starting optimization loop")
        
        optimization_trials = []
        min_good_trials = self.config.get_min_good_trials()
        good_trials_count = 0
        iteration = 0
        
        # Continue until we hit measurement budget or have enough good trials
        while (self.total_measurements < max_measurements and 
               self.total_measurements < self.config.get_max_total_measurements()):
            
            # Generate parameters using provided function
            parameters = param_generation_func(target_volume_ml, iteration)
            
            # Execute trial
            trial_result = self._execute_trial(
                parameters, target_volume_ml, f"{strategy_name.lower()}_{iteration}",
                strategy="optimization", liquid=self.config.get_liquid_name()
            )
            optimization_trials.append(trial_result)
            
            # Check if this is a good trial
            trial_passed = self._is_trial_successful(trial_result, target_volume_ml)
            if trial_passed:
                good_trials_count += 1
            
            # Log results
            self._log_optimization_progress(trial_result, iteration, good_trials_count, 
                                          max_measurements, strategy_name)
            
            # Check stopping criteria
            if good_trials_count >= min_good_trials:
                logger.info(f"Reached {good_trials_count} good trials, stopping optimization")
                break
                
            iteration += 1
        
        if self.total_measurements >= max_measurements:
            logger.info(f"Reached measurement budget ({max_measurements}), stopping optimization")
            
        return optimization_trials
        
    def _log_optimization_progress(self, trial_result: TrialResult, iteration: int, 
                                 good_trials_count: int, max_measurements: int, strategy_name: str):
        """Log optimization progress with meaningful metrics."""
        avg_volume_ul = trial_result.analysis.mean_volume_ml * 1000
        deviation_pct = abs(trial_result.analysis.deviation_pct)
        avg_time_s = trial_result.analysis.mean_duration_s
        
        logger.info(f"Trial {iteration + 1}: deviation={deviation_pct:.1f}%, good_trials={good_trials_count}")
        logger.info(f"{strategy_name} trial {iteration + 1}: "
                   f"{avg_volume_ul:.1f}uL measured ({deviation_pct:.1f}% dev, {avg_time_s:.1f}s), "
                   f"measurements={self.total_measurements}/{max_measurements}")

    def _create_volume_dependent_param_generator(self, base_parameters: 'PipettingParameters', 
                                                volume_dependent_params: List[str]):
        """Create a parameter generation function for volume-dependent optimization."""
        def generate_params(target_volume_ml: float, iteration: int) -> 'PipettingParameters':
            return self._generate_volume_dependent_parameters(
                base_parameters, target_volume_ml, volume_dependent_params, iteration
            )
        return generate_params

    def _create_inherited_parameters(self, target_volume_ml: float, 
                                   optimal_overaspirate_ml: float) -> 'PipettingParameters':
        """Create parameters for inherited trial with optimal overaspirate.
        
        Args:
            target_volume_ml: Target volume
            optimal_overaspirate_ml: Calculated optimal overaspirate volume
            
        Returns:
            PipettingParameters with optimal overaspirate and inherited hardware params
        """
        from .data_structures import PipettingParameters, CalibrationParameters
        
        # Use best parameters from previous volume as base
        if self.volume_results:
            base_params = self.volume_results[-1].optimal_parameters
            if base_params:
                return PipettingParameters(
                    calibration=CalibrationParameters(overaspirate_vol=optimal_overaspirate_ml),
                    hardware=base_params.hardware
                )
        
        # Fallback to generated defaults with optimal overaspirate
        fallback_params = self._generate_screening_parameters(target_volume_ml, 0)
        return PipettingParameters(
            calibration=CalibrationParameters(overaspirate_vol=optimal_overaspirate_ml),
            hardware=fallback_params.hardware
        )

    def _log_inherited_trial_result(self, trial_result: 'TrialResult', target_volume_ml: float):
        """Log inherited trial results with success/failure assessment."""
        is_successful = self._is_trial_successful(trial_result, target_volume_ml)
        
        if is_successful:
            logger.info("[SUCCESS] Inherited trial SUCCEEDED - using optimal parameters directly")
        else:
            logger.info("[FAILED] Inherited trial FAILED - will proceed with optimization")
            
        logger.info(f"  Measured: {trial_result.analysis.deviation_pct:.1f}% deviation")
        logger.info(f"  Variability: {trial_result.analysis.cv_volume_pct:.1f}% CV")
    
    def _setup_logging(self):
        """Configure experiment-specific logging."""
        # This would set up file logging, etc.
        pass
    
    def _calculate_volume_budget(self, volume_index: int, target_volumes: list) -> int:
        """
        Calculate measurement budget for current volume using adaptive allocation.
        
        First volume gets dedicated budget from config.
        Subsequent volumes share remaining budget equally.
        
        Args:
            volume_index: Current volume index (0-based)
            target_volumes: List of all target volumes
            
        Returns:
            Measurement budget for this volume
        """
        if volume_index == 0:
            # First volume gets dedicated budget
            return self.config.get_max_measurements_first_volume()
        
        # Calculate remaining budget for subsequent volumes
        total_budget = self.config.get_max_total_measurements()
        volumes_remaining = len(target_volumes) - volume_index
        measurements_remaining = total_budget - self.total_measurements
        
        # Allocate remaining measurements equally among remaining volumes
        volume_budget = measurements_remaining // volumes_remaining if volumes_remaining > 0 else 0
        
        return max(0, volume_budget)
    
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
                
                # Calculate adaptive budget for this volume
                volume_budget = self._calculate_volume_budget(volume_index, target_volumes)
                
                logger.info(f"Starting calibration for volume {target_volume_ml} mL "
                           f"({volume_index + 1}/{len(target_volumes)}) - Budget: {volume_budget} measurements")
                
                # Check if we have sufficient budget to proceed
                if volume_budget < 3:  # Need minimum measurements for meaningful calibration
                    logger.warning(f"Insufficient budget ({volume_budget}) for volume {target_volume_ml} mL, skipping")
                    break
                
                volume_result = self._calibrate_volume(target_volume_ml, volume_budget)
                self.volume_results.append(volume_result)
                
                # Check global budget constraints
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
            if self.protocol_module and self.protocol_state:
                try:
                    self.protocol_module.wrapup(self.protocol_state)
                except:
                    pass
            
            raise
    
    def _initialize_protocol(self):
        """Initialize the measurement protocol."""
        try:
            logger.info(f"Initializing protocol...")
            
            # Convert config to dict format for protocols - FAIL if hardware missing
            raw_config = self.config.get_raw_config()
            if 'hardware' not in raw_config:
                raise ValueError("Missing required 'hardware' configuration section")
            
            config_dict = {
                'experiment': {
                    'liquid': self.config.get_liquid_name()
                },
                'hardware': raw_config['hardware'],  # Fail if missing
                'random_seed': self.config.get_random_seed()
            }
            
            # Load protocol module from config
            protocol_name = self.config.get_protocol_module()
            self.protocol_module = load_hardware_protocol(protocol_name)
            self.protocol_state = self.protocol_module.initialize(config_dict)
            logger.info(f"Protocol initialized: {protocol_name}")
            
        except Exception as e:
            logger.error(f"Protocol initialization failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
            raise RuntimeError(f"Failed to initialize measurement protocol: {e}") from e
    
    def _finalize_protocol(self):
        """Clean up and finalize the protocol."""
        if self.protocol_module and self.protocol_state:
            success = self.protocol_module.wrapup(self.protocol_state)
            if not success:
                logger.warning("Protocol cleanup reported issues")
    
    def _execute_protocol_measurement(self, parameters: Dict[str, float], target_volume_ml: float) -> Dict:
        """Execute a measurement using the protocol module."""
        # Convert PipettingParameters to dict for protocol compatibility
        if hasattr(parameters, 'calibration') and hasattr(parameters, 'hardware'):
            # It's a PipettingParameters object - convert to dict
            params_dict = {
                'overaspirate_vol': parameters.overaspirate_vol,
                **parameters.hardware.__dict__  # Unpack all hardware parameters
            }
        else:
            # It's already a dict
            params_dict = parameters
            
        # Protocol returns a list of measurements (one per replicate)
        # We're calling with replicates=1, so extract the single measurement
        measurement_list = self.protocol_module.measure(self.protocol_state, target_volume_ml, params_dict, replicates=1)
        measurement_dict = measurement_list[0]  # Get single measurement dict
        
        # Convert to RawMeasurement object expected by analyzer
        return RawMeasurement(
            measurement_id=f"measurement_{int(time.time()*1000000)}_{measurement_dict.get('replicate', 1)}",
            parameters=parameters,  # Original PipettingParameters object
            target_volume_ml=target_volume_ml,
            measured_volume_ml=measurement_dict['volume'],
            duration_s=measurement_dict['elapsed_s'],
            replicate_id=measurement_dict.get('replicate', 1),
            metadata=measurement_dict
        )
    
    def _calibrate_volume(self, target_volume_ml: float, volume_budget: int) -> VolumeCalibrationResult:
        """
        Calibrate a single volume using proper Bayesian optimization.
        
        First volume: Full screening + multi-objective optimization
        Subsequent volumes: Transfer learning + single-objective optimization 
        
        Args:
            target_volume_ml: Target volume to calibrate
            volume_budget: Maximum measurements allocated for this volume
        """
        start_time = time.time()
        logger.info(f"Starting calibration for {target_volume_ml*1000:.1f}uL")
        
        # Phase 1: Screening (only for first volume)
        screening_trials = []
        if self.current_volume_index == 0:
            logger.info("Running screening phase (first volume)")
            screening_trials = self._run_screening_phase(target_volume_ml)
            # Store screening trials for two-point calibration access
            self._current_screening_trials = screening_trials
        else:
            logger.info("Skipping screening phase (using transfer learning from first volume)")
        
        # Phase 2: Two-point constraint calibration 
        constraint_update = None
        two_point_trials = []
        inherited_trial = None
        
        # For first volume: use best screening candidate as baseline
        # For subsequent volumes: use previous volume's optimal parameters as baseline
        if self.current_volume_index == 0 and screening_trials:
            logger.info("Running two-point constraint calibration using best screening candidate")
            constraint_update, two_point_trials = self._run_two_point_constraint_calibration(target_volume_ml, volume_budget)
        elif self.current_volume_index > 0:
            logger.info("Running two-point constraint calibration using previous volume parameters")
            constraint_update, two_point_trials = self._run_two_point_constraint_calibration(target_volume_ml, volume_budget)
        
        # Store two-point trials for subsequent optimizer seeding
        self._current_two_point_trials = two_point_trials
        
        # PAUSE: Allow inspection of two-point calibration results (if any were run)
        if two_point_trials:
            logger.info("=" * 60)
            logger.info("TWO-POINT CALIBRATION COMPLETE - PAUSING FOR INSPECTION")
            if constraint_update:
                logger.info(f"Constraint update: {constraint_update}")
                logger.info(f"Efficiency: {constraint_update.justification}")
            logger.info(f"Two-point trials executed: {len(two_point_trials)}")
            for i, trial in enumerate(two_point_trials):
                # Use trial_name instead of trial_id and correct volume attribute
                trial_name = getattr(trial, 'trial_name', f'trial_{i+1}')
                measured_vol = getattr(trial, 'measured_volume_ml', getattr(trial, 'volume_ml', 0.0))
                logger.info(f"  Trial {i+1}: {trial_name} -> {measured_vol*1000:.1f}uL (target: {trial.target_volume_ml*1000:.1f}uL)")
            logger.info("=" * 60)
        
        # Phase 2.5: Inherited trial (test optimal overaspirate value) - only for subsequent volumes
        if self.current_volume_index > 0:
            inherited_trial = self._run_inherited_trial(target_volume_ml, constraint_update)
            
        # Phase 3: Bayesian optimization (with updated constraints if available)
        # Skip optimization if inherited trial succeeded
        optimization_trials = []
        if inherited_trial is None or not self._is_trial_successful(inherited_trial, target_volume_ml):
            optimization_trials = self._run_optimization_phase(target_volume_ml, screening_trials, volume_budget, constraint_update, inherited_trial)
        
        # Removed old overaspirate calibration phase - replaced by two-point calibration
        
        # Combine all trials for this volume (include inherited trial if it exists)
        inherited_trials = [inherited_trial] if inherited_trial else []
        all_volume_trials = screening_trials + two_point_trials + inherited_trials + optimization_trials
        self.all_trials.extend(all_volume_trials)
        
        # Analyze results - if inherited trial succeeded, use it as the only best trial
        if inherited_trial and self._is_trial_successful(inherited_trial, target_volume_ml):
            logger.info("Using successful inherited trial as optimal result")
            best_trials = [inherited_trial]
            optimal_parameters = inherited_trial.parameters
        else:
            # Standard analysis for failed inherited trial or when no inherited trial
            best_trials = self.analyzer.find_best_trials(all_volume_trials, max_results=5)
            optimal_parameters = best_trials[0].parameters if best_trials else None
        
        volume_statistics = self.analyzer.calculate_trial_statistics(all_volume_trials)
        
        # Create volume result
        duration = time.time() - start_time
        measurement_count = sum(len(trial.measurements) for trial in all_volume_trials)
        
        volume_result = VolumeCalibrationResult(
            target_volume_ml=target_volume_ml,
            trials=all_volume_trials,
            best_trials=best_trials,
            optimal_parameters=optimal_parameters,
            statistics=volume_statistics,
            duration_s=duration,
            measurement_count=measurement_count
        )
        
        # Update state for next volume
        self.current_volume_index += 1
        
        logger.info(f"Volume {target_volume_ml*1000:.1f}uL calibration completed: "
                   f"{measurement_count} measurements, {len(best_trials)} good trials")
        
        return volume_result
        
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
    
    def _run_screening_phase(self, target_volume_ml: float) -> List[TrialResult]:
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
            trial_result = self._execute_trial(parameters, target_volume_ml, f"screening_{trial_idx}", 
                                              strategy="screening", liquid="water")
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
                    logger.warning(f"LLM screening failed, falling back to SOBOL: {e}")
        
        # Use SOBOL sampling via temporary Ax client (following calibration_sdl_simplified pattern)
        if not hasattr(self, '_screening_optimizer'):
            from .bayesian_recommender import create_optimizer, OptimizerType
            # Create temporary optimizer for SOBOL screening
            screening_trials = self.config.get_screening_trials()
            self._screening_optimizer = create_optimizer(
                self.config, 
                target_volume_ml, 
                optimizer_type=OptimizerType.MULTI_OBJECTIVE,
                fixed_params=None,
                volume_dependent_only=False,
                num_sobol_trials=screening_trials  # 5 SOBOL trials for screening
            )
            logger.info(f"Created SOBOL optimizer for screening phase ({screening_trials} trials)")
        
        # Get SOBOL parameter suggestion
        parameters = self._screening_optimizer.suggest_parameters()
        logger.info(f"Using SOBOL-generated parameters for screening trial {trial_idx}")
        
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
    

    
    def _run_first_volume_optimization(self, target_volume_ml: float, optimizer) -> List[TrialResult]:
        """
        Optimize ALL parameters for the first volume.
        This establishes the baseline parameter set for transfer learning.
        """
        logger.info("First volume: optimizing ALL parameters")
        
        max_measurements = self.config.get_max_measurements_first_volume()
        
        # Use generic optimization loop with first-volume parameter generation
        return self._run_optimization_loop(
            target_volume_ml=target_volume_ml,
            max_measurements=max_measurements,
            strategy_name="First volume optimization",
            param_generation_func=self._generate_optimization_parameters
        )
    
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
        
        max_measurements = self.config.get_max_measurements_per_volume()
        
        # Create parameter generator for volume-dependent optimization
        param_generator = self._create_volume_dependent_param_generator(
            base_parameters, volume_dependent_params
        )
        
        # Use generic optimization loop with volume-dependent parameter generation
        return self._run_optimization_loop(
            target_volume_ml=target_volume_ml,
            max_measurements=max_measurements,
            strategy_name="Volume-dependent optimization", 
            param_generation_func=param_generator
        )
    
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
        
        # Get noise factors from config instead of hardcoded values
        noise_config = self.config.get_parameter_exploration_config()
        initial_noise = noise_config['initial_noise_factor']
        decay_rate = noise_config['noise_decay_rate']
        min_noise = noise_config['minimum_noise_factor']
        
        # Reduce noise over iterations (start broad, get more focused)
        noise_factor = max(min_noise, initial_noise - iteration * decay_rate)
        param_range = max_val - min_val
        noise = np.random.normal(0, param_range * noise_factor)
        
        # Apply noise and clamp to bounds
        new_value = base_value + noise
        return max(min_val, min(max_val, new_value))
    
    def _run_optimization_phase(
        self, 
        target_volume_ml: float, 
        screening_trials: List[TrialResult], 
        volume_budget: int,
        constraint_update: Optional[ConstraintBoundsUpdate] = None,
        inherited_trial: Optional[TrialResult] = None
    ) -> List[TrialResult]:
        """
        Run Bayesian optimization phase using proper Ax integration.
        
        Replaces the old noise-based parameter generation with actual Bayesian optimization
        following the proven patterns from calibration_sdl_simplified.py.
        """
        logger.info("Starting Bayesian optimization phase")
        
        # Determine optimization strategy
        is_first_volume = self.current_volume_index == 0
        
        # Set up fixed parameters for transfer learning
        fixed_params = {}
        if not is_first_volume and self.config.is_transfer_learning_enabled():
            # For subsequent volumes, fix non-volume-dependent parameters
            volume_dependent_params = ["blowout_vol", "overaspirate_vol"]
            previous_best = self.volume_results[-1].optimal_parameters
            
            if previous_best:
                logger.info("Using transfer learning - fixing non-volume-dependent parameters")
                all_params = previous_best.to_protocol_dict()
                fixed_params = {k: v for k, v in all_params.items() 
                              if k not in volume_dependent_params}
        
        # Create optimizer
        optimizer_type = OptimizerType.MULTI_OBJECTIVE if is_first_volume else OptimizerType.SINGLE_OBJECTIVE
        try:
            optimizer = create_optimizer(
                config=self.config,
                target_volume_ml=target_volume_ml,
                optimizer_type=optimizer_type,
                fixed_params=fixed_params,
                volume_dependent_only=not is_first_volume,
                constraint_updates=[constraint_update] if constraint_update else None,
                num_sobol_trials=0  # 0 SOBOL trials - go straight to Bayesian optimization
            )
            
            logger.info(f"Created {optimizer_type.value} optimizer for volume {target_volume_ml*1000:.0f}uL")
        except RuntimeError as e:
            logger.error(f"Cannot run optimization without Ax: {e}")
            logger.error("Please install Ax with: pip install ax-platform")
            logger.error("For now, running simple screening-only approach...")
            
            # Instead of fallback optimization, just return screening results
            # This maintains the quality of the system without degrading to noise
            logger.info("Using screening trials as optimization results")
            return screening_trials
        
        # Load screening trials as historical data for Bayesian optimization
        if screening_trials:
            logger.info(f"Loading {len(screening_trials)} screening trials into optimizer")
            for trial in screening_trials:
                objectives = OptimizationObjectives.from_adaptive_result(trial.analysis)
                optimizer.seed_with_historical_data(trial.parameters, objectives)
        
        # Load inherited trial if it failed (so we can learn from it)
        if inherited_trial:
            logger.info("Loading inherited trial into optimizer for learning")
            objectives = OptimizationObjectives.from_adaptive_result(inherited_trial.analysis)
            optimizer.seed_with_historical_data(inherited_trial.parameters, objectives)
        
        # Load two-point calibration trials for subsequent volumes only
        # (First volume does multi-objective optimization including precision, 
        #  so single-replicate two-point trials would mess up the precision model)
        if not is_first_volume and hasattr(self, '_current_two_point_trials') and self._current_two_point_trials:
            logger.info(f"Loading {len(self._current_two_point_trials)} two-point calibration trials into optimizer")
            logger.info("Note: Only using for subsequent volumes (single-objective accuracy optimization)")
            for trial in self._current_two_point_trials:
                objectives = OptimizationObjectives.from_adaptive_result(trial.analysis)
                optimizer.seed_with_historical_data(trial.parameters, objectives)
                logger.debug(f"Added two-point trial: overaspirate={trial.parameters.calibration.overaspirate_vol*1000:.1f}uL, "
                           f"measured={trial.analysis.mean_volume_ml*1000:.1f}uL, deviation={trial.analysis.deviation_pct:.1f}%")
        
        # Run optimization iterations
        optimization_trials = []
        current_volume_measurements = sum(len(trial.measurements) for trial in screening_trials)
        if inherited_trial:
            current_volume_measurements += len(inherited_trial.measurements)
        
        iteration = 0
        while not optimizer.is_converged() and current_volume_measurements < volume_budget:
            iteration += 1
            logger.info(f"Optimization iteration {iteration}")
            
            # Check volume budget
            remaining_budget = volume_budget - current_volume_measurements
            if remaining_budget < 3:  # Need at least 3 measurements for meaningful trial
                logger.info(f"Insufficient volume budget remaining ({remaining_budget}) - stopping optimization")
                break
            
            # Get parameter suggestion from optimizer
            try:
                suggested_params = optimizer.suggest_parameters()
                logger.info(f"Generated parameter suggestion from Ax optimizer")
            except Exception as e:
                logger.error(f"Failed to get suggestion from optimizer: {e}")
                break
            
            # Run adaptive measurement
            try:
                trial = self._execute_trial(
                    parameters=suggested_params,
                    target_volume_ml=target_volume_ml,
                    trial_id=f"BAYESIAN_OPT_{iteration}",
                    strategy="optimization",
                    liquid="water"
                )
                
                optimization_trials.append(trial)
                current_volume_measurements += len(trial.measurements)
                
                # Update optimizer with result
                objectives = OptimizationObjectives.from_adaptive_result(trial.analysis)
                optimizer.update_with_result(trial.parameters, objectives)
                
                # Log progress
                summary = optimizer.get_summary()
                logger.info(f"Trial {iteration}: deviation={objectives.accuracy:.1f}%, "
                           f"good_trials={summary.get('good_trials', 0)}")
                
            except Exception as e:
                logger.error(f"Failed to run optimization trial {iteration}: {e}")
                break
        
        # Log final optimization state
        final_summary = optimizer.get_summary()
        
        logger.info("="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Status: {final_summary.get('status', 'unknown')}")
        logger.info(f"Convergence reason: {final_summary.get('convergence_reason', 'N/A')}")
        logger.info(f"Total trials: {final_summary.get('total_trials', 0)}")
        logger.info(f"Good trials (within tolerance): {final_summary.get('good_trials', 0)}")
        
        best_accuracy = final_summary.get('best_accuracy')
        best_params = final_summary.get('best_parameters')
        
        if best_accuracy is not None:
            logger.info(f"Best accuracy achieved: {best_accuracy:.1f}% deviation")
            
        if best_params:
            # Log best calibration parameters
            if hasattr(best_params, 'calibration') and hasattr(best_params.calibration, 'overaspirate_vol'):
                overasp = best_params.calibration.overaspirate_vol * 1000
                logger.info(f"Best overaspirate volume: {overasp:.1f}uL")
                
            # Log best hardware parameters
            if hasattr(best_params, 'hardware') and hasattr(best_params.hardware, 'parameters'):
                logger.info("Best hardware parameters:")
                hw_params = best_params.hardware.parameters
                for param_name, value in hw_params.items():
                    logger.info(f"  {param_name}: {value:.1f}")
        
        logger.info("="*60)
        
        # Show top 5 trials like calibration_sdl_simplified
        try:
            optimizer.state.show_top_trials(optimizer.config, num_trials=5)
        except Exception as e:
            logger.warning(f"Could not display top trials: {e}")
        
        logger.info(f"Optimization completed: {final_summary}")
        
        return optimization_trials

    def _get_calibration_baseline_parameters(self, target_volume_ml: float) -> Optional[PipettingParameters]:
        """Extract best parameters for two-point calibration baseline.
        
        Args:
            target_volume_ml: Target volume being calibrated
            
        Returns:
            PipettingParameters for baseline trial or None if not available
        """
        if self.current_volume_index == 0:
            # First volume: get best screening candidate
            if not hasattr(self, '_current_screening_trials') or not self._current_screening_trials:
                logger.error("No screening trials available for first volume two-point calibration")
                return None
                
            # Find best screening trial by SDL score
            best_screening_trials = self.analyzer.find_best_trials(self._current_screening_trials, max_results=1)
            if not best_screening_trials:
                # Fallback: if all screening trials have penalty precision (100%), 
                # rank by accuracy instead (deviation from target)
                logger.warning("No valid screening trials found (all have precision penalties)")
                logger.info("Falling back to best accuracy trial for two-point calibration")
                
                penalty_trials = [trial for trial in self._current_screening_trials if trial.objectives.precision >= 99.9]
                if penalty_trials:
                    # Sort by accuracy (lower deviation is better)
                    penalty_trials.sort(key=lambda t: t.objectives.accuracy)
                    best_trial = penalty_trials[0]
                    logger.info(f"Selected most accurate trial: {best_trial.objectives.accuracy:.1f}% deviation (vs 100% precision penalty)")
                else:
                    logger.error("No screening trials available at all for two-point calibration")
                    return None
            else:
                best_trial = best_screening_trials[0]
            
            logger.info(f"Using best screening candidate for {target_volume_ml*1000:.1f}uL two-point calibration")
            return best_trial.parameters
        else:
            # Subsequent volumes: use previous volume's optimal parameters
            if not self.volume_results:
                logger.error("No previous volume results available for subsequent volume calibration")
                return None
                
            first_volume_result = self.volume_results[0]
            if not first_volume_result.optimal_parameters:
                logger.error("No optimal parameters from first volume")
                return None
                
            logger.info(f"Using optimized parameters from {first_volume_result.target_volume_ml*1000:.1f}uL volume for {target_volume_ml*1000:.1f}uL calibration")
            return first_volume_result.optimal_parameters

    def _execute_two_point_measurement(self, 
                                     optimized_params: 'PipettingParameters',
                                     target_volume_ml: float) -> Tuple[List['TrialResult'], Optional['ConstraintBoundsUpdate']]:
        """Execute two-point calibration measurements and calculate constraint bounds.
        
        Args:
            optimized_params: Base parameters for Point 1
            target_volume_ml: Target volume for calibration
            
        Returns:
            Tuple of (calibration_trials, constraint_update)
        """
        import numpy as np
        
        calibration_trials = []
        
        # Initialize constraint calibrator using config-driven tolerance
        volume_tolerances = self.config.calculate_tolerances_for_volume(target_volume_ml)
        tolerance_pct = volume_tolerances.precision_tolerance_pct
        
        # Import ConstraintCalibrator locally to avoid circular imports
        try:
            from .constraint_calibration import ConstraintCalibrator
            calibrator = ConstraintCalibrator(tolerance_pct=tolerance_pct)
        except ImportError as e:
            logger.error(f"Could not import ConstraintCalibrator: {e}")
            return calibration_trials, None
        
        # Point 1: Test with base overaspirate from optimized parameters
        logger.info(f"Point 1: Testing with base overaspirate {optimized_params.overaspirate_vol*1000:.1f}uL")
        point_1_trial = self._execute_trial(
            optimized_params,
            target_volume_ml,
            f"two_point_cal_point1_{target_volume_ml}",
            force_replicates=1,  # Single measurement for two-point calibration
            strategy="calibration",
            liquid=self.config.get_liquid_name()
        )
        calibration_trials.append(point_1_trial)
        
        # Analyze Point 1 results to determine optimal Point 2 direction
        point_1_measured_ml = np.mean([m.measured_volume_ml for m in point_1_trial.measurements])
        shortfall_ml = target_volume_ml - point_1_measured_ml  # Positive = under-delivery
        
        # Calculate tolerance-based spread
        tolerance_buffer_ul = volume_tolerances.accuracy_tolerance_ul
        shortfall_ul = abs(shortfall_ml * 1000)
        spread_ul = max(shortfall_ul + tolerance_buffer_ul, 2.0)  # Minimum 2uL spread
        
        logger.info(f"Shortfall analysis: {shortfall_ml*1000:.1f}uL shortfall + {tolerance_buffer_ul:.1f}uL tolerance = {spread_ul:.1f}uL spread")
        
        # Adaptive direction: move toward target volume
        if shortfall_ml > 0:
            # Under-delivering: increase overaspirate
            point_2_overaspirate_ml = optimized_params.overaspirate_vol + (spread_ul / 1000)
            direction = "increased"
            direction_sign = "+"
        else:
            # Over-delivering: decrease overaspirate
            point_2_overaspirate_ml = optimized_params.overaspirate_vol - (spread_ul / 1000)
            direction = "decreased" 
            direction_sign = "-"
        
        # Allow negative overaspirate but set reasonable lower bound (-10uL)
        point_2_overaspirate_ml = max(-0.010, point_2_overaspirate_ml)
        
        # Execute Point 2
        from .data_structures import PipettingParameters, CalibrationParameters
        point_2_params = PipettingParameters(
            calibration=CalibrationParameters(overaspirate_vol=point_2_overaspirate_ml),
            hardware=optimized_params.hardware
        )
        
        logger.info(f"Point 2: Testing with {direction} overaspirate {point_2_overaspirate_ml*1000:.1f}uL ({direction_sign}{spread_ul:.1f}uL)")
        point_2_trial = self._execute_trial(
            point_2_params,
            target_volume_ml,
            f"two_point_cal_point2_{target_volume_ml}",
            force_replicates=1,
            strategy="calibration", 
            liquid=self.config.get_liquid_name()
        )
        calibration_trials.append(point_2_trial)
        
        # Process results and calculate bounds
        point_2_measured_ml = np.mean([m.measured_volume_ml for m in point_2_trial.measurements])
        
        # Extract variability for bounds calculation
        best_trial_variability_ml = None
        if self.current_volume_index == 0 and hasattr(self, '_current_screening_trials'):
            best_screening_trials = self.analyzer.find_best_trials(self._current_screening_trials, max_results=1)
            if best_screening_trials:
                best_trial_variability_ml = best_screening_trials[0].analysis.stdev_volume_ml
        elif self.volume_results and self.volume_results[0].best_trials:
            best_trial_variability_ml = self.volume_results[0].best_trials[0].analysis.stdev_volume_ml
        
        # Calculate constraint bounds
        existing_screening_trials = getattr(self, '_current_screening_trials', []) if self.current_volume_index == 0 else []
        calibration_result = calibrator.calculate_two_point_bounds(
            optimized_params=optimized_params,
            target_volume_ml=target_volume_ml,
            point_1_overaspirate_ml=optimized_params.overaspirate_vol,
            point_1_measured_ml=point_1_measured_ml,
            point_1_variability_pct=point_1_trial.analysis.cv_volume_pct,
            point_1_measurement_count=len(point_1_trial.measurements),
            point_2_overaspirate_ml=point_2_overaspirate_ml,
            point_2_measured_ml=point_2_measured_ml,
            point_2_variability_pct=point_2_trial.analysis.cv_volume_pct,
            point_2_measurement_count=len(point_2_trial.measurements),
            best_trial_variability_ml=best_trial_variability_ml,
            existing_trials=existing_screening_trials
        )
        
        # Convert to constraint update
        constraint_update = calibrator.create_constraint_update(calibration_result)
        
        return calibration_trials, constraint_update
    
    
    def _run_two_point_constraint_calibration(
        self, 
        target_volume_ml: float, 
        volume_budget: int
    ) -> Tuple[Optional[ConstraintBoundsUpdate], List[TrialResult]]:
        """
        Run two-point calibration to calculate precise overaspirate constraint bounds.
        
        This replaces the old overaspirate calibration with a more sophisticated
        approach that tests two distinct overvolume points to calculate delivery
        efficiency and set data-driven bounds for the next optimization.
        
        Args:
            target_volume_ml: Target volume for next optimization
            volume_budget: Available measurement budget
            
        Returns:
            Tuple of (constraint_update, calibration_trials)
            constraint_update is None if calibration fails
        """
        logger.info("Starting two-point constraint calibration")
        
        calibration_trials = []
        measurements_needed = 2  # 1 per point for two-point calibration
        
        # Check if we have enough budget
        if self.total_measurements + measurements_needed > self.config.get_max_total_measurements():
            logger.warning(f"Insufficient budget for two-point calibration ({measurements_needed} measurements needed)")
            return None, calibration_trials
        
        # Get best parameters for calibration baseline
        baseline_parameters = self._get_calibration_baseline_parameters(target_volume_ml)
        if baseline_parameters is None:
            return None, calibration_trials
        
        # Extract variability information for bounds adjustment
        optimized_params = baseline_parameters
        best_trial_variability_ml = None
        
        if self.current_volume_index == 0:
            # First volume: extract variability from best screening trial
            best_screening_trials = self.analyzer.find_best_trials(self._current_screening_trials, max_results=1)
            if best_screening_trials:
                best_trial = best_screening_trials[0]
                best_trial_variability_ml = best_trial.analysis.stdev_volume_ml
                logger.info(f"Best screening trial variability: {best_trial_variability_ml*1000:.2f}uL")
        else:
            # Subsequent volumes: extract variability from first volume results
            if self.volume_results and self.volume_results[0].best_trials:
                best_trial = self.volume_results[0].best_trials[0]
                best_trial_variability_ml = best_trial.analysis.stdev_volume_ml
                logger.info(f"Best trial variability from first volume: {best_trial_variability_ml*1000:.2f}uL")

        try:
            # Execute two-point calibration measurements and calculate bounds
            calibration_trials, constraint_update = self._execute_two_point_measurement(optimized_params, target_volume_ml)
            
            if constraint_update:
                logger.info(f"Two-point calibration successful:")
                logger.info(f"  New bounds: [{constraint_update.min_value*1000:.1f}, {constraint_update.max_value*1000:.1f}] uL")
                logger.info(f"  Justification: {constraint_update.justification}")
            
            return constraint_update, calibration_trials
            
        except Exception as e:
            logger.error(f"Two-point calibration failed: {e}")
            logger.error(traceback.format_exc())
            return None, calibration_trials
    
    def _run_inherited_trial(self, target_volume_ml: float, constraint_update: 'ConstraintBoundsUpdate') -> Optional['TrialResult']:
        """
        Run inherited trial using optimal overaspirate calculated from two-point calibration.
        
        Args:
            target_volume_ml: Target volume for measurement
            constraint_update: Contains optimal overaspirate value from linear interpolation
            
        Returns:
            TrialResult if successful, None if constraint_update missing
        """
        if not constraint_update or not hasattr(constraint_update, 'optimal_overaspirate_ml'):
            logger.warning("No constraint update available for inherited trial")
            return None
            
        logger.info("Running inherited trial with optimal overaspirate from two-point calibration")
        
        # Get optimal overaspirate value and create parameters
        optimal_overaspirate_ml = constraint_update.optimal_overaspirate_ml
        logger.info(f"Using optimal overaspirate: {optimal_overaspirate_ml*1000:.2f}uL")
        
        inherited_params = self._create_inherited_parameters(target_volume_ml, optimal_overaspirate_ml)
        
        # Execute the trial
        logger.info(f"Executing inherited trial: overaspirate={optimal_overaspirate_ml*1000:.2f}uL")
        trial_result = self._execute_trial(
            inherited_params,
            target_volume_ml,
            trial_id="inherited_optimal",
            force_replicates=3,  # Use 3 replicates for good statistics
            strategy="inherited",
            liquid=self.config.get_liquid_name()
        )
        
        # Log results with success assessment
        self._log_inherited_trial_result(trial_result, target_volume_ml)
        
        return trial_result
        
    def _is_trial_successful(self, trial: TrialResult, target_volume_ml: float) -> bool:
        """
        Check if a trial meets success criteria (within tolerance).
        
        Args:
            trial: Trial result to evaluate
            target_volume_ml: Target volume for tolerance calculation
            
        Returns:
            True if trial meets tolerance criteria
        """
        if not trial or not trial.analysis:
            return False
            
        # Get tolerance for this volume
        tolerances = self.config.calculate_tolerances_for_volume(target_volume_ml)
        # Convert uL tolerance to percentage
        target_volume_ul = target_volume_ml * 1000
        tolerance_pct = (tolerances.accuracy_tolerance_ul / target_volume_ul) * 100
        
        # Add debugging (can be removed later)
        logger.info(f"[DEBUG] Calculated tolerance: {tolerance_pct:.1f}% for {target_volume_ul:.0f}uL")
        
        # Check if BOTH deviation AND variability are within tolerance
        deviation_within_tolerance = abs(trial.analysis.deviation_pct) <= tolerance_pct
        variability_within_tolerance = trial.analysis.cv_volume_pct <= tolerances.precision_tolerance_pct
        
        success = deviation_within_tolerance and variability_within_tolerance
        
        logger.info(f"Trial success evaluation:")
        logger.info(f"  Deviation: {trial.analysis.deviation_pct:.1f}% (tolerance: +/-{tolerance_pct:.1f}%)")
        logger.info(f"  Variability: {trial.analysis.cv_volume_pct:.1f}% CV (tolerance: <={tolerances.precision_tolerance_pct:.1f}%)")
        logger.info(f"  Deviation check: {deviation_within_tolerance} ({abs(trial.analysis.deviation_pct):.1f} <= {tolerance_pct:.1f})")
        logger.info(f"  Variability check: {variability_within_tolerance} ({trial.analysis.cv_volume_pct:.1f} <= {tolerances.precision_tolerance_pct:.1f})")
        logger.info(f"  Result: {'PASS' if success else 'FAIL'}")
        
        return success
    
    def _execute_trial(self, parameters: PipettingParameters, 
                      target_volume_ml: float,
                      trial_id: str,
                      force_replicates: Optional[int] = None,
                      strategy: str = "optimization",
                      liquid: str = "water") -> TrialResult:
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
            measurement = self._execute_protocol_measurement(parameters, target_volume_ml)
            measurements.append(measurement)
            self.total_measurements += 1
        
        # Analyze and determine if more replicates are needed
        while use_adaptive and len(measurements) < force_replicates if force_replicates else 10:
            # Analyze current measurements
            trial_result = self.analyzer.analyze_trial(measurements, target_volume_ml, strategy, liquid)
            
            if not trial_result.needs_additional_replicates:
                break
            
            # Check budget
            if self.total_measurements >= self.config.get_max_total_measurements():
                logger.warning("Budget exhausted, stopping replicates")
                break
            
            # Add another replicate
            replicate_idx = len(measurements)
            measurement = self._execute_protocol_measurement(parameters, target_volume_ml)
            measurements.append(measurement)
            self.total_measurements += 1
            
            logger.debug(f"Added replicate {replicate_idx + 1} for trial {trial_id}")
        
        # Final analysis
        trial_result = self.analyzer.analyze_trial(measurements, target_volume_ml, strategy, liquid)
        
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
        
        # Calculate total measurements from actual data
        total_measurements = sum(vol.measurement_count for vol in self.volume_results)
        
        return ExperimentResults(
            experiment_name=self.config.get_experiment_name(),
            volume_results=self.volume_results,
            optimal_conditions=optimal_conditions,
            total_measurements=total_measurements,
            total_duration_s=time.time() - experiment_start_time,
            overall_statistics=overall_statistics,
            config_used=self.config.get_raw_config()
        )
    
    def _export_results(self, results: ExperimentResults):
        """Export results to files."""
        if not self.config.should_export_optimal_conditions():
            return
        
        # Export optimal conditions as structured JSON
        if results.optimal_conditions:
            optimal_json_path = self.output_dir / "optimal_conditions.json"
            import json
            with open(optimal_json_path, 'w') as f:
                json.dump(asdict(results.optimal_conditions), f, indent=2)
            logger.info(f"Exported optimal conditions JSON to {optimal_json_path}")
        
        # Export raw measurements as structured JSON if requested
        if self.config.should_save_raw_measurements():
            all_measurements = []
            for trial in self.all_trials:
                for measurement in trial.measurements:
                    measurement_dict = asdict(measurement)
                    measurement_dict['trial_score'] = trial.composite_score
                    measurement_dict['trial_quality'] = trial.quality.overall_quality
                    all_measurements.append(measurement_dict)
            
            if all_measurements:
                measurements_json_path = self.output_dir / "raw_measurements.json"
                import json
                with open(measurements_json_path, 'w') as f:
                    json.dump(all_measurements, f, indent=2)
                logger.info(f"Exported raw measurements JSON to {measurements_json_path}")
        
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
                    'strategy': trial.strategy,
                    'liquid': trial.liquid,
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
                    measurement_dict['strategy'] = trial.strategy
                    measurement_dict['liquid'] = trial.liquid
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
            
            clean_csvs = [f for f in self.output_dir.glob("*.csv")]
            if clean_csvs:
                logger.info(f"[SUCCESS] Generated {len(clean_csvs)} CSV files")
            
            if insights:
                logger.info("[SUCCESS] Generated analysis insights and recommendations")
                
        except Exception as e:
            logger.error(f"Error generating enhanced outputs: {e}")
            logger.debug(f"Enhanced outputs error details: {traceback.format_exc()}")