#!/usr/bin/env python3
"""
Ax-Based Bayesian Recommender for Modular Calibration System

Implements the proven Bayesian optimization pattern from calibration_sdl_simplified
with clean data structures and proper Ax integration.

Key Features:
- Multi-objective optimization (accuracy, precision, time)
- Dynamic parameter constraints based on volume
- Parameter inheritance for transfer learning
- Volume-dependent parameter re-optimization
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

# Handle both relative and absolute imports
try:
    from .data_structures import (
        PipettingParameters, TrialResult, ConstraintBoundsUpdate,
        CalibrationParameters, HardwareParameters
    )
    from .optimization_structures import (
        OptimizationConstraints, OptimizationConfig, OptimizerType,
        OptimizationObjectives, OptimizationTrial, OptimizationState
    )
    from .config_manager import ExperimentConfig
except ImportError:
    from data_structures import (
        PipettingParameters, TrialResult, ConstraintBoundsUpdate,
        CalibrationParameters, HardwareParameters
    )
    from optimization_structures import (
        OptimizationConstraints, OptimizationConfig, OptimizerType,
        OptimizationObjectives, OptimizationTrial, OptimizationState
    )
    from config_manager import ExperimentConfig

logger = logging.getLogger(__name__)
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import Ax components (using Ax 0.4.0 API - same as existing working code)
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties  
    from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
    from ax.modelbridge.factory import Models
    AX_AVAILABLE = True
except ImportError as e:
    logger.error(f"Ax not available: {e}")
    AxClient, ObjectiveProperties = None, None
    GenerationStep, GenerationStrategy, Models = None, None, None
    AX_AVAILABLE = False


class AxBayesianOptimizer:
    """
    Ax-based Bayesian optimizer following calibration_sdl_simplified patterns.
    
    Handles both multi-objective (first volume) and single-objective (subsequent volumes)
    optimization with proper parameter constraints and transfer learning.
    """
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer with configuration."""
        self.config = config
        self.state = OptimizationState()
        
        if not AX_AVAILABLE:
            raise RuntimeError("Ax not available - cannot initialize Bayesian optimizer")
        
        # Create Ax client
        self._create_ax_client()
        
        logger.info(f"Initialized {config.optimizer_type.value} optimizer")
        logger.info(f"Optimizing: {self._get_optimize_params()}")
        logger.info(f"Fixed: {config.constraints.fixed_parameters}")
    
    def _get_optimize_params(self) -> List[str]:
        """Get list of parameters to optimize."""
        constraints = self.config.constraints
        
        # Default to all calibration + hardware parameters
        all_params = ["overaspirate_vol"]  # Always include calibration parameters
        
        # Add hardware parameters from config (hardware-agnostic)
        if not self.config.experiment_config:
            raise ValueError(f"OptimizationConfig missing experiment_config - cannot access hardware parameters. Config type: {type(self.config)}")
        
        logger.debug(f"Using experiment_config: {type(self.config.experiment_config)}")
        hardware_params = self.config.experiment_config.get_hardware_parameter_names()
        logger.debug(f"Hardware parameters: {hardware_params}")
        all_params.extend(hardware_params)
        
        # Filter out fixed parameters
        if constraints.optimize_parameters:
            # Use explicitly specified parameters
            return [p for p in constraints.optimize_parameters 
                   if p not in constraints.fixed_parameters]
        else:
            # Use all non-fixed parameters
            return [p for p in all_params 
                   if p not in constraints.fixed_parameters]
    
    def _get_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter bounds for Ax (following calibration_sdl_simplified)."""
        constraints = self.config.constraints
        optimize_params = self._get_optimize_params()
        
        # Get hardware parameter bounds from config (hardware-agnostic)
        if not self.config.experiment_config:
            raise ValueError("OptimizationConfig missing experiment_config - cannot access hardware bounds")
        
        hardware_bounds = self.config.experiment_config.get_hardware_parameter_bounds()
        
        # Build default bounds dict using config
        default_bounds = {}
        
        # Add hardware parameters from config
        if not self.config.experiment_config:
            raise ValueError("OptimizationConfig missing experiment_config - cannot access hardware parameters")
            
        hardware_param_names = self.config.experiment_config.get_hardware_parameter_names()
        
        for param_name in hardware_param_names:
            bounds_tuple = hardware_bounds.get(param_name)
            if bounds_tuple:
                default_bounds[param_name] = {
                    "type": "range", 
                    "bounds": list(bounds_tuple)
                }
        
        # Add calibration parameters
        default_bounds["overaspirate_vol"] = {
            "type": "range", 
            "bounds": [constraints.min_overaspirate_ml, constraints.max_overaspirate_ml]
        }
        
        # Build parameters list for Ax
        parameters = []
        for param_name in optimize_params:
            if param_name in default_bounds:
                param_config = default_bounds[param_name].copy()
                param_config["name"] = param_name
                parameters.append(param_config)
        
        return parameters
    
    def _get_parameter_constraints(self) -> List[str]:
        """Get parameter constraints for Ax."""
        constraints = self.config.constraints
        optimize_params = self._get_optimize_params()
        constraint_list = []
        
        # Volume constraint: post_asp_air_vol + overaspirate_vol <= tip_volume - target_volume
        if ("post_asp_air_vol" in optimize_params and 
            "overaspirate_vol" in optimize_params):
            available_volume = constraints.get_volume_constraint_ml()
            constraint_str = f"post_asp_air_vol + overaspirate_vol <= {available_volume}"
            constraint_list.append(constraint_str)
            
            logger.info(f"Added volume constraint: {constraint_str}")
            logger.info(f"  Available volume: {available_volume*1000:.1f}uL")
        
        return constraint_list
    
    def _create_ax_client(self) -> None:
        """Create Ax client with custom generation strategy (Ax 0.4.0 API pattern)."""
        # Create custom generation strategy based on SOBOL trial count
        num_sobol_trials = self.config.num_initial_trials
        
        if num_sobol_trials > 0:
            # Strategy with SOBOL followed by Bayesian (like existing working code)
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=num_sobol_trials,
                        min_trials_observed=num_sobol_trials,  # Wait for all SOBOL trials
                    ),
                    GenerationStep(
                        model=Models.GPEI,  # Gaussian Process Expected Improvement
                        num_trials=-1,  # Continue indefinitely
                    ),
                ]
            )
        else:
            # Pure Bayesian optimization (skip SOBOL)
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.GPEI,  # Gaussian Process Expected Improvement
                        num_trials=-1,  # Continue indefinitely
                    ),
                ]
            )
        
        # Create Ax client with custom strategy
        ax_client = AxClient(
            generation_strategy=generation_strategy,
            verbose_logging=False
        )
        
        # Create experiment
        if self.config.optimizer_type == OptimizerType.MULTI_OBJECTIVE:
            objectives = {
                "deviation": ObjectiveProperties(minimize=True, threshold=50.0),
                "variability": ObjectiveProperties(minimize=True, threshold=10.0), 
                "time": ObjectiveProperties(minimize=True, threshold=90.0),
            }
        else:  # Single objective (accuracy only)
            objectives = {
                "deviation": ObjectiveProperties(minimize=True, threshold=50.0),
            }
        
        ax_client.create_experiment(
            parameters=self._get_parameter_bounds(),
            objectives=objectives,
            parameter_constraints=self._get_parameter_constraints(),
        )
        
        # Store references
        self.state.ax_client = ax_client
        logger.info(f"Created Ax client with {num_sobol_trials} SOBOL trials")
        if num_sobol_trials == 0:
            logger.info("  Mode: Pure Bayesian optimization (no exploration)")
        else:
            logger.info(f"  Mode: {num_sobol_trials} SOBOL trials -> Bayesian optimization")
    
    def suggest_parameters(self) -> PipettingParameters:
        """Get next parameter suggestion from Ax."""
        if not self.state.ax_client:
            raise RuntimeError("Ax client not initialized")
        
        # Get suggestion from Ax
        params, trial_index = self.state.ax_client.get_next_trial()
        
        # Check which generation method is being used (Ax 0.4.0 API)
        method = "Unknown"
        gs = self.state.ax_client.generation_strategy
        
        # Get current generation step in Ax 0.4.0
        if hasattr(gs, '_curr_step_idx') and hasattr(gs, 'steps'):
            current_step_idx = gs._curr_step_idx
            if current_step_idx < len(gs.steps):
                current_step = gs.steps[current_step_idx]
                if hasattr(current_step, 'model') and current_step.model:
                    model_name = str(current_step.model)
                    if 'SOBOL' in model_name:
                        method = "SOBOL"
                    elif 'GPEI' in model_name:
                        method = "Bayesian (GPEI)"
                    elif 'GP' in model_name:
                        method = "Bayesian (GP)"
                    else:
                        method = f"Model({model_name})"
                else:
                    # Check by step index - first step is usually SOBOL
                    if current_step_idx == 0 and self.config.num_initial_trials > 0:
                        method = "SOBOL"
                    else:
                        method = "Bayesian"
        
        logger.info(f"Generated suggestion (trial {trial_index}) using {method}")
        
        # Apply fixed parameters
        for param_name, fixed_value in self.config.constraints.fixed_parameters.items():
            if param_name in params:
                params[param_name] = fixed_value
        
        # Create PipettingParameters from Ax suggestion
        pipetting_params = self._ax_params_to_pipetting_parameters(params)
        
        # Store trial index for feedback
        self.state.trial_counter = trial_index
        
        return pipetting_params
    
    def update_with_result(self, parameters: PipettingParameters, 
                          objectives: OptimizationObjectives) -> None:
        """Update optimizer with trial result."""
        if not self.state.ax_client:
            raise RuntimeError("Ax client not initialized")
        
        # Create trial
        trial = OptimizationTrial(
            parameters=parameters,
            objectives=objectives, 
            trial_index=self.state.trial_counter
        )
        
        # Update Ax client
        trial_index, ax_objectives = trial.to_ax_result(self.config.optimizer_type)
        self.state.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_objectives
        )
        
        # Update state
        self.state.add_trial(trial, self.config)
        
        logger.info(f"Updated with result: deviation={objectives.accuracy:.1f}%, trial={trial_index}")
    
    def seed_with_historical_data(self, parameters: PipettingParameters, 
                                 objectives: OptimizationObjectives) -> None:
        """Add historical trial data to the optimizer for training AND ranking."""
        if not self.state.ax_client:
            raise RuntimeError("Ax client not initialized")
        
        # Convert parameters to Ax format
        ax_params = self._pipetting_parameters_to_ax_params(parameters)
        
        # Filter to only include parameters that are in the current search space
        # (important for transfer learning where some parameters are fixed)
        search_space_params = set(self.state.ax_client.experiment.search_space.parameters.keys())
        filtered_ax_params = {k: v for k, v in ax_params.items() if k in search_space_params}
        
        # Check if parameter values are within current search space bounds
        # (important when constraint updates have created tighter bounds than original screening)
        search_space = self.state.ax_client.experiment.search_space
        for param_name, param_value in filtered_ax_params.items():
            if param_name in search_space.parameters:
                param_def = search_space.parameters[param_name]
                # Check bounds for range parameters
                if hasattr(param_def, 'lower') and hasattr(param_def, 'upper'):
                    if not (param_def.lower <= param_value <= param_def.upper):
                        logger.info(f"Skipping historical trial: {param_name}={param_value:.6f} outside bounds [{param_def.lower:.6f}, {param_def.upper:.6f}]")
                        return  # Skip this trial entirely if any parameter is out of bounds
        
        # Convert objectives to Ax format  
        _, ax_objectives = OptimizationTrial(
            parameters=parameters,
            objectives=objectives,
            trial_index=-1  # Dummy index for conversion
        ).to_ax_result(self.config.optimizer_type)
        
        # Attach historical trial to Ax
        _, trial_index = self.state.ax_client.attach_trial(
            parameters=filtered_ax_params
        )
        
        # Complete the trial with results
        self.state.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_objectives
        )
        
        # IMPORTANT: Also add to state for ranking (not just Ax training)
        # Create an optimization trial for the historical data
        historical_trial = OptimizationTrial(
            parameters=parameters,
            objectives=objectives,
            trial_index=len(self.state.trials)  # Assign proper index
        )
        
        # Add to state for ranking consideration
        self.state.add_trial(historical_trial, self.config)
        
        logger.info(f"Seeded optimizer with historical data: deviation={objectives.accuracy:.1f}% (included in ranking)")
    
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        return self.state.is_converged
    
    def get_best_parameters(self) -> Optional[PipettingParameters]:
        """Get best parameters found so far."""
        if not self.state.trials:
            return None
        
        # Find best trial dynamically (simple accuracy-based)
        best_trial = min(self.state.trials, key=lambda t: t.objectives.accuracy)
        return best_trial.parameters
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return self.state.get_summary()
    
    def _ax_params_to_pipetting_parameters(self, ax_params: Dict[str, float]) -> PipettingParameters:
        """Convert Ax parameters to PipettingParameters."""
        # Split into calibration and hardware parameters
        cal_params = CalibrationParameters(
            overaspirate_vol=ax_params.get("overaspirate_vol", 0.005)
        )
        
        # Hardware parameters (everything except overaspirate_vol)
        hw_dict = {k: v for k, v in ax_params.items() if k != "overaspirate_vol"}
        
        # Add any fixed hardware parameters
        for param_name, fixed_value in self.config.constraints.fixed_parameters.items():
            if param_name != "overaspirate_vol":  # Don't override calibration params
                hw_dict[param_name] = fixed_value
        
        hw_params = HardwareParameters(parameters=hw_dict)
        
        return PipettingParameters(calibration=cal_params, hardware=hw_params)
    
    def _pipetting_parameters_to_ax_params(self, params: PipettingParameters) -> Dict[str, float]:
        """Convert PipettingParameters to Ax parameters dictionary with proper type handling."""
        ax_params = {}
        
        # Add calibration parameters
        ax_params["overaspirate_vol"] = params.calibration.overaspirate_vol
        
        # Add hardware parameters with type conversions for Ax requirements
        for name, value in params.hardware.parameters.items():
            # Convert to appropriate type based on parameter requirements
            if name in ['aspirate_speed', 'dispense_speed']:
                # These are integer parameters in Ax
                ax_params[name] = int(round(value))
            else:
                # Float parameters
                ax_params[name] = float(value)
        
        return ax_params


# Factory function for creating optimizers
def create_optimizer(config: ExperimentConfig, target_volume_ml: float,
                    optimizer_type: OptimizerType = OptimizerType.MULTI_OBJECTIVE,
                    fixed_params: Optional[Dict[str, float]] = None,
                    volume_dependent_only: bool = False,
                    constraint_updates: Optional[List['ConstraintBoundsUpdate']] = None,
                    num_sobol_trials: Optional[int] = None) -> AxBayesianOptimizer:
    """
    Create Bayesian optimizer with proper constraints.
    
    Args:
        config: Experiment configuration
        target_volume_ml: Target volume for optimization
        optimizer_type: Type of optimizer to create
        fixed_params: Parameters to keep fixed
        volume_dependent_only: If True, only optimize volume-dependent parameters
        constraint_updates: Optional constraint bounds updates from two-point calibration
        num_sobol_trials: Number of SOBOL trials (5 for screening, 0 for subsequent volumes)
    """
    # Get default overaspirate bounds from config (not calculated)
    cal_bounds = config.get_calibration_parameter_bounds()
    default_min_overaspirate_ml = cal_bounds.overaspirate_vol[0] 
    default_max_overaspirate_ml = cal_bounds.overaspirate_vol[1]
    
    # Apply constraint updates if available
    min_overaspirate_ml = default_min_overaspirate_ml
    max_overaspirate_ml = default_max_overaspirate_ml
    
    if constraint_updates:
        for update in constraint_updates:
            if update.parameter_name == "overaspirate_vol":
                min_overaspirate_ml = update.min_value
                max_overaspirate_ml = update.max_value
                logger.info(f"Applied constraint update for {update.parameter_name}: "
                           f"[{min_overaspirate_ml*1000:.1f}, {max_overaspirate_ml*1000:.1f}] uL")
                logger.info(f"Justification: {update.justification}")
                break
    else:
        # Log the default bounds from config
        logger.info(f"Using config default overaspirate bounds: "
                   f"[{min_overaspirate_ml*1000:.1f}, {max_overaspirate_ml*1000:.1f}] uL")
    
    # Volume-dependent parameters (matching calibration_sdl_simplified)
    volume_dependent_params = ["blowout_vol", "overaspirate_vol"]
    
    # Create constraints
    constraints = OptimizationConstraints(
        target_volume_ml=target_volume_ml,
        min_overaspirate_ml=min_overaspirate_ml,
        max_overaspirate_ml=max_overaspirate_ml,
        fixed_parameters=fixed_params or {},
        optimize_parameters=volume_dependent_params if volume_dependent_only else None
    )
    
    # Determine number of SOBOL trials
    if num_sobol_trials is not None:
        # Explicitly specified (5 for screening, 0 for subsequent volumes)
        sobol_count = num_sobol_trials
    else:
        # Default fallback (shouldn't happen in normal operation)
        sobol_count = 5 if not volume_dependent_only else 0
    
    # Create config
    optimization_config = OptimizationConfig(
        optimizer_type=optimizer_type,
        constraints=constraints,
        random_seed=config.get_random_seed() or 42,
        num_initial_trials=sobol_count,
        target_volume_ml=target_volume_ml,
        experiment_config=config
    )
    
    return AxBayesianOptimizer(optimization_config)


if __name__ == "__main__":
    # Test the optimizer
    from config_manager import ExperimentConfig
    
    # Mock config for testing
    class MockConfig:
        def get_random_seed(self):
            return 42
    
    config = MockConfig()
    
    # Test multi-objective optimizer
    print("Testing multi-objective optimizer...")
    optimizer = create_optimizer(
        config, 
        target_volume_ml=0.05,
        optimizer_type=OptimizerType.MULTI_OBJECTIVE
    )
    
    print(f"Created optimizer: {optimizer.get_summary()}")
    
    # Test parameter suggestion (would fail without Ax, but shows structure)
    try:
        params = optimizer.suggest_parameters() 
        print(f"Generated parameters: {params}")
    except Exception as e:
        print(f"Parameter suggestion failed (expected without Ax): {e}")