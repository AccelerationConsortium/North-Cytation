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
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from config_manager import ExperimentConfig
from data_structures import PipettingParameters, CalibrationParameters, HardwareParameters
from optimization_structures import (
    OptimizationObjectives, OptimizationTrial, OptimizationConstraints,
    OptimizationConfig, OptimizationState, OptimizerType
)

logger = logging.getLogger(__name__)

# Import Ax components (following calibration_sdl_simplified pattern)
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties  
    from ax.modelbridge.factory import Models
    from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    AX_AVAILABLE = True
except ImportError as e:
    logger.error(f"Ax not available: {e}")
    AxClient, ObjectiveProperties, Models = None, None, None
    GenerationStep, GenerationStrategy = None, None
    qNoisyExpectedHypervolumeImprovement, qLogNoisyExpectedImprovement = None, None
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
        # In real implementation, this would come from config_manager
        default_hw_params = [
            "aspirate_speed", "dispense_speed", "aspirate_wait_time", 
            "dispense_wait_time", "retract_speed", "blowout_vol", "post_asp_air_vol"
        ]
        all_params.extend(default_hw_params)
        
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
        
        # Default bounds (matching calibration_sdl_simplified)
        default_bounds = {
            "aspirate_speed": {"type": "range", "bounds": [10, 35]},
            "dispense_speed": {"type": "range", "bounds": [10, 35]}, 
            "aspirate_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
            "dispense_wait_time": {"type": "range", "bounds": [0.0, 30.0]},
            "retract_speed": {"type": "range", "bounds": [1.0, 15.0]},
            "blowout_vol": {"type": "range", "bounds": [0.0, 0.2]},
            "post_asp_air_vol": {"type": "range", "bounds": [0.0, 0.1]},
            "overaspirate_vol": {"type": "range", "bounds": [
                constraints.min_overaspirate_ml, 
                constraints.max_overaspirate_ml
            ]},
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
            logger.info(f"  Available volume: {available_volume*1000:.1f}μL")
        
        return constraint_list
    
    def _create_ax_client(self) -> None:
        """Create Ax client with proper configuration."""
        # Set up acquisition function based on optimizer type
        if self.config.optimizer_type == OptimizerType.MULTI_OBJECTIVE:
            model_gen_kwargs = {
                "botorch_acqf_class": qNoisyExpectedHypervolumeImprovement,
                "deduplicate": True,
            }
        else:  # Single objective
            model_gen_kwargs = {
                "botorch_acqf_class": qLogNoisyExpectedImprovement,
                "deduplicate": True,
            }
        
        # Create generation strategy
        steps = []
        if self.config.num_initial_trials > 0:
            steps.append(GenerationStep(
                model=Models.SOBOL,
                num_trials=self.config.num_initial_trials,
                min_trials_observed=self.config.num_initial_trials,
                max_parallelism=self.config.num_initial_trials,
                model_kwargs={"seed": self.config.random_seed},
                model_gen_kwargs=model_gen_kwargs,
            ))
        
        steps.append(GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            max_parallelism=self.config.bayesian_batch_size,
            model_gen_kwargs=model_gen_kwargs,
        ))
        
        gs = GenerationStrategy(steps=steps)
        
        # Create Ax client
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
        
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
        logger.info("Created Ax client successfully")
    
    def suggest_parameters(self) -> PipettingParameters:
        """Get next parameter suggestion from Ax."""
        if not self.state.ax_client:
            raise RuntimeError("Ax client not initialized")
        
        # Get suggestion from Ax
        params, trial_index = self.state.ax_client.get_next_trial()
        
        # Apply fixed parameters
        for param_name, fixed_value in self.config.constraints.fixed_parameters.items():
            if param_name in params:
                params[param_name] = fixed_value
        
        # Create PipettingParameters from Ax suggestion
        pipetting_params = self._ax_params_to_pipetting_parameters(params)
        
        # Store trial index for feedback
        self.state.trial_counter = trial_index
        
        logger.info(f"Generated suggestion (trial {trial_index})")
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
        trial_index, ax_objectives = trial.to_ax_result()
        self.state.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_objectives
        )
        
        # Update state
        self.state.add_trial(trial, self.config)
        
        logger.info(f"Updated with result: deviation={objectives.accuracy:.1f}%, trial={trial_index}")
    
    def is_converged(self) -> bool:
        """Check if optimization has converged."""
        return self.state.is_converged
    
    def get_best_parameters(self) -> Optional[PipettingParameters]:
        """Get best parameters found so far."""
        return self.state.best_trial.parameters if self.state.best_trial else None
    
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


# Factory function for creating optimizers
def create_optimizer(config: ExperimentConfig, target_volume_ml: float,
                    optimizer_type: OptimizerType = OptimizerType.MULTI_OBJECTIVE,
                    fixed_params: Optional[Dict[str, float]] = None,
                    volume_dependent_only: bool = False) -> AxBayesianOptimizer:
    """
    Create Bayesian optimizer with proper constraints.
    
    Args:
        config: Experiment configuration
        target_volume_ml: Target volume for optimization
        optimizer_type: Type of optimizer to create
        fixed_params: Parameters to keep fixed
        volume_dependent_only: If True, only optimize volume-dependent parameters
    """
    # Calculate overaspirate bounds (following calibration_sdl_simplified logic)
    max_overaspirate_fraction = 0.2  # 20% of target volume
    max_overaspirate_ml = min(0.01, target_volume_ml * max_overaspirate_fraction)
    
    # Volume-dependent parameters (matching calibration_sdl_simplified)
    volume_dependent_params = ["blowout_vol", "overaspirate_vol"]
    
    # Create constraints
    constraints = OptimizationConstraints(
        target_volume_ml=target_volume_ml,
        max_overaspirate_ml=max_overaspirate_ml,
        fixed_parameters=fixed_params or {},
        optimize_parameters=volume_dependent_params if volume_dependent_only else None
    )
    
    # Create config
    optimization_config = OptimizationConfig(
        optimizer_type=optimizer_type,
        constraints=constraints,
        random_seed=config.get_random_seed() or 42,
        num_initial_trials=5 if not volume_dependent_only else 3
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