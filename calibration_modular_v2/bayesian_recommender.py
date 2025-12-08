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
                param_type = self.config.experiment_config.get_parameter_type(param_name)
                ax_param_type = "range" if param_type == "float" else "range"  # Ax uses "range" for both
                value_type = "int" if param_type == "integer" else "float"
                
                default_bounds[param_name] = {
                    "type": ax_param_type, 
                    "bounds": list(bounds_tuple),
                    "value_type": value_type
                }
        
        # Add calibration parameters
        overaspirate_type = self.config.experiment_config.get_parameter_type("overaspirate_vol")
        overaspirate_value_type = "int" if overaspirate_type == "integer" else "float"
        default_bounds["overaspirate_vol"] = {
            "type": "range", 
            "bounds": [constraints.min_overaspirate_ml, constraints.max_overaspirate_ml],
            "value_type": overaspirate_value_type
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
        """Get parameter constraints for Ax from protocol and configuration."""
        constraint_list = []
        
        # Get constraints from protocol (hardware-specific dynamic constraints)
        if hasattr(self.config, 'protocol') and self.config.protocol:
            if hasattr(self.config.protocol, 'get_parameter_constraints'):
                try:
                    protocol_constraints = self.config.protocol.get_parameter_constraints(
                        self.config.constraints.target_volume_ml
                    )
                    
                    # Process constraints to handle fixed parameters
                    processed_constraints = self._process_constraints_with_fixed_params(protocol_constraints)
                    constraint_list.extend(processed_constraints)
                    
                    for constraint in processed_constraints:
                        logger.info(f"Added protocol constraint: {constraint}")
                        
                except Exception as e:
                    logger.warning(f"Failed to get constraints from protocol: {e}")
        
        # Get static constraints from configuration (if any remain)
        try:
            constraints_config = self.config.experiment_config.get_parameter_constraints()
            
            for constraint_def in constraints_config:
                constraint_expr = constraint_def.get('constraint')
                description = constraint_def.get('description', '')
                
                if constraint_expr:
                    # Check if all parameters in constraint are being optimized
                    optimize_params = self._get_optimize_params()
                    
                    # Simple check - if constraint contains parameter names that are being optimized
                    constraint_applicable = False
                    for param in optimize_params:
                        if param in constraint_expr:
                            constraint_applicable = True
                            break
                    
                    if constraint_applicable:
                        constraint_list.append(constraint_expr)
                        logger.info(f"Added config constraint: {constraint_expr}")
                        if description:
                            logger.info(f"  Description: {description}")
        except AttributeError:
            # No config constraints defined, that's fine
            pass
        
        return constraint_list
    
    def _process_constraints_with_fixed_params(self, constraints: List[str]) -> List[str]:
        """Process constraints by substituting fixed parameter values."""
        processed_constraints = []
        fixed_params = self.config.constraints.fixed_parameters
        
        for constraint in constraints:
            processed_constraint = constraint
            
            # Substitute fixed parameter values in the constraint
            for param_name, fixed_value in fixed_params.items():
                if param_name in constraint:
                    # Replace parameter name with its fixed value
                    processed_constraint = processed_constraint.replace(param_name, str(fixed_value))
            
            # Simplify the constraint if possible (basic arithmetic)
            try:
                processed_constraint = self._simplify_constraint(processed_constraint)
            except Exception as e:
                logger.warning(f"Could not simplify constraint '{processed_constraint}': {e}")
            
            processed_constraints.append(processed_constraint)
            
            if constraint != processed_constraint:
                logger.info(f"Rewrote constraint: '{constraint}' -> '{processed_constraint}'")
        
        return processed_constraints
    
    def _simplify_constraint(self, constraint: str) -> str:
        """Simplify constraints with arithmetic (e.g., '0.075 + x <= 0.190' -> 'x <= 0.115')."""
        import re
        
        # Handle constraints of the form: "number + variable <= number"
        # Example: "0.075 + overaspirate_vol <= 0.190"
        pattern = r'(\d+\.?\d*)\s*\+\s*(\w+)\s*<=\s*(\d+\.?\d*)'
        match = re.match(pattern, constraint.strip())
        
        if match:
            fixed_value = float(match.group(1))
            variable = match.group(2)
            limit = float(match.group(3))
            new_limit = limit - fixed_value
            
            # Handle floating-point precision errors - treat very small numbers as zero
            if abs(new_limit) < 1e-10:
                new_limit = 0.0
            
            return f"{variable} <= {new_limit:.6f}"
        
        # If we can't simplify, return as-is
        return constraint
    
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
        
        # DEBUG: Log raw Ax parameters before any processing
        logger.info(f"RAW AX PARAMETERS: {params}")

        # Apply parameter rounding IMMEDIATELY to prevent floating-point precision issues
        params = self._apply_parameter_rounding(params)
        
        # DEBUG: Log parameters after rounding
        logger.info(f"AFTER ROUNDING: {params}")

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

        # Apply fixed parameters (now using clean rounded values)
        for param_name, fixed_value in self.config.constraints.fixed_parameters.items():
            if param_name in params:
                logger.info(f"FIXING PARAMETER: {param_name} = {fixed_value} (was {params[param_name]})")
                params[param_name] = fixed_value
                
        # DEBUG: Log final parameters before creating PipettingParameters
        logger.info(f"FINAL PARAMETERS: {params}")

        # Create PipettingParameters from Ax suggestion
        pipetting_params = self._ax_params_to_pipetting_parameters(params)

        # Store trial index for feedback
        self.state.trial_counter = trial_index

        return pipetting_params
    
    def update_with_result(self, parameters: PipettingParameters, 
                          objectives: OptimizationObjectives,
                          measurement_count: int = 1,
                          is_successful: bool = False) -> None:
        """Update optimizer with trial result."""
        if not self.state.ax_client:
            raise RuntimeError("Ax client not initialized")
        
        # Create trial
        trial = OptimizationTrial(
            parameters=parameters,
            objectives=objectives, 
            measurement_count=measurement_count,
            trial_index=self.state.trial_counter
        )
        
        # Update Ax client
        trial_index, ax_objectives = trial.to_ax_result(self.config.optimizer_type)
        self.state.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=ax_objectives
        )
        
        # Update state - use the success flag from experiment instead of _is_good_trial
        self.state.add_trial(trial, self.config, is_successful=is_successful)
        
        logger.info(f"Updated with result: deviation={objectives.accuracy:.1f}%, trial={trial_index}")
    
    def seed_with_historical_data(self, parameters: PipettingParameters, 
                                 objectives: OptimizationObjectives,
                                 measurement_count: int = 1,
                                 is_successful: bool = False) -> None:
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
            measurement_count=measurement_count,
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
            measurement_count=measurement_count,
            trial_index=len(self.state.trials)  # Assign proper index
        )
        
        # Add to state for ranking consideration
        self.state.add_trial(historical_trial, self.config, is_successful=is_successful)
        
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
        
        # Add hardware parameters with correct type conversion
        for name, value in params.hardware.parameters.items():
            param_type = self.config.experiment_config.get_parameter_type(name)
            if param_type == "integer":
                ax_params[name] = int(round(value))
            else:
                ax_params[name] = float(value)
        
        return ax_params
    
    def _apply_parameter_rounding(self, ax_params: Dict[str, float]) -> Dict[str, float]:
        """Apply configured rounding to prevent floating-point precision issues."""
        rounded_params = {}
        
        # Get all parameter configurations (calibration + hardware)
        all_params_config = self.config.experiment_config.get_optimization_parameters()
        
        for param_name, value in ax_params.items():
            param_config = all_params_config.get(param_name, {})
            round_to = param_config.get('round_to_nearest')
            
            if round_to is not None and round_to > 0:
                # Round to nearest specified precision
                rounded_value = round(value / round_to) * round_to
                rounded_params[param_name] = rounded_value
                
                # Log significant rounding (difference > 1% of rounding precision)
                if abs(value - rounded_value) > round_to * 0.01:
                    logger.debug(f"Rounded {param_name}: {value:.8f} -> {rounded_value:.8f}")
            else:
                # No rounding specified, use original value
                rounded_params[param_name] = value
        
        return rounded_params


# Factory function for creating optimizers
def create_optimizer(config: ExperimentConfig, target_volume_ml: float,
                    optimizer_type: OptimizerType = OptimizerType.MULTI_OBJECTIVE,
                    fixed_params: Optional[Dict[str, float]] = None,
                    volume_dependent_only: bool = False,
                    constraint_updates: Optional[List['ConstraintBoundsUpdate']] = None,
                    num_sobol_trials: Optional[int] = None,
                    protocol_instance=None,
                    min_good_trials: Optional[int] = None) -> AxBayesianOptimizer:
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
        protocol_instance: Protocol instance for getting hardware constraints
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
    
    # Get volume-dependent parameters from configuration
    volume_dependent_params = config.get_volume_dependent_parameters()
    
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
        experiment_config=config,
        protocol=protocol_instance,  # Pass protocol for constraints
        min_good_trials=min_good_trials if min_good_trials is not None else config.get_min_good_trials()  # Use volume-specific value
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