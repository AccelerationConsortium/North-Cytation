#!/usr/bin/env python3
"""
Bayesian Recommender for Modular Calibration System

Single-file Bayesian optimization supporting 1-3 objectives with dynamic
parameter bounds and fixed parameter handling matching the original system.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from config_manager import ExperimentConfig
from data_structures import PipettingParameters

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Single optimization trial result."""
    parameters: PipettingParameters
    objectives: Dict[str, float]  # accuracy, precision, time
    
class BayesianRecommender:
    """
    Bayesian optimization recommender supporting 1-3 objectives.
    
    Handles:
    - Dynamic parameter bounds (overaspirate_vol constraint)
    - Fixed parameters (excluded from optimization)
    - Multi-objective (qNEHVI) or single-objective (qLogEI) optimization
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 target_volume_ml: float,
                 fixed_params: Optional[Dict[str, float]] = None):
        self.config = config
        self.target_volume_ml = target_volume_ml
        self.fixed_params = fixed_params or {}
        
        # Get optimization parameters (excluding fixed ones)
        self.optimize_params = self._get_optimization_parameters()
        
        # Setup parameter bounds
        self.param_bounds = self._get_parameter_bounds()
        
        # Optimization history
        self.X = []  # Parameter history
        self.Y = []  # Objective history
        
        # Lazy load optimizers to avoid import issues
        self._optimizer = None
        
        logger.info(f"Bayesian recommender initialized for {len(self.optimize_params)} parameters")
        logger.info(f"Optimizing: {list(self.optimize_params)}")
        if self.fixed_params:
            logger.info(f"Fixed: {self.fixed_params}")
    
    def _get_optimization_parameters(self) -> List[str]:
        """Get parameters to optimize (exclude fixed parameters)."""
        all_params = list(self.config._config['parameters'].keys())
        return [param for param in all_params if param not in self.fixed_params]
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds, applying dynamic constraints."""
        bounds = {}
        
        for param_name in self.optimize_params:
            param_config = self.config._config['parameters'][param_name]
            base_bounds = param_config['bounds']
            
            # Apply dynamic constraints
            if param_name == 'overaspirate_vol':
                max_fraction = param_config.get('max_fraction_of_target', 0.2)
                dynamic_max = min(base_bounds[1], self.target_volume_ml * max_fraction)
                bounds[param_name] = (base_bounds[0], dynamic_max)
            else:
                bounds[param_name] = tuple(base_bounds)
                
        return bounds
    
    def _initialize_optimizer(self):
        """Lazy initialization of BoTorch optimizer."""
        try:
            import botorch
            from botorch.models import SingleTaskGP
            from botorch.fit import fit_gpytorch_mll
            from botorch.acquisition import qNoisyExpectedImprovement, qLogNoisyExpectedImprovement
            from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
            from botorch.optim import optimize_acqf
            from gpytorch.mlls import ExactMarginalLogLikelihood
            
            self._botorch = botorch
            self._fit_gpytorch_mll = fit_gpytorch_mll
            self._optimize_acqf = optimize_acqf
            self._SingleTaskGP = SingleTaskGP
            self._ExactMarginalLogLikelihood = ExactMarginalLogLikelihood
            
            # Choose acquisition function based on objectives
            optimizer_type = self.config.get_optimizer_type()
            if optimizer_type == "multi_objective":
                self._acquisition_func = qNoisyExpectedHypervolumeImprovement
            else:
                self._acquisition_func = qLogNoisyExpectedImprovement
                
        except ImportError as e:
            logger.error(f"BoTorch not available: {e}")
            self._optimizer = None
            
    def _parameters_to_tensor(self, parameters: PipettingParameters) -> torch.Tensor:
        """Convert parameters to tensor for optimization."""
        values = []
        for param_name in self.optimize_params:
            values.append(getattr(parameters, param_name))
        return torch.tensor(values, dtype=torch.float64)
    
    def _tensor_to_parameters(self, tensor: torch.Tensor) -> PipettingParameters:
        """Convert tensor back to parameters."""
        param_dict = {}
        
        # Add optimized parameters
        for i, param_name in enumerate(self.optimize_params):
            param_dict[param_name] = tensor[i].item()
        
        # Add fixed parameters
        param_dict.update(self.fixed_params)
        
        # Fill in any missing parameters with defaults
        for param_name, param_config in self.config._config['parameters'].items():
            if param_name not in param_dict:
                param_dict[param_name] = param_config['default']
                
        return PipettingParameters(**param_dict)
    
    def _objectives_to_tensor(self, objectives: Dict[str, float]) -> torch.Tensor:
        """Convert objectives dict to tensor."""
        # Always use all three objectives, even if some are weighted to 0
        values = [
            objectives.get('accuracy', 0.0),    # Minimize deviation
            objectives.get('precision', 0.0),   # Minimize variability  
            objectives.get('time', 0.0)         # Minimize time
        ]
        # Negate for minimization (BoTorch maximizes)
        return torch.tensor([-v for v in values], dtype=torch.float64)
    
    def update_with_results(self, results: List[OptimizationResult]) -> None:
        """Update optimizer with new experimental results."""
        for result in results:
            X_tensor = self._parameters_to_tensor(result.parameters)
            Y_tensor = self._objectives_to_tensor(result.objectives)
            
            self.X.append(X_tensor)
            self.Y.append(Y_tensor)
        
        logger.info(f"Updated optimizer with {len(results)} results. Total: {len(self.X)} trials")
    
    def suggest_parameters(self, n_suggestions: int = 1) -> List[PipettingParameters]:
        """Suggest next parameter sets to try."""
        
        if len(self.X) == 0:
            # No data yet - use random suggestions within bounds
            return self._random_suggestions(n_suggestions)
        
        if self._optimizer is None:
            self._initialize_optimizer()
            
        if self._optimizer is None:
            # Fallback to random if BoTorch not available
            logger.warning("BoTorch not available, using random suggestions")
            return self._random_suggestions(n_suggestions)
        
        return self._bayesian_suggestions(n_suggestions)
    
    def _random_suggestions(self, n_suggestions: int) -> List[PipettingParameters]:
        """Generate random parameter suggestions within bounds."""
        suggestions = []
        
        for _ in range(n_suggestions):
            param_values = {}
            
            for param_name in self.optimize_params:
                bounds = self.param_bounds[param_name]
                param_values[param_name] = np.random.uniform(bounds[0], bounds[1])
            
            # Add fixed parameters
            param_values.update(self.fixed_params)
            
            # Fill defaults
            for param_name, param_config in self.config._config['parameters'].items():
                if param_name not in param_values:
                    param_values[param_name] = param_config['default']
                    
            suggestions.append(PipettingParameters(**param_values))
            
        return suggestions
    
    def _bayesian_suggestions(self, n_suggestions: int) -> List[PipettingParameters]:
        """Generate Bayesian optimization suggestions."""
        try:
            # Prepare training data
            X_train = torch.stack(self.X)
            Y_train = torch.stack(self.Y)
            
            # Fit GP model
            model = self._SingleTaskGP(X_train, Y_train)
            mll = self._ExactMarginalLogLikelihood(model.likelihood, model)
            self._fit_gpytorch_mll(mll)
            
            # Setup bounds for optimization
            bounds = torch.tensor([[self.param_bounds[param][0] for param in self.optimize_params],
                                 [self.param_bounds[param][1] for param in self.optimize_params]], 
                                dtype=torch.float64)
            
            # Optimize acquisition function
            acquisition_func = self._acquisition_func(model, ref_point=torch.zeros(Y_train.shape[-1]))
            
            candidates, _ = self._optimize_acqf(
                acquisition_func,
                bounds=bounds,
                q=n_suggestions,
                num_restarts=10,
                raw_samples=100,
            )
            
            # Convert to parameter objects
            suggestions = []
            for i in range(n_suggestions):
                suggestions.append(self._tensor_to_parameters(candidates[i]))
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}, falling back to random")
            return self._random_suggestions(n_suggestions)

if __name__ == "__main__":
    # Test the recommender
    from config_manager import ExperimentConfig
    
    config = ExperimentConfig.from_yaml("experiment_config.yaml")
    
    # Test with no fixed parameters
    recommender = BayesianRecommender(config, target_volume_ml=0.05)
    suggestions = recommender.suggest_parameters(3)
    
    print(f"Generated {len(suggestions)} suggestions:")
    for i, params in enumerate(suggestions):
        print(f"  {i+1}: {params}")
    
    # Test with fixed parameters - use config-driven parameter names
    hw_param_names = config.get_hardware_parameter_names()
    if len(hw_param_names) >= 2:
        # Use first two hardware parameters as an example
        fixed_params = {hw_param_names[0]: 15.0, hw_param_names[1]: 15.0}
        recommender_fixed = BayesianRecommender(config, target_volume_ml=0.05, fixed_params=fixed_params)
        suggestions_fixed = recommender_fixed.suggest_parameters(2)
        
        print(f"\nWith fixed params {fixed_params}:")
        for i, params in enumerate(suggestions_fixed):
            print(f"  {i+1}: {params}")