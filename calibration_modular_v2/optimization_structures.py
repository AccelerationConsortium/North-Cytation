"""
Optimization Data Structures for Modular Calibration System
=========================================================

Clean data structures for Bayesian optimization integration that mirror
the proven patterns from calibration_sdl_simplified.py.

These structures bridge between the modular experiment framework and 
the Ax optimization system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
from enum import Enum

from data_structures import PipettingParameters, RawMeasurement, AdaptiveMeasurementResult


class OptimizerType(Enum):
    """Types of optimizers available."""
    SOBOL = "sobol"           # Random exploration
    SINGLE_OBJECTIVE = "qLogEI"  # Single objective (accuracy only)  
    MULTI_OBJECTIVE = "qNEHVI"   # Multi-objective (accuracy + precision + time)


@dataclass
class OptimizationObjectives:
    """Objectives for optimization."""
    accuracy: float    # Deviation percentage (minimize)
    precision: float   # Variability percentage (minimize) 
    time: float        # Duration seconds (minimize)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for Ax interface."""
        return {
            "deviation": self.accuracy,
            "variability": self.precision, 
            "time": self.time
        }
    
    @classmethod
    def from_adaptive_result(cls, result: AdaptiveMeasurementResult) -> 'OptimizationObjectives':
        """Create objectives from adaptive measurement result."""
        return cls(
            accuracy=result.absolute_deviation_pct,
            precision=result.cv_volume_pct,
            time=result.mean_duration_s
        )


@dataclass
class OptimizationTrial:
    """Single trial result for optimization feedback."""
    parameters: PipettingParameters
    objectives: OptimizationObjectives
    trial_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_ax_result(self) -> Tuple[int, Dict[str, float]]:
        """Convert to format expected by Ax optimizer."""
        if self.trial_index is None:
            raise ValueError("trial_index required for Ax feedback")
        return self.trial_index, self.objectives.to_dict()


@dataclass
class OptimizationConstraints:
    """Parameter constraints for optimization."""
    # Volume-dependent constraints
    target_volume_ml: float
    tip_volume_ml: float = 1.0
    
    # Overaspirate bounds (dynamically calculated)
    min_overaspirate_ml: float = 0.0
    max_overaspirate_ml: float = 0.01  # 10μL default
    
    # Fixed parameters (not optimized)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    
    # Parameters to optimize (if None, optimize all non-fixed)
    optimize_parameters: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate constraints."""
        if self.target_volume_ml <= 0:
            raise ValueError("target_volume_ml must be positive")
        if self.tip_volume_ml <= 0:
            raise ValueError("tip_volume_ml must be positive") 
        if self.min_overaspirate_ml > self.max_overaspirate_ml:
            raise ValueError("min_overaspirate_ml cannot exceed max_overaspirate_ml")
    
    def get_volume_constraint_ml(self) -> float:
        """Calculate available volume for post_asp_air_vol + overaspirate_vol."""
        return self.tip_volume_ml - self.target_volume_ml
    
    def get_overaspirate_bounds_ul(self) -> Tuple[float, float]:
        """Get overaspirate bounds in microliters for display."""
        return (self.min_overaspirate_ml * 1000, self.max_overaspirate_ml * 1000)


@dataclass
class OptimizationConfig:
    """Configuration for optimization phase."""
    optimizer_type: OptimizerType
    constraints: OptimizationConstraints
    
    # Optimization parameters
    num_initial_trials: int = 5      # SOBOL exploration trials
    bayesian_batch_size: int = 1     # Parallel suggestions
    random_seed: int = 42
    
    # Stopping criteria
    max_trials: int = 20
    min_good_trials: int = 6
    
    # Objective thresholds (for "good" trial evaluation)
    accuracy_threshold_pct: float = 10.0
    precision_threshold_pct: float = 5.0
    time_threshold_s: float = 60.0
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_initial_trials < 0:
            raise ValueError("num_initial_trials must be non-negative")
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.min_good_trials <= 0:
            raise ValueError("min_good_trials must be positive")


@dataclass 
class OptimizationState:
    """State tracking for optimization process."""
    trials: List[OptimizationTrial] = field(default_factory=list)
    good_trials: List[OptimizationTrial] = field(default_factory=list)
    best_trial: Optional[OptimizationTrial] = None
    
    # Ax optimizer state
    ax_client: Optional[Any] = None  # Ax client instance
    trial_counter: int = 0
    
    # Stopping criteria tracking
    is_converged: bool = False
    convergence_reason: str = ""
    
    def add_trial(self, trial: OptimizationTrial, config: OptimizationConfig) -> None:
        """Add trial and update state."""
        self.trials.append(trial)
        
        # Check if trial is "good"
        if self._is_good_trial(trial, config):
            self.good_trials.append(trial)
        
        # Update best trial
        if (self.best_trial is None or 
            trial.objectives.accuracy < self.best_trial.objectives.accuracy):
            self.best_trial = trial
        
        # Check convergence
        self._check_convergence(config)
    
    def _is_good_trial(self, trial: OptimizationTrial, config: OptimizationConfig) -> bool:
        """Check if trial meets quality thresholds."""
        return (trial.objectives.accuracy <= config.accuracy_threshold_pct and
                trial.objectives.precision <= config.precision_threshold_pct and
                trial.objectives.time <= config.time_threshold_s)
    
    def _check_convergence(self, config: OptimizationConfig) -> None:
        """Check if optimization should stop."""
        if len(self.good_trials) >= config.min_good_trials:
            self.is_converged = True
            self.convergence_reason = f"Found {len(self.good_trials)} good trials"
        elif len(self.trials) >= config.max_trials:
            self.is_converged = True  
            self.convergence_reason = f"Reached maximum {config.max_trials} trials"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.trials:
            return {"status": "no_trials"}
        
        return {
            "status": "converged" if self.is_converged else "running",
            "convergence_reason": self.convergence_reason,
            "total_trials": len(self.trials),
            "good_trials": len(self.good_trials),
            "best_accuracy": self.best_trial.objectives.accuracy if self.best_trial else None,
            "best_parameters": self.best_trial.parameters if self.best_trial else None
        }


@dataclass
class VolumeOptimizationResult:
    """Results for optimizing one specific volume."""
    target_volume_ml: float
    config: OptimizationConfig
    state: OptimizationState
    
    # Summary statistics
    total_measurements: int
    duration_s: float
    success: bool
    
    # Best results
    optimal_parameters: Optional[PipettingParameters] = None
    optimal_objectives: Optional[OptimizationObjectives] = None
    
    def __post_init__(self):
        """Extract optimal results from state."""
        if self.state.best_trial:
            self.optimal_parameters = self.state.best_trial.parameters
            self.optimal_objectives = self.state.best_trial.objectives
            self.success = True
        else:
            self.success = False
    
    def get_transfer_learning_params(self, volume_dependent_params: List[str]) -> Dict[str, float]:
        """Get parameters for transfer learning to next volume."""
        if not self.optimal_parameters:
            return {}
        
        # Return all parameters except volume-dependent ones
        all_params = self.optimal_parameters.to_protocol_dict()
        return {k: v for k, v in all_params.items() 
                if k not in volume_dependent_params}


if __name__ == "__main__":
    # Test the data structures
    from data_structures import CalibrationParameters, HardwareParameters, PipettingParameters
    
    # Create test parameters
    cal_params = CalibrationParameters(overaspirate_vol=0.005)
    hw_params = HardwareParameters(parameters={
        "aspirate_speed": 20.0,
        "dispense_speed": 15.0
    })
    pipetting_params = PipettingParameters(calibration=cal_params, hardware=hw_params)
    
    # Create test objectives
    objectives = OptimizationObjectives(accuracy=5.2, precision=3.1, time=45.0)
    
    # Create test trial
    trial = OptimizationTrial(parameters=pipetting_params, objectives=objectives, trial_index=1)
    
    # Create constraints
    constraints = OptimizationConstraints(
        target_volume_ml=0.05,
        max_overaspirate_ml=0.008,
        fixed_parameters={"retract_speed": 8.0}
    )
    
    # Create config
    config = OptimizationConfig(
        optimizer_type=OptimizerType.MULTI_OBJECTIVE,
        constraints=constraints
    )
    
    # Create state and add trial
    state = OptimizationState()
    state.add_trial(trial, config)
    
    print("Test structures created successfully:")
    print(f"  Trial objectives: {trial.objectives.to_dict()}")
    print(f"  Constraints: {constraints.get_overaspirate_bounds_ul()}")
    print(f"  State summary: {state.get_summary()}")