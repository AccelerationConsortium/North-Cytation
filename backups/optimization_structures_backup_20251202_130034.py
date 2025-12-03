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
    
    def to_dict(self, optimizer_type: 'OptimizerType' = None) -> Dict[str, float]:
        """Convert to dictionary for Ax interface."""
        result = {
            "deviation": self.accuracy,
            "variability": self.precision, 
            "time": self.time
        }
        
        # Filter objectives based on optimizer type
        if optimizer_type == OptimizerType.SINGLE_OBJECTIVE:
            # Single objective only uses deviation (accuracy)
            return {"deviation": self.accuracy}
        else:
            # Multi-objective or None (default) returns all objectives
            return result
    
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
    strategy: str = "optimization"  # "screening" or "optimization" 
    liquid: str = "water"  # Liquid being pipetted
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_ax_result(self, optimizer_type: 'OptimizerType' = None) -> Tuple[int, Dict[str, float]]:
        """Convert to format expected by Ax optimizer."""
        if self.trial_index is None:
            raise ValueError("trial_index required for Ax feedback")
        return self.trial_index, self.objectives.to_dict(optimizer_type)


@dataclass
class OptimizationConstraints:
    """Parameter constraints for optimization."""
    # Volume-dependent constraints
    target_volume_ml: float
    tip_volume_ml: float = 1.0
    
    # Overaspirate bounds (dynamically calculated)
    min_overaspirate_ml: float = 0.0
    max_overaspirate_ml: float = 0.01  # 10uL default
    
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
    
    # Experiment context
    liquid: str = "water"            # Liquid being pipetted
    experiment_name: str = "calibration"  # Experiment identifier
    protocol: Any = None             # Protocol instance for constraints
    
    # Optimization parameters
    num_initial_trials: int = 5      # SOBOL exploration trials
    bayesian_batch_size: int = 1     # Parallel suggestions
    random_seed: int = 42
    
    # Stopping criteria
    max_trials: int = 20
    min_good_trials: int = 3  # Default changed to match common config usage
    
    # Objective thresholds (for "good" trial evaluation)
    accuracy_threshold_pct: float = 10.0
    precision_threshold_pct: float = 5.0
    time_threshold_s: float = 60.0
    
    # Volume-specific tolerance support
    target_volume_ml: Optional[float] = None
    experiment_config: Optional[Any] = None  # Will import properly to avoid circular imports
    
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
        
        # Simple trial tracking - no real-time best trial updates
        print(f"[TRIAL {len(self.trials)}] Accuracy: {trial.objectives.accuracy:.1f}% dev, Precision: {trial.objectives.precision:.1f}% CV, Time: {trial.objectives.time:.1f}s")
        
        # Check convergence
        self._check_convergence(config)
    
    def _is_good_trial(self, trial: OptimizationTrial, config: OptimizationConfig) -> bool:
        """Check if trial meets volume-specific tolerance thresholds."""
        # Use volume-specific tolerance instead of hardcoded threshold
        target_volume_ml = getattr(config, 'target_volume_ml', 0.05)  # Default to 50uL if not set
        
        # Calculate volume-specific tolerance (same logic as experiment._is_trial_successful)
        tolerances = config.experiment_config.calculate_tolerances_for_volume(target_volume_ml) if hasattr(config, 'experiment_config') else None
        
        if tolerances:
            target_volume_ul = target_volume_ml * 1000
            accuracy_threshold_pct = (tolerances.accuracy_tolerance_ul / target_volume_ul) * 100
            precision_threshold_pct = tolerances.precision_tolerance_pct
        else:
            # Fallback to config thresholds if volume-specific calculation fails
            accuracy_threshold_pct = config.accuracy_threshold_pct
            precision_threshold_pct = config.precision_threshold_pct
        
        return (trial.objectives.accuracy <= accuracy_threshold_pct and
                trial.objectives.precision <= precision_threshold_pct and
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
        """Get optimization summary with dynamically calculated best trial."""
        if not self.trials:
            return {"status": "no_trials"}
        
        # Find best trial dynamically (simple accuracy-based for summary)
        best_trial = min(self.trials, key=lambda t: t.objectives.accuracy)
        
        return {
            "status": "converged" if self.is_converged else "running",
            "convergence_reason": self.convergence_reason,
            "total_trials": len(self.trials),
            "good_trials": len(self.good_trials),
            "best_accuracy": best_trial.objectives.accuracy,
            "best_parameters": best_trial.parameters
        }

    def _calculate_sdl_simplified_score(self, trial: OptimizationTrial, config: OptimizationConfig) -> float:
        """Calculate composite score using exact calibration_sdl_simplified math."""
        import statistics
        
        # Get weights (use config values or defaults)
        if hasattr(config, 'experiment_config') and config.experiment_config is not None:
            weights = config.experiment_config.get_objective_weights()
            accuracy_weight = weights.accuracy_weight
            precision_weight = weights.precision_weight  
            time_weight = weights.time_weight
        else:
            # Fallback to config defaults - should only happen in tests
            # In production, experiment_config should always be provided
            accuracy_weight = 0.5
            precision_weight = 0.4
            time_weight = 0.1
        
        # Collect trial data, excluding penalty precision trials (100%) for normalization
        valid_trials = [t for t in self.trials if t.objectives.precision < 99.9]
        
        if not valid_trials:
            # If all trials have penalty precision, fall back to using all trials
            valid_trials = self.trials
        
        raw_accuracies = [t.objectives.accuracy for t in valid_trials]
        raw_precisions = [t.objectives.precision for t in valid_trials]
        raw_times = [t.objectives.time for t in valid_trials]
        
        # Calculate standard deviations (exact calibration_sdl_simplified math)
        acc_std = max(statistics.stdev(raw_accuracies) if len(raw_accuracies) > 1 else 0.1, 0.1)
        prec_std = max(statistics.stdev(raw_precisions) if len(raw_precisions) > 1 else 0.1, 0.1)
        time_std = max(statistics.stdev(raw_times) if len(raw_times) > 1 else 1.0, 1.0)
        
        # Calculate normalized scores (exact calibration_sdl_simplified math)
        acc_score = trial.objectives.accuracy / acc_std * 100
        prec_score = trial.objectives.precision / prec_std * 100  
        time_score = trial.objectives.time / time_std * 100
        
        # Weighted composite score (exact calibration_sdl_simplified math)
        composite_score = accuracy_weight * acc_score + precision_weight * prec_score + time_weight * time_score
        
        return composite_score
    
    def show_top_trials(self, config: OptimizationConfig, num_trials: int = 5) -> None:
        """Show top N trials ranked by SDL score like calibration_sdl_simplified.
        
        This mimics the end-of-batch ranking from the original calibration_sdl_simplified.
        """
        if len(self.trials) == 0:
            print("No trials to display.")
            return
        
        print(f"\n{'='*60}")
        print("TOP TRIALS RANKED BY SDL SCORE (End of Optimization)")
        print(f"{'='*60}")
        
        # Calculate scores for all trials, excluding penalty precision trials
        trial_scores = []
        for i, trial in enumerate(self.trials):
            # Exclude trials with 100% precision (penalty values)
            if trial.objectives.precision >= 99.9:  # Allow for floating point precision
                continue  # Skip penalty trials from ranking
                
            score = self._calculate_sdl_simplified_score(trial, config)
            trial_scores.append((i + 1, trial, score))
        
        # Sort by score (lower is better)
        trial_scores.sort(key=lambda x: x[2])
        
        # Show top N trials
        for rank, (trial_num, trial, score) in enumerate(trial_scores[:num_trials], 1):
            print(f"\nRank {rank} - Trial {trial_num}:")
            print(f"  Accuracy: {trial.objectives.accuracy:.1f}% deviation")
            print(f"  Precision: {trial.objectives.precision:.1f}% CV")
            print(f"  Time: {trial.objectives.time:.1f} seconds")
            print(f"  SDL Score: {score:.3f}")
            
            # Show key parameters
            if hasattr(trial.parameters, 'calibration') and hasattr(trial.parameters.calibration, 'overaspirate_vol'):
                overasp = trial.parameters.calibration.overaspirate_vol * 1000
                print(f"  Overaspirate: {overasp:.1f}uL")
            
            if hasattr(trial.parameters, 'hardware') and hasattr(trial.parameters.hardware, 'parameters'):
                hw_params = trial.parameters.hardware.parameters
                key_params = ['aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time']
                param_summary = []
                for param in key_params:
                    if param in hw_params:
                        param_summary.append(f"{param}={hw_params[param]:.1f}")
                if param_summary:
                    print(f"  Hardware: {', '.join(param_summary)}")
        
        print(f"\n{'='*60}")


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
        if self.state.trials:
            # Find best trial dynamically (simple accuracy-based)
            best_trial = min(self.state.trials, key=lambda t: t.objectives.accuracy)
            self.optimal_parameters = best_trial.parameters
            self.optimal_objectives = best_trial.objectives
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