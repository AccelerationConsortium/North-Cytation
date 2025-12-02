"""
Type-Safe Data Structures for Calibration System
================================================

This module defines all data structures used throughout the calibration system.
All structures use dataclasses for type safety, validation, and clear interfaces.

Key Features:
- Comprehensive validation in __post_init__ methods
- Unit-explicit field naming (no ambiguous units)
- Immutable parameters (frozen=True for PipettingParameters)
- Optional fields with sensible defaults
- Rich metadata support for extensibility

Data Flow Hierarchy:
    PipettingParameters -> RawMeasurement -> AdaptiveMeasurementResult -> 
    TrialResult -> VolumeCalibrationResult -> ExperimentResults
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


@dataclass(frozen=True)
class CalibrationParameters:
    """
    Mandatory calibration parameters that all hardware must support.
    
    These are essential for volume accuracy and are hardware-agnostic.
    """
    overaspirate_vol: float  # mL - mandatory for volume precision
    
    def __post_init__(self):
        """Validate mandatory parameters."""
        # Note: Physical limits for overaspirate_vol should be validated by hardware-specific config
        # Removed hardcoded limits (-0.02 to 1.0 mL) to support different hardware systems
        pass


@dataclass(frozen=True)
class HardwareParameters:
    """
    Hardware-specific parameters with flexible field names.
    
    Field names and validation rules are determined by hardware configuration.
    All parameters are optional with hardware-specific defaults.
    """
    parameters: Dict[str, float]
    
    def __post_init__(self):
        """Validate parameters using hardware-specific rules."""
        # Validation delegated to hardware protocol or config manager
        pass
    
    def get(self, name: str, default: float = None) -> float:
        """Get parameter value with optional default."""
        return self.parameters.get(name, default)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for protocol interface."""
        return self.parameters.copy()


@dataclass(frozen=True) 
class PipettingParameters:
    """
    Complete parameter set combining mandatory calibration and optional hardware parameters.
    
    Clean hardware-agnostic architecture:
    - calibration: Mandatory parameters (overaspirate_vol) - all hardware must support
    - hardware: Optional parameters (config-driven names) - hardware-specific
    """
    calibration: CalibrationParameters
    hardware: HardwareParameters
    
    # Convenient access methods
    @property
    def overaspirate_vol(self) -> float:
        """Get the mandatory overaspirate volume parameter."""
        return self.calibration.overaspirate_vol
    
    def get_hardware_param(self, name: str, default: float = None) -> float:
        """Get hardware parameter by name with optional default."""
        return self.hardware.get(name, default)
    
    def to_protocol_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for protocol interface."""
        result = self.hardware.to_dict()
        result['overaspirate_vol'] = self.calibration.overaspirate_vol
        return result


@dataclass
class VolumeTolerances:
    """Volume-dependent tolerance thresholds."""
    accuracy_tolerance_ul: float
    precision_tolerance_pct: float 
    
    def __post_init__(self):
        """Validate tolerance values."""
        if self.accuracy_tolerance_ul <= 0:
            raise ValueError(f"accuracy_tolerance_ul must be positive, got {self.accuracy_tolerance_ul}")
        if self.precision_tolerance_pct <= 0:
            raise ValueError(f"precision_tolerance_pct must be positive, got {self.precision_tolerance_pct}")


@dataclass  
class RawMeasurement:
    """Single raw measurement from protocol execution."""
    
    # Core measurement data (required fields first)
    measurement_id: str
    parameters: PipettingParameters
    target_volume_ml: float
    measured_volume_ml: float  # What was actually measured (not "actual")
    duration_s: float
    
    # Optional fields with defaults
    timestamp: float = field(default_factory=time.time)
    replicate_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate measurement data."""
        if self.target_volume_ml <= 0:
            raise ValueError(f"target_volume_ml must be positive, got {self.target_volume_ml}")
        if self.measured_volume_ml < 0:
            raise ValueError(f"measured_volume_ml cannot be negative, got {self.measured_volume_ml}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if not self.measurement_id.strip():
            raise ValueError("measurement_id cannot be empty")


@dataclass
class AdaptiveMeasurementResult:
    """Statistical analysis of replicate measurements."""
    
    # Core statistics
    target_volume_ml: float
    num_replicates: int
    mean_volume_ml: float
    stdev_volume_ml: float
    cv_volume_pct: float
    
    # Accuracy metrics
    deviation_ml: float
    deviation_pct: float  
    absolute_deviation_pct: float
    
    # Timing statistics
    mean_duration_s: float
    stdev_duration_s: float
    
    # Additional statistics
    min_volume_ml: float
    max_volume_ml: float
    median_volume_ml: float
    
    def __post_init__(self):
        """Validate statistical results."""
        if self.num_replicates <= 0:
            raise ValueError(f"num_replicates must be positive, got {self.num_replicates}")
        if self.target_volume_ml <= 0:
            raise ValueError(f"target_volume_ml must be positive, got {self.target_volume_ml}")
        if self.mean_volume_ml < 0:
            raise ValueError(f"mean_volume_ml cannot be negative, got {self.mean_volume_ml}")
        if self.stdev_volume_ml < 0:
            raise ValueError(f"stdev_volume_ml cannot be negative, got {self.stdev_volume_ml}")


@dataclass
class QualityEvaluation:
    """Quality assessment against tolerance thresholds."""
    
    # Quality flags
    accuracy_good: bool
    precision_good: bool
    overall_quality: str  # "within_tolerance", "partial_tolerance", "outside_tolerance"
    
    # Tolerance thresholds used
    accuracy_tolerance_ul: float
    precision_tolerance_pct: float
    
    # Measured values
    measured_accuracy_ul: float
    measured_precision_pct: float  
    measured_time_s: float
    
    def __post_init__(self):
        """Validate quality evaluation."""
        valid_qualities = {"within_tolerance", "partial_tolerance", "outside_tolerance"}
        if self.overall_quality not in valid_qualities:
            raise ValueError(f"overall_quality must be one of {valid_qualities}, got {self.overall_quality}")


@dataclass
class TrialResult:
    """Complete analysis result for one parameter set."""
    
    # Core trial data
    parameters: PipettingParameters
    target_volume_ml: float
    measurements: List[RawMeasurement]
    analysis: AdaptiveMeasurementResult
    quality: QualityEvaluation
    composite_score: float
    tolerances_used: VolumeTolerances
    
    # Experiment context
    strategy: str = "optimization"  # "screening" or "optimization"
    liquid: str = "water"          # Liquid being pipetted
    
    # Adaptive measurement state
    needs_additional_replicates: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trial result consistency."""
        if not self.measurements:
            raise ValueError("Trial must have at least one measurement")
        if len(self.measurements) != self.analysis.num_replicates:
            raise ValueError(f"Measurement count {len(self.measurements)} != analysis replicates {self.analysis.num_replicates}")
        
        # Validate measurements are consistent with parameters
        for measurement in self.measurements:
            if measurement.parameters != self.parameters:
                raise ValueError("All measurements must use same parameters")
            if abs(measurement.target_volume_ml - self.target_volume_ml) > 1e-6:
                raise ValueError("All measurements must have same target volume")


@dataclass
class VolumeCalibrationResult:
    """Results for calibrating one specific volume."""
    
    target_volume_ml: float
    trials: List[TrialResult]
    best_trials: List[TrialResult]  # Ranked by score
    optimal_parameters: Optional[PipettingParameters]
    statistics: Dict[str, float]
    duration_s: float
    measurement_count: int
    
    def __post_init__(self):
        """Validate volume calibration results."""
        if self.target_volume_ml <= 0:
            raise ValueError(f"target_volume_ml must be positive, got {self.target_volume_ml}")
        if not self.trials:
            raise ValueError("Volume calibration must have at least one trial")
        if self.measurement_count != sum(len(trial.measurements) for trial in self.trials):
            raise ValueError("measurement_count must equal sum of trial measurements")


@dataclass  
class ExperimentResults:
    """Complete experiment results across all volumes."""
    
    experiment_name: str
    volume_results: List[VolumeCalibrationResult]
    optimal_conditions: Optional[PipettingParameters]
    total_measurements: int
    total_duration_s: float
    overall_statistics: Dict[str, float]
    config_used: Dict[str, Any]
    
    def __post_init__(self):
        """Validate experiment results."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")
        if not self.volume_results:
            raise ValueError("Experiment must have at least one volume result")
        
        # Validate measurement count consistency
        calculated_measurements = sum(vol.measurement_count for vol in self.volume_results)
        if self.total_measurements != calculated_measurements:
            raise ValueError(f"total_measurements {self.total_measurements} != calculated {calculated_measurements}")


# === TWO-POINT CALIBRATION DATA STRUCTURES ===

@dataclass(frozen=True)
class TwoPointCalibrationPoint:
    """Single data point from two-point calibration measurement."""
    
    overaspirate_vol_ml: float
    measured_volume_ml: float
    measurement_count: int
    variability_pct: float
    parameters_used: PipettingParameters
    
    def __post_init__(self):
        """Validate calibration point."""
        if self.measured_volume_ml <= 0:
            raise ValueError(f"measured_volume_ml must be positive, got {self.measured_volume_ml}")
        if self.measurement_count < 1:
            raise ValueError(f"measurement_count must be >= 1, got {self.measurement_count}")
        if not (0 <= self.variability_pct <= 100):
            raise ValueError(f"variability_pct must be 0-100%, got {self.variability_pct}")


@dataclass(frozen=True)
class TwoPointCalibrationResult:
    """Complete two-point calibration results with constraint bounds."""
    
    target_volume_ml: float
    point_1: TwoPointCalibrationPoint  # Base overvolume from previous optimization
    point_2: TwoPointCalibrationPoint  # Base + shortfall (or +5uL minimum)
    
    # Calculated results
    volume_efficiency_ul_per_ul: float  # Slope: Delta_Volume / Delta_Overaspirate
    shortfall_ml: float                 # target - measured at point 1
    optimal_overaspirate_ml: float      # Optimal value from linear interpolation
    
    # Constraint bounds for next optimization
    min_overaspirate_ml: float
    max_overaspirate_ml: float
    tolerance_range_ml: float  # +/-tolerance for target volume
    
    def __post_init__(self):
        """Validate two-point calibration results."""
        if self.target_volume_ml <= 0:
            raise ValueError(f"target_volume_ml must be positive, got {self.target_volume_ml}")
        
        # Validate points have different overaspirate values
        overaspirate_diff = abs(self.point_2.overaspirate_vol_ml - self.point_1.overaspirate_vol_ml)
        if overaspirate_diff < 0.001:  # 1uL minimum difference
            raise ValueError(f"Two points must have different overaspirate values (diff: {overaspirate_diff:.4f})")
        
        # Validate bounds are reasonable
        if self.tolerance_range_ml <= 0:
            raise ValueError(f"tolerance_range_ml must be positive, got {self.tolerance_range_ml}")


@dataclass
class ConstraintBoundsUpdate:
    """Request to update optimization constraint bounds."""
    
    parameter_name: str
    min_value: float
    max_value: float
    justification: str  # Why these bounds were chosen
    source_calibration: Optional[TwoPointCalibrationResult] = None
    optimal_overaspirate_ml: Optional[float] = None  # Optimal value from linear interpolation
    
    def __post_init__(self):
        """Validate constraint update."""
        if not self.parameter_name.strip():
            raise ValueError("parameter_name cannot be empty")
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value {self.min_value} must be < max_value {self.max_value}")
        if not self.justification.strip():
            raise ValueError("justification cannot be empty")