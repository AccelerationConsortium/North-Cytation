"""
Calibration Modular V2 - Next Generation Calibration System
===========================================================

A modular, type-safe calibration system combining the best of both
next_gen architecture and simplified workflow logic.

Key Features:
- Type-safe data structures with comprehensive validation
- Clean protocol abstraction (simulation/hardware)
- Multi-objective Bayesian optimization
- Adaptive measurement with conditional replicates
- Transfer learning between volumes
- Comprehensive analysis and quality evaluation

Components:
- data_structures: Type-safe data classes for all experiment data
- config_manager: YAML configuration loading and validation
- protocols: Hardware abstraction for simulation/real execution
- analysis: Statistical analysis and quality evaluation
- experiment: Main workflow orchestration

Quick Start:
    from calibration_modular_v2 import CalibrationExperiment, ExperimentConfig
    
    config = ExperimentConfig.from_yaml("experiment_config.yaml")
    experiment = CalibrationExperiment(config)
    results = experiment.run()
    
    print(f"Best parameters: {results.optimal_conditions}")
"""

from .config_manager import ExperimentConfig
from .experiment import CalibrationExperiment
from .analysis import CalibrationAnalyzer
from .protocol_loader import create_protocol
from .external_data import ExternalDataLoader
from .data_structures import (
    PipettingParameters,
    RawMeasurement,
    AdaptiveMeasurementResult,
    TrialResult,
    QualityEvaluation,
    VolumeTolerances,
    ExperimentResults,
    VolumeCalibrationResult
)

__version__ = "2.0.0"
__author__ = "Calibration System Team"

# Main exports
__all__ = [
    # Core classes
    'ExperimentConfig',
    'CalibrationExperiment', 
    'CalibrationAnalyzer',
    'ExternalDataLoader',
    
    # Factory functions
    'create_protocol',
    
    # Data structures
    'PipettingParameters',
    'RawMeasurement',
    'AdaptiveMeasurementResult', 
    'TrialResult',
    'QualityEvaluation',
    'VolumeTolerances',
    'ExperimentResults',
    'VolumeCalibrationResult'
]