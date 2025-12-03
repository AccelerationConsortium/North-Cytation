"""
Configuration Management for Modular Calibration System
======================================================

This module handles loading, validation, and access to experiment configuration.
Provides type-safe access to all experiment parameters with comprehensive validation.

Key Features:
- YAML configuration loading with validation
- Type-safe access to all parameters
- Unit-explicit parameter naming
- Tolerance calculation with volume dependencies
- Protocol selection (simulation vs hardware)
- Parameter inheritance for transfer learning

Example Usage:
    config = ExperimentConfig.from_yaml("experiment_config.yaml")
    targets = config.get_target_volumes_ml()
    params = config.get_parameter_bounds()
    tolerances = config.calculate_tolerances_for_volume(0.05)  # 50uL
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from data_structures import (
    PipettingParameters, VolumeTolerances, 
    CalibrationParameters, HardwareParameters
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParameterBounds:
    """Bounds for mandatory calibration parameters."""
    overaspirate_vol: Tuple[float, float]


@dataclass 
class HardwareParameterBounds:
    """Bounds for optional hardware-specific parameters."""
    parameters: Dict[str, Tuple[float, float]]
    
    def get(self, name: str) -> Optional[Tuple[float, float]]:
        """Get bounds for named parameter."""
        return self.parameters.get(name)


@dataclass
class ObjectiveWeights:
    """Multi-objective optimization weights."""
    accuracy_weight: float
    precision_weight: float
    time_weight: float
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.accuracy_weight + self.precision_weight + self.time_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Objective weights must sum to 1.0, got {total}")


class ExperimentConfig:
    """
    Type-safe configuration manager for calibration experiments.
    
    Handles loading from YAML, parameter validation, and provides
    convenient access to all configuration values.
    """
    
    def __init__(self, config_dict: Dict[str, Any], config_path: Optional[str] = None):
        """Initialize from configuration dictionary."""
        self._config = config_dict
        self._config_path = config_path  # Store config file path for relative path resolution
        self._validate_config()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return cls(config_dict, str(path.absolute()))
    
    def _validate_config(self):
        """Validate required configuration sections and values."""
        required_sections = [
            'experiment', 'execution', 'volumes', 'budget', 
            'tolerances', 'optimization'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate parameter sections (support both old and new formats)
        has_old_params = 'parameters' in self._config
        has_new_params = ('calibration_parameters' in self._config or 
                         'hardware_parameters' in self._config)
        
        if not (has_old_params or has_new_params):
            raise ValueError("Must have either 'parameters' section (old format) or " +
                           "'calibration_parameters'/'hardware_parameters' sections (new format)")
        
        # Validate objective weights
        weights = self._config['optimization']['objectives']
        total_weight = (weights['accuracy_weight'] + 
                       weights['precision_weight'] + 
                       weights['time_weight'])
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Objective weights must sum to 1.0, got {total_weight}")
            
        # Validate volume targets
        volumes = self._config['volumes']['targets_ml']
        if not volumes or not isinstance(volumes, list):
            raise ValueError("Must specify at least one target volume")
        if any(v <= 0 for v in volumes):
            raise ValueError("All target volumes must be positive")
            
        logger.info(f"Configuration validated successfully")
        logger.info(f"Experiment: {self.get_experiment_name()}")
        logger.info(f"Target volumes: {self.get_target_volumes_ml()} mL")
        logger.info(f"Simulation mode: {self.is_simulation()}")
    
    # Experiment metadata
    def get_experiment_name(self) -> str:
        """Get human-readable experiment name."""
        return self._config['experiment']['name']
    
    def get_liquid_name(self) -> str:
        """Get target liquid name."""
        return self._config['experiment']['liquid']
    
    def get_description(self) -> str:
        """Get experiment description."""
        return self._config['experiment']['description']
    
    # Execution configuration
    def is_simulation(self) -> bool:
        """Check if running in simulation mode."""
        return self._config['execution']['simulate']
    
    def get_random_seed(self) -> Optional[int]:
        """Get random seed for reproducibility."""
        return self._config['execution'].get('random_seed')
    
    # Volume configuration
    def get_target_volumes_ml(self) -> List[float]:
        """Get list of target volumes in mL."""
        return self._config['volumes']['targets_ml']
    
    def get_max_total_measurements(self) -> int:
        """Get maximum total measurements across experiment."""
        return self._config['budget']['max_total_measurements']
    
    def get_max_measurements_first_volume(self) -> int:
        """Get maximum measurements for first volume."""
        return self._config['budget']['max_measurements_first_volume']
    
    def get_max_measurements_per_volume(self) -> int:
        """Get maximum measurements for subsequent volumes (volume-dependent optimization)."""
        # Default to 1/3 of first volume budget for subsequent volumes
        return self._config['budget'].get('max_measurements_per_volume', 
                                         self.get_max_measurements_first_volume() // 3)
    
    # Parameter bounds - NEW ARCHITECTURE ONLY
    # Use get_calibration_parameter_bounds() and get_hardware_parameter_bounds() instead
    
    # Parameter defaults - NEW ARCHITECTURE ONLY  
    # Use get_default_calibration_parameters() and get_default_hardware_parameters() instead
    
    def get_volume_dependent_parameters(self) -> List[str]:
        """Get list of parameters that are re-optimized per volume."""
        # Check parameter_inheritance config
        param_inheritance = self._config.get('optimization', {}).get('parameter_inheritance', {})
        if not param_inheritance.get('enabled', False):
            return []
        return param_inheritance.get('volume_dependent_params', [])
    
    def get_overaspirate_max_fraction(self) -> float:
        """Get maximum overaspirate volume as fraction of target."""
        return self._config['calibration_parameters']['overaspirate_vol'].get('max_fraction_of_target', 0.2)
    
    # New separated parameter methods
    def get_calibration_parameter_bounds(self) -> CalibrationParameterBounds:
        """Get bounds for mandatory calibration parameters."""
        cal_params = self._config['calibration_parameters']
        return CalibrationParameterBounds(
            overaspirate_vol=tuple(cal_params['overaspirate_vol']['bounds'])
        )
    
    def get_hardware_parameter_bounds(self) -> HardwareParameterBounds:
        """Get bounds for optional hardware parameters."""
        hw_params = self._config.get('hardware_parameters', {})
        bounds_dict = {}
        for param_name, param_config in hw_params.items():
            bounds_dict[param_name] = tuple(param_config['bounds'])
        return HardwareParameterBounds(parameters=bounds_dict)
    
    def get_default_calibration_parameters(self) -> CalibrationParameters:
        """Get default values for mandatory calibration parameters."""
        cal_params = self._config['calibration_parameters']
        return CalibrationParameters(
            overaspirate_vol=cal_params['overaspirate_vol']['default']
        )
    
    def get_default_hardware_parameters(self) -> HardwareParameters:
        """Get default values for optional hardware parameters."""
        hw_params = self._config.get('hardware_parameters', {})
        defaults_dict = {}
        for param_name, param_config in hw_params.items():
            defaults_dict[param_name] = param_config['default']
        return HardwareParameters(parameters=defaults_dict)
    
    def get_hardware_parameter_names(self) -> List[str]:
        """Get list of available hardware parameter names."""
        return list(self._config.get('hardware_parameters', {}).keys())
    
    def is_volume_dependent_parameter(self, param_name: str) -> bool:
        """Check if a parameter should be re-optimized for each volume."""
        # Check calibration parameters
        if param_name == 'overaspirate_vol':
            return self._config['calibration_parameters']['overaspirate_vol'].get('volume_dependent', False)
        
        # Check hardware parameters
        hw_params = self._config.get('hardware_parameters', {})
        if param_name in hw_params:
            return hw_params[param_name].get('volume_dependent', False)
        
        return False
    
    def get_optimization_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all optimization parameters (calibration + hardware) with their configuration."""
        all_params = {}
        
        # Add calibration parameters
        cal_params = self._config.get('calibration_parameters', {})
        for name, config in cal_params.items():
            all_params[name] = config
            
        # Add hardware parameters  
        hw_params = self._config.get('hardware_parameters', {})
        for name, config in hw_params.items():
            all_params[name] = config
            
        return all_params
    
    def get_parameter_type(self, param_name: str) -> str:
        """Get the optimizer type (integer/float) for a parameter."""
        all_params = self.get_optimization_parameters()
        return all_params.get(param_name, {}).get('type', 'float')  # Default to float
    
    # Tolerance calculation
    def calculate_tolerances_for_volume(self, target_volume_ml: float) -> VolumeTolerances:
        """Calculate volume-specific tolerances using explicit ranges."""
        tolerances = self._config['tolerances']
        volume_ul = target_volume_ml * 1000  # Convert to uL
        
        # Find matching volume range
        tolerance_pct = 5.0  # Default fallback
        for range_def in tolerances['volume_ranges']:
            if range_def['volume_min_ul'] <= volume_ul <= range_def['volume_max_ul']:
                tolerance_pct = range_def['tolerance_pct']
                break
        
        # Apply simulation multiplier if in simulation mode
        if self.is_simulation() and 'simulation' in tolerances:
            sim_config = tolerances['simulation']
            dev_multiplier = sim_config.get('deviation_multiplier', 1.0)
            var_multiplier = sim_config.get('variability_multiplier', 1.0)
            
            accuracy_tolerance_ul = (target_volume_ml * 1000 * tolerance_pct / 100) * dev_multiplier
            precision_tolerance_pct = tolerance_pct * var_multiplier
        else:
            accuracy_tolerance_ul = target_volume_ml * 1000 * tolerance_pct / 100
            precision_tolerance_pct = tolerance_pct
        
        return VolumeTolerances(
            accuracy_tolerance_ul=accuracy_tolerance_ul,
            precision_tolerance_pct=precision_tolerance_pct
        )
    
    # Optimization configuration
    def get_objective_weights(self) -> ObjectiveWeights:
        """Get multi-objective optimization weights."""
        objectives = self._config['optimization']['objectives']
        return ObjectiveWeights(
            accuracy_weight=objectives['accuracy_weight'],
            precision_weight=objectives['precision_weight'],
            time_weight=objectives['time_weight']
        )
    
    def get_optimizer_type(self) -> str:
        """Get optimizer type (multi_objective or single_objective)."""
        return self._config['optimization']['optimizer']['type']
    
    def get_optimizer_backend(self) -> str:
        """Get optimizer backend for first volume."""
        return self._config['optimization']['optimizer']['backend']
    
    def get_optimizer_backend_subsequent(self) -> str:
        """Get optimizer backend for subsequent volumes."""
        return self._config['optimization']['optimizer'].get(
            'backend_subsequent', self.get_optimizer_backend())
    
    def get_objective_thresholds(self) -> Dict[str, float]:
        """Get objective thresholds for gradient learning."""
        return self._config['optimization']['objective_thresholds']
    
    # LLM optimization
    def is_llm_optimization_enabled(self) -> bool:
        """Check if LLM-based optimization is enabled."""
        return self._config.get('optimization', {}).get('llm_optimization', {}).get('enabled', False)
    
    def get_llm_config_path(self) -> Optional[str]:
        """Get LLM configuration file path."""
        return self._config.get('optimization', {}).get('llm_optimization', {}).get('config_path')
    
    # Adaptive measurement
    def is_adaptive_measurement_enabled(self) -> bool:
        """Check if adaptive measurement is enabled."""
        return self._config.get('adaptive_measurement', {}).get('enabled', False)
    
    def get_adaptive_measurement_config(self) -> Dict[str, Any]:
        """Get full adaptive measurement configuration."""
        return self._config.get('adaptive_measurement', {})
    
    # Protocol selection
    def get_protocol_module(self) -> str:
        """Get appropriate protocol module based on execution mode."""
        protocol_config = self._config.get('protocol', {})
        
        # Check for forced module override first
        if 'module' in protocol_config:
            return protocol_config['module']
        
        # Select based on simulation mode using config values
        if self.is_simulation():
            if 'simulation_module' not in protocol_config:
                raise ValueError("Missing 'simulation_module' in protocol configuration")
            return protocol_config['simulation_module']
        else:
            if 'hardware_module' not in protocol_config:
                raise ValueError("Missing 'hardware_module' in protocol configuration")
            return protocol_config['hardware_module']
    
    # Phase configuration
    def get_screening_trials(self) -> int:
        """Get number of screening trials."""
        return self._config.get('screening', {}).get('num_trials', 5)
    
    def get_min_good_trials(self) -> int:
        """Get minimum good trials for stopping criteria."""
        return self._config.get('optimization', {}).get('stopping_criteria', {}).get('min_good_trials', 3)
    
    def use_llm_for_screening(self) -> bool:
        """Check if LLM should be used for screening phase."""
        return self._config.get('screening', {}).get('use_llm_suggestions', False)
    
    def get_screening_llm_config_path(self) -> Optional[str]:
        """Get LLM configuration file path for screening phase."""
        return self._config.get('screening', {}).get('llm_config_path')
    
    # Transfer learning (now called parameter inheritance)
    def is_transfer_learning_enabled(self) -> bool:
        """Check if parameter inheritance is enabled."""
        return self._config.get('optimization', {}).get('parameter_inheritance', {}).get('enabled', False)
    
    def carry_optimizer_history(self) -> bool:
        """Check if optimizer state should be carried across volumes."""
        return self._config.get('optimization', {}).get('parameter_inheritance', {}).get('carry_optimizer_state', False)
    
    # First volume final calibration
    def is_first_volume_final_calibration_enabled(self) -> bool:
        """Check if final overaspirate calibration is enabled for first volume."""
        return self._config.get('optimization', {}).get('first_volume_final_calibration', {}).get('enabled', False)
    
    def should_skip_final_calibration_if_good_trial(self) -> bool:
        """Check if final calibration should be skipped when current best trial is already good."""
        return self._config.get('optimization', {}).get('first_volume_final_calibration', {}).get('skip_if_good_trial', True)
    
    # External data (now under screening section)
    def is_external_data_enabled(self) -> bool:
        """Check if external data integration is enabled."""
        return self._config.get('screening', {}).get('external_data', {}).get('enabled', False)
    
    def get_external_data_path(self) -> Optional[str]:
        """Get path to external data file."""
        return self._config.get('screening', {}).get('external_data', {}).get('data_path')
    
    def get_external_data_volume_filter(self) -> Optional[float]:
        """Get volume filter for external data."""
        return self._config.get('screening', {}).get('external_data', {}).get('volume_filter_ml')
    
    def get_external_data_liquid_filter(self) -> Optional[str]:
        """Get liquid filter for external data."""
        return self._config.get('screening', {}).get('external_data', {}).get('liquid_filter')
    
    def get_external_data_required_columns(self) -> List[str]:
        """Get required columns for external data - hardware agnostic."""
        # Base columns that are always required
        base_columns = ["volume_ml", "deviation_pct", "duration_s", "overaspirate_vol_ml"]
        
        # Add hardware-specific parameter columns from config
        hw_param_names = self.get_hardware_parameter_names()
        
        # Return base + configured hardware parameters
        return base_columns + hw_param_names
    
    # Advanced features
    def use_range_based_variability(self) -> bool:
        """Check if range-based variability calculation should be used."""
        return self._config.get('advanced', {}).get('use_range_based_variability', False)
    
    # Output configuration
    def get_output_directory(self) -> str:
        """Get base output directory, resolved relative to config file location."""
        base_directory = self._config.get('output', {}).get('base_directory', 'output/calibration_v2_runs')
        
        # If we have a config file path, make base_directory relative to it
        if self._config_path:
            config_dir = Path(self._config_path).parent
            resolved_path = config_dir / base_directory
            return str(resolved_path.absolute())
        else:
            # Fallback to relative path (for programmatically created configs)
            return base_directory
    
    def should_save_raw_measurements(self) -> bool:
        """Check if raw measurements should be saved."""
        return self._config.get('output', {}).get('save_raw_measurements', True)
    
    def should_generate_plots(self) -> bool:
        """Check if visualization plots should be generated."""
        return self._config.get('output', {}).get('generate_plots', True)
    
    def should_export_optimal_conditions(self) -> bool:
        """Check if optimal conditions should be exported."""
        return self._config.get('output', {}).get('export_optimal_conditions', True)
    
    # Validation helpers - HARDWARE AGNOSTIC
    def validate_parameter_set(self, params: PipettingParameters) -> bool:
        """Validate that parameter set is within bounds - hardware agnostic."""
        # Validate calibration parameters
        cal_bounds = self.get_calibration_parameter_bounds()
        cal_valid = (cal_bounds.overaspirate_vol[0] <= params.overaspirate_vol <= 
                    cal_bounds.overaspirate_vol[1])
        
        # Validate hardware parameters
        hw_bounds = self.get_hardware_parameter_bounds()
        hw_valid = True
        for param_name, bounds in hw_bounds.parameters.items():
            param_value = params.get_hardware_param(param_name)
            if param_value is not None:
                if not (bounds[0] <= param_value <= bounds[1]):
                    hw_valid = False
                    break
        
        return cal_valid and hw_valid
    
    def apply_volume_constraints(self, params: PipettingParameters, 
                               target_volume_ml: float) -> PipettingParameters:
        """Apply volume-specific constraints to parameters - hardware agnostic."""
        # Apply overaspirate constraint (the main volume-dependent constraint)
        cal_bounds = self.get_calibration_parameter_bounds()
        max_overaspirate = min(
            cal_bounds.overaspirate_vol[1],
            target_volume_ml * self.get_overaspirate_max_fraction()
        )
        
        adjusted_overaspirate = min(params.overaspirate_vol, max_overaspirate)
        
        # Create new parameters with adjusted overaspirate
        new_cal = CalibrationParameters(overaspirate_vol=adjusted_overaspirate)
        
        return PipettingParameters(
            calibration=new_cal,
            hardware=params.hardware  # Keep hardware params unchanged
        )
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary for advanced use cases."""
        """Get raw configuration dictionary for advanced use cases."""
        return self._config.copy()
    
    def get_hardware_specific_warnings(self) -> str:
        """Get hardware-specific warnings for LLM prompts."""
        warnings = self._config.get('hardware', {}).get('parameter_warnings', [])
        if warnings:
            return "\\n".join(warnings)
        # Default warnings if not specified
        return "IMPORTANT: Higher speed values = SLOWER operation (counterintuitive scaling)\\nNOTE: Speeds are relative units, not absolute velocities\\nCAUTION: Retract speed behaves differently - higher = faster"