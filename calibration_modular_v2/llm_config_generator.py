#!/usr/bin/env python3
"""
LLM Configuration Generator for Modular Calibration System

Generates LLM configuration files from experiment_config.yaml, templating
parameter names and hardware-specific warnings for portability.
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from config_manager import ExperimentConfig

class LLMConfigGenerator:
    """Generate LLM configuration from experiment config."""
    
    def __init__(self, experiment_config: ExperimentConfig, 
                 hardware_config_path: Optional[str] = None):
        self.config = experiment_config
        self.hardware_config = self._load_hardware_config(hardware_config_path)
        
    def _load_hardware_config(self, hardware_config_path: Optional[str]) -> Dict[str, Any]:
        """Load hardware-specific configuration."""
        if hardware_config_path is None:
            hardware_config_path = "hardware_config_default.json"
            
        try:
            with open(hardware_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default if no hardware config found
            return {
                "parameter_warnings": [],
                "parameter_specifics": {}
            }
    
    def get_time_affecting_params(self) -> List[str]:
        """Get list of parameter names that affect timing."""
        time_params = []
        
        # Check hardware parameters for time_affecting flag
        hw_params = self.config._config.get('hardware_parameters', {})
        for param_name, param_config in hw_params.items():
            if param_config.get('time_affecting', False):
                time_params.append(param_name)
                
        return time_params
    
    def get_non_fixed_params(self, fixed_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get parameters that are not fixed (for LLM to suggest)."""
        if fixed_params is None:
            fixed_params = {}
            
        llm_params = {}
        
        # Combine calibration and hardware parameters
        all_params = {}
        all_params.update(self.config._config.get('calibration_parameters', {}))
        all_params.update(self.config._config.get('hardware_parameters', {}))
        
        for param_name, param_config in all_params.items():
            if param_name not in fixed_params:
                # Convert to LLM format
                bounds = param_config['bounds']
                param_type = "float"
                # Could make this configurable per parameter in the future
                # if param_name in self.config.get_integer_parameters():
                #     param_type = "integer"
                    
                llm_params[param_name] = {
                    "type": param_type,
                    "unit": self._get_param_unit(param_name),
                    "range": bounds,
                    "description": param_config.get('description', f"{param_name} parameter")
                }
                
                # Add hardware-specific notes
                if param_name in self.hardware_config.get('parameter_specifics', {}):
                    specifics = self.hardware_config['parameter_specifics'][param_name]
                    llm_params[param_name]["description"] += f". {specifics.get('note', '')}"
                    
                # Add safety limits if present
                if 'safety_limit' in param_config:
                    llm_params[param_name]["safety_limit"] = param_config['safety_limit']
                    
        return llm_params
    
    def _get_param_unit(self, param_name: str) -> str:
        """Get appropriate unit for parameter."""
        if 'speed' in param_name:
            return "relative_units"
        elif 'wait' in param_name or 'time' in param_name:
            return "seconds" 
        elif 'vol' in param_name:
            return "mL"
        else:
            return "units"
    
    def generate_llm_config(self, 
                           template_path: str = "calibration_screening_llm_template.json",
                           target_volume_ml: float = 0.05,
                           batch_size: int = 5,
                           fixed_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate complete LLM configuration."""
        
        # Load template
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        # Get time-affecting parameters for templating
        time_affecting_params = self.get_time_affecting_params()
        time_params_str = ", ".join(time_affecting_params)
        
        # Get hardware warnings
        hardware_warnings = "\n".join(self.hardware_config.get('parameter_warnings', []))
        
        # Get non-fixed parameters
        llm_parameters = self.get_non_fixed_params(fixed_params)
        
        # Template substitution
        config = template.copy()
        
        # Template system message
        system_message = []
        for line in template['system_message_template']:
            line = line.replace('{TIME_AFFECTING_PARAMS}', time_params_str)
            line = line.replace('{HARDWARE_SPECIFIC_WARNINGS}', hardware_warnings)
            system_message.append(line)
        
        config['system_message'] = system_message
        del config['system_message_template']  # Remove template
        
        # Template experimental setup
        config['experimental_setup']['target_volume_ul'] = int(target_volume_ml * 1000)
        config['experimental_setup']['device_serial'] = self.hardware_config.get('device_serial', 'UNKNOWN')
        
        # Set batch size
        config['batch_size'] = batch_size
        
        # Set parameters (only non-fixed ones)
        config['parameters'] = llm_parameters
        
        return config
    
    def save_llm_config(self, output_path: str, **kwargs) -> None:
        """Generate and save LLM configuration to file."""
        config = self.generate_llm_config(**kwargs)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

if __name__ == "__main__":
    # Example usage
    experiment_config = ExperimentConfig.from_yaml("experiment_config.yaml")
    generator = LLMConfigGenerator(experiment_config)
    
    # Generate config for screening
    config = generator.generate_llm_config(
        target_volume_ml=0.05,
        batch_size=5,
        fixed_params={}  # No fixed parameters for screening
    )
    
    print("Generated LLM config with parameters:", list(config['parameters'].keys()))
    print("Time-affecting parameters:", generator.get_time_affecting_params())