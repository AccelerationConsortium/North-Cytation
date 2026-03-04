"""
LLM-based Parameter Recommender for Calibration Experiments

This module provides LLM-based parameter suggestions for pipetting calibration.
It integrates with the template-based LLM configuration system and supports
both screening and optimization phases.
"""

import json
import logging
import requests
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from data_structures import PipettingParameters, VolumeCalibrationResult, CalibrationParameters, HardwareParameters
from .config_manager import ExperimentConfig
from llm_config_generator import LLMConfigGenerator


class LLMRecommender:
    """LLM-based parameter recommender using template-generated configurations."""
    
    def __init__(self, config: ExperimentConfig, llm_template_path: str, phase: str = "screening"):
        """
        Initialize LLM recommender.
        
        Args:
            config: Experiment configuration
            llm_template_path: Path to LLM configuration template file
            phase: "screening" or "optimization" phase
        """
        self.config = config
        self.phase = phase
        self.logger = logging.getLogger(__name__)
        
        # Load template and generate hardware-specific LLM configuration
        self.llm_config = self._load_and_process_template(llm_template_path)
        
        # Extract parameter information
        self._setup_parameters()
        
        # Initialize LLM connection info (using your remote Ollama setup)
        self.ollama_url = None
        self.ollama_model = None
        
        # Initialize conversation history
        self.conversation_history = []
        
    def _load_and_process_template(self, template_path: str) -> Dict[str, Any]:
        """Load template and substitute hardware-specific values."""
        # Handle relative paths by making them relative to this file's directory
        if not os.path.isabs(template_path):
            template_path = os.path.join(os.path.dirname(__file__), template_path)
            
        try:
            with open(template_path, 'r') as f:
                template = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load LLM template from {template_path}: {e}")
        
        # Get hardware-specific parameters and substitutions
        time_affecting_params = self.config.get_time_affecting_parameters()
        hardware_warnings = self.config.get_hardware_specific_warnings()
        
        # Build parameters section from config
        parameters = {}
        all_params = {}
        all_params.update(self.config._config.get('calibration_parameters', {}))
        all_params.update(self.config._config.get('hardware_parameters', {}))
        
        for param_name, param_config in all_params.items():
            parameters[param_name] = {
                "type": param_config.get("type", "float"),
                "unit": param_config.get("unit", ""),
                "range": param_config["bounds"],
                "description": param_config.get("description", ""),
            }
            if "safety_limit" in param_config:
                parameters[param_name]["safety_limit"] = param_config["safety_limit"]
        
        # Perform template substitutions
        processed_config = self._substitute_template_values(template, {
            "TIME_AFFECTING_PARAMS": ", ".join(time_affecting_params),
            "HARDWARE_SPECIFIC_WARNINGS": hardware_warnings,
            "PARAMETERS_SECTION": parameters,
            "TARGET_VOLUME_UL": int(self.config.experiment.target_volume_ml * 1000),
            "DEVICE_SERIAL": getattr(self.config.experiment, 'device_serial', 'UNKNOWN'),
            "BATCH_SIZE": 5
        })
        
        return processed_config
    
    def _substitute_template_values(self, template: Dict[str, Any], substitutions: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute template placeholders with actual values."""
        
        def substitute_recursive(obj):
            """Recursively substitute in nested structures."""
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Check if the entire string is a single placeholder
                for key, value in substitutions.items():
                    placeholder = "{" + key + "}"
                    if obj == placeholder:
                        # Return the value directly (preserves type)
                        return value
                
                # Otherwise, do string substitution for partial matches
                for key, value in substitutions.items():
                    placeholder = "{" + key + "}"
                    if placeholder in obj:
                        if isinstance(value, (dict, list)):
                            # For complex values, convert to JSON
                            obj = obj.replace(placeholder, json.dumps(value, indent=2))
                        else:
                            # For simple values, convert to string
                            obj = obj.replace(placeholder, str(value))
                return obj
            else:
                return obj
        
        return substitute_recursive(template)
    
    def _setup_parameters(self):
        """Setup parameter bounds and constraints."""
        self.optimize_params = []
        self.fixed_params = {}
        self.param_bounds = {}
        
        # Combine calibration and hardware parameters
        all_params = {}
        all_params.update(self.config._config.get('calibration_parameters', {}))
        all_params.update(self.config._config.get('hardware_parameters', {}))
        
        for param_name, param_config in all_params.items():
            if param_config.get('fixed') is not None:
                self.fixed_params[param_name] = param_config['fixed']
            else:
                self.optimize_params.append(param_name)
                self.param_bounds[param_name] = param_config['bounds']
    
    def _is_time_affecting_parameter(self, param_name: str) -> bool:
        """Check if parameter affects timing based on config."""
        # Check hardware parameters first
        hw_params = self.config._config.get('hardware_parameters', {})
        if param_name in hw_params:
            return hw_params[param_name].get('time_affecting', False)
        
        # Check calibration parameters (less common to have time_affecting)
        cal_params = self.config._config.get('calibration_parameters', {})
        if param_name in cal_params:
            return cal_params[param_name].get('time_affecting', False)
        
        return False
    
    def suggest_parameters(self, n_suggestions_or_volume: Union[int, float] = 1, 
                          previous_results_or_trial: Optional[Union[List[VolumeCalibrationResult], int]] = None) -> Union[List[PipettingParameters], PipettingParameters]:
        """
        Generate LLM-based parameter suggestions.
        
        Args:
            n_suggestions_or_volume: Number of parameter sets (new API) OR target volume (legacy API)
            previous_results_or_trial: Previous calibration results (new API) OR trial index (legacy API)
            
        Returns:
            List of suggested parameter sets (new API) OR single parameter set (legacy API)
        """
        # Handle legacy API: suggest_parameters(target_volume_ml, trial_idx)
        if isinstance(n_suggestions_or_volume, float) and isinstance(previous_results_or_trial, (int, type(None))):
            # Legacy mode: return single parameter set
            self.logger.debug(f"Legacy API call: target_volume={n_suggestions_or_volume}, trial_idx={previous_results_or_trial}")
            suggestions = self._suggest_parameters_new_api(n_suggestions=1, previous_results=None)
            if suggestions:
                return suggestions[0]
            else:
                raise RuntimeError("LLM failed to generate parameters")
        
        # Handle new API: suggest_parameters(n_suggestions, previous_results)
        elif isinstance(n_suggestions_or_volume, int):
            return self._suggest_parameters_new_api(n_suggestions_or_volume, previous_results_or_trial)
        
        else:
            raise ValueError(f"Invalid arguments: {type(n_suggestions_or_volume)}, {type(previous_results_or_trial)}")
    
    def _suggest_parameters_new_api(self, n_suggestions: int = 1, 
                          previous_results: Optional[List[VolumeCalibrationResult]] = None) -> List[PipettingParameters]:
        """
        Generate LLM-based parameter suggestions (new API implementation).
        
        Args:
            n_suggestions: Number of parameter sets to suggest
            previous_results: Previous calibration results for context
            
        Returns:
            List of suggested parameter sets
        """
        # Build context from previous results
        context = self._build_context(previous_results)
        
        # Generate LLM prompt
        prompt = self._generate_prompt(n_suggestions, context)
        
        # Try actual LLM API call first, fall back to simulation
        try:
            suggestions = self._call_llm_api(prompt, n_suggestions, context)
            self.logger.info(f"LLM API generated {len(suggestions)} suggestions for {self.phase} phase")
        except Exception as e:
            self.logger.warning(f"LLM API call failed: {e}. Falling back to intelligent simulation")
            suggestions = self._simulate_llm_response(n_suggestions, context)
            self.logger.info(f"LLM simulation generated {len(suggestions)} suggestions for {self.phase} phase")
        
        return suggestions
    
    def _build_context(self, previous_results: Optional[List[VolumeCalibrationResult]]) -> Dict[str, Any]:
        """Build context dictionary from previous results."""
        if not previous_results:
            return {
                "phase": self.phase,
                "has_previous_data": False,
                "liquid": self.config.experiment.liquid,  # Use experiment.liquid directly
                "target_volume": self.config.experiment.target_volume_ml  # Use configured target volume
            }
        
        # Analyze previous results
        best_trials = []
        for result in previous_results:
            best_trials.extend(result.best_trials)
        
        if not best_trials:
            return {
                "phase": self.phase,
                "has_previous_data": False,
                "liquid": self.config.experiment.liquid,
                "target_volume": previous_results[0].target_volume_ml if previous_results else self.config.experiment.target_volume_ml
            }
        
        # Find overall best trial
        best_trial = min(best_trials, key=lambda t: t.score)
        
        # Calculate summary statistics
        all_scores = [t.score for t in best_trials]
        all_durations = [t.duration_s for t in best_trials]
        
        results_summary = {
            "best_score": min(all_scores),
            "avg_score": np.mean(all_scores),
            "avg_duration": np.mean(all_durations),
            "num_trials": len(best_trials),
            "num_volumes": len(previous_results)
        }
        
        return {
            "phase": self.phase,
            "has_previous_data": True,
            "liquid": self.config.experiment.liquid,
            "target_volume": previous_results[-1].target_volume_ml,  # Most recent volume
            "results_summary": results_summary,
            "best_parameters": best_trial.parameters.to_protocol_dict()  # Hardware-agnostic parameter access
        }
    
    def _generate_prompt(self, n_suggestions: int, context: Dict[str, Any]) -> str:
        """Generate LLM prompt using the template-based approach for hardware agnosticism."""
        
        # Get system message from processed template
        system_message_raw = self.llm_config.get("system_message_template", [])
        
        if isinstance(system_message_raw, list):
            system_message = "\n".join(system_message_raw)
        else:
            system_message = system_message_raw
        
        # Get parameters section from processed template
        parameters = self.llm_config.get("parameters", {})
        
        # Build user prompt - different for phases with/without previous data
        if context["has_previous_data"]:
            # Optimization phase with previous experimental results
            user_prompt = f"""Please suggest {n_suggestions} parameter set(s) for {context['liquid']} calibration based on previous experimental results.

Previous results summary:
- Best score: {context['results_summary']['best_score']:.3f}
- Average score: {context['results_summary']['avg_score']:.3f}
- Average duration: {context['results_summary']['avg_duration']:.1f}s
- Number of trials: {context['results_summary']['num_trials']}
- Volumes tested: {context['results_summary']['num_volumes']}

Current target volume: {context['target_volume']:.3f} mL

Best performing parameters so far:
{json.dumps(context['best_parameters'], indent=2)}

Please suggest improvements or variations that might achieve better performance.

PARAMETERS TO OPTIMIZE:
{json.dumps(parameters, indent=2)}

Generate {n_suggestions} parameter combinations that improve upon the best results for {context['liquid']}."""
        else:
            # Initial screening phase - use material properties from template
            material_info = self.llm_config.get("material_properties", {}).get(context['liquid'], {})
            material_description = material_info.get("description", f"{context['liquid']} liquid")
            material_focus = material_info.get("focus", "Balanced parameter exploration")
            
            user_prompt = f"""Please suggest {n_suggestions} parameter set(s) for {context['liquid']} calibration.

This is the {context['phase']} phase with no previous experimental data.

Material: {material_description}
Focus: {material_focus}
Target volume: {context['target_volume']:.3f} mL

PARAMETERS TO OPTIMIZE:
{json.dumps(parameters, indent=2)}

Generate {n_suggestions} diverse parameter combinations that explore different regions of the parameter space for {context['liquid']}."""

        return f"{system_message}\n\n{user_prompt}"
    
    def _call_llm_api(self, prompt: str, n_suggestions: int, context: Dict[str, Any]) -> List[PipettingParameters]:
        """Call the actual LLM API using your remote Ollama setup."""
        
        # Set up connection (matches your existing pattern from llm_optimizer.py)
        api_settings = self.llm_config.get("api_settings", {})
        model = api_settings.get("model", "online_server")
        temperature = api_settings.get("temperature", 1.0)
        max_tokens = api_settings.get("max_tokens", 8000)
        
        # Use remote server (matches your existing setup)
        if model == "online_server":
            self.ollama_url = "https://mac-llm.tail2a00e9.ts.net/ollama"
        else:
            self.ollama_url = "http://localhost:11434"
        
        # Auto-select model from remote server (matches your existing logic)
        if model == "online_server":
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    available_models = [m["name"] for m in response.json().get("models", [])]
                    
                    # Prefer certain models (matches your existing logic)
                    preferred_models = ["gpt-oss:20b", "llama2", "mistral"]
                    selected_model = None
                    for pref in preferred_models:
                        if pref in available_models:
                            selected_model = pref
                            break
                    
                    if not selected_model and available_models:
                        selected_model = available_models[0]
                    
                    model = selected_model or "gpt-oss:20b"
                    self.logger.info(f"Auto-selected model: {model}")
                else:
                    model = "gpt-oss:20b"
            except Exception as e:
                self.logger.warning(f"Failed to auto-select model: {e}")
                model = "gpt-oss:20b"
        
        # Make API call (matches your existing format from llm_optimizer.py)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        self.logger.info(f"Calling LLM API with model: {model}")
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM API error (status {response.status_code}): {response.text}")
        
        result = response.json()
        response_content = result.get("response", "")
        
        if not response_content:
            raise Exception("Empty response from LLM API")
        
        self.logger.info(f"LLM API response received ({len(response_content)} chars)")
        
        # Save prompt and response (matches your existing system)
        self._save_prompt_and_response(prompt, response_content, model)
        
        # Parse the response
        suggestions = self._parse_llm_response(response_content, n_suggestions, context)
        return suggestions
    
    def _save_prompt_and_response(self, prompt: str, response_content: str, model: str):
        """Save prompt and response to timestamped files (matches your existing LLM system)."""
        
        # Get log directory from config (matches your existing pattern)
        log_base_dir = self.llm_config.get("logging", {}).get("log_directory", "LLM_prompts")
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_base_dir, f"calibration_llm_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Save prompt
        prompt_file = os.path.join(log_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Phase: {self.phase}\n")
            f.write(f"Liquid: {self.config.get_liquid_name()}\n")
            f.write("="*50 + "\n")
            f.write(prompt)
        
        # Save response content
        response_file = os.path.join(log_dir, "response.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Phase: {self.phase}\n")
            f.write(f"Liquid: {self.config.get_liquid_name()}\n")
            f.write("="*50 + "\n")
            f.write(response_content)
        
        # Save metadata
        metadata_file = os.path.join(log_dir, "calibration_metadata.json")
        metadata = {
            "timestamp": timestamp,
            "model": model,
            "phase": self.phase,
            "liquid": self.config.get_liquid_name(),
            "prompt_length": len(prompt),
            "response_length": len(response_content),
            "config_file": "calibration_screening_llm_template.json"
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Prompt and response saved to: {log_dir}")
    
    def _parse_llm_response(self, response_content: str, n_suggestions: int, context: Dict[str, Any]) -> List[PipettingParameters]:
        """Parse LLM response to extract parameter suggestions."""
        
        try:
            # Try to parse as structured JSON with explanations
            import re
            
            # Look for complete JSON object with explanation and suggestions
            json_match = re.search(r'\{[^{}]*(?:"explanation"[^{}]*"suggestions"[^{}]*|\{[^{}]*\}[^{}]*)*\}', response_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_response = json.loads(json_str)
                    
                    # Log the overall explanation
                    if "explanation" in parsed_response:
                        self.logger.info(f"LLM Explanation: {parsed_response['explanation']}")
                    
                    # Extract suggestions with individual reasoning
                    suggestions_data = parsed_response.get("suggestions", [])
                    if not suggestions_data:
                        # Fallback: try to use as simple array
                        suggestions_data = [{"parameters": parsed_response, "reasoning": "No specific reasoning provided"}]
                    
                except json.JSONDecodeError:
                    # Fallback to simple JSON array parsing
                    json_array_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', response_content, re.DOTALL)
                    if json_array_match:
                        json_str = json_array_match.group()
                        suggestions_data = [{"parameters": params, "reasoning": "No reasoning provided"} 
                                          for params in json.loads(json_str)]
                    else:
                        # Last resort: look for individual objects
                        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_content)
                        suggestions_data = []
                        for json_str in json_matches:
                            try:
                                params = json.loads(json_str)
                                suggestions_data.append({"parameters": params, "reasoning": "No reasoning provided"})
                            except json.JSONDecodeError:
                                continue
            else:
                raise Exception("No valid JSON structure found")
            
            # Convert to PipettingParameters objects
            suggestions = []
            all_params = {}
            all_params.update(self.config._config.get('calibration_parameters', {}))
            all_params.update(self.config._config.get('hardware_parameters', {}))
            
            for suggestion_data in suggestions_data:
                if len(suggestions) >= n_suggestions:
                    break
                    
                try:
                    params = suggestion_data.get("parameters", suggestion_data)  # Handle both formats
                    reasoning = suggestion_data.get("reasoning", "No reasoning provided")
                    
                    # Log the individual reasoning
                    self.logger.info(f"Parameter set {len(suggestions)+1} reasoning: {reasoning}")
                    
                    # Validate and process parameters
                    param_values = {}
                    for param_name, param_config in all_params.items():
                        if param_name in params:
                            value = float(params[param_name])
                            bounds = param_config['bounds']
                            # Clamp to bounds
                            value = max(bounds[0], min(bounds[1], value))
                            param_values[param_name] = value
                        else:
                            # Use default if missing
                            param_values[param_name] = param_config['default']
                    
                    # Create parameter objects
                    calibration_params = CalibrationParameters(
                        overaspirate_vol=param_values.get('overaspirate_vol', 0.004)
                    )
                    
                    hw_param_dict = {k: v for k, v in param_values.items() if k != 'overaspirate_vol'}
                    hardware_params = HardwareParameters(parameters=hw_param_dict)
                    
                    suggestions.append(PipettingParameters(
                        calibration=calibration_params,
                        hardware=hardware_params
                    ))
                    
                except (ValueError, KeyError) as e:
                    self.logger.debug(f"Failed to process parameter set: {e}")
                    continue
            
            if suggestions:
                return suggestions
            else:
                raise Exception("No valid parameter sets found in LLM response")
                
        except Exception as e:
            raise Exception(f"Failed to parse LLM response: {e}")
    
    def _simulate_llm_response(self, n_suggestions: int, context: Dict[str, Any]) -> List[PipettingParameters]:
        """
        Simulate intelligent LLM response.
        In production, replace with actual LLM API call.
        """
        suggestions = []
        
        for i in range(n_suggestions):
            param_values = {}
            
            # Generate intelligent suggestions based on context
            if context["has_previous_data"] and context["phase"] == "optimization":
                # Optimization phase: refine around best parameters
                best_params = context["best_parameters"]
                for param_name in self.optimize_params:
                    bounds = self.param_bounds[param_name]
                    if param_name in best_params:
                        # Add noise around best value
                        best_val = best_params[param_name]
                        noise_factor = 0.1  # 10% variation
                        range_size = bounds[1] - bounds[0]
                        noise = np.random.normal(0, noise_factor * range_size)
                        suggested_val = np.clip(best_val + noise, bounds[0], bounds[1])
                    else:
                        # Random if no previous data for this parameter
                        suggested_val = np.random.uniform(bounds[0], bounds[1])
                    param_values[param_name] = suggested_val
            else:
                # Screening phase: intelligent exploration
                for param_name in self.optimize_params:
                    bounds = self.param_bounds[param_name]
                    
                    # Use domain knowledge for better initial suggestions
                    if self._is_time_affecting_parameter(param_name):
                        # Time-affecting parameters: favor moderate values for balance of speed/accuracy
                        mid_point = (bounds[0] + bounds[1]) / 2
                        param_values[param_name] = np.random.normal(mid_point, (bounds[1] - bounds[0]) / 6)
                        param_values[param_name] = np.clip(param_values[param_name], bounds[0], bounds[1])
                    else:
                        # Non-time-affecting parameters: uniform exploration
                        param_values[param_name] = np.random.uniform(bounds[0], bounds[1])
            
            # Add fixed parameters
            param_values.update(self.fixed_params)
            
            # Fill any missing parameters with defaults
            all_params = {}
            all_params.update(self.config._config.get('calibration_parameters', {}))
            all_params.update(self.config._config.get('hardware_parameters', {}))
            
            for param_name, param_config in all_params.items():
                if param_name not in param_values:
                    param_values[param_name] = param_config['default']
            
            # Create proper data structures
            calibration_params = CalibrationParameters(
                overaspirate_vol=param_values.get('overaspirate_vol', 0.004)
            )
            
            # Extract hardware parameters
            hw_param_dict = {}
            for param_name, value in param_values.items():
                if param_name != 'overaspirate_vol':  # Exclude calibration parameter
                    hw_param_dict[param_name] = value
            
            hardware_params = HardwareParameters(parameters=hw_param_dict)
            
            suggestions.append(PipettingParameters(
                calibration=calibration_params,
                hardware=hardware_params
            ))
        
        return suggestions
    
    def update_with_results(self, results: List[VolumeCalibrationResult]):
        """Update recommender with new experimental results."""
        # In production LLM system, this would update conversation history
        # For simulation, we just log the update
        self.logger.info(f"Updated LLM recommender with {len(results)} new results")
        
        # Store results for future context
        if not hasattr(self, 'all_results'):
            self.all_results = []
        self.all_results.extend(results)


def create_llm_recommender(config: ExperimentConfig, phase: str = "screening") -> Optional[LLMRecommender]:
    """
    Create LLM recommender if enabled in configuration.
    
    Args:
        config: Experiment configuration
        phase: "screening" or "optimization"
        
    Returns:
        LLMRecommender instance or None if disabled
    """
    if phase == "screening":
        if not config.use_llm_for_screening():
            return None
        config_path = config.get_screening_llm_config_path()
    else:  # optimization
        if not config.is_llm_optimization_enabled():
            return None  
        config_path = config.get_llm_config_path()
    
    if not config_path:
        # Generate LLM config on-the-fly
        generator = LLMConfigGenerator(config, "hardware_config_north_c9.json")
        config_path = f"generated_llm_config_{phase}.json"
        generator.save_llm_config(
            config_path, 
            template_path="calibration_screening_llm_template.json",
            target_volume_ml=0.05,
            batch_size=5
        )
    
    return LLMRecommender(config, config_path, phase)