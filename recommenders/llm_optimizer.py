import pandas as pd
from openai import OpenAI
import json
import os
from typing import Dict, Any, List
import argparse
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

class LLMOptimizer:
    def __init__(self, api_key: str = None, backend: str = "auto", ollama_model: str = None, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the LLM Optimizer
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            backend: Backend to use ("openai", "ollama", or "auto"). Auto will try OpenAI first, fallback to Ollama
            ollama_model: Ollama model to use (e.g., "gpt-oss:20b", "gemma3:1b") or "online_server" for remote
            ollama_url: Ollama server URL (local or remote)
        """
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.client = None
        
        if backend == "auto":
            # Try OpenAI first, fallback to Ollama
            try:
                if api_key and api_key != "demo-key":
                    self.client = OpenAI(api_key=api_key)
                    self.backend = "openai"
                else:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        self.client = OpenAI(api_key=api_key)
                        self.backend = "openai"
                    else:
                        raise ValueError("No OpenAI API key found")
            except:
                print("⚠️  OpenAI API key not found, falling back to Ollama")
                self.backend = "ollama"
                self._setup_ollama()
        elif backend == "openai":
            if api_key and api_key != "demo-key":
                self.client = OpenAI(api_key=api_key)
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
                self.client = OpenAI(api_key=api_key)
        elif backend == "ollama":
            self._setup_ollama()
        else:
            raise ValueError(f"Invalid backend: {backend}. Must be 'openai', 'ollama', or 'auto'")
    
    def _setup_ollama(self):
        """Setup Ollama backend (local or remote)"""
        try:
            # Handle special case for remote server
            if self.ollama_model == "online_server":
                # Use colleague's remote server
                self.ollama_url = "https://mac-llm.tail2a00e9.ts.net/ollama"
                print(f"🌐 Using remote Ollama server: {self.ollama_url}")
            
            # Test if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                server_type = "remote" if "tail2a00e9.ts.net" in self.ollama_url else "local"
                print(f"🦙 Ollama {server_type} server connected! Available models: {available_models}")
                
                # Auto-select model if not specified or if using online_server
                if not self.ollama_model or self.ollama_model == "online_server":
                    if available_models:
                        # Prefer certain models if available
                        preferred_models = ["gpt-oss:20b", "llama2", "mistral"]
                        selected_model = None
                        for pref in preferred_models:
                            if pref in available_models:
                                selected_model = pref
                                break
                        
                        if not selected_model:
                            selected_model = available_models[0]
                        
                        self.ollama_model = selected_model
                        print(f"🎯 Auto-selected model: {self.ollama_model}")
                    else:
                        raise ValueError("No Ollama models available")
                elif self.ollama_model not in available_models:
                    print(f"⚠️  Requested model {self.ollama_model} not found. Available: {available_models}")
                    if available_models:
                        self.ollama_model = available_models[0]
                        print(f"🎯 Using fallback model: {self.ollama_model}")
            else:
                raise ValueError(f"Ollama server not responding (status: {response.status_code})")
        except requests.exceptions.RequestException as e:
            if "online_server" in str(self.ollama_model):
                raise ValueError(f"Remote Ollama server not accessible: {e}")
            else:
                raise ValueError(f"Ollama server not running. Start it with: ollama serve. Error: {e}")
    
    def list_data_files(self, data_folder: str) -> list:
        """
        List available CSV files in the data folder
        
        Args:
            data_folder: Path to the data folder
            
        Returns:
            List of CSV filenames
        """
        if not os.path.exists(data_folder):
            return []
        
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        return sorted(csv_files)
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            pandas DataFrame
        """
        return pd.read_csv(csv_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Dictionary containing configuration
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_optimization_prompt(self, df: pd.DataFrame, config: Dict[str, Any]) -> str:
        """
        Create the optimization prompt for the LLM
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary
            
        Returns:
            Formatted prompt string
        """
        system_message_raw = config.get("system_message", "You are an optimization expert.")
        
        # Handle system_message as either string or array
        if isinstance(system_message_raw, list):
            system_message = "\n".join(system_message_raw)
        else:
            system_message = system_message_raw
        
        # Convert DataFrame to string representation
        # Check if we have meaningful experimental data (more than just liquid/material type)
        has_experimental_data = False
        if not df.empty:
            # Check if we have any parameter columns with actual experimental results
            experimental_columns = ['aspirate_speed', 'dispense_speed', 'deviation', 'variability', 'time']
            has_experimental_data = any(col in df.columns for col in experimental_columns)
        
        if df.empty or not has_experimental_data:
            data_str = "No existing experimental data - generating initial parameter recommendations"
            is_initial_generation = True
        else:
            data_str = df.to_string(index=False)
            is_initial_generation = False
        
        # Extract parameter information if available
        parameters_info = ""
        if "parameters" in config:
            parameters_info = "\n\nParameter Definitions:\n"
            for param_name, param_info in config["parameters"].items():
                param_type = param_info.get("type", "unknown")
                unit = param_info.get("unit", "")
                param_range = param_info.get("range", "not specified")
                description = param_info.get("description", "")
                safety_limit = param_info.get("safety_limit", "not specified")
                
                if param_type == "categorical":
                    options = param_info.get("options", [])
                    parameters_info += f"- {param_name} ({param_type}): {description} {unit}\n  Options: {', '.join(options)}\n"
                elif param_type == "discrete":
                    options = param_info.get("options", [])
                    parameters_info += f"- {param_name} ({param_type}): {description} {unit}\n  Range: {param_range}, Options: {options}\n"
                else:
                    parameters_info += f"- {param_name} ({param_type}): {description} {unit}\n  Range: {param_range}, Safety limit: {safety_limit}\n"
        
        # Extract output metrics information if available
        metrics_info = ""
        if "output_metrics" in config:
            metrics_info = "\n\nOutput Metrics to Optimize:\n"
            for metric_name, metric_info in config["output_metrics"].items():
                target = metric_info.get("target", "unknown")
                unit = metric_info.get("unit", "")
                description = metric_info.get("description", "")
                target_value = metric_info.get("target_value", "")
                
                if target == "match" and target_value:
                    metrics_info += f"- {metric_name}: {target} to {target_value} {unit} ({description})\n"
                else:
                    metrics_info += f"- {metric_name}: {target} {description} {unit}\n"
        
        # Extract optimization guidelines if available
        guidelines_info = ""
        if "optimization_guidelines" in config:
            guidelines = config["optimization_guidelines"]
            guidelines_info = "\n\nOptimization Guidelines:\n"
            
            # Handle both list and dict formats
            if isinstance(guidelines, list):
                for guideline in guidelines:
                    guidelines_info += f"- {guideline}\n"
            elif isinstance(guidelines, dict):
                if guidelines.get("avoid_infeasible_experiments"):
                    guidelines_info += "- Avoid suggesting infeasible experiments (outside parameter bounds or safety limits)\n"
                if guidelines.get("minimize_experiment_count"):
                    guidelines_info += "- Minimize the number of experiments needed\n"
                if guidelines.get("avoid_previous_combinations"):
                    guidelines_info += "- Avoid suggesting previously tested parameter combinations\n"
                if guidelines.get("consider_physical_meaning"):
                    guidelines_info += "- Consider the physical/chemical meaning of observed data\n"
                if guidelines.get("safety_first"):
                    guidelines_info += "- Prioritize safety in all recommendations\n"
        
        # Extract batch size
        batch_size = config.get("batch_size", 3)
        
        # Extract experimental setup information if available
        setup_info = ""
        if "experimental_setup" in config:
            setup_config = config["experimental_setup"]
            setup_info = f"\n\nExperimental Setup:\n"
            for key, value in setup_config.items():
                # Convert key from snake_case to readable format
                readable_key = key.replace('_', ' ').title()
                setup_info += f"- {readable_key}: {value}\n"
        
        # Add material/condition type information from config
        material_type_info = ""
        material_col = None
        
        # Get material column name from config if specified
        if config.get("data_columns", {}).get("material_column"):
            material_col = config["data_columns"]["material_column"]
        else:
            # Fallback to detecting common column names
            for potential_col in ["liquid", "material", "condition", "sample", "treatment"]:
                if potential_col in df.columns:
                    material_col = potential_col
                    break
                    
        if material_col and material_col in df.columns:
            material_types = df[material_col].unique().tolist()
            material_type_info = f"\n\nMaterial/Condition Information:\n"
            material_type_info += f"- Types in data: {material_types}\n"
            material_type_info += f"- IMPORTANT: Keep the SAME {material_col} type as in the input data\n"
            
            # Add configurable material properties if available
            if "material_properties" in config:
                material_type_info += f"- Consider material-specific properties:\n"
                for material_name, properties in config["material_properties"].items():
                    if material_name in material_types:
                        prop_desc = properties.get("description", "No description")
                        optimization_notes = properties.get("optimization_notes", "")
                        material_type_info += f"  * For {material_name}: {prop_desc}"
                        if optimization_notes:
                            material_type_info += f" - {optimization_notes}"
                        material_type_info += "\n"
        
        
        # Legacy liquid properties support (for backward compatibility)
        liquid_properties_info = ""
        if "parameters" in config and "liquid" in config["parameters"]:
            liquid_param = config["parameters"]["liquid"]
            if "properties" in liquid_param:
                liquid_properties_info = "\n\nLiquid Properties:\n"
                for liquid_name, properties in liquid_param["properties"].items():
                    liquid_properties_info += f"- {liquid_name}: "
                    prop_list = []
                    for prop, value in properties.items():
                        prop_list.append(f"{prop}={value}")
                    liquid_properties_info += ", ".join(prop_list) + "\n"
        
        # Extract material/condition types for the prompt (using configurable column name)
        material_info = ""
        if material_col and material_col in df.columns:
            material_types_list = df[material_col].unique().tolist()
            material_types = ", ".join(material_types_list)
            material_info = f"\nMATERIAL TYPE: {material_types}"
            
            # Add detailed material properties if available
            if "material_properties" in config:
                material_info += "\nMATERIAL PROPERTIES:\n"
                for material_name in material_types_list:
                    if material_name in config["material_properties"]:
                        properties = config["material_properties"][material_name]
                        prop_desc = properties.get("description", "No description")
                        focus = properties.get("focus", "")
                        material_info += f"- {material_name}: {prop_desc}"
                        if focus:
                            material_info += f" | Focus: {focus}"
                        material_info += "\n"
                    else:
                        material_info += f"- {material_name}: Use general liquid handling principles\n"
            else:
                material_info += " - Consider relevant material properties"
            material_info += "\n"
        
        # Generate dynamic JSON format based on config parameters
        param_names = list(config["parameters"].keys())
        json_fields = ',\n            '.join([f'"{param}": value' for param in param_names])
        
        # Generate appropriate instructions based on whether we have data or not
        if is_initial_generation:
            instructions = f"""INSTRUCTIONS: 
1. Generate {batch_size} diverse initial parameter combinations based on the parameter ranges and guidelines
2. Use your understanding of liquid handling physics and the material properties
3. Consider the experimental setup and optimization objectives
4. Provide reasoning based on expected behavior and best practices"""
            reasoning_description = "physics-based explanation for parameter choice and expected behavior"
        else:
            # Generate dynamic optimization instructions based on metrics config
            metrics_instruction = "1. Identify which experiments performed best"
            if "metrics" in config:
                metrics_goals = []
                for metric, details in config["metrics"].items():
                    goal = details.get("goal", "minimize")
                    priority = details.get("priority", "medium")
                    if goal == "minimize":
                        metrics_goals.append(f"{goal} {metric}")
                    else:
                        metrics_goals.append(f"{goal} {metric}")
                
                if metrics_goals:
                    metrics_instruction = f"1. Identify which experiments performed best ({', '.join(metrics_goals)})"
            
            instructions = f"""INSTRUCTIONS: 
{metrics_instruction}
2. Spot patterns: what parameter values correlate with good/bad performance?
3. Recommend {batch_size} new parameter combinations based on your analysis
4. Reference specific data points in your reasoning"""
            reasoning_description = "data-driven explanation citing specific experimental results"
            
        prompt = f"""{system_message}

PARAMETERS (ranges): {', '.join([f"{param} {details['range']} {details.get('unit', '')}" for param, details in config["parameters"].items()])}
{material_info}
EXPERIMENTAL DATA TO ANALYZE:
{data_str}

{instructions}

JSON Response Format:
{{
    "recommendations": [
        {{
            {json_fields},
            "confidence": "high/medium/low",
            "reasoning": "{reasoning_description}",
            "expected_improvement": "quantified prediction based on observed patterns"
        }}
    ],
    "summary": "Key insights and strategy for parameter selection"
}}"""
        return prompt
    
    def call_llm(self, prompt: str, config: Dict[str, Any] = None, model: str = None) -> str:
        """
        Call LLM (OpenAI or Ollama) with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            config: Configuration dictionary (used for logging settings)
            model: Model to use (overrides default)
            
        Returns:
            Response from the LLM
        """
        if self.backend == "openai":
            return self._call_openai(prompt, config, model or "gpt-3.5-turbo")
        elif self.backend == "ollama":
            return self._call_ollama(prompt, config, model or self.ollama_model)
        else:
            raise ValueError(f"Invalid backend: {self.backend}")
    
    # Keep the old method name for backward compatibility
    def call_openai(self, prompt: str, config: Dict[str, Any] = None, model: str = "gpt-3.5-turbo") -> str:
        """Legacy method name for backward compatibility"""
        return self.call_llm(prompt, config, model)
    
    def _call_openai(self, prompt: str, config: Dict[str, Any] = None, model: str = "gpt-3.5-turbo") -> str:
        """
        Call OpenAI API with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            config: Configuration dictionary (used for logging settings)
            model: OpenAI model to use
            
        Returns:
            Response from the LLM
        """
        try:
            # Get model and API parameters from config if available
            api_config = config.get("api_settings", {}) if config else {}
            temperature = api_config.get("temperature", 0.7)
            max_tokens = api_config.get("max_tokens", 2000)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            response_content = response.choices[0].message.content
            
            # Save prompt and response to timestamped folder
            self._save_prompt_and_response(prompt, response_content, model, response, config)
            
            return response_content
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")
    
    def _call_ollama(self, prompt: str, config: Dict[str, Any] = None, model: str = None) -> str:
        """
        Call Ollama API with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            config: Configuration dictionary (used for logging settings)
            model: Ollama model to use
            
        Returns:
            Response from the LLM
        """
        try:
            model = model or self.ollama_model
            
            # Get API parameters from config if available
            api_config = config.get("api_settings", {}) if config else {}
            temperature = api_config.get("temperature", 0.7)
            
            # Ollama API payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            # Add max_tokens equivalent for Ollama if specified
            max_tokens = api_config.get("max_tokens")
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            print(f"🦙 Calling Ollama with model: {model}")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300  # 5 minute timeout for local generation
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error (status {response.status_code}): {response.text}")
            
            result = response.json()
            response_content = result.get("response", "")
            
            if not response_content:
                raise Exception("Empty response from Ollama")
            
            # Create a mock response object for logging compatibility
            mock_response = type('MockResponse', (), {
                'choices': [type('MockChoice', (), {'message': type('MockMessage', (), {'content': response_content})()})()],
                'usage': type('MockUsage', (), {
                    'prompt_tokens': len(prompt.split()),  # Rough estimate
                    'completion_tokens': len(response_content.split()),  # Rough estimate
                    'total_tokens': len(prompt.split()) + len(response_content.split())
                })(),
                'id': result.get('created_at', 'ollama-response'),
                'created': result.get('created_at', 'unknown')
            })()
            
            # Save prompt and response to timestamped folder
            self._save_prompt_and_response(prompt, response_content, model, mock_response, config)
            
            return response_content
            
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. Try a smaller model or increase timeout.")
        except Exception as e:
            raise Exception(f"Error calling Ollama API: {str(e)}")
    
    def _save_prompt_and_response(self, prompt: str, response_content: str, model: str, full_response, config: Dict[str, Any] = None):
        """
        Save the full prompt and response to timestamped files
        
        Args:
            prompt: The prompt sent to the LLM
            response_content: The response content from the LLM
            model: Model used
            full_response: Full API response object
            config: Configuration dictionary (used for logging settings)
        """
        from datetime import datetime
        import json
        
        # Get log directory from config, default to "LLM_prompts"
        log_base_dir = "LLM_prompts"
        if config and "logging" in config:
            log_base_dir = config["logging"].get("log_directory", "LLM_prompts")
        
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_base_dir, f"llm_session_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Save prompt
        prompt_file = os.path.join(log_dir, "prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n")
            f.write(prompt)
        
        # Save response content
        response_file = os.path.join(log_dir, "response.txt")
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n")
            f.write(response_content)
        
        # Save full API response metadata
        metadata_file = os.path.join(log_dir, "api_metadata.json")
        metadata = {
            "model": model,
            "timestamp": timestamp,
            "usage": {
                "prompt_tokens": full_response.usage.prompt_tokens,
                "completion_tokens": full_response.usage.completion_tokens,
                "total_tokens": full_response.usage.total_tokens
            },
            "response_id": full_response.id,
            "created": full_response.created
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 Prompt and response saved to: {log_dir}")
        
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response and extract recommendations
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Parsed dictionary containing recommendations
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: return the raw response
                return {
                    "recommendations": [],
                    "summary": response,
                    "raw_response": response
                }
        except json.JSONDecodeError as e:
            return {
                "recommendations": [],
                "summary": response,
                "raw_response": response,
                "parse_error": str(e)
            }
    
    def create_recommendations_csv(self, parsed_response: Dict[str, Any], output_path: str):
        """
        Create CSV file with recommendations
        
        Args:
            parsed_response: Parsed LLM response
            output_path: Path for the output CSV file
        """
        recommendations = parsed_response.get("recommendations", [])
        
        if not recommendations:
            # Create a simple CSV with the summary if no specific recommendations
            df = pd.DataFrame({
                "type": ["summary"],
                "content": [parsed_response.get("summary", "No recommendations provided")]
            })
        else:
            # Create DataFrame from recommendations - each row is a complete parameter set
            df = pd.DataFrame(recommendations)
            
            # Add a recommendation ID for each row
            df.insert(0, 'recommendation_id', range(1, len(df) + 1))
        
        df.to_csv(output_path, index=False)
        print(f"Recommendations saved to: {output_path}")
    
    def optimize(self, data, config, output_path: str, model: str = None):
        """
        Main optimization method
        
        Args:
            data: DataFrame with input data or path to CSV file
            config: Dictionary with configuration or path to JSON file
            output_path: Path for output CSV file
            model: OpenAI model to use (if None, will use config or default)
        """
        print("Loading data and configuration...")
        
        # Load input data and configuration
        if isinstance(data, str):
            df = self.load_csv(data)
        else:
            df = data
            
        if isinstance(config, str):
            config = self.load_config(config)
        else:
            config = config
        
        # Get model from config if not specified
        if model is None:
            if self.backend == "ollama":
                config_model = config.get("api_settings", {}).get("model", self.ollama_model)
                # If config specifies "online_server", use the actual selected model
                if config_model == "online_server":
                    model = self.ollama_model
                else:
                    model = config_model
            else:
                model = config.get("api_settings", {}).get("model", "gpt-3.5-turbo")
        
        print(f"Loaded {len(df)} rows of data")
        print(f"Using model: {model}")
        
        # Create optimization prompt
        prompt = self.create_optimization_prompt(df, config)
        
        print(f"Calling {self.backend.upper()} API...")
        
        # Call LLM API
        response = self.call_llm(prompt, config, model)
        
        print("Parsing response...")
        
        # Parse response
        parsed_response = self.parse_llm_response(response)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        
        # Create output CSV
        self.create_recommendations_csv(parsed_response, output_path)
        
        print("Optimization complete!")
        return parsed_response

def main():
    parser = argparse.ArgumentParser(description="LLM-based optimization tool")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    parser.add_argument("--output", required=True, help="Path for output CSV file")
    parser.add_argument("--model", help="Model to use (e.g., 'gpt-3.5-turbo' for OpenAI or 'gpt-oss:20b' for Ollama)")
    parser.add_argument("--api-key", help="OpenAI API key (optional if set in environment)")
    parser.add_argument("--backend", default="auto", choices=["openai", "ollama", "auto"], 
                       help="Backend to use: 'openai', 'ollama', or 'auto' (tries OpenAI first, falls back to Ollama)")
    parser.add_argument("--ollama-model", help="Ollama model to use (e.g., 'gpt-oss:20b', 'gemma3:1b')")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = LLMOptimizer(
        api_key=args.api_key,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url
    )
    
    # Load data and config
    data = optimizer.load_csv(args.csv)
    config = optimizer.load_config(args.config)
    
    # Run optimization
    result = optimizer.optimize(
        data=data,
        config=config,
        output_path=args.output,
        model=args.model
    )
    
    print("\nOptimization Summary:")
    print(result.get("summary", "No summary available"))

if __name__ == "__main__":
    main()
