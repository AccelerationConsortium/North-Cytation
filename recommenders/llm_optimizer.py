import pandas as pd
from openai import OpenAI
import json
import os
from typing import Dict, Any, List
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMOptimizer:
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM Optimizer
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
        """
        if api_key and api_key != "demo-key":
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)
    
    def list_data_files(self, data_folder: str = "../data_analysis/data") -> list:
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
        system_message = config.get("system_message", "You are an optimization expert.")
        
        # Convert DataFrame to string representation
        data_str = df.to_string(index=False)
        
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
        
        # Extract liquid-specific information if available
        liquid_info = ""
        if "liquid_handling_specific" in config:
            lh_config = config["liquid_handling_specific"]
            liquid_info = f"\n\nLiquid Handling Configuration:\n"
            liquid_info += f"- Target volume: {lh_config.get('target_volume_ul', 'not specified')} μL\n"
            liquid_info += f"- Device: {lh_config.get('device_serial', 'not specified')}\n"
            liquid_info += f"- Iterations per run: {lh_config.get('iterations_per_run', 'not specified')}\n"
            liquid_info += f"- Encoding method: {lh_config.get('encoding_method', 'not specified')}\n"
        
        # Add liquid type information
        liquid_type_info = ""
        if "liquid" in df.columns:
            liquid_types = df["liquid"].unique().tolist()
            liquid_type_info = f"\n\nLiquid Type Information:\n"
            liquid_type_info += f"- Liquid types in data: {liquid_types}\n"
            liquid_type_info += f"- IMPORTANT: Keep the SAME liquid type as in the input data - do not change the liquid type\n"
            liquid_type_info += f"- ANALYZE liquid properties and adjust recommendations accordingly:\n"
            liquid_type_info += f"  * For glycerol_30%: Consider higher viscosity, slower flow rates, longer settling times\n"
            liquid_type_info += f"  * For water: Standard pipetting parameters, moderate flow rates\n"
            liquid_type_info += f"  * For ethanol: Consider volatility, moderate flow rates, appropriate air volumes\n"
            liquid_type_info += f"  * For toluene: Consider density and surface tension effects\n"
            liquid_type_info += f"  * Use your knowledge of liquid properties to optimize parameters for the specific liquid type\n"
        
        
        # Extract liquid properties if available
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
        
        prompt = f"""
{system_message}
{parameters_info}{metrics_info}{liquid_info}{liquid_properties_info}{liquid_type_info}{guidelines_info}

Current experimental data:
{data_str}

Please analyze this data and provide optimization recommendations. 
Consider the parameter definitions, ranges, and safety limits provided above.
Follow the optimization guidelines strictly.

Return your response in JSON format with the following structure:
{{
    "recommendations": [
        {{
            "aspirate_speed": "recommended_value",
            "dispense_speed": "recommended_value", 
            "aspirate_wait_time": "recommended_value",
            "dispense_wait_time": "recommended_value",
            "retract_speed": "recommended_value",
            "pre_asp_air_vol": "recommended_value",
            "post_asp_air_vol": "recommended_value",
            "overaspirate_vol": "recommended_value",
            "confidence": "high/medium/low",
            "reasoning": "explanation for this parameter combination",
            "expected_improvement": "quantified improvement expected"
        }}
    ],
    "summary": "Overall optimization strategy and key insights"
}}

CRITICAL REQUIREMENTS:
- You MUST provide exactly {batch_size} complete parameter combinations
- Each combination must include ALL parameters: aspirate_speed, dispense_speed, aspirate_wait_time, dispense_wait_time, retract_speed, pre_asp_air_vol, post_asp_air_vol, overaspirate_vol
- Each row in the recommendations array should be a complete parameter set for one experiment
- Use your general knowledge about liquid handling to optimize the parameters
- For liquid: Use the EXACT liquid type from the input data - do not change the liquid type
- Ensure all suggested values are within the defined parameter ranges
- Avoid suggesting combinations that appear in the current data
- Consider physical/chemical feasibility of all recommendations
- Prioritize safety and equipment capabilities

EXAMPLE FORMAT (this is what you must return):
{{
    "recommendations": [
        {{
            "aspirate_speed": 15,
            "dispense_speed": 20,
            "aspirate_wait_time": 2.5,
            "dispense_wait_time": 1.0,
            "retract_speed": 8.0,
            "pre_asp_air_vol": 0.05,
            "post_asp_air_vol": 0.02,
            "overaspirate_vol": 0.01,
            "confidence": "high",
            "reasoning": "This combination optimizes for speed while maintaining accuracy",
            "expected_improvement": "20% faster with maintained precision"
        }}
    ],
    "summary": "Optimization strategy focused on balanced speed and accuracy"
}}

Please provide specific, actionable recommendations based on the data provided and the parameter constraints.


OPTIMIZATION GUIDANCE:
- Focus on balancing accuracy, precision, and speed
- Consider the trade-offs between different parameters
- CRITICAL: Analyze the liquid type properties and adjust recommendations accordingly
- For glycerol: Use slower flow rates, longer settling times due to higher viscosity
- For water: Use standard pipetting parameters with moderate flow rates
- For ethanol: Consider volatility effects on air volume settings
- For toluene: Consider density and surface tension effects on pipetting
- Use your knowledge of liquid handling principles to optimize the parameters
- Ensure all recommendations are within the defined safety limits
"""
        return prompt
    
    def call_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Call OpenAI API with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            model: OpenAI model to use
            
        Returns:
            Response from the LLM
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")
    
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
    
    def optimize(self, data, config, output_path: str, model: str = "gpt-3.5-turbo"):
        """
        Main optimization method
        
        Args:
            data: DataFrame with input data or path to CSV file
            config: Dictionary with configuration or path to JSON file
            output_path: Path for output CSV file
            model: OpenAI model to use
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
        
        print(f"Loaded {len(df)} rows of data")
        
        # Create optimization prompt
        prompt = self.create_optimization_prompt(df, config)
        
        print("Calling OpenAI API...")
        
        # Call OpenAI API
        response = self.call_openai(prompt, model)
        
        print("Parsing response...")
        
        # Parse response
        parsed_response = self.parse_llm_response(response)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create output CSV
        self.create_recommendations_csv(parsed_response, output_path)
        
        print("Optimization complete!")
        return parsed_response

def main():
    parser = argparse.ArgumentParser(description="LLM-based optimization tool")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    parser.add_argument("--output", required=True, help="Path for output CSV file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--api-key", help="OpenAI API key (optional if set in environment)")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = LLMOptimizer(api_key=args.api_key)
    
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
