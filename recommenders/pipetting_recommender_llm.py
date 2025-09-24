"""
LLM-based Pipetting Recommender for UToronto Demo

This module provides a clean interface to get AI-powered pipetting recommendations
for integration into the UToronto automation demo.

Usage:
    recommender = PipettingRecommenderLLM()
    recommendations = recommender.get_new_recs("path/to/experiment_data.csv")
"""

import os
import sys
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the llm_optimizer source to path
# The llm_optimizer.py is in the same directory (recommenders folder)
try:
    from llm_optimizer import LLMOptimizer
except ImportError:
    # Try importing from the same directory
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from llm_optimizer import LLMOptimizer


class PipettingRecommenderLLM:
    """
    LLM-based pipetting parameter recommender for UToronto demo.
    
    Provides AI-powered recommendations for liquid handling optimization
    based on experimental data.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the LLM-based recommender.
        
        Args:
            api_key: OpenAI API key. If None, will try to load from environment
            config_path: Path to custom config file. If None, uses default UToronto config
        """
        self.optimizer = LLMOptimizer(api_key=api_key)
        self.config_path = config_path
        self._config = None
        
        print("🤖 LLM Pipetting Recommender initialized")
        
    def _get_config(self) -> Dict[str, Any]:
        """Get the configuration for optimization."""
        if self._config is None:
            if self.config_path and os.path.exists(self.config_path):
                # Load custom config file
                self._config = self.optimizer.load_config(self.config_path)
                print(f"📄 Loaded custom config from: {self.config_path}")
            else:
                # Use default UToronto demo configuration
                self._config = self._get_default_utoronto_config()
                print("⚙️ Using default UToronto demo configuration")
        
        return self._config
    
    def _get_default_utoronto_config(self) -> Dict[str, Any]:
        """
        Default configuration optimized for UToronto demo environment.
        
        Returns:
            Dictionary containing UToronto-specific optimization configuration
        """
        return {
            "system_message": (
                "You are an expert liquid handling optimization consultant for the UToronto "
                "Acceleration Consortium demo. Your task is to analyze experimental pipetting data "
                "and provide specific, actionable recommendations for improving pipetting performance. "
                "Focus on balancing accuracy, precision, and speed for demo reliability. "
                "Consider the physical properties of liquids and provide clear reasoning."
            ),
            
            "model": "gpt-4",
            "temperature": 0.7,
            "batch_size": 3,  # 3 recommendations for demo
            
            "optimization_guidelines": {
                "demo_reliability": True,
                "avoid_extreme_parameters": True,
                "prioritize_consistency": True,
                "safety_first": True,
                "avoid_previous_combinations": True,
                "consider_physical_meaning": True
            },
            
            "parameters": {
                "aspirate_speed": {
                    "type": "continuous",
                    "unit": "μL/s",
                    "range": [5, 35],
                    "description": "Speed for liquid aspiration operations",
                    "safety_limit": 40
                },
                "dispense_speed": {
                    "type": "continuous",
                    "unit": "μL/s",
                    "range": [5, 35],
                    "description": "Speed for liquid dispensing operations",
                    "safety_limit": 40
                },
                "aspirate_wait_time": {
                    "type": "continuous",
                    "unit": "seconds",
                    "range": [0.0, 30.0],
                    "description": "Wait time after aspiration before tip movement",
                    "safety_limit": 35.0
                },
                "dispense_wait_time": {
                    "type": "continuous",
                    "unit": "seconds",
                    "range": [0.0, 30.0],
                    "description": "Wait time after dispensing before tip movement",
                    "safety_limit": 35.0
                },
                "retract_speed": {
                    "type": "continuous",
                    "unit": "μL/s",
                    "range": [1.0, 15.0],
                    "description": "Speed for retracting tip from liquid surface",
                    "safety_limit": 20.0
                },
                "pre_asp_air_vol": {
                    "type": "continuous",
                    "unit": "mL",
                    "range": [0.0, 0.1],
                    "description": "Air volume aspirated before liquid to prevent contamination",
                    "safety_limit": 0.15
                },
                "post_asp_air_vol": {
                    "type": "continuous",
                    "unit": "mL",
                    "range": [0.0, 0.1],
                    "description": "Air volume aspirated after liquid to prevent dripping",
                    "safety_limit": 0.15
                },
                "overaspirate_vol": {
                    "type": "continuous",
                    "unit": "mL",
                    "range": [0.0, 0.125],
                    "description": "Extra volume aspirated beyond target to compensate for retention",
                    "safety_limit": 0.25
                }
            },
            
            "metrics": {
                "volume_deviation": {
                    "goal": "minimize",
                    "unit": "% deviation",
                    "description": "Volume accuracy as percentage deviation from target"
                },
                "variability": {
                    "goal": "minimize", 
                    "unit": "% relative std",
                    "description": "Precision as relative standard deviation"
                },
                "time": {
                    "goal": "minimize",
                    "unit": "seconds",
                    "description": "Total time for pipetting operation"
                }
            },
            
            "utoronto_demo_specific": {
                "target_volume_ul": 100,
                "device_name": "UToronto Demo Pipettor",
                "demo_environment": True,
                "reliability_priority": "high",
                "max_experiment_time": 300
            }
        }
    
    def get_new_recs(self, existing_data_file_ref: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Get new pipetting recommendations based on existing experimental data.
        
        Args:
            existing_data_file_ref: Path to CSV file containing experimental data
            output_file: Optional path to save recommendations CSV. If None, auto-generates filename
            
        Returns:
            Dictionary containing recommendations and metadata
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data file format is invalid
        """
        print(f"🔍 Analyzing experimental data: {existing_data_file_ref}")
        
        # Validate input file
        if not os.path.exists(existing_data_file_ref):
            raise FileNotFoundError(f"Data file not found: {existing_data_file_ref}")
        
        try:
            # Load the experimental data
            data = self.optimizer.load_csv(existing_data_file_ref)
            print(f"📊 Loaded {len(data)} experimental data points")
            
            # Get configuration
            config = self._get_config()
            
            # Generate output filename if not provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"llm_recommendations_{timestamp}.csv"
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            print("🧠 Generating AI-powered recommendations...")
            
            # Get recommendations from LLM
            result = self.optimizer.optimize(
                data=data,
                config=config, 
                output_path=output_file
            )
            
            # Parse recommendations for easy access
            recommendations = []
            if result and 'recommendations' in result:
                recommendations = result['recommendations']
            
            print(f"✅ Generated {len(recommendations)} recommendations")
            print(f"💾 Saved to: {output_file}")
            
            # Return structured result
            return {
                "success": True,
                "num_recommendations": len(recommendations),
                "recommendations": recommendations,
                "summary": result.get('summary', 'No summary available'),
                "output_file": output_file,
                "input_data_points": len(data),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the optimization configuration.
        
        Args:
            config_updates: Dictionary of configuration parameters to update
        """
        if self._config is None:
            self._config = self._get_default_utoronto_config()
        
        # Deep update of configuration
        self._deep_update(self._config, config_updates)
        print("⚙️ Configuration updated")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing key configuration parameters
        """
        config = self._get_config()
        return {
            "model": config.get("model", "unknown"),
            "batch_size": config.get("batch_size", "unknown"),
            "num_parameters": len(config.get("parameters", {})),
            "optimization_guidelines": list(config.get("optimization_guidelines", {}).keys()),
            "demo_specific": config.get("utoronto_demo_specific", {})
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the PipettingRecommenderLLM class.
    This shows how to integrate it into your UToronto demo.
    """
    
    # Initialize the recommender
    recommender = PipettingRecommenderLLM()
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    summary = recommender.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 Recommender ready for integration!")
    print("\nUsage in your demo:")
    print("  result = recommender.get_new_recs('path/to/your/data.csv')")
    print("  recommendations = result['recommendations']")