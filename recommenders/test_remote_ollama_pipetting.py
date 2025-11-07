#!/usr/bin/env python3
"""
Test script to generate 5 pipetting parameter suggestions using the remote Ollama server.
"""

import sys
import os
import pandas as pd

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_optimizer import LLMOptimizer

def create_sample_pipetting_data():
    """Create sample calibration data for testing."""
    sample_data = {
        'liquid': ['water', 'water', 'glycerol', 'glycerol', 'ethanol', 'ethanol'],
        'aspirate_speed': [15, 25, 20, 30, 12, 18],
        'dispense_speed': [15, 25, 20, 30, 12, 18],
        'aspirate_wait_time': [1.0, 5.0, 10.0, 15.0, 0.5, 2.0],
        'dispense_wait_time': [1.0, 5.0, 10.0, 15.0, 0.5, 2.0],
        'retract_speed': [5.0, 8.0, 6.0, 4.0, 10.0, 7.0],
        'blowout_vol': [0.02, 0.05, 0.08, 0.10, 0.01, 0.03],
        'post_asp_air_vol': [0.02, 0.04, 0.06, 0.08, 0.01, 0.03],
        'overaspirate_vol': [0.005, 0.010, 0.015, 0.020, 0.002, 0.008],
        'deviation': [2.1, 4.5, 3.2, 6.8, 1.8, 3.9],  # % error
        'variability': [1.5, 3.2, 2.8, 5.1, 1.2, 2.6],  # % CV
        'time': [12.3, 18.7, 25.4, 32.1, 9.8, 15.2]  # seconds
    }
    return pd.DataFrame(sample_data)

def test_remote_ollama_pipetting(liquid_type="water"):
    """Test the remote Ollama integration for initial pipetting parameter generation."""
    print(f"=== Testing Remote Ollama for Initial Pipetting Parameters ({liquid_type}) ===")
    
    try:
        # Initialize optimizer with ollama backend and online_server model
        print("🔧 Initializing LLM Optimizer with remote Ollama...")
        optimizer = LLMOptimizer(
            backend="ollama",
            ollama_model="online_server"  # This will trigger remote server usage
        )
        
        # Create empty DataFrame for initial suggestions with specified liquid
        print(f"📊 Using empty dataset for initial parameter generation (liquid: {liquid_type})...")
        empty_data = pd.DataFrame({'liquid': [liquid_type]})  # Specify the liquid type
        print(f"Data shape: {empty_data.shape} (initial generation mode for {liquid_type})")
        
        # Load config
        print("\n⚙️  Loading calibration config...")
        config_path = os.path.join(os.path.dirname(__file__), "calibration_initial_config.json")
        config = optimizer.load_config(config_path)
        print(f"Config loaded. Batch size: {config['batch_size']} initial suggestions")
        
        # Generate initial recommendations
        print(f"\n🤖 Generating initial pipetting parameter recommendations for {liquid_type}...")
        print("This may take a moment as we query the remote Ollama server...")
        
        output_path = f"initial_pipetting_recommendations_{liquid_type}.csv"
        result = optimizer.optimize(
            data=empty_data,
            config=config,
            output_path=output_path
        )
        
        print("\n✅ Initial generation complete!")
        print(f"\nSummary: {result.get('summary', 'No summary available')}")
        
        # Show recommendations if available
        if 'recommendations' in result and result['recommendations']:
            print(f"\n📋 Generated {len(result['recommendations'])} initial parameter sets for {liquid_type}:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n--- Initial Set {i} ---")
                print(f"Confidence: {rec.get('confidence', 'unknown')}")
                print(f"Reasoning: {rec.get('reasoning', 'No reasoning provided')[:150]}...")
                
                # Show key parameter values
                key_params = ['aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time']
                param_values = []
                for param in key_params:
                    if param in rec:
                        param_values.append(f"{param}={rec[param]}")
                if param_values:
                    print(f"Key Parameters: {', '.join(param_values)}")
        
        print(f"\n📁 Full initial recommendations saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("\nPossible issues:")
        print("- Remote Ollama server is not accessible")
        print("- Network connectivity problems")
        print("- Server is not running any models")
        return False

if __name__ == "__main__":
    # Test with glycerol (change to "water" for water test)
    liquid_to_test = "4%_hyaluronic_acid_water"  # Change this to "water" for water test
    
    print(f"🧪 Testing with {liquid_to_test.upper()}...")
    success = test_remote_ollama_pipetting(liquid_to_test)
    
    if success:
        print(f"\n🎉 Remote Ollama integration test successful for {liquid_to_test}!")
    else:
        print(f"\n💥 Remote Ollama integration test failed for {liquid_to_test}!")
        sys.exit(1)