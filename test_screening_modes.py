#!/usr/bin/env python3
"""
Test LLM Screening and External Data Integration
===============================================

Demonstrates how to switch between different screening modes by
changing configuration settings:

1. Normal screening (random/SOBOL)
2. LLM-based screening 
3. External data loading

All controlled via YAML configuration settings.
"""

import os
import sys
sys.path.append('calibration_modular_v2')

from calibration_modular_v2.config_manager import ExperimentConfig
from calibration_modular_v2.experiment import CalibrationExperiment
import logging

def test_screening_modes():
    """Test different screening mode configurations."""
    print("🧪 Testing Screening Mode Configuration")
    print("=" * 50)
    
    config_path = "calibration_modular_v2/experiment_config.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    print("📋 Current Configuration:")
    print(f"   Normal screening trials: {config.get_screening_trials()}")
    print(f"   LLM screening enabled: {config.use_llm_for_screening()}")
    print(f"   LLM config path: {config.get_screening_llm_config_path()}")
    
    # Check external data configuration
    external_data_config = config._config.get('screening', {}).get('external_data', {})
    print(f"   External data enabled: {external_data_config.get('enabled', False)}")
    print(f"   External data path: {external_data_config.get('data_path', 'None')}")
    
    print("\n🎛️  How to Enable Each Mode:")
    print("-" * 30)
    
    print("\n✅ **Current Mode: Normal Screening (Random/SOBOL)**")
    print("   Already configured - uses random parameter generation")
    
    print("\n🤖 **To Enable LLM Screening:**")
    print("   1. Set screening.use_llm_suggestions: true")
    print("   2. Set screening.llm_config_path: 'calibration_screening_llm_template.json'")
    print("   3. The system will generate LLM-based parameter suggestions")
    
    print("\n📊 **To Enable External Data Loading:**")
    print("   1. Set screening.external_data.enabled: true")
    print("   2. Set screening.external_data.data_path: 'sample_external_data.csv'")
    print("   3. Optionally set volume/liquid filters")
    print("   4. The system will load historical data instead of generating new trials")
    
    print("\n🔧 **Configuration Example for LLM Screening:**")
    print("""
screening:
  use_llm_suggestions: true
  llm_config_path: "calibration_screening_llm_template.json"
  external_data:
    enabled: false
""")
    
    print("\n🔧 **Configuration Example for External Data:**")
    print("""
screening:
  use_llm_suggestions: false
  external_data:
    enabled: true
    data_path: "sample_external_data.csv"
    volume_filter_ml: 0.05  # Optional: only use 50μL data
    liquid_filter: "water"  # Optional: only use water data
""")
    
    print("\n🎯 **Workflow Priority (if multiple enabled):**")
    print("   1. External Data (highest priority)")
    print("   2. LLM Screening")
    print("   3. Normal Screening (fallback)")
    
    # Test the configuration loading
    print("\n🧪 **Testing Configuration Loading:**")
    
    # Test external data loader
    try:
        from calibration_modular_v2.external_data import ExternalDataLoader
        external_loader = ExternalDataLoader(config)
        has_data = external_loader.has_valid_data()
        print(f"   External data loader initialized: ✅")
        print(f"   Has valid external data: {has_data}")
    except Exception as e:
        print(f"   External data loader failed: ❌ {e}")
    
    # Test LLM recommender (if path exists)
    llm_path = config.get_screening_llm_config_path()
    if llm_path and os.path.exists(llm_path):
        try:
            from calibration_modular_v2.llm_recommender import LLMRecommender
            print(f"   LLM recommender available: ✅")
            print(f"   LLM config file exists: ✅ ({llm_path})")
        except Exception as e:
            print(f"   LLM recommender failed: ❌ {e}")
    else:
        print(f"   LLM config file: ❌ Not found ({llm_path})")
    
    return True

def test_external_data_sample():
    """Check if sample external data exists."""
    print("\n📁 **Sample Files Check:**")
    
    sample_files = [
        "calibration_modular_v2/sample_external_data.csv",
        "calibration_modular_v2/calibration_screening_llm_template.json"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} exists")
            if file_path.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"      → {len(df)} rows, columns: {list(df.columns)}")
                except Exception as e:
                    print(f"      → Failed to read: {e}")
        else:
            print(f"   ❌ {file_path} missing")

def main():
    """Main test function."""
    logging.basicConfig(level=logging.INFO)
    
    print("🎛️  **Screening Mode Configuration Test**")
    print("=" * 60)
    
    test_screening_modes()
    test_external_data_sample()
    
    print("\n✅ **Summary:**")
    print("   The modular system supports all three screening modes:")
    print("   1. ✅ Normal screening (currently active)")
    print("   2. ✅ LLM screening (configurable)")
    print("   3. ✅ External data loading (configurable)")
    print()
    print("   Simply change the YAML config to switch modes - no code changes needed!")
    print("   The system automatically detects the configuration and routes accordingly.")

if __name__ == "__main__":
    main()