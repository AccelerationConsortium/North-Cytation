#!/usr/bin/env python3
"""
Debug script to test LLM material properties configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommenders import llm_optimizer as llm_opt
import json

def debug_config():
    print("🔍 Debugging LLM Config...")
    
    # Initialize LLM optimizer
    optimizer = llm_opt.LLMOptimizer(backend="ollama", ollama_model="online_server")
    
    # Load config
    config_path = "recommenders/calibration_initial_config.json"
    config = optimizer.load_config(config_path)
    
    print(f"\n📄 Config loaded from: {config_path}")
    
    # Check experimental setup
    if "experimental_setup" in config:
        print(f"\n🧪 Experimental setup: {config['experimental_setup']}")
    else:
        print("\n❌ No experimental_setup in config")
    
    # Check material properties
    if "material_properties" in config:
        print(f"\n🧬 Material properties: {json.dumps(config['material_properties'], indent=2)}")
        if "water" in config["material_properties"]:
            print(f"\n💧 Water properties: {config['material_properties']['water']}")
        else:
            print("\n❌ No 'water' in material_properties")
    else:
        print("\n❌ No material_properties in config")
    
    # Test setting current liquid
    config['experimental_setup']['current_liquid'] = 'water'
    print(f"\n✅ Set current_liquid to 'water'")
    print(f"Updated experimental_setup: {config['experimental_setup']}")
    
    # Test the logic that generates material info
    current_liquid = config['experimental_setup']['current_liquid']
    print(f"\n🔍 Testing lookup for '{current_liquid}' (lowercase: '{current_liquid.lower()}')")
    
    if current_liquid.lower() in config['material_properties']:
        properties = config['material_properties'][current_liquid.lower()]
        print(f"✅ Found properties: {properties}")
    else:
        print(f"❌ '{current_liquid.lower()}' not found in material_properties")
        print(f"Available keys: {list(config['material_properties'].keys())}")

if __name__ == "__main__":
    debug_config()