#!/usr/bin/env python3
"""
Quick Start Script Template for North Robotics Workflows

This script helps you quickly set up and run new experiments.
Copy this file and modify it for your specific workflow.

Usage:
    python quick_start_template.py --help
    python quick_start_template.py --simulate  # Test mode
    python quick_start_template.py --run       # Real experiment
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the utoronto_demo directory to the Python path
sys.path.append("../utoronto_demo")

def setup_experiment_directory(project_name: str):
    """Create a new experiment directory with proper structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{project_name}_{timestamp}")
    
    # Create directory structure
    directories = [
        exp_dir / "config",
        exp_dir / "data" / "raw", 
        exp_dir / "data" / "processed",
        exp_dir / "data" / "plots",
        exp_dir / "protocols",
        exp_dir / "scripts"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    print(f"📁 Created experiment directory: {exp_dir}")
    return exp_dir

def copy_templates(exp_dir: Path):
    """Copy template files to the new experiment directory."""
    import shutil
    
    # Copy templates
    templates = [
        ("workflow_template.py", exp_dir / "scripts" / "workflow.py"),
        ("vial_template.csv", exp_dir / "config" / "vials.csv"),
        ("config_template.json", exp_dir / "config" / "parameters.json"),
        ("PROJECT_TEMPLATE.md", exp_dir / "README.md")
    ]
    
    for src, dst in templates:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"📋 Copied template: {src} → {dst}")
        else:
            print(f"⚠️  Template not found: {src}")

def validate_setup():
    """Validate that the system is ready for experiments."""
    checks = []
    
    # Check Python imports
    try:
        from master_usdl_coordinator import Lash_E
        checks.append(("✅", "Lash_E import"))
    except ImportError as e:
        checks.append(("❌", f"Lash_E import failed: {e}"))
    
    # Check critical directories
    critical_dirs = ["../utoronto_demo/status", "../utoronto_demo/workflows"]
    for dir_path in critical_dirs:
        if os.path.exists(dir_path):
            checks.append(("✅", f"Directory exists: {dir_path}"))
        else:
            checks.append(("❌", f"Missing directory: {dir_path}"))
    
    # Check for protocol files (if applicable)
    protocol_dir = "C:\\Protocols"
    if os.path.exists(protocol_dir):
        protocols = list(Path(protocol_dir).glob("*.prt"))
        checks.append(("✅", f"Found {len(protocols)} Cytation protocols"))
    else:
        checks.append(("⚠️", f"Protocol directory not found: {protocol_dir}"))
    
    print("\n🔍 System Validation:")
    for status, message in checks:
        print(f"   {status} {message}")
    
    return all(status == "✅" for status, _ in checks)

def run_simulation_test():
    """Run a quick simulation test to verify the system works."""
    print("\n🧪 Running simulation test...")
    
    try:
        # Import the workflow template
        sys.path.append("./workflows")
        from workflow_template import your_workflow_name
        
        # Run a quick test
        result = your_workflow_name(param1=0.1, param2=2, simulate=True)
        print("✅ Simulation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

def interactive_setup():
    """Interactive setup wizard for new projects."""
    print("\n🚀 North Robotics Experiment Setup Wizard")
    print("=" * 50)
    
    # Get project information
    project_name = input("Project name (no spaces): ").strip().replace(" ", "_")
    if not project_name:
        project_name = f"experiment_{datetime.now().strftime('%Y%m%d')}"
    
    author = input("Author name: ").strip()
    description = input("Brief description: ").strip()
    
    # Get experimental parameters
    print("\n📊 Experimental Parameters:")
    try:
        num_samples = int(input("Number of samples (default: 6): ") or "6")
        replicates = int(input("Replicates per sample (default: 3): ") or "3")
        volume_ml = float(input("Sample volume in mL (default: 0.5): ") or "0.5")
    except ValueError:
        print("Invalid input, using defaults...")
        num_samples, replicates, volume_ml = 6, 3, 0.5
    
    # Use temperature controller?
    use_temp = input("Use temperature controller? (y/n, default: n): ").lower().startswith('y')
    temp_celsius = 25.0
    if use_temp:
        try:
            temp_celsius = float(input("Target temperature (°C, default: 25): ") or "25")
        except ValueError:
            temp_celsius = 25.0
    
    # Use photoreactor?
    use_photo = input("Use photoreactor? (y/n, default: n): ").lower().startswith('y')
    
    print(f"\n📋 Setup Summary:")
    print(f"   Project: {project_name}")
    print(f"   Author: {author}")
    print(f"   Samples: {num_samples} × {replicates} replicates")
    print(f"   Volume: {volume_ml} mL per sample")
    print(f"   Temperature: {temp_celsius}°C" if use_temp else "   Temperature: Not used")
    print(f"   Photoreactor: {'Yes' if use_photo else 'No'}")
    
    confirm = input("\nProceed with setup? (y/n): ").lower().startswith('y')
    if not confirm:
        print("Setup cancelled.")
        return None
    
    # Create experiment directory
    exp_dir = setup_experiment_directory(project_name)
    copy_templates(exp_dir)
    
    # Update configuration with user inputs
    config_path = exp_dir / "config" / "parameters.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update with user inputs
        config["project_info"]["name"] = project_name
        config["project_info"]["author"] = author
        config["project_info"]["description"] = description
        config["project_info"]["date_created"] = datetime.now().strftime("%Y-%m-%d")
        
        config["experimental_parameters"]["replication"]["technical_replicates"] = replicates
        config["experimental_parameters"]["volumes"]["transfer_volume_mL"] = volume_ml
        config["experimental_parameters"]["processing"]["temperature_celsius"] = temp_celsius
        
        config["instrument_settings"]["temperature_controller"]["enabled"] = use_temp
        config["instrument_settings"]["temperature_controller"]["target_temperature_celsius"] = temp_celsius
        config["instrument_settings"]["photoreactor"]["enabled"] = use_photo
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"\n✅ Experiment setup complete!")
    print(f"📁 Directory: {exp_dir}")
    print(f"📝 Next steps:")
    print(f"   1. Edit {exp_dir}/config/vials.csv with your vial setup")
    print(f"   2. Modify {exp_dir}/scripts/workflow.py for your specific needs")
    print(f"   3. Test with: cd {exp_dir} && python scripts/workflow.py --simulate")
    
    return exp_dir

def main():
    parser = argparse.ArgumentParser(
        description="North Robotics Quick Start - Setup and run experiments easily"
    )
    parser.add_argument("--setup", action="store_true", 
                       help="Run interactive setup wizard")
    parser.add_argument("--validate", action="store_true",
                       help="Validate system setup")
    parser.add_argument("--test", action="store_true",
                       help="Run simulation test")
    parser.add_argument("--simulate", action="store_true",
                       help="Run workflow in simulation mode")
    parser.add_argument("--run", action="store_true", 
                       help="Run actual experiment")
    
    args = parser.parse_args()
    
    if args.setup:
        interactive_setup()
    elif args.validate:
        validate_setup()
    elif args.test:
        run_simulation_test()
    elif args.simulate:
        print("🔬 Running workflow in simulation mode...")
        # Add your workflow import and call here
        pass
    elif args.run:
        print("⚠️  Running REAL experiment - make sure everything is ready!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() == "yes":
            print("🤖 Starting real experiment...")
            # Add your workflow import and call here
        else:
            print("Experiment cancelled.")
    else:
        print("🚀 North Robotics Quick Start")
        print("\nUsage:")
        print("  python quick_start_template.py --setup     # Interactive setup wizard")
        print("  python quick_start_template.py --validate  # Check system status")  
        print("  python quick_start_template.py --test      # Run simulation test")
        print("  python quick_start_template.py --simulate  # Test your workflow")
        print("  python quick_start_template.py --run       # Run real experiment")

if __name__ == "__main__":
    main()