#!/usr/bin/env python3
"""
XGBoost Installation Diagnostic
==============================

Check current XGBoost installation state and provide fix recommendations.
"""

import subprocess
import sys

def check_installation_method():
    """Check how XGBoost was installed - conda vs pip."""
    print("=== XGBoost Installation Diagnosis ===")
    
    try:
        # Check conda list
        print("1. Checking conda installations...")
        result = subprocess.run(['conda', 'list', 'xgboost'], capture_output=True, text=True)
        if result.returncode == 0 and 'xgboost' in result.stdout:
            print("   ✓ XGBoost found in conda:")
            for line in result.stdout.strip().split('\n')[2:]:  # Skip header
                if 'xgboost' in line:
                    print(f"     {line}")
        else:
            print("   ❌ XGBoost not found in conda")
        
        # Check pip list  
        print("\n2. Checking pip installations...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            xgb_lines = [line for line in result.stdout.split('\n') if 'xgboost' in line.lower()]
            if xgb_lines:
                print("   ✓ XGBoost found in pip:")
                for line in xgb_lines:
                    print(f"     {line}")
            else:
                print("   ❌ XGBoost not found in pip")
        
        # Check what python thinks it has
        print("\n3. Checking what Python sees...")
        try:
            import xgboost as xgb
            print(f"   ✓ XGBoost importable: version {xgb.__version__}")
            print(f"   ✓ Install path: {xgb.__file__}")
        except Exception as e:
            print(f"   ❌ XGBoost import failed: {e}")
            
    except Exception as e:
        print(f"Error checking installations: {e}")

def get_fix_recommendations():
    """Provide specific fix recommendations."""
    print("\n=== Fix Recommendations ===")
    
    print("The safest approach is usually:")
    print("1. Remove ALL xgboost installations")
    print("2. Install with ONE method only")
    print()
    
    print("🎯 RECOMMENDED FIX:")
    print("   conda remove xgboost --force")
    print("   pip uninstall xgboost")
    print("   conda install xgboost")
    print()
    
    print("Alternative (pip only):")
    print("   conda remove xgboost --force") 
    print("   pip uninstall xgboost")
    print("   pip install xgboost")
    print()
    
    print("⚠️  BEFORE DOING ANYTHING:")
    print("   conda list > backup_packages.txt")
    print("   (Save your package list in case something breaks)")

if __name__ == "__main__":
    check_installation_method()
    get_fix_recommendations()