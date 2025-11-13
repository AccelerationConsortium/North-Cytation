#!/usr/bin/env python3
"""
Quick test of parameter generation using the full CSV generation.
"""

import sys
import os

# Just test by running one of the full workflows and checking the output
def test_csv_generation():
    """Generate a few trials and check if parameters are diverse."""
    print("🧪 Testing Parameter Generation via CSV Export")
    print("=" * 50)
    
    # Import and run test
    from test_complete_workflow import run_quick_test
    
    print("Running workflow test to check parameter diversity...")
    try:
        result = run_quick_test()
        if result:
            print("✅ Workflow test completed successfully")
            
            # Check if output CSV was created  
            import os
            if os.path.exists("calibration_modular_v2/data/multi_volume_calibration_test_results.csv"):
                print("📊 Checking generated parameters in CSV...")
                
                import pandas as pd
                df = pd.read_csv("calibration_modular_v2/data/multi_volume_calibration_test_results.csv")
                
                print(f"Generated {len(df)} trials")
                if len(df) >= 2:
                    # Check if parameters vary
                    varying_params = []
                    for col in df.columns:
                        if col.startswith(('aspirate_', 'dispense_', 'plunger_', 'mix_')):
                            if len(set(df[col].round(6))) > 1:
                                varying_params.append(col)
                    
                    if varying_params:
                        print(f"✅ Parameters showing variation: {varying_params}")
                        return True
                    else:
                        print("❌ All parameters appear identical")
                        return False
                else:
                    print("⚠️  Not enough trials generated to check diversity")
                    return False
            else:
                print("❌ No CSV output found")
                return False
        else:
            print("❌ Workflow test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_csv_generation()
    exit(0 if success else 1)