#!/usr/bin/env python3
"""
Test SHAP Analysis on Existing Data
==================================

This script tests the SHAP analysis functionality to diagnose what's wrong
with the XGBoost/SHAP libraries.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_shap_imports():
    """Test if SHAP and XGBoost can be imported properly."""
    print("=== Testing SHAP/XGBoost Imports ===")
    
    try:
        print("1. Testing basic imports...")
        import shap
        print("   ✓ shap imported successfully")
        
        import xgboost as xgb
        print("   ✓ xgboost imported successfully")
        
        from sklearn.preprocessing import StandardScaler
        print("   ✓ sklearn imported successfully")
        
        print("\n2. Testing XGBRegressor instantiation...")
        model = xgb.XGBRegressor()
        print("   ✓ XGBRegressor created successfully")
        
        print("\n3. Testing basic model training...")
        # Create dummy data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model.fit(X, y)
        print("   ✓ Model training successful")
        
        print("\n4. Testing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        print("   ✓ SHAP TreeExplainer created successfully")
        
        shap_values = explainer.shap_values(X[:10])
        print(f"   ✓ SHAP values calculated: shape {shap_values.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def simple_parameter_analysis():
    """Simple parameter effect analysis using just pandas/numpy - no risky libraries."""
    print("\n=== Safe Parameter Analysis (No XGBoost Required) ===")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load the actual calibration data
        data_file = "calibration_modular_v2/external_calibration_data.csv"
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} measurements")
        
        # Define parameters and targets
        param_cols = [
            'aspirate_speed', 'dispense_speed', 'aspirate_wait_time', 'dispense_wait_time',
            'overaspirate_vol_ml', 'pre_asp_air_vol', 'post_asp_air_vol', 'blowout_vol', 'retract_speed'
        ]
        
        target_cols = ['deviation_pct', 'duration_s', 'variability_pct']
        
        # Filter to available columns
        available_params = [col for col in param_cols if col in df.columns]
        available_targets = [col for col in target_cols if col in df.columns]
        
        print(f"Analyzing {len(available_params)} parameters vs {len(available_targets)} targets")
        
        results = {}
        
        for target in available_targets:
            print(f"\n--- {target.replace('_', ' ').title()} Analysis ---")
            target_data = pd.to_numeric(df[target], errors='coerce')
            
            param_effects = {}
            
            for param in available_params:
                param_data = pd.to_numeric(df[param], errors='coerce')
                
                # Remove NaN values
                valid_mask = ~(pd.isna(target_data) | pd.isna(param_data))
                if valid_mask.sum() < 3:
                    continue
                    
                param_clean = param_data[valid_mask]
                target_clean = target_data[valid_mask]
                
                # Calculate multiple effect metrics
                correlation = abs(np.corrcoef(param_clean, target_clean)[0,1])
                param_range = param_clean.max() - param_clean.min()
                target_range = target_clean.max() - target_clean.min()
                
                # Calculate actual IMPACT: how much output changes per unit input change
                if param_range > 1e-6:  # Avoid division by zero
                    # Find the actual range of output values caused by this parameter
                    param_sorted_idx = np.argsort(param_clean)
                    target_sorted = target_clean.iloc[param_sorted_idx] if hasattr(target_clean, 'iloc') else target_clean[param_sorted_idx]
                    
                    # Calculate slope (impact per unit): output range / input range
                    output_range = target_sorted.max() - target_sorted.min()
                    impact_per_unit = output_range / param_range
                    
                    # Weight by correlation (strong correlation = more reliable impact estimate)
                    weighted_impact = abs(impact_per_unit) * correlation
                else:
                    impact_per_unit = 0
                    weighted_impact = 0
                
                param_effects[param] = {
                    'correlation': correlation,
                    'impact_per_unit': impact_per_unit,
                    'weighted_impact': weighted_impact,
                    'param_range': param_range,
                    'output_range': target_sorted.max() - target_sorted.min() if param_range > 1e-6 else 0
                }
            
            # Sort by weighted impact (impact per unit × correlation reliability)
            sorted_effects = sorted(param_effects.items(), 
                                  key=lambda x: x[1]['weighted_impact'], reverse=True)
            
            print("Parameter Impact (Output Change per Unit Input):")
            for param, metrics in sorted_effects[:6]:  # Top 6
                corr = metrics['correlation']
                impact = metrics['impact_per_unit']
                weighted = metrics['weighted_impact']
                param_name = param.replace('_', ' ').replace('ml', '').title()
                
                # Format units properly
                if 'vol' in param.lower():
                    unit_str = f"{impact:.1f}% per mL"
                elif 'speed' in param.lower():
                    unit_str = f"{impact:.2f}% per unit"
                elif 'time' in param.lower():
                    unit_str = f"{impact:.2f}% per sec"
                else:
                    unit_str = f"{impact:.2f}% per unit"
                    
                print(f"  {param_name:20} {unit_str:15} (Corr: {corr:5.3f})")
            
            results[target] = dict(sorted_effects)
        
        return results
        
    except Exception as e:
        print(f"Error in simple analysis: {e}")
        return None

def create_insights_file(analysis_results):
    """Create a basic insights file compatible with the GUI."""
    if not analysis_results:
        return None
        
    print("\n=== Creating Insights File ===")
    
    try:
        import json
        from pathlib import Path
        
        # Create insights structure compatible with GUI
        insights = {
            'parameter_sensitivity': {
                'shap_importance': {}
            },
            'analysis_method': 'correlation_based',
            'description': 'Simple correlation and range-based parameter analysis'
        }
        
        # Convert analysis results to SHAP-compatible format
        for target, param_effects in analysis_results.items():
            target_key = target.replace('_pct', '').replace('_s', '')  # 'accuracy', 'duration', etc
            
            # Use weighted_impact as importance score (impact per unit × correlation)
            importance_dict = {}
            for param, metrics in param_effects.items():
                importance_dict[param] = metrics['weighted_impact']
            
            insights['parameter_sensitivity']['shap_importance'][target_key] = importance_dict
        
        # Save to calibration_modular_v2 output folder (create if needed)
        output_dir = Path("calibration_modular_v2/output/test_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        insights_file = output_dir / "experiment_insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
            
        print(f"✓ Insights saved to: {insights_file}")
        print("This file can now be loaded by the GUI for parameter importance visualization!")
        
        return str(insights_file)
        
    except Exception as e:
        print(f"Error creating insights file: {e}")
        return None

def main():
    """Run safe parameter analysis without risky XGBoost installation."""
    print("Safe Parameter Effect Analysis")
    print("=" * 40)
    
    # Skip risky SHAP imports, go straight to safe analysis
    print("Using correlation and range-based analysis (no ML libraries needed)")
    
    # Run safe parameter analysis
    results = simple_parameter_analysis()
    
    if results:
        print("\n🎉 Parameter analysis completed successfully!")
        
        # Create insights file for GUI
        insights_file = create_insights_file(results) 
        
        if insights_file:
            print("\n✨ You can now load this analysis in the GUI:")
            print("   1. Copy the insights file to your optimization output folder")
            print("   2. The GUI will show parameter importance without needing SHAP!")
        
    else:
        print("\n❌ Parameter analysis failed")
        
if __name__ == "__main__":
    main()