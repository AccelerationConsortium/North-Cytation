#!/usr/bin/env python3
"""
Standalone SHAP analyzer for calibration optimization data.
Runs in isolated environment to avoid package conflicts.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_with_shap(csv_file, output_file, plots_dir=None):
    """
    Analyze parameter importance using SHAP and save results as JSON.
    Also save 3 specific SHAP plots if plots_dir is provided: accuracy, time, precision.
    """
    try:
        # Import SHAP (will fail gracefully if not available)
        import xgboost as xgb
        import shap
        import matplotlib.pyplot as plt
        
        # Load data
        data = pd.read_csv(csv_file)
        
        # Read config to find what's actually being optimized
        config_file = Path(csv_file).parent / "experiment_config_used.yaml"
        optimized_params = []
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                # Get parameters that have bounds (being optimized) - NO SILENT DEFAULTS
                if 'hardware_parameters' not in config:
                    print("[WARNING] No hardware_parameters section found in config")
                else:
                    hw_params = config['hardware_parameters']  # Fail if missing
                    for param, settings in hw_params.items():
                        if 'bounds' in settings:
                            optimized_params.extend([param, f'hardware_parameters_{param}'])
                
                if 'calibration_parameters' not in config:
                    print("[WARNING] No calibration_parameters section found in config")  
                else:
                    cal_params = config['calibration_parameters']  # Fail if missing
                    for param, settings in cal_params.items():
                        if 'bounds' in settings:
                            optimized_params.extend([param, f'calibration_{param}'])
                
                print(f"[DEBUG] Config-based optimized parameters: {optimized_params}")
            except ImportError:
                print("[WARNING] yaml module not available, using fallback parameters")
                # Use the actual column names from the data
                optimized_params = ['hardware_parameters_aspirate_speed', 'hardware_parameters_dispense_speed', 
                                 'hardware_parameters_aspirate_wait_time', 'hardware_parameters_pre_asp_air_vol', 
                                 'hardware_parameters_post_asp_air_vol', 'hardware_parameters_blowout_vol',
                                 'hardware_parameters_dispense_wait_time', 'hardware_parameters_retract_speed',
                                 'hardware_parameters_post_retract_wait_time', 'hardware_parameters_asp_disp_cycles',
                                 'calibration_overaspirate_vol']
                print(f"[DEBUG] Using fallback parameters (yaml not available): {optimized_params}")
        
        if not optimized_params:
            # Fallback to common parameters if no config or no optimized params found  
            optimized_params = ['hardware_parameters_aspirate_speed', 'hardware_parameters_dispense_speed', 
                             'hardware_parameters_aspirate_wait_time', 'hardware_parameters_pre_asp_air_vol', 
                             'hardware_parameters_post_asp_air_vol', 'hardware_parameters_blowout_vol',
                             'hardware_parameters_dispense_wait_time', 'hardware_parameters_retract_speed',
                             'hardware_parameters_post_retract_wait_time', 'hardware_parameters_asp_disp_cycles',
                             'calibration_overaspirate_vol']
            print(f"[DEBUG] Using fallback parameters (no config): {optimized_params}")
        
        # Find parameters that exist in the data
        param_cols = [col for col in data.columns if col in optimized_params]
        
        print(f"[DEBUG] Parameters found in data: {param_cols}")
        
        # Prepare target variables for 3 different analyses - NO SILENT DEFAULTS
        targets = {}
        
        # 1. Accuracy target - deviation from target volume (REQUIRED)
        if 'deviation_pct' in data.columns:
            targets['accuracy'] = 100 - abs(data['deviation_pct'])  # Higher = better accuracy
            print(f"[DEBUG] Using deviation_pct for accuracy target")
        elif 'measured_volume_ul' in data.columns and 'volume_target_ul' in data.columns:
            deviation_pct = 100 * abs(data['measured_volume_ul'] - data['volume_target_ul']) / data['volume_target_ul']
            targets['accuracy'] = 100 - deviation_pct
            print(f"[DEBUG] Calculated accuracy from measured/target volumes")
        else:
            print("[ERROR] No accuracy data found - missing deviation_pct or volume columns")
        
        # 2. Time target - measurement duration (REQUIRED, NO DEFAULTS)
        if 'duration_mean_s' in data.columns:
            targets['time'] = data['duration_mean_s'].max() - data['duration_mean_s']  # Invert so higher = faster
            print(f"[DEBUG] Using duration_mean_s for time target")
        else:
            print("[WARNING] No time data found - missing duration_mean_s column")
        
        # 3. Precision target - coefficient of variation (REQUIRED, NO DEFAULTS)
        if 'precision_cv_pct' in data.columns:
            targets['precision'] = 100 - data['precision_cv_pct']  # Higher = better precision
            print(f"[DEBUG] Using precision_cv_pct for precision target")
        else:
            print("[WARNING] No precision data found - missing precision_cv_pct column")
        
        print(f"[DEBUG] Targets available: {list(targets.keys())}")
        if not targets:
            print("[CRITICAL ERROR] No target variables found - cannot proceed with analysis")
        
        # Filter to only parameters that actually vary - NO FABRICATED ANALYSIS
        varying_params = []
        for col in param_cols:
            unique_values = data[col].nunique()
            if unique_values > 1:  # Only include parameters that change
                varying_params.append(col)
                print(f"[DEBUG] Parameter {col}: {unique_values} unique values - INCLUDED")
            else:
                print(f"[DEBUG] Parameter {col}: {unique_values} unique values - EXCLUDED (no variation)")
    
        print(f"[DEBUG] Final varying parameters: {varying_params}")
        
        if len(varying_params) < 2:
            print(f"[CRITICAL ERROR] Only {len(varying_params)} varying parameters found. Need at least 2 for meaningful SHAP analysis.")
            result = {
                'status': 'insufficient_data',
                'message': f'Need at least 2 varying parameters for SHAP analysis. Found {len(varying_params)}: {varying_params}',
                'parameter_importance': {},
                'debug_info': {
                    'all_parameters_checked': param_cols,
                    'varying_parameters': varying_params,
                    'parameter_unique_counts': {col: data[col].nunique() for col in param_cols}
                }
            }
        elif not targets:
            print(f"[CRITICAL ERROR] No target variables found. Required columns missing:")
            print(f"  - For accuracy: 'deviation_pct' OR ('measured_volume_ul' AND 'volume_target_ul')")  
            print(f"  - For time: 'duration_mean_s'")
            print(f"  - For precision: 'precision_cv_pct'")
            print(f"Available columns: {list(data.columns)}")
            result = {
                'status': 'no_targets',
                'message': 'No suitable target variables found (accuracy, time, precision)',
                'parameter_importance': {},
                'debug_info': {
                    'available_columns': list(data.columns),
                    'required_for_accuracy': ['deviation_pct', 'measured_volume_ul+volume_target_ul'],
                    'required_for_time': ['duration_mean_s'],
                    'required_for_precision': ['precision_cv_pct']
                }
            }
        else:
            # Prepare features
            X = data[varying_params]
            
            # Analyze each target and store results
            target_results = {}
            
            for target_name, target_values in targets.items():
                try:
                    # Train XGBoost model
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42,
                        objective='reg:squarederror'
                    )
                    model.fit(X, target_values)
                    
                    # Calculate SHAP values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                    
                    # Calculate feature importance (mean absolute SHAP values)
                    importance_scores = np.abs(shap_values).mean(axis=0)
                    importance_dict = dict(zip(varying_params, importance_scores))
                    
                    # Sort by importance and convert to regular Python floats for JSON serialization
                    sorted_importance = dict(sorted(importance_dict.items(), 
                                                 key=lambda x: x[1], reverse=True))
                    sorted_importance = {k: float(v) for k, v in sorted_importance.items()}
                    
                    target_results[target_name] = {
                        'parameter_importance': sorted_importance,
                        'model_score': float(model.score(X, target_values)),
                        'shap_values': shap_values,
                        'features': X
                    }
                    
                except Exception as e:
                    print(f"[WARNING] Failed to analyze {target_name}: {e}")
                    continue
            
            if target_results:
                # Return results for all targets, not just primary
                all_target_importance = {}
                model_scores = {}
                
                for target_name, target_data in target_results.items():
                    all_target_importance[target_name] = target_data['parameter_importance']
                    model_scores[target_name] = target_data['model_score']
                
                # Use accuracy as primary for backward compatibility but include all
                primary_target = 'accuracy' if 'accuracy' in target_results else list(target_results.keys())[0]
                
                result = {
                    'status': 'success',
                    'parameter_importance': target_results[primary_target]['parameter_importance'],  # Primary for compatibility
                    'all_targets': all_target_importance,  # All targets
                    'model_scores': model_scores,  # All model scores
                    'primary_target': primary_target,
                    'n_samples': len(data),
                    'n_features': len(varying_params),
                    'targets_analyzed': list(target_results.keys())
                }
                
                # Save individual target plots if plots_dir provided
                if plots_dir and Path(plots_dir).exists():
                    save_target_specific_plots(target_results, varying_params, plots_dir)
            else:
                result = {
                    'status': 'analysis_error',
                    'message': 'Failed to analyze any target variables',
                    'parameter_importance': {}
                }
            
    except ImportError as e:
        result = {
            'status': 'import_error',
            'message': f'SHAP/XGBoost not available: {str(e)}',
            'parameter_importance': {}
        }
    except Exception as e:
        result = {
            'status': 'analysis_error', 
            'message': f'Analysis failed: {str(e)}',
            'parameter_importance': {}
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def save_target_specific_plots(target_results, feature_names, plots_dir):
    """Save separate SHAP importance plots for accuracy, time, and precision."""
    try:
        import shap
        import matplotlib.pyplot as plt
        
        plots_path = Path(plots_dir)
        
        for target_name, target_data in target_results.items():
            # 1. Create parameter importance BAR PLOT for this target
            plt.figure(figsize=(10, 6))
            
            importance_dict = target_data['parameter_importance']
            # Get top 8 parameters for better readability
            sorted_items = list(importance_dict.items())[:8]
            
            if sorted_items:
                params, values = zip(*sorted_items)
                
                # Clean parameter names for display
                clean_params = [p.replace('hardware_parameters_', '').replace('_', ' ').title() 
                              for p in params]
                
                y_pos = np.arange(len(params))
                bars = plt.barh(y_pos, values, color='steelblue', alpha=0.8)
                
                plt.yticks(y_pos, clean_params)
                plt.xlabel('SHAP Importance (Impact on Performance)')
                
                # Customize title based on target
                title_map = {
                    'accuracy': 'Parameter Importance for Volume Accuracy',
                    'time': 'Parameter Importance for Measurement Speed', 
                    'precision': 'Parameter Importance for Measurement Precision'
                }
                plt.title(title_map.get(target_name, f'Parameter Importance for {target_name.title()}'), 
                         fontsize=12, fontweight='bold')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    plt.text(bar.get_width() + max(values) * 0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', ha='left', va='center', fontsize=9)
                
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                
                # Save bar chart
                bar_filename = f'shap_{target_name}_importance.png'
                plt.savefig(plots_path / bar_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {target_name} importance bar chart: {bar_filename}")
            
            # 2. Create SHAP SUMMARY PLOT (red/blue scatter) for this target
            if 'shap_values' in target_data and 'features' in target_data:
                plt.figure(figsize=(10, 8))
                
                shap_values = target_data['shap_values']
                features = target_data['features']
                
                # Generate SHAP summary plot with feature values colored
                shap.summary_plot(shap_values, features, 
                                feature_names=feature_names, 
                                show=False, max_display=8)
                
                # Customize title
                summary_title_map = {
                    'accuracy': 'SHAP Summary: Volume Accuracy Impact',
                    'time': 'SHAP Summary: Measurement Speed Impact',
                    'precision': 'SHAP Summary: Measurement Precision Impact'
                }
                plt.title(summary_title_map.get(target_name, f'SHAP Summary: {target_name.title()} Impact'), 
                         fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                
                # Save summary plot  
                summary_filename = f'shap_{target_name}_summary.png'
                plt.savefig(plots_path / summary_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {target_name} SHAP summary plot: {summary_filename}")
        
    except Exception as e:
        print(f"Failed to save target-specific plots: {e}")

def save_shap_plots(shap_values, X, feature_names, plots_dir):
    """Save SHAP plots to the plots directory (legacy function for compatibility)."""
    try:
        import shap
        import matplotlib.pyplot as plt
        
        plots_path = Path(plots_dir)
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(plots_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        importance_scores = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, importance_scores))
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
        plt.xlabel('Mean |SHAP value|')
        plt.title('Parameter Importance for Pipetting Performance')
        plt.tight_layout()
        plt.savefig(plots_path / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SHAP Waterfall plot for best sample (if we have samples)
        if len(shap_values) > 0:
            plt.figure(figsize=(10, 8))
            # Find best performing sample
            best_idx = np.argmax(np.sum(shap_values, axis=1))
            shap.waterfall_plot(shap.Explanation(values=shap_values[best_idx], 
                                               base_values=np.mean(shap_values), 
                                               data=X.iloc[best_idx].values,
                                               feature_names=feature_names), show=False)
            plt.tight_layout()
            plt.savefig(plots_path / 'shap_waterfall_best.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"SHAP plots saved to {plots_path}")
        
    except Exception as e:
        print(f"Failed to save SHAP plots: {e}")
        plt.close()
        
        # 2. SHAP Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        importance_scores = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, importance_scores))
        sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
        plt.xlabel('Mean |SHAP value|')
        plt.title('Parameter Importance for Pipetting Performance')
        plt.tight_layout()
        plt.savefig(plots_path / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. SHAP Waterfall plot for best sample (if we have samples)
        if len(shap_values) > 0:
            plt.figure(figsize=(10, 8))
            # Find best performing sample
            best_idx = np.argmax(np.sum(shap_values, axis=1))
            shap.waterfall_plot(shap.Explanation(values=shap_values[best_idx], 
                                               base_values=np.mean(shap_values), 
                                               data=X.iloc[best_idx].values,
                                               feature_names=feature_names), show=False)
            plt.tight_layout()
            plt.savefig(plots_path / 'shap_waterfall_best.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"SHAP plots saved to {plots_path}")
        
    except Exception as e:
        print(f"Failed to save SHAP plots: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python shap_analyzer.py <input_csv> <output_json> [plots_dir]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    plots_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not Path(input_csv).exists():
        print(f"Error: Input file {input_csv} not found")
        sys.exit(1)
    
    print(f"Analyzing {input_csv}...")
    result = analyze_with_shap(input_csv, output_json, plots_dir)
    print(f"Results saved to {output_json}")
    if plots_dir:
        print(f"Plots saved to {plots_dir}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Found {len(result['parameter_importance'])} important parameters")
        for param, importance in list(result['parameter_importance'].items())[:3]:
            print(f"  {param}: {importance:.4f}")