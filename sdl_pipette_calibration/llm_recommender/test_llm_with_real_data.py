#!/usr/bin/env python3
"""
Simple LLM Test Program - Feed real calibration data to test LLM recommendations

Tests LLM with actual experimental data from external_calibration_data.csv
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports (like other v2 programs)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal fake classes for LLM compatibility (keep in test only)
class FakeParameters:
    """Minimal fake parameters object with to_protocol_dict method."""
    def __init__(self, param_dict):
        self.param_dict = param_dict
    
    def to_protocol_dict(self):
        return self.param_dict

class FakeTrialResult:
    """Minimal fake trial result with detailed metrics for LLM analysis."""
    def __init__(self, param_dict, score, duration_s, deviation_pct=None, cv_pct=None, measured_vol=None, target_vol=None):
        # Convert numpy types to Python types for JSON compatibility
        clean_params = {}
        for key, value in param_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                clean_params[key] = value.item()  # Convert to Python type
            else:
                clean_params[key] = value
        
        self.parameters = FakeParameters(clean_params)
        self.score = float(score)  # Ensure Python float
        self.duration_s = float(duration_s)  # Ensure Python float
        
        # Store individual metrics for detailed LLM analysis
        self.deviation_pct = float(deviation_pct) if deviation_pct is not None else None
        self.cv_pct = float(cv_pct) if cv_pct is not None else None  
        self.measured_vol_ml = float(measured_vol) if measured_vol is not None else None
        self.target_vol_ml = float(target_vol) if target_vol is not None else None

class FakeVolumeCalibrationResult:
    """Minimal fake volume result with just what LLM needs."""
    def __init__(self, target_volume_ml, trial_summaries):
        self.target_volume_ml = target_volume_ml
        self.best_trials = []
        
        for trial in trial_summaries:
            # Convert deviation and CV to a simple score (lower = better)  
            score = trial['deviation_pct'] + trial['cv_pct']
            fake_trial = FakeTrialResult(
                trial['parameters'], 
                score, 
                trial['mean_time_s'],
                deviation_pct=trial['deviation_pct'],
                cv_pct=trial['cv_pct'], 
                measured_vol=trial['measured_volume_ml'],
                target_vol=trial['target_volume_ml']
            )
            self.best_trials.append(fake_trial)

def test_llm_with_real_data():
    """Test LLM with real calibration data."""
    
    try:
        # Import LLM components (now with proper path setup)
        from calibration_modular_v2.llm_recommender import LLMRecommender
        from calibration_modular_v2.config_manager import ExperimentConfig
        print("✓ LLM modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LLM modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Load real calibration data
        data_file = "calibration_modular_v2/external_calibration_data.csv"
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} measurements from {data_file}")
        print(f"  Liquid: {df['liquid_type'].unique()[0]}")
        print(f"  Target volumes: {df['target_volume_ml'].unique()}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"✗ Failed to load calibration data: {e}")
        return False
    
    try:
        # Load config
        config = ExperimentConfig.from_yaml("calibration_modular_v2/experiment_config.yaml")
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    try:
        # Create LLM recommender for optimization phase (with previous data)
        template_path = "calibration_screening_llm_template.json"
        llm_recommender = LLMRecommender(config, template_path, phase="optimization")
        print(f"✓ LLM recommender created with template: {template_path}")
    except Exception as e:
        print(f"✗ Failed to create LLM recommender: {e}")
        return False
        
    try:
        # Convert real data to simple format for LLM context
        # Group by measurement_id and calculate basic stats
        trial_summaries = []
        for measurement_id, group in df.groupby('measurement_id'):
            measured_vols = group['measured_volume_ml'].values
            times = group['measurement_time_s'].values
            target_vol = group['target_volume_ml'].iloc[0]
            
            # Calculate basic stats like the real analyzer would
            mean_vol = measured_vols.mean()
            deviation_pct = abs((mean_vol - target_vol) / target_vol) * 100
            cv_pct = (measured_vols.std() / mean_vol) * 100 if mean_vol > 0 else 100
            mean_time = times.mean()
            
            # Extract parameters
            params = group.iloc[0]
            
            trial_summary = {
                'measurement_id': measurement_id,
                'target_volume_ml': target_vol,
                'measured_volume_ml': mean_vol,
                'deviation_pct': deviation_pct,
                'cv_pct': cv_pct,
                'mean_time_s': mean_time,
                'parameters': {
                    'overaspirate_vol': params['overaspirate_vol'],
                    'aspirate_speed': params['aspirate_speed'], 
                    'dispense_speed': params['dispense_speed'],
                    'aspirate_wait_time': params['aspirate_wait_time'],
                    'dispense_wait_time': params['dispense_wait_time'],
                    'blowout_vol': params['blowout_vol'],
                    'pre_asp_air_vol': params['pre_asp_air_vol'],
                    'post_asp_air_vol': params['post_asp_air_vol'],
                    'post_retract_wait_time': params['post_retract_wait_time']
                }
            }
            trial_summaries.append(trial_summary)
        
        print(f"✓ Processed {len(trial_summaries)} experimental trials")
        
        # Show sample of the data we'll send to LLM
        print("\\n📊 Sample experimental data for LLM:")
        for i, trial in enumerate(trial_summaries[:3]):
            print(f"  Trial {i+1}: {trial['measured_volume_ml']*1000:.1f}uL measured "
                  f"({trial['deviation_pct']:.1f}% dev, {trial['cv_pct']:.1f}% CV, {trial['mean_time_s']:.1f}s)")
            
    except Exception as e:
        print(f"✗ Failed to process calibration data: {e}")
        return False
    
    try:
        print("\\n🤖 Testing LLM parameter suggestions...")
        
        # Convert simple trial data to fake objects for LLM compatibility
        fake_volume_result = FakeVolumeCalibrationResult(0.05, trial_summaries)
        
        # Test LLM with real experimental context
        suggestions = llm_recommender.suggest_parameters(
            3,  # n_suggestions_or_volume
            [fake_volume_result]  # previous_results_or_trial (list of volume results)
        )
        
        if not suggestions:
            print("✗ LLM returned no suggestions")
            return False
            
        print(f"✓ LLM generated {len(suggestions)} parameter suggestions!")
        
        # Display suggestions
        print("\\n🎯 LLM Parameter Recommendations:")
        for i, params in enumerate(suggestions, 1):
            print(f"  Suggestion {i}:")
            print(f"    • overaspirate_vol: {params.calibration.overaspirate_vol*1000:.2f} uL") 
            print(f"    • aspirate_speed: {params.hardware.get('aspirate_speed', 'N/A')}")
            print(f"    • dispense_speed: {params.hardware.get('dispense_speed', 'N/A')}")
            print(f"    • aspirate_wait_time: {params.hardware.get('aspirate_wait_time', 'N/A'):.1f}s")
            print(f"    • dispense_wait_time: {params.hardware.get('dispense_wait_time', 'N/A'):.1f}s")
            print(f"    • blowout_vol: {params.hardware.get('blowout_vol', 0)*1000:.1f} uL")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM suggestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_screening_mode():
    """Test LLM in screening mode (no previous data)."""
    
    try:
        from calibration_modular_v2.llm_recommender import LLMRecommender
        from calibration_modular_v2.config_manager import ExperimentConfig
        
        config = ExperimentConfig.from_yaml("calibration_modular_v2/experiment_config.yaml")
        template_path = "calibration_screening_llm_template.json"
        
        llm_recommender = LLMRecommender(config, template_path, phase="screening")
        print("\\n🔍 Testing LLM in screening mode (no previous data)...")
        
        # Test screening suggestions (legacy API)
        target_volume = 0.05  # 50 uL
        trial_idx = 0
        parameters = llm_recommender.suggest_parameters(target_volume, trial_idx)
        
        print(f"✓ LLM screening suggestion for {target_volume*1000:.0f}uL:")
        print(f"  • overaspirate_vol: {parameters.calibration.overaspirate_vol*1000:.2f} uL")
        print(f"  • aspirate_speed: {parameters.hardware.get('aspirate_speed', 'N/A')}")
        print(f"  • dispense_speed: {parameters.hardware.get('dispense_speed', 'N/A')}")
        print(f"  • aspirate_wait_time: {parameters.hardware.get('aspirate_wait_time', 'N/A'):.1f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM screening test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 LLM Calibration Test Program")
    print("="*50)
    
    # Test 1: Optimization mode with real data
    print("\\n[TEST 1] LLM Optimization Mode")
    success1 = test_llm_with_real_data()
    
    # Test 2: Screening mode 
    print("\\n[TEST 2] LLM Screening Mode")  
    success2 = test_llm_screening_mode()
    
    print("\\n" + "="*50)
    if success1 and success2:
        print("✅ ALL TESTS PASSED - LLM is working!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        
    print("\\nTo enable LLM in real experiments:")
    print("  1. Set optimization.llm_optimization.enabled: true")
    print("  2. Set optimization.llm_optimization.config_path: calibration_modular_v2/calibration_screening_llm_template.json")
    print("  3. Or set screening.use_llm_suggestions: true for screening phase")
    print("\nTo run this test: python calibration_modular_v2/test_llm_with_real_data.py")