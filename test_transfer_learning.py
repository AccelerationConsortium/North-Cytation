#!/usr/bin/env python3
"""
Quick test to verify transfer learning setup works correctly.
"""

# Add paths for imports
import sys
sys.path.append("../North-Cytation")

# Import the workflow
from workflows.calibration_sdl_simplified import (
    create_transfer_learning_optimizer, USE_TRANSFER_LEARNING, 
    VOLUMES, get_current_config_summary
)

def test_transfer_learning():
    """Test basic transfer learning functionality."""
    print("🧪 TESTING TRANSFER LEARNING SETUP")
    print("="*50)
    
    # Show current config
    get_current_config_summary()
    
    if not USE_TRANSFER_LEARNING:
        print("\n❌ Transfer learning is disabled")
        return False
        
    print(f"\n🎯 Creating transfer learning optimizer for volumes: {[f'{v*1000:.0f}μL' for v in VOLUMES]}")
    
    try:
        # Create the global optimizer
        global_ax_client = create_transfer_learning_optimizer()
        
        if global_ax_client is None:
            print("❌ Failed to create transfer learning optimizer")
            return False
            
        print("✅ Transfer learning optimizer created successfully!")
        print(f"   Experiment: {global_ax_client.experiment.name}")
        print(f"   Parameters: {list(global_ax_client.experiment.search_space.parameters.keys())}")
        print(f"   Objectives: {list(global_ax_client.experiment.optimization_config.objective.objectives.keys())}")
        
        # Test getting a suggestion with fixed volume
        print(f"\n🔍 Testing suggestion generation...")
        
        # Import get_suggestions
        from recommenders.pipetting_optimizer_3objectives import get_suggestions
        
        # Test suggestion for 50μL
        suggestions = get_suggestions(global_ax_client, n=1, fixed_features={"volume": 0.05})
        params, trial_idx = suggestions[0]
        
        print(f"   ✅ Generated suggestion for 50μL:")
        print(f"   Trial {trial_idx}: {params}")
        
        # Verify volume is not in the returned parameters (it should be fixed)
        if "volume" in params:
            print(f"   ⚠️  Warning: Volume included in returned parameters")
        else:
            print(f"   ✅ Volume correctly handled as fixed feature")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing transfer learning: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transfer_learning()
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")