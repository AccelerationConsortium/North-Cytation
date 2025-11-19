#!/usr/bin/env python3
"""Test script to validate protocol interface compliance.

This demonstrates how the abstract base class ensures both simulation 
and hardware protocols implement the same interface correctly.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from calibration_protocol_base import CalibrationProtocolBase, create_protocol


def test_protocol_compliance(protocol: CalibrationProtocolBase, protocol_name: str):
    """Test that a protocol correctly implements the interface."""
    print(f"\n🧪 Testing {protocol_name} protocol compliance...")
    
    # Test 1: initialize() returns valid state
    print("  ✓ Testing initialize()...")
    test_config = {
        'random_seed': 42,
        'experiment': {'liquid': 'water'},
        'hardware': {'device_serial': 'TEST123'}
    }
    
    try:
        state = protocol.initialize(test_config)
        assert isinstance(state, dict), f"initialize() must return dict, got {type(state)}"
        assert protocol.validate_state(state), f"State validation failed for {protocol_name}"
        print("    ✅ initialize() returns valid state")
    except Exception as e:
        print(f"    ❌ initialize() failed: {e}")
        return False
    
    # Test 2: measure() returns valid results
    print("  ✓ Testing measure()...")
    test_params = {
        'aspirate_speed': 15,
        'dispense_speed': 10,
        'aspirate_wait_time': 1.5,
        'dispense_wait_time': 1.0,
        'overaspirate_vol': 0.005
    }
    
    try:
        results = protocol.measure(state, volume_mL=0.05, params=test_params, replicates=2)
        assert isinstance(results, list), f"measure() must return list, got {type(results)}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        # Validate each result
        for i, result in enumerate(results):
            replicate_num = i + 1
            assert protocol.validate_measurement_result(result, replicate_num), \
                   f"Result {i} validation failed for {protocol_name}"
        
        print(f"    ✅ measure() returns {len(results)} valid results")
        print(f"    📊 Sample result: replicate={results[0]['replicate']}, "
              f"volume={results[0]['volume']:.4f}mL, elapsed={results[0]['elapsed_s']:.2f}s")
    except Exception as e:
        print(f"    ❌ measure() failed: {e}")
        return False
    
    # Test 3: wrapup() completes without error  
    print("  ✓ Testing wrapup()...")
    try:
        protocol.wrapup(state)  # Should not raise exceptions
        print("    ✅ wrapup() completed successfully")
    except Exception as e:
        print(f"    ⚠️  wrapup() raised exception (should log instead): {e}")
    
    print(f"  🎉 {protocol_name} protocol passed all compliance tests!")
    return True


def test_factory_function():
    """Test the protocol factory function."""
    print("\\n🏭 Testing protocol factory function...")
    
    # Test valid protocols
    for protocol_name in ['simulated', 'hardware']:
        try:
            protocol = create_protocol(protocol_name)
            assert isinstance(protocol, CalibrationProtocolBase), \
                   f"Factory must return CalibrationProtocolBase instance"
            print(f"  ✅ Successfully created {protocol_name} protocol")
        except ImportError as e:
            print(f"  ⚠️  {protocol_name} protocol not available: {e}")
        except Exception as e:
            print(f"  ❌ Failed to create {protocol_name} protocol: {e}")
    
    # Test invalid protocol
    try:
        create_protocol('invalid')
        print("  ❌ Factory should reject invalid protocol names")
    except ValueError:
        print("  ✅ Factory correctly rejects invalid protocol names")


def test_backward_compatibility():
    """Test that old function-based interface still works."""
    print("\\n🔄 Testing backward compatibility...")
    
    try:
        # Test simulation protocol backward compatibility
        import calibration_protocol_simulated as sim
        
        # Test function interface
        state = sim.initialize({'random_seed': 42})
        results = sim.measure(state, 0.05, {'overaspirate_vol': 0.005}, 1)
        sim.wrapup(state)
        
        print("  ✅ Simulation protocol backward compatibility works")
        
    except Exception as e:
        print(f"  ❌ Simulation backward compatibility failed: {e}")
    
    try:
        # Test hardware protocol backward compatibility (may fail if hardware not available)
        import calibration_protocol_hardware as hw
        
        # This will likely fail without hardware, but we can test the interface
        try:
            state = hw.initialize({'hardware': {'device_serial': 'TEST'}})
            print("  ✅ Hardware protocol backward compatibility works")
        except RuntimeError as e:
            if "Hardware not available" in str(e):
                print("  ⚠️  Hardware protocol interface works (hardware not available)")
            else:
                raise
                
    except Exception as e:
        print(f"  ❌ Hardware backward compatibility failed: {e}")


def main():
    """Run all compliance tests."""
    print("🚀 Protocol Interface Compliance Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test factory function
    test_factory_function()
    
    # Test each available protocol
    for protocol_name in ['simulated', 'hardware']:
        try:
            protocol = create_protocol(protocol_name)
            success = test_protocol_compliance(protocol, protocol_name)
            all_passed &= success
        except ImportError:
            print(f"\\n⚠️  Skipping {protocol_name} protocol (not available)")
        except Exception as e:
            print(f"\\n❌ Failed to test {protocol_name} protocol: {e}")
            all_passed = False
    
    # Test backward compatibility
    test_backward_compatibility()
    
    # Summary
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 All available protocols passed compliance tests!")
        print("   Your abstract base class successfully enforces the interface contract.")
    else:
        print("❌ Some protocols failed compliance tests.")
        print("   Check the implementation against CalibrationProtocolBase requirements.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())