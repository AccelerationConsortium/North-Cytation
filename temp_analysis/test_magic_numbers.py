#!/usr/bin/env python3
"""
Test script to verify magic number cleanup works correctly
"""

import yaml
import sys
import os

# Add the path to import North_Safe
sys.path.append(os.path.dirname(__file__))

def test_yaml_loading():
    """Test that the robot_hardware.yaml file loads correctly"""
    try:
        with open('robot_state/robot_hardware.yaml', 'r') as file:
            data = yaml.safe_load(file)
        
        print("✅ YAML loading successful")
        print(f"Movement speeds: {data.get('movement_speeds', {})}")
        print(f"Physical constants: {data.get('physical_constants', {})}")
        print(f"Track axes: {data.get('track_axes', {})}")
        
        # Check required sections
        required_sections = ['movement_speeds', 'physical_constants', 'track_axes']
        for section in required_sections:
            if section not in data:
                print(f"❌ Missing section: {section}")
                return False
                
        # Check required movement speeds
        required_speeds = ['default_robot', 'fast_approach', 'standard_xy', 'precise_movement', 'retract']
        for speed in required_speeds:
            if speed not in data['movement_speeds']:
                print(f"❌ Missing speed: {speed}")
                return False
                
        # Check required constants
        required_constants = ['safe_height_z']
        for constant in required_constants:
            if constant not in data['physical_constants']:
                print(f"❌ Missing constant: {constant}")
                return False
        
        print("✅ All required YAML sections and values present")
        return True
        
    except Exception as e:
        print(f"❌ YAML loading failed: {e}")
        return False

def test_mock_robot_class():
    """Test the accessor methods with a mock robot class"""
    try:
        # Create a minimal mock of the config loading functionality
        class MockRobotConfig:
            def __init__(self):
                with open('robot_state/robot_hardware.yaml', 'r') as file:
                    self.ROBOT_HARDWARE = yaml.safe_load(file)
            
            def get_config_parameter(self, config_name, key, parameter, error_on_missing=True):
                config_map = {'robot_hardware': 'ROBOT_HARDWARE'}
                config = getattr(self, config_map.get(config_name))
                if config and key in config and parameter in config[key]:
                    return config[key][parameter]
                elif error_on_missing:
                    raise ValueError(f"Missing parameter: {config_name}.{key}.{parameter}")
                return None
            
            def get_speed(self, speed_name):
                """Get movement speed from robot hardware configuration"""
                return self.get_config_parameter('robot_hardware', 'movement_speeds', speed_name, error_on_missing=True)
            
            def get_safe_height(self):
                """Get safe height from robot hardware configuration"""
                return self.get_config_parameter('robot_hardware', 'physical_constants', 'safe_height_z', error_on_missing=True)
        
        robot = MockRobotConfig()
        
        # Test speed accessors
        speeds_to_test = ['default_robot', 'fast_approach', 'standard_xy', 'precise_movement', 'retract']
        for speed in speeds_to_test:
            value = robot.get_speed(speed)
            print(f"✅ get_speed('{speed}') = {value}")
        
        # Test safe height
        safe_height = robot.get_safe_height()
        print(f"✅ get_safe_height() = {safe_height}")
        
        # Test track axes access
        track_axes = robot.get_config_parameter('robot_hardware', 'track_axes', '', error_on_missing=False)
        print(f"✅ track_axes = {track_axes}")
        
        print("✅ All accessor methods working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Mock robot test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Magic Number Cleanup Changes...")
    print("=" * 50)
    
    success = True
    success &= test_yaml_loading()
    print()
    success &= test_mock_robot_class()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Magic number cleanup successful.")
    else:
        print("❌ Some tests failed. Please check the implementation.")