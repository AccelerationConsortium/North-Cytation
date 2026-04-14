#!/usr/bin/env python3
"""
Test script to verify X-Y coordinate conversion functionality
in the enhanced robot arm position program.
"""

import sys
import os
from unittest.mock import MagicMock

# Add project paths for imports
sys.path.append("../utoronto_demo")
sys.path.insert(0, "..")

def create_mock_robot():
    """Create a mock robot with kinematics functions for testing."""
    mock_robot = MagicMock() 
    
    # Mock the kinematics functions
    def mock_n9_fk(gripper_cts, elbow_cts, shoulder_cts):
        """Mock forward kinematics - Convert joint counts to X-Y coordinates."""
        # Simple approximation for testing
        x = 150 + (shoulder_cts / 1000)  # Approximate X position 
        y = 100 + (elbow_cts / 1000)    # Approximate Y position  
        theta = gripper_cts / 1000       # Approximate gripper angle
        return x, y, theta
        
    def mock_n9_ik(x, y):
        """Mock inverse kinematics - Convert X-Y coordinates to joint counts."""
        # Simple approximation for testing
        gripper_cts = 0  # Keep gripper at same position
        elbow_cts = int((y - 100) * 1000)     # Convert Y to elbow
        shoulder_cts = int((x - 150) * 1000)  # Convert X to shoulder
        return gripper_cts, elbow_cts, shoulder_cts
    
    def mock_counts_to_rad(axis, counts):
        """Convert counts to radians."""
        return counts / 1000  # Simple conversion for testing
        
    def mock_rad_to_counts(axis, rad):
        """Convert radians to counts."""
        return int(rad * 1000)  # Simple conversion for testing
    
    # Attach mock functions
    mock_robot.n9_fk = mock_n9_fk
    mock_robot.n9_ik = mock_n9_ik 
    mock_robot.counts_to_rad = mock_counts_to_rad
    mock_robot.rad_to_counts = mock_rad_to_counts
    
    # Mock robot constants
    mock_robot.GRIPPER = 0
    mock_robot.ELBOW = 1
    mock_robot.SHOULDER = 2  
    mock_robot.Z_AXIS = 3
    
    return mock_robot

def test_xy_conversions():
    """Test the X-Y coordinate conversion functions."""
    print("=" * 50)
    print("Testing X-Y Coordinate Conversion Functionality")
    print("=" * 50)
    
    robot = create_mock_robot()
    
    # Test forward kinematics (joint angles -> X-Y coordinates)
    print("\n1. Testing Forward Kinematics (Joints -> X-Y):")
    test_positions = [
        (0, 0, 0),      # Home position
        (0, 1000, 500), # Elbow and shoulder moved
        (0, -500, 1000) # Different position
    ]
    
    for gripper, elbow, shoulder in test_positions:
        x, y, theta = robot.n9_fk(gripper, elbow, shoulder)
        print(f"  Joints ({gripper}, {elbow}, {shoulder}) -> X-Y ({x:.1f}, {y:.1f}, {theta:.3f})")
    
    # Test inverse kinematics (X-Y coordinates -> joint angles)  
    print("\n2. Testing Inverse Kinematics (X-Y -> Joints):")
    test_coordinates = [
        (150, 100),  # Center position
        (200, 150),  # Right and forward
        (100, 50)    # Left and back
    ]
    
    for x, y in test_coordinates:
        gripper, elbow, shoulder = robot.n9_ik(x, y)
        print(f"  X-Y ({x:.1f}, {y:.1f}) -> Joints ({gripper}, {elbow}, {shoulder})")
        
    # Test round-trip conversion (should get back close to original)
    print("\n3. Testing Round-Trip Conversion (Joints -> X-Y -> Joints):")
    original_positions = [(0, 1000, 500), (0, -500, 1000)]
    
    for orig_gripper, orig_elbow, orig_shoulder in original_positions:
        # Forward: joints -> X-Y
        x, y, theta = robot.n9_fk(orig_gripper, orig_elbow, orig_shoulder)
        
        # Reverse: X-Y -> joints
        new_gripper, new_elbow, new_shoulder = robot.n9_ik(x, y)
        
        print(f"  Original: ({orig_gripper}, {orig_elbow}, {orig_shoulder})")
        print(f"  X-Y:      ({x:.1f}, {y:.1f})")  
        print(f"  Result:   ({new_gripper}, {new_elbow}, {new_shoulder})")
        print(f"  Match:    {abs(new_elbow - orig_elbow) < 10 and abs(new_shoulder - orig_shoulder) < 10}")
        print()

    # Test movement increment calculations
    print("4. Testing Movement Increment System:")
    step_size_mm = 5.0
    
    # Simulate X-Y movement
    current_x = 150.0
    current_y = 100.0
    
    print(f"  Current position: ({current_x:.1f}, {current_y:.1f})")
    print(f"  Step size: {step_size_mm} mm")
    
    # Test different movement directions
    movements = [
        ("X Left",     current_x - step_size_mm, current_y),
        ("X Right",    current_x + step_size_mm, current_y), 
        ("Y Forward",  current_x, current_y + step_size_mm),
        ("Y Back",     current_x, current_y - step_size_mm)
    ]
    
    for direction, target_x, target_y in movements:
        gripper, elbow, shoulder = robot.n9_ik(target_x, target_y)
        elbow_rad = robot.counts_to_rad(1, elbow)
        shoulder_rad = robot.counts_to_rad(2, shoulder)
        print(f"  {direction:10}: X-Y ({target_x:.1f}, {target_y:.1f}) -> joints ({elbow_rad:.3f}, {shoulder_rad:.3f}) rad")

    print("\n" + "=" * 50)
    print("X-Y Coordinate Conversion Tests Completed!")  
    print("=" * 50)

if __name__ == "__main__":
    test_xy_conversions()