#!/usr/bin/env python3
"""
Demo showing the clean separation between liquid handling and robot behavior parameters.
"""

import sys
sys.path.append(".")
from pipetting_data.pipetting_parameters import PipettingParameters

def main():
    print("=== PipettingParameters Demo ===")
    print("\nPipettingParameters now contains ONLY liquid handling parameters:")
    
    # Example 1: Water-optimized parameters
    water_params = PipettingParameters(
        aspirate_speed=15,
        dispense_speed=10,
        aspirate_wait_time=1.0,
        dispense_wait_time=0.5,
        pre_asp_air_vol=0.005,  # 5 µL air cushion
        post_asp_air_vol=0.002  # 2 µL air gap
    )
    
    print(f"\nWater parameters:")
    print(f"  Aspirate speed: {water_params.aspirate_speed}")
    print(f"  Dispense speed: {water_params.dispense_speed}")
    print(f"  Wait times: {water_params.aspirate_wait_time}s / {water_params.dispense_wait_time}s")
    print(f"  Air volumes: {water_params.pre_asp_air_vol} / {water_params.post_asp_air_vol}")
    
    # Example 2: Viscous liquid parameters
    glycerol_params = PipettingParameters(
        aspirate_speed=5,       # Slower for viscous liquids
        dispense_speed=3,       # Much slower dispensing
        aspirate_wait_time=3.0, # Longer settling time
        dispense_wait_time=2.0, # Wait for viscous liquid to settle
        blowout_vol=0.001,      # Small blowout to clear viscous residue
        asp_disp_cycles=2       # Pre-mixing for better accuracy
    )
    
    print(f"\nGlycerol parameters:")
    print(f"  Aspirate speed: {glycerol_params.aspirate_speed}")
    print(f"  Dispense speed: {glycerol_params.dispense_speed}")  
    print(f"  Wait times: {glycerol_params.aspirate_wait_time}s / {glycerol_params.dispense_wait_time}s")
    print(f"  Blowout volume: {glycerol_params.blowout_vol}")
    print(f"  Mixing cycles: {glycerol_params.asp_disp_cycles}")
    
    print("\n=== Robot Behavior Parameters ===")
    print("Robot behavior is now controlled by method parameters:")
    print("  - move_to_aspirate=True/False")
    print("  - track_height=True/False") 
    print("  - move_up=True/False")
    print("  - initial_move=True/False")
    print("  - measure_weight=True/False")
    print("  - specified_tip='small_tip'/'large_tip'")
    
    print("\n=== Usage Examples ===")
    print("Standard aspiration:")
    print("  robot.aspirate_from_vial('source', 1.0, parameters=water_params)")
    print()
    print("Aspiration without movement (for in-place mixing):")
    print("  robot.aspirate_from_vial('source', 0.5, parameters=water_params,")
    print("                          move_to_aspirate=False, track_height=False)")
    print()
    print("Dispensing with weight measurement:")
    print("  robot.dispense_into_vial('dest', 1.0, parameters=glycerol_params,")
    print("                          measure_weight=True)")
    
    print("\n=== Benefits ===")
    print("✓ Liquid handling parameters are reusable across operations")
    print("✓ Robot behavior parameters are explicit and context-specific")
    print("✓ No confusion between liquid properties and robot actions")
    print("✓ Clean separation of concerns")
    print("✓ Method signatures are clear about what they control")

if __name__ == "__main__":
    main()