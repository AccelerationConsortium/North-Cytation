"""
Example of smart parameter lookup system based on liquid type and volume

This shows how you could load parameters from CSV and interpolate optimal conditions
"""

import pandas as pd
import numpy as np
from pipetting_data.pipetting_parameters import PipettingParameters

# Example CSV structure you might have:
example_csv_data = """
liquid_type,min_volume,max_volume,aspirate_speed,dispense_speed,aspirate_wait_time,dispense_wait_time,pre_asp_air_vol,blowout_vol,notes
water,0.001,1.0,10,10,1.0,0.0,0.0,0.0,Standard aqueous solutions
ethanol,0.001,1.0,8,6,1.5,0.5,0.02,0.05,Volatile solvent - slower dispense
DMSO,0.001,1.0,5,3,3.0,2.0,0.01,0.1,Viscous - long wait times
protein_buffer,0.001,0.5,6,6,2.0,1.0,0.005,0.02,Delicate biological samples
organic_solvent,0.001,1.0,7,5,1.2,0.8,0.03,0.08,General organic solvents
small_molecule,0.001,0.2,3,3,2.5,1.5,0.005,0.01,Precise small volumes
"""

class SmartPipettingParams:
    """
    Smart parameter lookup system that can interpolate optimal pipetting conditions
    based on liquid type and volume.
    """
    
    def __init__(self, csv_path=None):
        """Initialize with parameter database"""
        if csv_path:
            self.param_db = pd.read_csv(csv_path)
        else:
            # Use example data for demo
            from io import StringIO
            self.param_db = pd.read_csv(StringIO(example_csv_data))
    
    def get_parameters_for_liquid(self, liquid_type: str, volume_mL: float) -> PipettingParameters:
        """
        Get optimal pipetting parameters for a specific liquid and volume.
        
        Args:
            liquid_type: Type of liquid (e.g., 'water', 'ethanol', 'DMSO')
            volume_mL: Volume to pipette in mL
            
        Returns:
            PipettingParameters object with optimized settings
        """
        # Find matching liquid type
        liquid_params = self.param_db[self.param_db['liquid_type'] == liquid_type]
        
        if liquid_params.empty:
            print(f"⚠️  Unknown liquid type '{liquid_type}', using water defaults")
            liquid_params = self.param_db[self.param_db['liquid_type'] == 'water']
        
        # Check if volume is in range
        row = liquid_params.iloc[0]
        if volume_mL < row['min_volume'] or volume_mL > row['max_volume']:
            print(f"⚠️  Volume {volume_mL} mL outside recommended range [{row['min_volume']}-{row['max_volume']}] for {liquid_type}")
        
        # Apply volume-based adjustments
        adjusted_params = self._adjust_for_volume(row, volume_mL)
        
        return PipettingParameters(
            aspirate_speed=int(adjusted_params['aspirate_speed']),
            dispense_speed=int(adjusted_params['dispense_speed']),
            aspirate_wait_time=adjusted_params['aspirate_wait_time'],
            dispense_wait_time=adjusted_params['dispense_wait_time'],
            pre_asp_air_vol=adjusted_params['pre_asp_air_vol'],
            blowout_vol=adjusted_params['blowout_vol']
        )
    
    def _adjust_for_volume(self, base_params, volume_mL):
        """Apply volume-based adjustments to base parameters"""
        adjusted = base_params.copy()
        
        # Volume-based speed adjustments
        if volume_mL < 0.01:  # Very small volumes - slower
            adjusted['aspirate_speed'] *= 0.6
            adjusted['dispense_speed'] *= 0.6
            adjusted['aspirate_wait_time'] *= 1.5
        elif volume_mL > 0.5:  # Large volumes - can be faster
            adjusted['aspirate_speed'] *= 1.2
            adjusted['dispense_speed'] *= 1.2
            adjusted['aspirate_wait_time'] *= 0.8
        
        # Air gap scaling with volume
        if adjusted['pre_asp_air_vol'] > 0:
            adjusted['pre_asp_air_vol'] = max(0.002, volume_mL * 0.02)  # 2% of volume, min 2µL
        
        if adjusted['blowout_vol'] > 0:
            adjusted['blowout_vol'] = max(0.005, volume_mL * 0.1)  # 10% of volume, min 5µL
        
        return adjusted
    
    def get_available_liquids(self):
        """Get list of available liquid types"""
        return self.param_db['liquid_type'].tolist()
    
    def add_liquid_type(self, liquid_type: str, **params):
        """Add a new liquid type to the database"""
        new_row = {
            'liquid_type': liquid_type,
            'min_volume': params.get('min_volume', 0.001),
            'max_volume': params.get('max_volume', 1.0),
            'aspirate_speed': params.get('aspirate_speed', 10),
            'dispense_speed': params.get('dispense_speed', 10),
            'aspirate_wait_time': params.get('aspirate_wait_time', 1.0),
            'dispense_wait_time': params.get('dispense_wait_time', 0.0),
            'pre_asp_air_vol': params.get('pre_asp_air_vol', 0.0),
            'blowout_vol': params.get('blowout_vol', 0.0),
            'notes': params.get('notes', 'User-defined liquid type')
        }
        self.param_db = pd.concat([self.param_db, pd.DataFrame([new_row])], ignore_index=True)


def demo_smart_parameters():
    """Demonstrate the smart parameter system"""
    
    print("🧪 Smart Pipetting Parameter System Demo")
    print("=" * 50)
    
    # Initialize the smart parameter system
    smart_params = SmartPipettingParams()
    
    print(f"📋 Available liquid types: {smart_params.get_available_liquids()}")
    print()
    
    # Test different scenarios
    test_cases = [
        ("water", 0.5),
        ("ethanol", 0.1),
        ("DMSO", 0.05),
        ("protein_buffer", 0.002),  # Very small volume
        ("organic_solvent", 0.8),   # Large volume
        ("unknown_liquid", 0.1),    # Unknown type
    ]
    
    for liquid, volume in test_cases:
        print(f"🔬 {liquid} @ {volume} mL:")
        params = smart_params.get_parameters_for_liquid(liquid, volume)
        print(f"   → aspirate_speed: {params.aspirate_speed}")
        print(f"   → dispense_speed: {params.dispense_speed}")
        print(f"   → aspirate_wait_time: {params.aspirate_wait_time}")
        print(f"   → dispense_wait_time: {params.dispense_wait_time}")
        print(f"   → pre_asp_air_vol: {params.pre_asp_air_vol}")
        print(f"   → blowout_vol: {params.blowout_vol}")
        print()
    
    print("💡 Usage in robot code:")
    print("   smart = SmartPipettingParams('my_liquid_database.csv')")
    print("   params = smart.get_parameters_for_liquid('ethanol', 0.1)")
    print("   robot.aspirate_from_vial('source', 0.1, parameters=params)")


if __name__ == "__main__":
    demo_smart_parameters()