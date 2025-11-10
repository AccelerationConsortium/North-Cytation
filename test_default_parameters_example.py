"""
Example modification to calibration_validation.py to test default parameters:

def validate_volumes_with_defaults(lash_e, volumes, replicates, simulate):
    '''Test with system defaults instead of calibrated parameters'''
    
    for volume in volumes:
        # Use system defaults - no wizard lookup
        params_dict = {
            'aspirate_speed': 20,  # System default
            'dispense_speed': 20,  # System default  
            'aspirate_wait_time': 1.0,
            'dispense_wait_time': 1.0,
            'blowout_vol': 0.05,
            'overaspirate_vol': 0.01,
            'post_asp_air_vol': 0.05
        }
        
        result = pipet_and_measure(
            lash_e=lash_e,
            source_vial="liquid_source_0",
            dest_vial="measurement_vial_0", 
            volume=volume,
            params=params_dict,
            expected_measurement=volume * 1.0,  # Assume water density
            expected_time=30.0,
            replicate_count=replicates,
            simulate=simulate,
            raw_path="validation_defaults.csv",
            raw_measurements=[],
            liquid="water",  # For density calculation only
            new_pipet_each_time=False,
            trial_type="DEFAULT_VALIDATION"
        )
        
        # The robot methods will use:
        # defaults → no liquid calibration (liquid=None internally) → user params
        
# Or test the robot methods directly with liquid=None:
lash_e.nr_robot.aspirate_from_vial('vial1', 0.1, liquid=None)  # ✅ Pure defaults
"""

print(__doc__)