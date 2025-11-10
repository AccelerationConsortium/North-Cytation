# Quick test of tolerance calculation

VOLUME_TOLERANCE_RANGES = [
    {'min_ul': 200, 'max_ul': 1000, 'tolerance_pct': 1.0, 'name': 'large_volume'},
    {'min_ul': 60,  'max_ul': 200,  'tolerance_pct': 2.0, 'name': 'medium_large_volume'},
    {'min_ul': 20,  'max_ul': 60,   'tolerance_pct': 3.0, 'name': 'medium_volume'},
    {'min_ul': 1,   'max_ul': 20,   'tolerance_pct': 5.0, 'name': 'small_volume'},
    {'min_ul': 0,   'max_ul': 1,    'tolerance_pct': 10.0, 'name': 'micro_volume'},
]

def get_volume_dependent_tolerances(volume_ml):
    """Calculate volume-dependent tolerances."""
    volume_ul = volume_ml * 1000
    
    # Find appropriate tolerance range
    tolerance_pct = None
    range_name = 'unknown'
    
    print(f"Volume tolerance lookup for {volume_ul}μL")
    for vol_range in VOLUME_TOLERANCE_RANGES:
        in_range = vol_range['min_ul'] <= volume_ul < vol_range['max_ul']
        print(f"  Range {vol_range['name']}: [{vol_range['min_ul']}, {vol_range['max_ul']}) = {vol_range['tolerance_pct']}% → {in_range}")
        if in_range:
            tolerance_pct = vol_range['tolerance_pct']
            range_name = vol_range['name']
            break
    
    if tolerance_pct is None:
        tolerance_pct = 10.0
        range_name = 'fallback'
    
    base_tolerance_ul = volume_ul * (tolerance_pct / 100.0)
    
    return {
        'deviation_ul': base_tolerance_ul,
        'variation_ul': base_tolerance_ul,
        'tolerance_percent': tolerance_pct,
        'range_name': range_name
    }

# Test different volumes
for vol_ml in [0.025, 0.05, 0.1]:
    print(f"\n=== Testing {vol_ml*1000}μL ===")
    tolerances = get_volume_dependent_tolerances(vol_ml)
    print(f"Result: {tolerances}")