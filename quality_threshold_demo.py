#!/usr/bin/env python3
"""
Example showing configurable quality thresholds for mass monitoring
"""

# Example usage for different quality requirements:

# Standard quality (default - suitable for most pipetting)
standard_threshold = 0.001  # 1.0mg standard deviation

# Strict quality (for very low volume or critical measurements) 
strict_threshold = 0.0005   # 0.5mg standard deviation

# Lenient quality (for quick/rough measurements)
lenient_threshold = 0.002   # 2.0mg standard deviation

def demonstrate_quality_levels():
    """Show how different thresholds would affect quality assessment."""
    
    # Example stability data from your measurement
    example_stability_info = {
        'pre_stable_count': 5,
        'pre_total_count': 5, 
        'pre_baseline_std': 0.000000,  # Perfect pre-baseline
        'post_stable_count': 2,
        'post_total_count': 5,
        'post_baseline_std': 0.000612  # Some noise in post-baseline
    }
    
    print("=== Quality Assessment Example ===")
    print(f"Pre-baseline: {example_stability_info['pre_stable_count']}/{example_stability_info['pre_total_count']} stable, "
          f"std={example_stability_info['pre_baseline_std']:.6f}g")
    print(f"Post-baseline: {example_stability_info['post_stable_count']}/{example_stability_info['post_total_count']} stable, "
          f"std={example_stability_info['post_baseline_std']:.6f}g")
    print()
    
    # Test different threshold levels
    thresholds = [
        ("Strict", strict_threshold),
        ("Standard", standard_threshold), 
        ("Lenient", lenient_threshold)
    ]
    
    for name, threshold in thresholds:
        # Pre-baseline check
        pre_stable_pct = (example_stability_info['pre_stable_count'] / example_stability_info['pre_total_count']) * 100
        pre_stable = (pre_stable_pct > 50.0) or (example_stability_info['pre_baseline_std'] < threshold)
        
        # Post-baseline check
        post_stable_pct = (example_stability_info['post_stable_count'] / example_stability_info['post_total_count']) * 100
        post_stable = (post_stable_pct > 50.0) or (example_stability_info['post_baseline_std'] < threshold)
        
        # Overall result
        is_acceptable = pre_stable and post_stable
        
        print(f"{name:8} ({threshold:.6f}g): {'✅ PASS' if is_acceptable else '❌ FAIL'}")
        print(f"         Pre: {'✅' if pre_stable else '❌'} "
              f"({pre_stable_pct:.0f}% stable OR std<{threshold:.6f}g)")
        print(f"         Post: {'✅' if post_stable else '❌'} "
              f"({post_stable_pct:.0f}% stable OR std<{threshold:.6f}g)")
        print()
    
    print("=== Usage Examples ===")
    print("# For very low volume measurements (1-5uL):")
    print("validate_pipetting_accuracy(..., quality_std_threshold=0.0003)  # 0.3mg")
    print()
    print("# For high precision experiments:")
    print("validate_pipetting_accuracy(..., quality_std_threshold=0.0005)  # 0.5mg")
    print()
    print("# For quick/rough validation:")
    print("validate_pipetting_accuracy(..., quality_std_threshold=0.002)   # 2.0mg")
    print()
    print("# In calibration protocol initialization:")
    print("protocol = HardwareCalibrationProtocol(lash_e, quality_std_threshold=0.0005)")

if __name__ == "__main__":
    demonstrate_quality_levels()