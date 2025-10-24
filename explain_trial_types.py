#!/usr/bin/env python3
"""
SIMPLE EXPLANATION: How Trial Types Should Work

This explains the trial type system clearly and shows where the issue is.
"""

print("🎯 HOW TRIAL TYPES SHOULD WORK:")
print("=" * 50)
print()

print("📋 THE 4 TRIAL TYPES:")
print("   1. SCREENING   - Initial parameter exploration (first volume only)")
print("   2. OPTIMIZATION - Parameter refinement using Bayesian optimization")  
print("   3. PRECISION    - Final validation with multiple replicates")
print("   4. OVERVOLUME_ASSAY - Calibration tests for overaspirate volumes")
print()

print("🔄 TYPICAL WORKFLOW:")
print("   Volume 1 (100µL):")
print("      → 5 SCREENING trials (explore parameter space)")
print("      → 1-3 OPTIMIZATION trials (refine best parameters)")
print("      → 4 PRECISION trials (validate final parameters)")
print()
print("   Volume 2 (50µL):")
print("      → 0 SCREENING trials (use Volume 1 baseline)")
print("      → 1-5 OPTIMIZATION trials (adapt parameters for 50µL)")
print("      → 4 PRECISION trials (validate adapted parameters)")
print()
print("   Volume 3 (25µL):")
print("      → 0 SCREENING trials (use Volume 1 baseline)")
print("      → 1-5 OPTIMIZATION trials (adapt parameters for 25µL)")  
print("      → 4 PRECISION trials (validate adapted parameters)")
print()

print("❌ WHAT WE'RE SEEING:")
print("   • NO precision trials at all")
print("   • Most trials labeled as OPTIMIZATION")
print("   • This suggests precision tests are running but getting mislabeled")
print()

print("🔍 THE PROBLEM:")
print("   The precision test function is being called, but somewhere the")
print("   trial_type is getting overridden or not passed correctly.")
print()

print("🔧 HOW TO DEBUG:")
print("   1. Check if run_precision_test() is actually being called")
print("   2. Check if pipet_and_measure() gets the correct trial_type")
print("   3. Check if clean_raw_measurement_data() overrides it")
print()

print("💡 THE FIX:")
print("   Find where precision trials are getting mislabeled as OPTIMIZATION")
print("   and ensure the trial_type='PRECISION' parameter is preserved.")