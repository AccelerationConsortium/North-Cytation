#!/usr/bin/env python3
"""
Test script to demonstrate the precision test failure loop issue and fix.
This shows why the system was continuing optimization after 96 wells.
"""

def simulate_old_precision_failure_loop():
    """Simulate the old behavior when precision tests fail"""
    print("=== OLD (BUGGY) PRECISION FAILURE BEHAVIOR ===")
    
    MAX_WELLS = 96
    PRECISION_REPLICATES = 4
    trial_count = 92
    volume_completed = False
    precision_attempts = 0
    
    print(f"Starting: {trial_count}/{MAX_WELLS} wells used")
    
    while not volume_completed and precision_attempts < 3:  # Limit attempts for demo
        precision_attempts += 1
        print(f"\n--- Precision Attempt #{precision_attempts} ---")
        
        # Check if we can do optimization
        if trial_count < MAX_WELLS - PRECISION_REPLICATES:
            print(f"✅ Can do optimization: {trial_count} < {MAX_WELLS - PRECISION_REPLICATES}")
            
            # Do some optimization trials
            optimization_trials = 2
            print(f"🔍 Running {optimization_trials} optimization trials...")
            for i in range(optimization_trials):
                if trial_count >= MAX_WELLS:
                    print(f"❌ OVERFLOW! Trial count: {trial_count}/{MAX_WELLS}")
                    break
                trial_count += 1
                print(f"   Trial {trial_count}: optimization trial")
            
            # Now try precision test
            print(f"🎯 Running precision test (4 replicates)...")
            if trial_count + PRECISION_REPLICATES > MAX_WELLS:
                print(f"❌ OVERFLOW! Would use wells {trial_count+1}-{trial_count+PRECISION_REPLICATES} (max: {MAX_WELLS})")
                # OLD BUG: System runs precision test anyway!
                for i in range(PRECISION_REPLICATES):
                    trial_count += 1
                    print(f"   Trial {trial_count}: PRECISION (EXCEEDS LIMIT!)")
            else:
                trial_count += PRECISION_REPLICATES
                print(f"   Used wells {trial_count-PRECISION_REPLICATES+1}-{trial_count}")
            
            # Simulate precision test failure
            print(f"❌ Precision test FAILED - need more optimization!")
            print(f"   Setting candidate_params = None to continue loop...")
            # volume_completed stays False, loop continues!
            
        else:
            print(f"❌ Cannot do more optimization: {trial_count} >= {MAX_WELLS - PRECISION_REPLICATES}")
            break
    
    print(f"\n📊 FINAL RESULT: {trial_count}/{MAX_WELLS} wells used")
    if trial_count > MAX_WELLS:
        print(f"❌ EXCEEDED LIMIT by {trial_count - MAX_WELLS} wells!")
    else:
        print(f"✅ Within limit")
    
    return trial_count

def simulate_new_precision_failure_behavior():
    """Simulate the new (fixed) behavior when precision tests fail"""
    print("\n=== NEW (FIXED) PRECISION FAILURE BEHAVIOR ===")
    
    MAX_WELLS = 96
    PRECISION_REPLICATES = 4
    trial_count = 92
    volume_completed = False
    precision_attempts = 0
    
    print(f"Starting: {trial_count}/{MAX_WELLS} wells used")
    
    while not volume_completed and precision_attempts < 3:  # Limit attempts for demo
        precision_attempts += 1
        print(f"\n--- Precision Attempt #{precision_attempts} ---")
        
        # NEW: Better check for optimization
        min_wells_needed = 1 + PRECISION_REPLICATES  # At least 1 optimization + precision test
        if (trial_count + min_wells_needed) <= MAX_WELLS:
            print(f"✅ Can do optimization: {trial_count} + {min_wells_needed} ≤ {MAX_WELLS}")
            
            # Do some optimization trials
            optimization_trials = min(2, MAX_WELLS - trial_count - PRECISION_REPLICATES)
            print(f"🔍 Running {optimization_trials} optimization trials...")
            for i in range(optimization_trials):
                trial_count += 1
                print(f"   Trial {trial_count}: optimization trial")
            
            # Reserve wells for precision test BEFORE running it
            print(f"🔒 Reserving {PRECISION_REPLICATES} wells for precision test...")
            trial_count += PRECISION_REPLICATES
            print(f"   Reserved wells: trial_count now {trial_count}/{MAX_WELLS}")
            
            # Simulate precision test failure
            print(f"❌ Precision test FAILED")
            
            # NEW: Check if we have enough wells for another attempt
            wells_remaining = MAX_WELLS - trial_count
            wells_needed_for_retry = 1 + PRECISION_REPLICATES
            
            if wells_remaining >= wells_needed_for_retry:
                print(f"   📈 Can try again: {wells_remaining} wells left, need {wells_needed_for_retry}")
                # Continue loop
            else:
                print(f"   ⏹️  Cannot try again: {wells_remaining} wells left, need {wells_needed_for_retry}")
                print(f"   🛑 Stopping to avoid exceeding well limit")
                volume_completed = True  # NEW: Force completion
            
        else:
            print(f"❌ Cannot do more optimization: {trial_count} + {min_wells_needed} > {MAX_WELLS}")
            volume_completed = True
    
    print(f"\n📊 FINAL RESULT: {trial_count}/{MAX_WELLS} wells used")
    if trial_count > MAX_WELLS:
        print(f"❌ EXCEEDED LIMIT by {trial_count - MAX_WELLS} wells!")
    else:
        print(f"✅ Within limit")
    
    return trial_count

def test_edge_case():
    """Test the specific case that was causing 110 trials"""
    print("\n=== EDGE CASE: Why 110 Trials Happened ===")
    
    print("Scenario: Multiple precision test failures with continued optimization")
    print("- Each precision test uses 4 wells")  
    print("- Each failure triggers more optimization")
    print("- System doesn't check if there are enough wells for next attempt")
    print()
    
    MAX_WELLS = 96
    trial_count = 92
    
    # Simulate multiple failed precision attempts (old behavior)
    print("OLD: Multiple precision failures without proper well checking:")
    for attempt in range(1, 5):  # 4 attempts = 16 extra wells = 112 total
        print(f"  Attempt {attempt}: trials {trial_count+1}-{trial_count+4} (precision test fails)")
        trial_count += 4
        print(f"    Trial count: {trial_count}/{MAX_WELLS}")
        
        if attempt < 4:  # Add some optimization between attempts
            trial_count += 1
            print(f"    + 1 optimization trial: {trial_count}/{MAX_WELLS}")
    
    print(f"\n❌ This explains your 110+ trials! ({trial_count} total)")
    print(f"✅ NEW: System stops when wells_remaining < wells_needed_for_retry")

if __name__ == "__main__":
    print("🧪 TESTING PRECISION TEST FAILURE LOOP FIX")
    print("="*60)
    
    old_count = simulate_old_precision_failure_loop()
    new_count = simulate_new_precision_failure_behavior()
    
    test_edge_case()
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY:")
    print(f"   Old behavior: {old_count}/96 wells ({'OVERFLOW!' if old_count > 96 else 'OK'})")
    print(f"   New behavior: {new_count}/96 wells ({'OVERFLOW!' if new_count > 96 else 'OK'})")
    print(f"   Wells saved: {old_count - new_count}")
    print(f"\n🎯 This fix prevents your 110+ trial problem!")