# Simplified Calibration Workflow - Data Flow Documentation

## Overview
This document maps out what data is determined in each phase and where it flows through the simplified calibration workflow.

## Phase 1: Screening (First Volume Only)
**Purpose**: Initial exploration of parameter space using SOBOL or LLM suggestions

**Input**:
- `ALL_PARAMS` (8 parameters to optimize)
- `INITIAL_SUGGESTIONS` (default: 5 trials)
- First volume from `VOLUMES` list

**Process**:
- Creates 3-objective optimizer with all 8 parameters
- Runs `INITIAL_SUGGESTIONS` trials using adaptive measurement
- Each trial: single measurement → conditional 2 additional replicates if deviation ≤ 10%

**Output Data Structure**:
```python
screening_results = [
    {
        # Parameter values
        "aspirate_speed": 25,
        "dispense_speed": 20,
        "aspirate_wait_time": 5.0,
        "dispense_wait_time": 3.0,
        "retract_speed": 8.0,
        "blowout_vol": 0.08,
        "post_asp_air_vol": 0.05,
        "overaspirate_vol": 0.015,
        
        # Performance metrics
        "volume": 0.05,
        "deviation": 7.2,         # Accuracy (% deviation from target)
        "variability": 3.1,       # Precision (range-based variability %)
        "time": 18.5,             # Average time per measurement
        
        # Metadata
        "trial_index": 1,
        "strategy": "SCREENING",
        "liquid": "glycerol",
        "time_reported": "2025-10-29T...",
        "replicate_count": 3,
        "raw_measurements": [0.0472, 0.0478, 0.0475]  # Individual volume measurements
    },
    # ... 4 more screening results
]
```

**Data Flow**: `screening_results` → fed to 3-objective optimizer AND stored in `all_results`

## Phase 2: Candidate Ranking & Selection
**Purpose**: Rank screening candidates to select best for overaspirate calibration

**Input**: 
- `screening_results` from Phase 1
- `tolerances` (volume-dependent accuracy/precision thresholds)

**Process**:
- Calls `rank_candidates_by_priority(screening_results, volume, tolerances)`
- Sorting priority: 1st accuracy → 2nd precision → 3rd time
- Evaluates each candidate using `evaluate_trial_quality()`

**Output**:
```python
ranked_candidates = [
    {
        # All original screening data PLUS:
        "quality_evaluation": {
            "is_good": True,
            "accuracy_ok": True,
            "precision_ok": True,
            "precision_value": 3.1,
            "accuracy_deviation_ul": 3.6,
            "accuracy_tolerance_ul": 5.0
        },
        "accuracy_score": 7.2,   # Lower is better
        "precision_score": 3.1,  # Lower is better  
        "time_score": 18.5       # Lower is better
    },
    # ... ranked best to worst  
]
```

**Data Flow**: `ranked_candidates[0]` (best) → used for overaspirate calibration

## Phase 3: Overaspirate Calibration (Multi-Volume Testing)
**Purpose**: Test best candidate parameters on remaining volumes to calibrate overaspirate formula

**Input**:
- `ranked_candidates[0]` (best parameters from screening)
- `remaining_volumes` (e.g., [0.025, 0.01] if first volume was 0.05)

**Process**:
- Extracts best parameters: `{k: v for k, v in best_candidate.items() if k in ALL_PARAMS}`
- Tests these parameters on each remaining volume (single measurement, no replicates)
- Fits linear regression to shortfalls: `shortfall = slope * volume + intercept`
- Calculates new overaspirate formula: `overaspirate = base_ul + scaling_percent * volume`

**Output**:
```python
calibration_data = [
    {
        'volume_set': 50.0,           # Target volume in µL
        'volume_measured': 46.4,      # Actual measured volume in µL  
        'deviation_pct': 7.2,         # Deviation percentage
        'existing_overaspirate_ul': 15.0,  # Overaspirate used in µL
        'slope': 0.0324,              # Linear fit slope
        'intercept': 2.15             # Linear fit intercept
    },
    {
        'volume_set': 25.0,
        'volume_measured': 23.1,
        'deviation_pct': 7.6,
        'existing_overaspirate_ul': 15.0,
        'slope': 0.0324,
        'intercept': 2.15
    },
    # ... one entry per tested volume
]

new_base_ul = 18.2        # Updated base overaspirate 
new_scaling_percent = 3.24 # Updated scaling percentage
```

**Data Flow**: 
- `new_base_ul`, `new_scaling_percent` → update global `OVERASPIRATE_BASE_UL`, `OVERASPIRATE_SCALING_PERCENT`
- These updated values used for all subsequent volume optimization

## Phase 4: 3-Objective Optimization (First Volume)
**Purpose**: Continue optimizing first volume with updated overaspirate parameters

**Input**:
- Updated overaspirate parameters from Phase 3
- `all_results` (includes screening results)
- Stopping criteria: `MAX_MEASUREMENTS` (60) OR `MIN_GOOD_TRIALS` (6)

**Process**:
- Continues with 3-objective optimizer from Phase 1
- Uses adaptive measurement strategy
- Checks `check_stopping_criteria()` after each trial
- Stops when either criterion is met

**Output**:
```python
# Additional entries added to all_results:
{
    # Same structure as screening, but:
    "strategy": "OPTIMIZATION_1", "OPTIMIZATION_2", etc.
    # Performance data continues to be collected
}
```

**Data Flow**: Additional trials → `all_results` → fed to 3-objective optimizer

## Phase 5: Best Candidate Selection (First Volume)
**Purpose**: Select final optimal parameters for first volume

**Input**:
- All first volume trials from `all_results` (screening + optimization)
- Excludes precision tests: `strategy not in ['PRECISION_TEST', 'PRECISION']`

**Process**:
- Filters: `first_volume_trials = [r for r in all_results if r.get('volume') == volume and ...]`
- Ranks using same `rank_candidates_by_priority()` system
- Selects `ranked_candidates[0]` (best overall)

**Output**:
```python
best_params = {
    "aspirate_speed": 23,
    "dispense_speed": 22,
    "aspirate_wait_time": 4.5,
    "dispense_wait_time": 2.8,
    "retract_speed": 7.8,
    "blowout_vol": 0.085,
    "post_asp_air_vol": 0.048,
    "overaspirate_vol": 0.018
}
```

**Data Flow**: `best_params` → used as baseline for subsequent volumes

## Phase 6: Subsequent Volume Optimization
**Purpose**: Optimize only volume-dependent parameters for remaining volumes

**Input**:
- `best_params` from first volume (fixed baseline)
- Current volume to optimize
- `VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]`

**Process**:
1. **Precision Test**: Test baseline parameters on new volume
2. **If Pass**: Done, use baseline parameters
3. **If Fail**: Run single-objective optimization (deviation only)
   - `fixed_params = {k: v for k, v in best_params.items() if k not in volume_dependent_params}`
   - Only optimize `blowout_vol` and `overaspirate_vol`
   - Uses proper single-objective optimizer (not 3-objective hack)
4. **Repeat**: Precision test → optimization cycles (max 3 attempts)

**Output**:
```python
volume_specific_params = {
    # Fixed from first volume
    "aspirate_speed": 23,
    "dispense_speed": 22,
    "aspirate_wait_time": 4.5,
    "dispense_wait_time": 2.8,
    "retract_speed": 7.8,
    "post_asp_air_vol": 0.048,
    
    # Optimized for this volume
    "blowout_vol": 0.065,      # Different from first volume
    "overaspirate_vol": 0.012  # Different from first volume
}
```

**Data Flow**: `volume_specific_params` → stored in `optimal_conditions` for final results

## Final Output Structure
```python
optimal_conditions = [
    {
        'volume_ml': 0.05,
        'volume_ul': 50.0,
        'status': 'success',
        # All 8 optimized parameters for this volume
        'aspirate_speed': 23,
        'dispense_speed': 22,
        # ... etc
    },
    {
        'volume_ml': 0.025,
        'volume_ul': 25.0,
        'status': 'success',
        # 6 fixed + 2 volume-specific parameters
        'aspirate_speed': 23,      # Fixed from first volume
        'blowout_vol': 0.065,     # Optimized for this volume
        # ... etc
    }
]
```

## Key Data Flow Dependencies

1. **Screening** determines parameter space exploration quality
2. **Ranking** ensures best candidate selection for overaspirate calibration
3. **Overaspirate calibration** provides updated parameters for subsequent optimization
4. **First volume optimization** provides baseline parameters for selective optimization
5. **Subsequent volumes** inherit most parameters, only optimize volume-dependent ones

## Critical Data Integrity Points

- Parameter copying: Always use `.copy()` to prevent contamination
- Volume conversion: Consistent mL ↔ µL conversions throughout
- Trial filtering: Proper exclusion of precision tests from optimization data
- Stopping criteria: Count measurements vs. parameter sets correctly