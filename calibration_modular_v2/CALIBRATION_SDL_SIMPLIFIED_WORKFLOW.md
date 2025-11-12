# Calibration SDL Simplified - Main Process Workflow

This document describes the proven workflow implemented in `calibration_sdl_simplified.py` that we need to replicate in the modular v2 system.

## 🎯 **Overall Architecture**

### **Global State Management**
- **Global measurement counter**: `global_measurement_count` - tracks ALL measurements across entire experiment
- **Hard budget enforcement**: `MAX_MEASUREMENTS` (96 total) - HARD STOP when exceeded
- **Volume-specific calibrations**: `volume_overaspirate_calibrations` - stores overaspirate constraints per volume
- **Global transfer learning**: `global_ax_client` - optional shared optimizer across volumes

### **Key Configuration Parameters**
```python
MAX_MEASUREMENTS = 96                    # TOTAL for entire experiment
MAX_MEASUREMENTS_INITIAL_VOLUME = 60    # Max for first volume
MIN_GOOD_PARAMETER_SETS = 6             # Stopping criteria
PRECISION_MEASUREMENTS = 3              # Replicates for good results
INITIAL_PARAMETER_SETS = 5              # Screening phase size
ADAPTIVE_DEVIATION_THRESHOLD = 10.0     # % threshold for replicates
BAYESIAN_MODEL_TYPE = 'qNEHVI'          # Multi-objective for 1st volume
BAYESIAN_MODEL_TYPE_SUBSEQUENT = 'qLogEI' # Single-objective for 2nd+ volumes
```

## 🔄 **Main Workflow Loop**

### **For Each Volume:**

```python
for volume_index, volume in enumerate(VOLUMES):
    if volume_index == 0:
        # FIRST VOLUME: Full optimization
        success, best_params, best_candidate = optimize_first_volume(...)
        
        # Post-optimization overaspirate calibration for subsequent volumes
        volume_calibrations = calibrate_overvolume_post_optimization(...)
        
    else:
        # SUBSEQUENT VOLUMES: Budget-aware optimization
        measurements_budget = calculate_measurements_per_volume(...)
        success, best_params, status = optimize_subsequent_volume_budget_aware(...)
```

## 📋 **Phase 1: First Volume Optimization**

### **1.1 External Data Loading OR Screening**
```python
# Try external data first, fall back to screening
screening_results = load_external_data_or_run_screening(...)

# External data loading:
if USE_EXTERNAL_DATA and external_csv_exists:
    screening_results = load_external_calibration_data(volume, liquid)
    # Converts CSV data to trial format, filters by volume/liquid
else:
    # Traditional screening with SOBOL/LLM
    screening_results = run_screening_phase(...)
```

**Screening Phase Details:**
- **LLM vs SOBOL**: If `USE_LLM_FOR_SCREENING=True`, get LLM suggestions, else use Ax SOBOL
- **Adaptive measurement**: Each screening trial uses conditional replicates
- **Fixed parameters**: Apply `FIXED_PARAMETERS` to all suggestions
- **Result format**: Store as trial dictionaries with parameters + performance

### **1.2 First Volume Constraint Calculation**
```python
# Use best screening candidate to calculate overaspirate constraint
best_candidate = rank_candidates_by_priority(screening_results)[0]
min_overaspirate_ml, max_overaspirate_ml = calculate_first_volume_constraint(best_candidate, volume)

# Recreate optimizer with updated constraints
ax_client = optimizer_3obj.create_model(
    min_overaspirate_ul=min_overaspirate_ml * 1000,
    max_overaspirate_ul=max_overaspirate_ml * 1000
)
```

### **1.3 3-Objective Optimization Loop**
```python
while True:
    # Check stopping criteria (measurements OR good trials)
    stopping_result = check_stopping_criteria(all_results, volume, tolerances)
    if stopping_result['should_stop']:
        break
    
    # Get Bayesian suggestion (qNEHVI multi-objective)
    params, trial_index = optimizer_3obj.get_suggestions(ax_client, volume, n=1)[0]
    
    # Apply fixed parameters
    for param_name, fixed_value in FIXED_PARAMETERS.items():
        params[param_name] = fixed_value
    
    # Run adaptive measurement
    adaptive_result = run_adaptive_measurement(...)
    
    # Update Bayesian optimizer
    model_result = {
        "deviation": adaptive_result['deviation'],
        "variability": adaptive_result['variability'], 
        "time": adaptive_result['time']
    }
    optimizer_3obj.add_result(ax_client, trial_index, model_result)
```

**Stopping Criteria:**
- **Measurement limit**: `>= MAX_MEASUREMENTS_INITIAL_VOLUME` individual measurements
- **Quality threshold**: `>= MIN_GOOD_PARAMETER_SETS` good parameter sets
- **Budget enforcement**: Global `>= MAX_MEASUREMENTS` total measurements

### **1.4 Best Candidate Selection**
```python
# Rank all first volume trials (screening + optimization)
first_volume_trials = [r for r in all_results if r.get('volume') == volume]
ranked_candidates = rank_candidates_by_priority(first_volume_trials, volume, tolerances)
best_candidate = ranked_candidates[0]
```

**Ranking System:**
- **Multi-metric scoring**: Accuracy (50%) + Precision (40%) + Time (10%)
- **Normalization**: Standard deviation-based to compress small differences
- **Quality evaluation**: Against volume-dependent tolerances

## 📋 **Phase 2: Post-Optimization Overaspirate Calibration**

```python
def calibrate_overvolume_post_optimization(optimized_params, remaining_volumes, ...):
    """Test optimized parameters on remaining volumes to calculate volume-specific constraints"""
    
    volume_calibrations = {}
    for volume in remaining_volumes:
        # Test optimized params with PRECISION_MEASUREMENTS replicates
        for rep in range(PRECISION_MEASUREMENTS):
            result = pipet_and_measure_tracked(...)
        
        # Calculate volume-specific shortfall and adjustment
        shortfall_ul = target_volume_ul - measured_volume_ul
        existing_overaspirate_ul = optimized_params.get('overaspirate_vol') * 1000
        
        # Calculate constraints: existing + shortfall + buffer
        guess_overaspirate_ul = existing_overaspirate_ul + shortfall_ul
        max_overaspirate_ul = guess_overaspirate_ul + OVERVOLUME_CALIBRATION_BUFFER_UL
        
        volume_calibrations[volume] = {
            'guess_ml': guess_overaspirate_ul / 1000,
            'max_ml': max_overaspirate_ul / 1000
        }
    
    return volume_calibrations
```

## 📋 **Phase 3: Subsequent Volume Optimization**

### **3.1 Budget Allocation**
```python
def calculate_measurements_per_volume(global_measurement_count, volumes_remaining):
    measurements_remaining = MAX_MEASUREMENTS - global_measurement_count
    measurements_per_volume = measurements_remaining // volumes_remaining
    return max(0, measurements_per_volume)
```

### **3.2 Inherited Parameter Testing**
```python
# Use parameters from first volume + volume-specific overaspirate if available
test_params = successful_params.copy()
if volume in volume_overaspirate_calibrations:
    test_params['overaspirate_vol'] = volume_calibrations[volume]['guess_ml']

# Test inherited parameters with adaptive measurement
inherited_result = run_adaptive_measurement(...)

if inherited_result['deviation'] <= ADAPTIVE_DEVIATION_THRESHOLD:
    # Good result - run additional replicates
    for rep in range(PRECISION_MEASUREMENTS - 1):
        additional_result = pipet_and_measure_tracked(...)
    
    # Check if tolerance met - if yes, stop optimization
    if tolerance_met:
        return True, test_params, 'success'
```

### **3.3 Volume-Dependent Parameter Optimization**
```python
# Only optimize volume-dependent parameters: ['overaspirate_vol', 'blowout_vol']
# Fix all other parameters from successful first volume

if use_single_objective:
    # qLogEI - optimize deviation only  
    ax_client = optimizer_single.create_model(
        optimize_params=['overaspirate_vol', 'blowout_vol'],
        fixed_params=non_volume_dependent_params
    )
else:
    # qNEHVI - optimize deviation + variability
    ax_client = optimizer_3obj.create_model(
        optimize_params=['overaspirate_vol', 'blowout_vol'], 
        fixed_params=non_volume_dependent_params
    )

# Run optimization within remaining budget
while measurements_used < budget:
    params = ax_client.get_suggestions(...)[0]
    adaptive_result = run_adaptive_measurement(...)
    ax_client.add_result(...)
```

## 🔬 **Adaptive Measurement System**

### **Core Logic:**
```python
def run_adaptive_measurement(deviation_threshold=10.0):
    # Step 1: Single initial measurement
    initial_result = pipet_and_measure_tracked(...)
    initial_deviation = initial_result['deviation']
    
    if initial_deviation > deviation_threshold:
        # Poor accuracy - don't waste replicates
        return {
            'deviation': initial_deviation,
            'variability': ADAPTIVE_PENALTY_VARIABILITY,  # 100.0 penalty
            'replicate_count': 1
        }
    else:
        # Good accuracy - run additional replicates
        for rep in range(PRECISION_MEASUREMENTS - 1):
            additional_result = pipet_and_measure_tracked(...)
        
        # Calculate variability from volume measurements
        variability = (max_vol - min_vol) / (2 * avg_vol) * 100
        
        return {
            'deviation': avg_deviation,
            'variability': variability,
            'replicate_count': PRECISION_MEASUREMENTS
        }
```

**Key Features:**
- **Conditional replicates**: Only run multiple measurements for promising results
- **Budget preservation**: Avoid wasting measurements on poor parameter sets
- **Quality signal**: Penalty variability clearly marks single-measurement trials

## 🎯 **Critical Integration Points for Modular v2**

### **1. Measurement Tracking**
- **Global counter**: Every `pipet_and_measure_tracked()` call increments `global_measurement_count`
- **Hard budget**: Return `None` if `global_measurement_count >= MAX_MEASUREMENTS`
- **Volume allocation**: Calculate per-volume budgets from remaining total

### **2. Ax/BoTorch Integration**  
- **Multi-objective first volume**: Use qNEHVI with deviation + variability + time
- **Single-objective subsequent**: Use qLogEI with deviation only  
- **Parameter constraints**: Volume-dependent overaspirate bounds from post-optimization calibration
- **Transfer learning**: Fixed parameters from first volume, optimize only volume-dependent

### **3. Adaptive Measurement**
- **Threshold-based**: 10% deviation threshold for conditional replicates
- **Penalty system**: 100.0 variability penalty for single measurements
- **Range-based variability**: `(max-min)/(2*mean)*100` for multiple measurements

### **4. Result Structures**
- **Trial format**: Dict with parameters + performance metrics
- **Ranking system**: Weighted composite scoring (50% accuracy, 40% precision, 10% time)  
- **Quality evaluation**: Boolean tolerance checks against volume-dependent thresholds

### **5. External Data Integration**
- **CSV format**: Standard columns for parameters + performance
- **Screening replacement**: Load external data instead of SOBOL/LLM screening  
- **Volume/liquid filtering**: Target specific conditions from historical data

## 📊 **Success Metrics**

### **Proven Performance:**
- **Transfer learning efficiency**: 67% reduction in trials (19 vs 32 for 3 volumes)
- **Budget compliance**: Hard enforcement of 96 measurement limit
- **Quality assurance**: Volume-dependent tolerance validation
- **Bayesian optimization**: Proven qNEHVI/qLogEI performance with Ax integration

### **Output Requirements:**
- **CSV exports**: Trial results, raw measurements, optimal conditions
- **Performance tracking**: Measured volumes, deviations, timing
- **Configuration logging**: All hyperparameters and settings
- **Statistical analysis**: Automated insights and recommendations

This workflow has been proven in production and should be replicated exactly in the modular v2 system to maintain the same efficiency and reliability.