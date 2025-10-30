# Calibration SDL Simplified - System Overview

## Executive Summary

The Calibration SDL Simplified system is an automated pipetting parameter optimization workflow that uses Bayesian multi-objective optimization to find optimal liquid handling parameters across multiple volumes. It combines intelligent screening, adaptive measurement strategies, and progressive optimization to achieve accurate and precise pipetting for various liquids (water, glycerol, etc.) while maintaining strict measurement budgets.

## System Architecture

### Core Dependencies
- **Master Controller**: `Lash_E` (from `master_usdl_coordinator.py`) - orchestrates hardware
- **Hardware**: North Robot (pipetting), North Track (wellplate handling), Cytation 5 (measurements)
- **Optimization**: Bayesian optimizers using Ax/BoTorch framework
- **Simulation Support**: Full simulation mode for development/testing

### Key Modules
- **Optimizers**: 3-objective (`deviation + variability + time`) and single-objective optimizers
- **LLM Integration**: Optional LLM-based parameter suggestions for initial exploration
- **Vial Management**: Automated liquid source and measurement vial handling
- **Analysis Pipeline**: Performance evaluation and ranking systems

## Workflow Architecture

### High-Level Process Flow
1. **First Volume**: Full 3-objective optimization (all parameters)
2. **Overaspirate Calibration**: Cross-volume parameter adjustment (optional)
3. **Subsequent Volumes**: 2-objective optimization (volume-dependent parameters only)
4. **Results Generation**: Performance analysis and optimal parameter export

### Volume Processing Strategy
- **First Volume**: Optimize all 8 pipetting parameters with maximum budget allocation
- **Subsequent Volumes**: Inherit speed/timing parameters, optimize only volume-dependent parameters (`overaspirate_vol`, `blowout_vol`)
- **Budget-Aware**: Dynamic measurement allocation across volumes

## Core Configuration System

### Measurement Budget Management
```python
DEFAULT_MAX_MEASUREMENTS = 96                    # Total for entire calibration
DEFAULT_MAX_MEASUREMENTS_INITIAL_VOLUME = 60     # Maximum for first volume  
DEFAULT_PRECISION_MEASUREMENTS = 3               # Replicates when deviation ≤ 10%
```

### Volume Configuration
```python
DEFAULT_VOLUMES = [0.05, 0.025, 0.01]           # mL - decreasing volume sequence
VOLUME_TOLERANCE_RANGES = [                      # Volume-dependent accuracy tolerances
    {'min_ul': 200, 'max_ul': 1000, 'tolerance_pct': 1.0},
    {'min_ul': 60,  'max_ul': 200,  'tolerance_pct': 2.0},
    {'min_ul': 20,  'max_ul': 60,   'tolerance_pct': 3.0},
    {'min_ul': 1,   'max_ul': 20,   'tolerance_pct': 5.0},
    {'min_ul': 0,   'max_ul': 1,    'tolerance_pct': 10.0}
]
```

### Optimization Strategy Configuration
```python
# Separate model types for different optimization phases
DEFAULT_BAYESIAN_MODEL_TYPE = 'qNEHVI'           # First volume (3-objective)
DEFAULT_BAYESIAN_MODEL_TYPE_SUBSEQUENT = 'qNEHVI' # Subsequent volumes (2-objective)

# Optimizer objective thresholds (higher = more learning gradient)
DEFAULT_OPTIMIZER_DEVIATION_THRESHOLD = 50.0     # 50% - anything worse treated equally
DEFAULT_OPTIMIZER_VARIABILITY_THRESHOLD = 25.0   # 25% - accommodates challenging liquids
DEFAULT_OPTIMIZER_TIME_THRESHOLD = 120.0         # 120s - reasonable upper limit
```

### Adaptive Measurement System
```python
DEFAULT_ADAPTIVE_DEVIATION_THRESHOLD = 10.0      # Threshold for running replicates
DEFAULT_ADAPTIVE_PENALTY_VARIABILITY = 100.0     # Penalty for poor single measurements
```

## First Volume Optimization (Comprehensive)

### Phase 1: Screening
- **Purpose**: Initial parameter space exploration
- **Method**: SOBOL quasi-random sampling OR LLM-guided suggestions
- **Budget**: 5 parameter sets (configurable via `INITIAL_PARAMETER_SETS`)
- **Parameters**: All 8 pipetting parameters optimized
- **Measurement Strategy**: Adaptive (1-3 measurements per parameter set based on initial accuracy)

### Phase 2: Overaspirate Calibration (Optional)
- **Trigger**: `AUTO_CALIBRATE_OVERVOLUME = True` and multiple volumes configured
- **Process**: Test best screening candidate on remaining volumes
- **Purpose**: Calculate volume-dependent overaspirate formula: `overaspirate = base_ul + scaling_pct * volume`
- **Algorithm**: Linear regression on volume shortfalls
- **Budget**: 1 measurement per remaining volume
- **Output**: Updated `OVERASPIRATE_BASE_UL` and `OVERASPIRATE_SCALING_PERCENT`

### Phase 3: Multi-Objective Optimization
- **Method**: 3-objective Bayesian optimization (qNEHVI)
- **Objectives**: 
  - `deviation` (accuracy) - minimize % error from target
  - `variability` (precision) - minimize measurement spread
  - `time` (speed) - minimize pipetting duration
- **Parameters**: All 8 parameters with updated overaspirate bounds
- **Stopping Criteria**: 
  - 60 total measurements for first volume OR
  - 6 "GOOD" parameter sets found
- **"GOOD" Definition**: Both accuracy and precision within volume-dependent tolerances

### Phase 4: Best Candidate Selection
- **Ranking Method**: Weighted composite scoring
  - Accuracy: 50% weight
  - Precision: 40% weight  
  - Time: 10% weight
- **Normalization**: Based on actual candidate pool ranges
- **Output**: Best parameter set for subsequent volume inheritance

## Adaptive Measurement Strategy

### Core Logic
```python
def run_adaptive_measurement():
    # Step 1: Single measurement
    initial_result = pipet_and_measure(params)
    
    # Step 2: Conditional replicates
    if initial_result.deviation <= 10.0:  # Good accuracy
        # Run 2 additional replicates (3 total)
        run_additional_replicates(2)
        variability = calculate_range_based_variability()
    else:  # Poor accuracy
        # Don't waste replicates, assign penalty
        variability = 100.0  # Penalty value
    
    return averaged_results
```

### Variability Calculation
```python
# Range-based variability for small sample sizes (3 replicates)
variability = (max_volume - min_volume) / (2 * avg_volume) * 100
```

### Key Benefits
- **Budget Efficiency**: Don't waste replicates on poor parameter sets
- **Statistical Validity**: Get precision data only when accuracy justifies it
- **Optimizer Feedback**: Clear signal between good (calculated variability) and bad (penalty) regions

## Subsequent Volume Optimization (Selective)

### Strategy Rationale
- **Speed/Timing Parameters**: Inherited from first volume (proven to work)
- **Volume-Dependent Parameters**: Re-optimized (`overaspirate_vol`, `blowout_vol`)
- **Optimization Type**: 2-objective (deviation + variability, time set to neutral 30s)

### Budget-Aware Process
1. **Inherited Parameter Test**: Test first volume parameters on new volume
2. **Conditional Optimization**: If inherited parameters fail tolerance, run constrained optimization
3. **Dynamic Budget**: Remaining global budget allocated across remaining volumes
4. **Early Termination**: Stop if good-enough parameters found within budget

### Implementation Details
```python
def optimize_subsequent_volume():
    # Test inherited parameters
    result = test_inherited_parameters()
    
    if meets_tolerance(result):
        return success, inherited_params
    else:
        # Run budget-constrained 2-objective optimization
        return run_optimization_within_budget()
```

## Measurement Budget System

### Global Budget Enforcement
```python
global_measurement_count = 0  # Tracks all measurements across entire calibration

def pipet_and_measure_tracked():
    if global_measurement_count >= MAX_MEASUREMENTS:
        return None  # Hard budget stop
    
    result = pipet_and_measure()
    global_measurement_count += 1
    return result
```

### Budget Allocation Strategy
- **First Volume**: Up to 60 measurements (screening + optimization)
- **Remaining Budget**: Distributed equally among remaining volumes
- **Minimum Protection**: Ensure at least basic optimization possible for each volume

## Performance Evaluation System

### Trial Quality Assessment
```python
def evaluate_trial_quality(trial_results, volume_ml, tolerances):
    # Accuracy check
    deviation_ul = (deviation_pct / 100.0) * volume_ul
    accuracy_ok = deviation_ul <= tolerances['deviation_ul']
    
    # Precision check  
    precision_ok = variability_pct <= tolerances['tolerance_percent']
    
    return {'is_good': accuracy_ok and precision_ok}
```

### Ranking System
- **Multi-criteria scoring** with data-driven normalization
- **Composite weighting**: Accuracy (50%) + Precision (40%) + Time (10%)
- **Volume-dependent tolerances**: Stricter requirements for larger volumes

## Hardware Integration

### Robot Control (`Lash_E`)
- **Pipetting**: North Robot with tip selection and parameter control
- **Transport**: North Track for wellplate management  
- **Measurement**: Cytation 5 for mass/volume determination
- **Vial Management**: Automated liquid source and waste handling

### Safety Features
- **Budget limits**: Hard stops to prevent runaway experiments
- **Error handling**: Graceful failure with Slack notifications
- **Simulation mode**: Complete workflow testing without hardware
- **Parameter bounds**: Physical safety limits on all parameters

## Output and Results

### Generated Files
- **Raw measurements**: `raw_measurements.csv` - every individual measurement
- **Optimization results**: `all_results.csv` - parameter sets and performance
- **Optimal conditions**: `optimal_conditions.csv` - best parameters per volume
- **Experiment summary**: `experiment_summary.txt` - human-readable report
- **Configuration**: `run_config.yaml` - exact hyperparameters used

### Performance Metrics
- **Success Rate**: Percentage of volumes meeting tolerance
- **Measurement Efficiency**: Results achieved vs budget used
- **Parameter Transfer**: How well first volume parameters work on subsequent volumes

## Simulation and Development Support

### Simulation Mode
```python
SIMULATE = True  # Full workflow without hardware
```
- **Complete workflow testing** without robot access
- **Deterministic but realistic** measurement simulation
- **Parameter validation** and algorithm development
- **Budget and timing analysis**

### Configuration Management
- **Default reset capability** for reproducible experiments
- **Environment variable overrides** for batch processing
- **Liquid-specific configurations** for different experimental conditions

## Key Design Principles

### 1. Budget-First Design
Every measurement counts toward global budget. No unlimited exploration.

### 2. Progressive Complexity
Simple screening → focused optimization → selective refinement

### 3. Adaptive Intelligence  
Measurement strategy adapts based on real-time performance assessment

### 4. Multi-Objective Reality
Balances accuracy, precision, and speed rather than optimizing single metric

### 5. Robust Failure Handling
Graceful degradation when conditions are challenging or budgets constrained

### 6. Experimental Realism
Thresholds and tolerances based on actual laboratory experience with different liquids

## Future Restructuring Considerations

### Modular Opportunities
- **Optimizer Engine**: Abstract interface for different optimization strategies
- **Measurement Manager**: Centralized budget and adaptive measurement logic
- **Volume Strategy**: Pluggable strategies for multi-volume optimization
- **Hardware Abstraction**: Clean separation between workflow logic and robot control

### Configuration System
- **Liquid Profiles**: Predefined configurations for different liquid types
- **Experiment Templates**: Common workflow patterns (screening-only, optimization-only, etc.)
- **Resource Profiles**: Different budget/time constraints for different scenarios

### Extension Points
- **Custom Objective Functions**: Domain-specific performance metrics
- **Alternative Optimizers**: Integration with other Bayesian optimization libraries
- **Real-time Analysis**: Live performance monitoring and early stopping
- **Multi-liquid Workflows**: Comparative optimization across liquid types