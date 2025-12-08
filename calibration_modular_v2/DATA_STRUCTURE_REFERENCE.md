# Data Structure Reference - Calibration System

## Current Data Structure Analysis (December 4, 2025)

### Core Domain Objects (data_structures.py)

#### Input/Parameter Objects
- **`CalibrationParameters`** - overaspirate_vol only
- **`HardwareParameters`** - Dict of hardware settings (aspirate_speed, dispense_speed, etc.)
- **`PipettingParameters`** - Combines calibration + hardware parameters
  - Used everywhere as the main parameter container

#### Measurement/Results Objects  
- **`RawMeasurement`** - Single measurement (timestamp, volume, duration)
- **`AdaptiveMeasurementResult`** - Analysis of multiple measurements (mean, std, CV)
- **`QualityEvaluation`** - Pass/fail assessment (overall_quality, meets_accuracy_threshold)
- **`TrialResult`** - MAIN result object: parameters + measurements + analysis + quality

#### Collection/Summary Objects
- **`VolumeTolerances`** - accuracy_tolerance_ul + precision_tolerance_pct  
- **`VolumeCalibrationResult`** - Results for one volume
- **`ExperimentResults`** - Results for entire experiment

#### Specialized Objects
- **`TwoPointCalibrationPoint`** - Single point in constraint calibration
- **`TwoPointCalibrationResult`** - Linear fit results from constraint calibration
- **`ConstraintBoundsUpdate`** - New bounds for optimization parameters

### Optimization Objects (optimization_structures.py)

#### Configuration Objects
- **`OptimizerType`** - Enum: MULTI_OBJECTIVE vs SINGLE_OBJECTIVE
- **`OptimizationConstraints`** - Parameter bounds, fixed_parameters, target_volume
- **`OptimizationConfig`** - Full optimizer configuration

#### Runtime Objects  
- **`OptimizationObjectives`** - accuracy, precision, time (DUPLICATE of trial analysis!)
- **`OptimizationTrial`** - parameters + objectives + metadata (DUPLICATE of TrialResult!)
- **`OptimizationState`** - List of trials, convergence tracking
- **`VolumeOptimizationResult`** - Results from optimization phase

### IDENTIFIED PROBLEMS

#### 1. DUPLICATE TRIAL REPRESENTATIONS
- **`TrialResult`** (data_structures.py) - Used by experiment system
- **`OptimizationTrial`** (optimization_structures.py) - Used by Bayesian optimizer
- **Same concept, different data structures, causes evaluation bugs**

#### 2. DUPLICATE ANALYSIS REPRESENTATIONS  
- **`AdaptiveMeasurementResult.analysis`** - deviation_pct, cv_volume_pct in TrialResult
- **`OptimizationObjectives`** - accuracy, precision in OptimizationTrial
- **Same data in different formats**

#### 3. TOO MANY RESULT CLASSES
- `TrialResult`, `VolumeCalibrationResult`, `VolumeOptimizationResult`, `ExperimentResults`
- **Unclear separation of responsibilities**

#### 4. SIMPLE WRAPPERS
- `VolumeTolerances` - just 2 numbers
- `QualityEvaluation` - could be inline in TrialResult

### USAGE ANALYSIS

#### Where OptimizationTrial is Used:
1. **bayesian_recommender.py** - Creating trials for Ax optimizer interface
2. **optimization_structures.py** - State management and convergence tracking
3. **experiment.py** - Indirectly through optimizer.update_with_result()

#### Why It Exists:
- **Ax optimizer needs specific data format** (parameters + objectives)
- **TrialResult has rich analysis data** but optimizer only needs summary metrics
- **Historical artifact** from when optimizer was separate from experiment

### CONSOLIDATION STRATEGY OPTIONS

#### Option 1: Keep Both (Current State)
- ✅ Minimal changes required
- ❌ Duplicate evaluation logic (the bug we just fixed)
- ❌ Confusing data flow

#### Option 2: Eliminate OptimizationTrial
- ✅ Single source of truth
- ✅ Unified evaluation logic  
- ❌ Need to modify Ax interface
- ❌ Risk of breaking optimizer

#### Option 3: Create Reference Documentation
- ✅ Understand current mess
- ✅ Guide future refactoring
- ❌ Doesn't fix underlying issues

### RECOMMENDATION

**For now: Document and stabilize**
1. Create this reference doc
2. Keep both structures but ensure they stay in sync
3. Plan future refactoring when system is stable

**Future: Gradual consolidation**
1. Add conversion methods between structures
2. Slowly migrate to single trial representation
3. Eliminate redundant classes

### CURRENT STATUS
- Fixed evaluation bug by using experiment._is_trial_successful everywhere
- Both trial structures exist but evaluation is unified
- System should work correctly despite architectural debt