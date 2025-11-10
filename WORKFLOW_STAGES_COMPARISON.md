# Workflow Stages Comparison: calibration_sdl_simplified vs Modular System

## ✅ **calibration_sdl_simplified Workflow (PROVEN)**

### Complete 6-Stage Process:
1. **External Data Loading OR Screening** (SOBOL/LLM exploration)
2. **Overaspirate Calibration** (volume-dependent parameter optimization)
3. **3-objective Optimization** (deviation, variability, time) 
4. **Simple Stopping** (60 measurements OR 6 "GOOD" parameter sets)
5. **Best Candidate Selection** (rank by accuracy → precision → time)
6. **Single Precision Test** (final validation with multiple replicates)

### Key Features:
- ✅ **Overaspirate calibration**: Dedicated stage for volume-dependent parameters
- ✅ **Precision testing**: Final validation phase with multiple measurements
- ✅ **Quality-based stopping**: Stops when enough good trials found
- ✅ **External data integration**: Replaces screening when available
- ✅ **Multi-volume support**: First volume = all params, subsequent = volume-dependent only

## ❌ **Our Modular System (INCOMPLETE)**

### Current 2-Stage Process:
1. **Screening Phase** ✅ (with external data loading)
2. **Optimization Phase** ✅ (Bayesian optimization)
3. ❌ **Missing: Overaspirate Calibration**
4. ❌ **Missing: Precision Test Phase**

### What's Actually Implemented:

#### ✅ **Screening Phase** (WORKING):
```python
def _run_screening_phase(self, target_volume_ml, optimizer):
    # Check external data first
    if self.external_data_loader.has_valid_data():
        return external_trials
    # Fall back to normal screening
    screening_trials = []
    # Generate screening parameters...
```

#### ✅ **Optimization Phase** (WORKING):
```python
def _run_optimization_phase(self, target_volume_ml, optimizer):
    # Bayesian optimization with stopping criteria
    while (measurements < max_measurements and good_trials < min_good):
        # Generate next parameters using Bayesian optimizer
        # Execute trial, check quality
```

#### ❌ **Missing: Overaspirate Calibration Phase**
- calibration_sdl_simplified has dedicated overaspirate calibration
- Our system: Just treats overaspirate_vol as another parameter
- Missing: Volume-dependent parameter optimization logic

#### ❌ **Missing: Precision Test Phase**  
- calibration_sdl_simplified runs final precision test with best parameters
- Our system: Just picks best trial, no final validation
- Missing: Multiple replicate measurements for precision assessment

## 🚨 **Critical Missing Stages**

### 1. **Overaspirate Calibration**
```python
# calibration_sdl_simplified has:
def run_overaspirate_calibration_new(lash_e, state, volume, liquid):
    # Dedicated calibration for volume-dependent parameters
    # Finds optimal overaspirate for this specific volume
    
# Our system: Missing this entirely
```

### 2. **Precision Test**
```python
# calibration_sdl_simplified has:
def run_precision_test(best_params, volume):
    # Run multiple replicates with best parameters
    # Calculate final precision metrics
    # PRECISION_MEASUREMENTS = 10  # Multiple measurements
    
# Our system: Missing this entirely
```

### 3. **Multi-Volume Logic**
```python
# calibration_sdl_simplified has:
# First volume: optimize ALL parameters
# Subsequent volumes: only volume-dependent parameters (blowout_vol, overaspirate_vol)

# Our system: Same optimization for all volumes (incorrect)
```

## 📊 **Data Storage Point**

**You're absolutely right about raw data storage!**

### ✅ **calibration_sdl_simplified approach**:
```python
# Hardware saves raw data internally:
autosave_raw_path = "raw_data/measurements.csv"  # Hardware-level storage
# Process only gets clean results:
return {"volume": volume_ml, "time": time_s, "deviation": deviation_pct}
```

### ❌ **Our modular system**:
```python
# All raw data flows through process via metadata
metadata=protocol_result  # Everything leaks through!
```

## 🎯 **Required Implementation**

### Phase 1: Add Missing Workflow Stages
1. ✅ Screening (done)
2. **TODO**: Overaspirate calibration phase  
3. ✅ Optimization (done)
4. **TODO**: Precision test phase

### Phase 2: Fix Data Flow
1. **TODO**: Clean protocol returns (volume + time only)
2. **TODO**: Move raw data storage to hardware protocols
3. **TODO**: Remove metadata leakage

### Phase 3: Multi-Volume Logic
1. **TODO**: First volume = all parameters
2. **TODO**: Subsequent volumes = volume-dependent only

## **Answer: Our system is missing 2 of 6 critical stages!**

We have screening and optimization, but we're missing the overaspirate calibration and precision test phases that are essential to the proven workflow.