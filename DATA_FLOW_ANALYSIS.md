# Data Flow Analysis: calibration_sdl_simplified vs Modular System

## Current State: Data Flow Issues

### ❌ **Problem: Hardware-Specific Data Leaking Through**

#### What calibration_sdl_simplified Does RIGHT:
```python
# PROCESS level - only cares about density for conversion
expected_mass = volume * LIQUIDS[liquid]["density"]

# Hardware level (in pipet_and_measure) - handles mass measurement
actual_mass = raw_measurements[-1]['mass']  
actual_volume_ml = actual_mass / LIQUIDS[liquid]["density"]

# PROCESS gets back clean data
return {
    "volume": actual_volume_ml,  # Clean volume 
    "time": elapsed_time,        # Clean timing
    "deviation": deviation,      # Process-calculated
    # NO MASS in process-level data!
}
```

#### What Our Modular System Does WRONG:
```python
# protocol_loader.py - Line 183
return RawMeasurement(
    volume_ml=protocol_result['volume'],        # ✅ Good
    target_volume_ml=target_volume_ml,          # ✅ Good  
    parameters=parameters,                      # ✅ Good
    timestamp=time.time(),                      # ✅ Good
    duration_s=protocol_result['elapsed_s'],    # ✅ Good
    metadata=protocol_result  # ❌ PROBLEM! All hardware data bleeds through
)

# calibration_protocol_example.py returns:
{
    "volume": measured_volume,     # ✅ Should be in process
    "elapsed_s": elapsed,          # ✅ Should be in process  
    "mass_g": mass_g,             # ❌ Hardware-specific, shouldn't leak
    "density_g_per_ml": density,  # ❌ Hardware-specific, shouldn't leak
    "new_pipet_each_time": flag,  # ❌ Hardware-specific, shouldn't leak
    **params                      # ❌ Echoed parameters clutter data
}
```

## ✅ **What Should Flow Through the System**

### Core Data That Should Move Through Process:
1. **Parameters**: `aspirate_speed`, `dispense_speed`, `aspirate_wait_time`, etc.
2. **Target volumes**: What we're trying to pipette
3. **Measured volumes**: What we actually got (volume only, not mass)
4. **Time**: How long it took
5. **Process-calculated metrics**: `deviation`, `accuracy`, `precision`, `variability`

### Hardware Data That Should Stay in Hardware:
1. **Mass measurements**: `mass_g`, density calculations
2. **Hardware state**: Vial names, pipet status, robot positions
3. **Hardware-specific parameters**: Liquid density, refill flags
4. **Debug info**: Timestamps, replicate IDs, hardware diagnostics

## 🔧 **Required Fixes**

### 1. Clean Protocol Return Interface
```python
# HARDWARE protocols should return ONLY:
{
    "volume": measured_volume_ml,  # The answer we need
    "elapsed_s": timing_seconds,   # Time it took
    # NO mass, density, vial names, hardware state, etc.
}
```

### 2. Keep Process-Level Calculations in Process
```python
# In PROCESS (experiment.py), calculate:
deviation = measured_volume - target_volume
deviation_pct = (deviation / target_volume) * 100
accuracy = abs(deviation_pct)
# These are PROCESS metrics, not HARDWARE metrics
```

### 3. No Hardware Data in RawMeasurement
```python
# Clean RawMeasurement should contain:
RawMeasurement(
    target_volume_ml=target,
    actual_volume_ml=measured,  # From hardware, but just the volume
    duration_s=timing,          # From hardware, but just the time
    parameters=params,          # Parameters used
    # NO metadata with hardware internals
)
```

## 📊 **Workflow Steps Comparison**

### calibration_sdl_simplified (CORRECT):
1. **Screening**: Parameter exploration → results with volume + time only
2. **Optimization**: Bayesian optimization on clean parameters + results
3. **Analysis**: Process calculates deviation, precision, etc. from clean data
4. **Output**: Clean optimal parameters + performance metrics

### Our Modular System (NEEDS FIXING):
1. **Screening**: ✅ Same logic (when we implement it)
2. **Optimization**: ✅ Same Bayesian logic  
3. **Analysis**: ❌ Works with cluttered data including hardware internals
4. **Output**: ❌ Potentially includes hardware-specific fields

## ✅ **What We Have Right**

### Process Structure (experiment.py):
- ✅ Correct workflow steps (screening → optimization → analysis)
- ✅ Bayesian optimization integration
- ✅ External data loading capability
- ✅ Configuration management

### Hardware Abstraction:
- ✅ Single protocol files with 3 functions
- ✅ Hardware state managed internally in protocols
- ✅ Clean initialize/measure/wrapup interface

## ❌ **What Needs Immediate Fixing**

### 1. Protocol Return Data
- Remove mass, density, hardware state from protocol returns
- Return only volume + time + essential process data

### 2. Data Structure Cleanup  
- Remove metadata field that carries hardware internals
- Keep RawMeasurement focused on process-relevant data

### 3. Variable Name Consistency
- Ensure no renaming between protocol → process → analysis
- Use same field names throughout pipeline

## 🎯 **Action Items**

1. **Fix protocol returns**: Remove hardware-specific data from protocol results
2. **Clean RawMeasurement**: Remove metadata that leaks hardware internals  
3. **Verify workflow steps**: Ensure all calibration_sdl_simplified steps are implemented
4. **Test data flow**: Verify only process-relevant data flows through system

The goal: **Same clean data flow as calibration_sdl_simplified but with modular hardware.**