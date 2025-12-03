# Hardware-Agnostic Calibration System - TODO List

## 🚨 Critical Issues to Fix

### Issue #4: Pipetting Wizard Parameter Type Assumptions
**Location:** `pipetting_wizard_v2.py` lines 243, 269
**Problem:** 
```python
if param in ['aspirate_speed', 'dispense_speed', 'retract_speed']:
    value = int(round(value))
```
**Solution Needed:** Make parameter type conversion hardware-agnostic. Should either:
- Auto-detect appropriate types from config/bounds
- Apply consistent type conversion rules for all parameters
- Make type conversion configurable per parameter

**Risk:** Different hardware parameters may need different type handling, could cause incorrect optimization.

---

## ✅ Issues Fixed

### ✅ Issue #1: Hardware-Specific Parameters in Display Code
**Fixed:** `optimization_structures.py` - Now shows ALL hardware parameters instead of hardcoded list

### ✅ Issue #2: Silent Error Swallowing  
**Fixed:** `experiment.py` - Protocol cleanup failures now logged as warnings

### ✅ Issue #3: LLM Recommender Hardcoded Parameters
**Fixed:** `llm_recommender.py` - Now uses `parameters.to_protocol_dict()` for hardware-agnostic access

### ✅ Issue #5: Simulated Protocol Hardcoded Parameter Logic  
**Fixed:** `calibration_protocol_simulated.py` - Now uses generic parameter categories instead of hardcoded North Robot parameter names
- ✅ Keeps overaspirate_vol effects (mandatory parameter) 
- ✅ Applies subtle generic effects based on parameter name patterns (speed, wait, volume, etc.)
- ✅ Works with any hardware parameter names (`pump_rate`, `injection_speed`, `dwell_time`, etc.)
- ✅ No crashes on unknown parameters - graceful fallback to minimal generic effects

---

## 📝 Implementation Notes

- **Issue #6 (Config file)** - Intentionally not fixed. Users should define their hardware parameters in `experiment_config.yaml`
- **All protocol files** - Anything in `calibration_protocol_northrobot.py` is hardware-specific and should remain unchanged

---

## 🎯 Priority Order
1. **Issue #5** (Simulation) - Higher impact, affects testing/development
2. **Issue #4** (Pipetting Wizard) - Medium impact, affects parameter optimization accuracy