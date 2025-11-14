# LLM Integration Todo List

## Current Status
LLM integration is partially implemented but not working in the actual workflow. The system falls back to SOBOL sampling.

## Issues to Fix

### 🔧 1. Fix LLM template path resolution
**Status:** Not Started  
**Description:** The workflow can't find `calibration_screening_llm_template.json`. Need to fix path resolution in experiment.py so it finds the template file properly.  
**Error:** `[Errno 2] No such file or directory: 'calibration_screening_llm_template.json'`

### 🔧 2. Fix LLM liquid context integration  
**Status:** Not Started  
**Description:** LLM is not reading the liquid type from `experiment_config.yaml` properly. The context building in `_build_context()` needs to correctly get the liquid from the config.  
**Issue:** LLM suggestions don't change when liquid is changed from water to glycerol.

### 🔧 3. Fix method signature compatibility
**Status:** Not Started  
**Description:** `Experiment.py` calls `suggest_parameters(target_volume, trial_idx)` but `LLMRecommender` expects `suggest_parameters(n_suggestions, previous_results)`. The compatibility wrapper needs debugging.  
**Problem:** API mismatch between workflow expectations and LLM implementation.

### ✅ 4. Clean up test files  
**Status:** Completed  
**Description:** ~~Delete redundant test files: test_water_llm.py, test_real_llm_call.py, test_workflow_integration.py, test_simple_template.py, test_hardware_agnostic_llm.py. Keep only test_template_substitution.py if it uses real config.~~

### 🔧 5. Verify template substitution works
**Status:** Not Started  
**Description:** Ensure the hardware-agnostic template system actually substitutes `TIME_AFFECTING_PARAMS` and `HARDWARE_SPECIFIC_WARNINGS` correctly from the real config file.  
**Goal:** Template should be hardware-agnostic and work across different systems.

### 🔧 6. Test end-to-end workflow
**Status:** Not Started  
**Description:** Once fixed, test that changing `liquid: 'water'` to `liquid: 'glycerol'` in `experiment_config.yaml` actually generates different LLM parameter suggestions via `run_calibration.py`.  
**Success Criteria:** Different liquids produce appropriate parameter suggestions (faster for water, slower for glycerol).

## Files Involved
- `calibration_modular_v2/llm_recommender.py` - LLM integration class
- `calibration_modular_v2/experiment.py` - Main workflow integration
- `calibration_modular_v2/experiment_config.yaml` - Configuration with LLM settings
- `calibration_modular_v2/calibration_screening_llm_template.json` - Hardware-agnostic template
- `calibration_modular_v2/config_manager.py` - Config loading and validation

## Priority Order
1. Fix path resolution (critical - prevents LLM from loading)
2. Fix method compatibility (critical - prevents parameter generation)  
3. Fix liquid context integration (important - ensures material-specific suggestions)
4. Verify template substitution (important - ensures hardware agnosticism)
5. End-to-end testing (validation)

## Goal
Working LLM integration that:
- Loads properly in the workflow
- Generates different parameters for different liquids
- Uses hardware-agnostic templates
- Integrates seamlessly with existing calibration system