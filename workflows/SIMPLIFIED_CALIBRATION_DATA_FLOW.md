# Simplified Calibration Workflow - Data Flow Documentation

## Overview
This document maps out the actual data flows in the simplified calibration workflow based on code analysis (November 2025). It identifies critical data inconsistencies, field naming issues, and proposes solutions for streamlining data handling.

**⚠️ CRITICAL FINDINGS**: The current system has significant data flow fragility due to inconsistent field naming, mixed units, and missing field validation.

**⚠️ CRITICAL FINDINGS**: The current system has significant data flow fragility due to inconsistent field naming, mixed units, and missing field validation.

---

## 🔄 **Data Flow Hierarchy**

The system has **5 primary data structures** that flow through the workflow:

1. **Raw Hardware Data** → Individual measurements from Cytation 5
2. **Adaptive Results** → Aggregated measurements with conditional replicates  
3. **Trial Results** → Full experimental records with parameters + performance
4. **Ranked Candidates** → Enhanced trial results with quality evaluation
5. **Constraint Data** → Volume-specific optimization bounds

---

## 📊 **Data Flow 1: Raw Hardware → Adaptive Results**

### **Input: Raw Hardware Measurements**
```python
# From pipet_and_measure_tracked() - individual measurement
raw_measurement = {
    'deviation': 8.2,        # Single measurement deviation %
    'time': 15.3,           # Single measurement time (s)
    'mass': 0.04128,        # Raw mass from Cytation 5 (g)
    'measured_volume': 0.0472  # Calculated volume (mass/density) in mL
}
```

### **Process: Adaptive Measurement Strategy**
```python
# run_adaptive_measurement() logic:
# 1. Single measurement
# 2. If deviation ≤ 10%: run 2 more replicates (3 total)
# 3. If deviation > 10%: assign penalty variability (100.0)
# 4. Aggregate all measurements
```

### **Output: Adaptive Results**
```python
adaptive_result = {
    'deviation': 7.9,           # AVERAGED across replicates
    'variability': 3.2,         # CALCULATED: (max-min)/(2*avg)*100 OR penalty
    'time': 16.1,              # AVERAGED across replicates
    'replicate_count': 3,       # Actual number of measurements taken
    'all_measurements': [0.0472, 0.0478, 0.0475],  # Individual volumes (mL)
    'all_deviations': [8.2, 7.1, 8.4],             # Individual deviations (%)
    'all_times': [15.3, 16.2, 16.8],               # Individual times (s)
    'measured_volume': 0.0475   # ⚠️ CRITICAL: Averaged measured volume (mL)
}
```

**🚨 Data Issues Identified:**
- ❌ **Field naming inconsistency**: `all_measurements` here becomes `raw_measurements` later
- ⚠️ **Missing validation**: No check that `measured_volume` calculation succeeded
- ⚠️ **Unit assumptions**: Code assumes volumes are in mL but doesn't validate

---

## 📋 **Data Flow 2: Adaptive Results → Trial Results**

### **Process: Full Trial Record Creation**
```python
# In screening/optimization functions:
full_result = dict(params)  # Start with parameter dict
full_result.update({
    # Target information
    "volume": volume,              # Target volume (mL)
    
    # Performance metrics (from adaptive_result)
    "deviation": adaptive_result['deviation'],     # Averaged deviation %
    "variability": adaptive_result['variability'], # Calculated/penalty variability %
    "time": adaptive_result['time'],               # Averaged time (s)
    
    # Metadata
    "trial_index": trial_index,
    "strategy": "SCREENING",      # or "OPTIMIZATION", "PRECISION_TEST", etc.
    "liquid": liquid,
    "time_reported": datetime.now().isoformat(),
    "replicate_count": adaptive_result['replicate_count'],
    
    # Raw data preservation
    "raw_measurements": adaptive_result['all_measurements'],  # ❌ NAME CHANGE!
    "measured_volume": adaptive_result.get('measured_volume', 0)  # ⚠️ Critical field
})
```

### **Output: Trial Results Structure**
```python
trial_result = {
    # PIPETTING PARAMETERS (8 parameters)
    "aspirate_speed": 25,
    "dispense_speed": 20,
    "aspirate_wait_time": 5.0,
    "dispense_wait_time": 3.0,
    "retract_speed": 8.0,
    "blowout_vol": 0.08,
    "post_asp_air_vol": 0.05,
    "overaspirate_vol": 0.015,
    
    # PERFORMANCE METRICS
    "volume": 0.05,              # Target volume (mL)
    "deviation": 7.9,            # Averaged deviation (%)
    "variability": 3.2,          # Calculated variability (%) OR penalty (100.0)
    "time": 16.1,               # Averaged time (s)
    
    # METADATA
    "trial_index": 1,
    "strategy": "SCREENING",     
    "liquid": "glycerol",
    "time_reported": "2025-11-09T...",
    "replicate_count": 3,
    
    # CRITICAL RAW DATA
    "raw_measurements": [0.0472, 0.0478, 0.0475],  # Individual volumes (mL)
    "measured_volume": 0.0475    # ⚠️ CRITICAL: Averaged measured volume (mL)
}
```

**🚨 Data Issues Identified:**
- ❌ **Inconsistent field names**: `all_measurements` → `raw_measurements`
- ❌ **Fragile parameter extraction**: `dict(params)` can introduce unexpected fields
- ⚠️ **Missing validation**: No verification that all required fields are present
- ⚠️ **Units not explicit**: Fields contain mL values but units not in field names

### **Storage: all_results**
All trial results are stored in the global `all_results` list and used for:
- Optimizer feedback
- Stopping criteria evaluation  
- Best candidate selection
- Final results generation

---

## 🏆 **Data Flow 3: Trial Results → Ranked Candidates**

### **Input: Multiple Trial Results**
```python
# From all_results list - multiple trial results
candidates = [trial_result1, trial_result2, ...]
```

### **Process: Quality Evaluation & Ranking**
```python
# rank_candidates_by_priority() performs:
# 1. Quality evaluation for each candidate
# 2. Composite scoring with normalization
# 3. Ranking by composite score (lower = better)
```

### **Output: Enhanced Candidates**
```python
ranked_candidate = {
    # ALL ORIGINAL TRIAL DATA +
    
    # QUALITY EVALUATION
    'quality_evaluation': {
        'is_good': True,                    # Meets accuracy AND precision tolerance
        'accuracy_ok': True,                # |deviation| ≤ tolerance  
        'precision_ok': True,               # variability ≤ tolerance%
        'precision_value': 3.2,             # ❌ REDUNDANT with 'variability'
        'accuracy_deviation_ul': 3.95,      # Deviation in μL (converted)
        'accuracy_tolerance_ul': 5.0        # Tolerance in μL (converted)
    },
    
    # RANKING SCORES (normalized, lower = better)
    'raw_accuracy': 7.9,      # Original deviation %
    'raw_precision': 3.2,     # Original variability %  
    'raw_time': 16.1,         # Original time s
    'accuracy_score': 79.0,   # Normalized accuracy score
    'precision_score': 32.0,  # Normalized precision score
    'time_score': 16.1,       # Normalized time score
    'composite_score': 52.3   # Weighted composite: 50% acc + 40% prec + 10% time
}
```

**🚨 Data Issues Identified:**
- ❌ **Redundant fields**: `precision_value` vs `raw_precision` vs `variability`
- ❌ **Unit mixing**: μL calculations in quality evaluation, mL everywhere else
- ⚠️ **Field explosion**: Adding many derived fields to original data structure
- ⚠️ **Assumption dependency**: Quality evaluation assumes specific field names exist

---

## 🎯 **Data Flow 4: Best Candidates → Constraint Calculations**

### **Input: Best Ranked Candidate**
```python
best_candidate = ranked_candidates[0]  # Top-ranked candidate
```

### **Process: Overaspirate Constraint Calculation**
```python
# calculate_first_volume_constraint() logic:
# 1. Extract measured_volume from best_candidate
# 2. Calculate shortfall: target - measured
# 3. Account for existing overaspirate
# 4. Add buffer for optimization range
```

### **Output: Constraint Bounds**
```python
# ⚠️ RETURNS TUPLE - was causing TypeError before fix
min_overaspirate_ml, max_overaspirate_ml = calculate_first_volume_constraint(...)

# Example values:
min_overaspirate_ml = 0.0      # Minimum bound (mL)
max_overaspirate_ml = 0.0182   # Maximum bound (mL)
```

**🚨 Data Issues Identified:**
- ✅ **Recently fixed**: Tuple unpacking was causing format string errors
- ⚠️ **Unit conversions**: Function works in mL but optimizer may expect different units
- ❌ **Global state dependency**: Results sometimes stored in global variables

---

## 🔄 **Data Flow 5: Parameter Inheritance & Transfer**

### **First Volume → Subsequent Volumes**
```python
# Extract best parameters from first volume
best_params = {k: v for k, v in best_candidate.items() if k in ALL_PARAMS}

# Split into fixed vs optimizable
VOLUME_DEPENDENT_PARAMS = ["blowout_vol", "overaspirate_vol"]
fixed_params = {k: v for k, v in best_params.items() 
                if k not in VOLUME_DEPENDENT_PARAMS}
```

### **Volume-Specific Calibrations**
```python
# Global storage for volume-specific constraints
volume_overaspirate_calibrations = {
    0.025: {
        'guess_ml': 0.0156,      # Estimated optimal overaspirate (mL)
        'max_ml': 0.0206,        # Upper bound for optimization (mL)
        'shortfall_ul': 2.3,     # Measured under-delivery (μL) ❌ UNIT MIX
        'measured_volume_ul': 22.7  # Actual measured volume (μL) ❌ UNIT MIX
    }
}
```

**🚨 Data Issues Identified:**
- ❌ **Unit mixing**: mL and μL values in same data structure
- ❌ **Global state**: Volume calibrations stored globally, hard to track
- ⚠️ **Parameter extraction fragility**: Depends on field names matching exactly

---

## 🚨 **Critical Data Flow Problems Summary**

### **1. Inconsistent Field Naming**
| Function | Field Name | Contents |
|----------|------------|----------|
| `run_adaptive_measurement()` | `all_measurements` | Individual volumes (mL) |
| `trial_result` creation | `raw_measurements` | Same data, different name |
| `quality_evaluation` | `precision_value` | Variability % |
| `trial_result` | `variability` | Same data, different name |

### **2. Mixed Units Throughout System**
| Context | Volume Unit | Deviation Unit |
|---------|-------------|----------------|
| Hardware measurements | mL | % |
| Constraint calculations | mL and μL | μL absolute |
| Quality evaluation | μL | μL absolute |
| Display output | μL | % |

### **3. Missing Field Validation**
```python
# Current pattern (FRAGILE):
measured_volume = candidate.get('measured_volume', 0)  # Defaults to 0!

# Better pattern (ROBUST):
if 'measured_volume' not in candidate:
    raise ValueError("Critical field 'measured_volume' missing from candidate")
```

### **4. Data Structure Proliferation**
- Start with 8 parameters + 4 performance metrics = 12 fields
- End with 25+ fields after ranking enhancements
- Many redundant/derived fields stored permanently

---

## ✅ **Proposed Solutions for Streamlining**

### **1. Standardized Data Classes**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class MeasurementData:
    """Single hardware measurement"""
    volume_ml: float
    deviation_pct: float  
    time_s: float
    mass_g: float

@dataclass
class TrialResult:
    """Complete trial with parameters and performance"""
    # Parameters
    parameters: dict  # 8 pipetting parameters
    
    # Target
    target_volume_ml: float
    
    # Performance (always averaged from replicates)
    measured_volume_ml: float    # ⚠️ CRITICAL FIELD
    deviation_pct: float         # Accuracy
    variability_pct: float       # Precision  
    time_s: float               # Speed
    
    # Raw data
    individual_measurements: List[MeasurementData]
    replicate_count: int
    
    # Metadata
    trial_index: str
    strategy: str
    liquid: str
    timestamp: str
    
    def is_valid(self) -> bool:
        """Validate all required fields are present and reasonable"""
        return (
            self.measured_volume_ml > 0 and
            0 <= self.deviation_pct <= 100 and
            self.time_s > 0 and
            len(self.individual_measurements) == self.replicate_count
        )

@dataclass  
class RankedCandidate:
    """Trial result with quality evaluation and ranking scores"""
    trial: TrialResult
    
    # Quality assessment
    meets_tolerance: bool
    accuracy_ok: bool
    precision_ok: bool
    
    # Ranking scores (lower = better)
    composite_score: float
    
    def get_parameters(self) -> dict:
        """Safe parameter extraction"""
        return self.trial.parameters.copy()
```

### **2. Unit-Safe Field Naming**
```python
# CURRENT (AMBIGUOUS):
measured_volume = 0.0475  # mL? μL? 

# PROPOSED (EXPLICIT):
measured_volume_ml = 0.0475
target_volume_ul = 50.0
deviation_pct = 7.9
accuracy_tolerance_ul = 5.0
```

### **3. Data Validation Layer**
```python
class DataValidator:
    @staticmethod
    def validate_trial_result(trial: TrialResult) -> None:
        """Comprehensive validation with clear error messages"""
        if not trial.is_valid():
            raise ValueError(f"Invalid trial result: {trial}")
            
        # Unit range checks
        if not (0.001 <= trial.measured_volume_ml <= 1.0):
            raise ValueError(f"measured_volume_ml out of range: {trial.measured_volume_ml}")
            
        if not (0 <= trial.deviation_pct <= 100):
            raise ValueError(f"deviation_pct out of range: {trial.deviation_pct}")
    
    @staticmethod
    def validate_candidate_list(candidates: List[RankedCandidate]) -> None:
        """Ensure all candidates have required fields for ranking"""
        for candidate in candidates:
            DataValidator.validate_trial_result(candidate.trial)
```

### **4. Centralized Data Processing**
```python
class DataProcessor:
    """Handle all data transformations in one place"""
    
    @staticmethod
    def adaptive_to_trial(adaptive_result: dict, params: dict, metadata: dict) -> TrialResult:
        """Convert adaptive measurement result to standardized trial result"""
        individual_measurements = [
            MeasurementData(
                volume_ml=vol,
                deviation_pct=dev, 
                time_s=time,
                mass_g=vol * LIQUIDS[metadata['liquid']]['density']
            )
            for vol, dev, time in zip(
                adaptive_result['all_measurements'],
                adaptive_result['all_deviations'], 
                adaptive_result['all_times']
            )
        ]
        
        return TrialResult(
            parameters=params.copy(),
            target_volume_ml=metadata['volume'],
            measured_volume_ml=adaptive_result['measured_volume'],
            deviation_pct=adaptive_result['deviation'],
            variability_pct=adaptive_result['variability'],
            time_s=adaptive_result['time'],
            individual_measurements=individual_measurements,
            replicate_count=adaptive_result['replicate_count'],
            trial_index=metadata['trial_index'],
            strategy=metadata['strategy'],
            liquid=metadata['liquid'],
            timestamp=metadata['time_reported']
        )
    
    @staticmethod
    def extract_parameters(candidate: RankedCandidate) -> dict:
        """Safe parameter extraction with validation"""
        return candidate.get_parameters()
```

### **5. Eliminate Global State**
```python
# CURRENT (GLOBAL):
global volume_overaspirate_calibrations

# PROPOSED (EXPLICIT):
class CalibrationState:
    def __init__(self):
        self.volume_constraints: Dict[float, dict] = {}
        self.all_trials: List[TrialResult] = []
        self.measurement_count = 0
    
    def add_trial(self, trial: TrialResult) -> None:
        DataValidator.validate_trial_result(trial)
        self.all_trials.append(trial)
        self.measurement_count += trial.replicate_count
```

---

## 🎯 **Implementation Priority**

### **Phase 1: Critical Fixes (Immediate)**
1. ✅ **Fix tuple unpacking** (already done)
2. 🔲 **Standardize field names**: `all_measurements` → `measured_volumes_ml`
3. 🔲 **Add unit suffixes**: `measured_volume` → `measured_volume_ml`
4. 🔲 **Validate critical fields**: Check `measured_volume` exists before use

### **Phase 2: Data Structure Cleanup (Short-term)**  
1. 🔲 **Implement TrialResult dataclass**
2. 🔲 **Add DataValidator class**
3. 🔲 **Centralize data processing**
4. 🔲 **Remove redundant fields**

### **Phase 3: Architecture Improvements (Long-term)**
1. 🔲 **Eliminate global state**
2. 🔲 **Implement CalibrationState class**  
3. 🔲 **Create data flow interfaces**
4. 🔲 **Add comprehensive testing**

---

## 💡 **Key Insights for Reconfiguration**

### **1. Single Source of Truth**
- **Problem**: Same data stored in multiple formats throughout workflow
- **Solution**: Use dataclasses with clear transformation points

### **2. Explicit Over Implicit**
- **Problem**: Field names don't indicate units or data type
- **Solution**: Unit-aware field naming (`_ml`, `_ul`, `_pct`, `_s`)

### **3. Fail Fast Validation**
- **Problem**: Missing fields cause cryptic errors late in workflow  
- **Solution**: Validate data at boundaries between workflow phases

### **4. Functional Data Flow**
- **Problem**: Global state makes debugging difficult
- **Solution**: Pass state explicitly through function parameters

### **5. Type Safety**
- **Problem**: Runtime errors from wrong data types
- **Solution**: Use dataclasses and type hints for compile-time checking

The current system works but is **fragile due to data inconsistencies**. The proposed changes would significantly reduce the likelihood of data flow errors while making the code more maintainable and debuggable.