# Universal Calibration System - Project Goals

## Architecture Flow Chart

### System Components

#### 1. CONFIG File
- **Input**: Parameter definitions with bounds
- **Contents**: 
  - Any parameters with any bounds (flexible parameter system)
  - Volume-dependent parameters (flagged as such)
  - Time-affecting parameters (flagged as such)
  - Overaspirate volume (specifically referenced by PROCESS)
- **Format**: YAML configuration

#### 2. PROCESS (Main Logic)
- **Source**: Based on `calibration_sdl_simplified.py` proven workflow
- **Inputs**: CONFIG file parameters and definitions
- **Hardware Interface**: Calls HARDWARE methods only through defined interface
- **Responsibilities**:
  - Calibration-specific workflow logic
  - Parameter optimization algorithms
  - Volume-dependent parameter handling
  - Stopping criteria and best candidate selection
- **Hardware Agnostic**: Doesn't know HOW volume is obtained, only that it receives volume + time
- **Outputs**: OPTIMAL PARAMETERS file

#### 3. HARDWARE (Single File)
- **File Count**: Exactly ONE file per hardware type
- **Contents**: ALL hardware-specific references (vials, robots, mass calculations, etc.)
- **Required Methods**:
  - `init()`: Initialize hardware, return state
  - `measure(parameters) -> {volume, time, ...}`: Execute measurement, return results
  - `cleanup()`: Safely shutdown hardware
- **Examples**: 
  - `north_robot_hardware.py` (for North Robot)
  - `simulation_hardware.py` (for simulation)
  - `tecan_hardware.py` (for different robot)

### Data Flow

```
CONFIG → PROCESS → HARDWARE
   ↓        ↓         ↓
Parameters → init() → Hardware Setup
   ↓        ↓         ↓
   └──→ measure() → {volume, time}
            ↓         ↓
         Optimize ← Results
            ↓
         cleanup()
            ↓
    OPTIMAL PARAMETERS
```

### Key Principles

1. **PROCESS is hardware agnostic**: Only knows about volume, time, and parameters
2. **HARDWARE encapsulates everything physical**: Mass calculations, vial management, robot control
3. **Clean interface**: Only 3 methods between PROCESS and HARDWARE
4. **Single file rule**: All hardware complexity in one swappable file
5. **Simulation as hardware**: Simulation is just another HARDWARE implementation

### Critical Design Decision
- **Overaspirate volume**: Only parameter specifically referenced in PROCESS (volume-dependent optimization)
- **Everything else**: Generic parameter handling, bounds from CONFIG
- **Hardware specifics**: PROCESS doesn't know about mass, density, vial rotation, etc.

## Overview
We are building a universal calibration system that maintains the proven structure and process of `calibration_sdl_simplified.py` while ensuring true hardware modularity and clean data flow.

## Goal 1: Universal Calibration Process
**Objective**: Create a calibration system that can use different hardware and pipetting parameters while running the same flexible process.

### Requirements:
- Support multiple hardware types (North Robot, other liquid handlers)
- Accept different pipetting parameter sets and ranges
- Maintain consistent optimization workflow regardless of hardware
- Support both simulation and real hardware modes
- Use the same external data integration capabilities

### Success Criteria:
- A researcher can switch from North Robot to different hardware by changing only one configuration file
- The same optimization algorithms (Bayesian, LLM) work with any hardware
- Pipetting parameter optimization follows the same workflow patterns

## Goal 2: Hardware Compartmentalization 
**Objective**: ALL hardware-specific elements must be compartmentalized into single files with three minimum functions.

### Required Functions:
1. **`initialize(config) -> state`**: Set up hardware, return state object
2. **`get_data(state, parameters, target_volume) -> measurement_results`**: Execute measurement and return data
3. **`cleanup(state) -> None`**: Safely shut down hardware

### Requirements:
- Each hardware type gets exactly one file (e.g., `north_robot_protocol.py`, `tecan_protocol.py`)
- No hardware-specific code anywhere else in the system
- Protocol files are self-contained - edit only one file to adapt to new hardware
- Clear interface contracts between protocol and main system

### Success Criteria:
- Adding new hardware requires creating only one new protocol file
- Existing protocols work without modification when system is updated
- No `if hardware_type == "north"` logic scattered through codebase

## Goal 3: Clean Data Flow
**Objective**: Prevent exponential multiplication and transformation of data streams that cause data flow issues.

### Requirements:
- Single, consistent data format throughout the system
- No data format conversions between system components
- Clear data schema that doesn't change between pipeline stages
- Minimal data transformations - prefer passing data through unchanged

### Data Flow Principles:
- Raw measurements → Analysis → Recommendations → Next parameters
- Each stage receives data in same format it expects
- No hidden data transformations that change schema
- Explicit data validation at system boundaries only

### Success Criteria:
- Data debugging is straightforward - same format everywhere
- Adding new analysis or recommendation components doesn't require data format changes
- System performance scales linearly with data volume

## Goal 4: Preserve Proven Structure
**Objective**: Maintain the same structure and process as `calibration_sdl_simplified.py` which works in production.

### What to Preserve:
- Overall workflow: External data → Screening → Optimization → Selection
- Configuration system (YAML-based)
- External data integration capabilities
- Multiple optimizer support (Bayesian, LLM)
- Liquid-specific parameter handling
- Volume-dependent parameter optimization

### What to Improve:
- Hardware abstraction (make it truly modular)
- Code organization (cleaner separation of concerns)
- Documentation (clearer interfaces)

### Success Criteria:
- Existing workflows migrate to new system without functionality loss
- Performance characteristics remain the same or better
- All current features continue to work
- Migration path is clear and low-risk

## Goal 5: Unified Variable Naming
**Objective**: Maintain consistent variable names throughout the entire process to avoid confusion and data flow issues.

### Requirements:
- No variable renaming between system components
- Parameter names stay consistent from input → processing → output
- Data field names remain the same across all pipeline stages
- No translation layers that change variable names

### Examples of What to Avoid:
- `aspirate_wait_time` becoming `aspirate_wait_time_s` in different components
- `overaspirate_vol` becoming `overaspirate_vol_ml` in protocol files
- `target_volume` becoming `volume_mL` becoming `vol_ml` in different stages

### Success Criteria:
- Same variable names used in configuration files, protocol functions, and analysis outputs
- No name mapping or translation code needed between components
- Variables can be traced through entire pipeline without name changes
- New team members can follow variable usage without confusion

## Implementation Strategy

### Phase 1: Analysis and Documentation
- [ ] Document exactly how `calibration_sdl_simplified.py` works
- [ ] Identify all hardware-specific code sections
- [ ] Map current data flow patterns
- [ ] Define protocol interface specification

### Phase 2: Protocol Abstraction
- [ ] Create protocol interface specification
- [ ] Build North Robot protocol following 3-function pattern
- [ ] Build simulation protocol following same pattern
- [ ] Verify protocols work with existing optimization logic

### Phase 3: System Integration
- [ ] Integrate protocols with main calibration workflow
- [ ] Preserve all existing features and capabilities
- [ ] Test with real hardware and simulation
- [ ] Validate data flow consistency

### Phase 4: Validation
- [ ] Compare results with original `calibration_sdl_simplified.py`
- [ ] Verify hardware switching works correctly
- [ ] Test external data integration
- [ ] Performance benchmarking

## Success Metrics
1. **Hardware Independence**: Can switch hardware by changing one config parameter
2. **Feature Preservation**: All features from `calibration_sdl_simplified.py` work
3. **Clean Architecture**: Hardware code isolated to protocol files only
4. **Data Consistency**: Same data format AND variable names throughout pipeline
5. **Performance**: No degradation in optimization speed or quality
6. **Variable Consistency**: No variable renaming between system components

## Current Status
- ❌ **Analysis Phase**: Understanding existing systems and their strengths
- ⏸️ **Protocol Design**: Defining clean hardware abstraction interface
- ⏸️ **Implementation**: Building new system with proven patterns
- ⏸️ **Testing**: Validation against working system