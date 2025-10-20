# North Robotics Laboratory Automation System

## Architecture Overview

This is a self-driving laboratory (SDL) system integrating:
- **North Robot**: Liquid handling robot with pipetting and vial manipulation
- **North Track**: Automated wellplate transport system
- **Cytation 5**: Biotek plate reader for UV-Vis measurements  
- **Photoreactor**: RPi Pico-controlled synthesis chamber
- **Lash_E Coordinator**: Master controller class (`master_usdl_coordinator.py`) that orchestrates all instruments

The system runs **closed-loop optimization workflows** where Bayesian optimizers (BayBe/Ax) suggest experimental conditions, robots execute experiments, analyzers process results, and recommenders suggest the next round.

## Core Development Patterns

### 1. Simulation-First Development
```python
# ALWAYS support simulation mode for development/testing
SIMULATE = True  # Set in workflow config
lash_e = Lash_E(vial_file, simulate=SIMULATE)

# Conditional hardware initialization
if not simulate:
    from north import NorthC9
    c9 = NorthC9("A", network_serial="AU06CNCF")
else:
    from unittest.mock import MagicMock
    c9 = MagicMock()
```

### 2. YAML-Driven Configuration
All robot state and configuration stored in YAML files under `robot_state/`:
- `robot_status.yaml` - Dynamic robot state (pipet usage, gripper status)
- `track_status.yaml` - Wellplate positions and counts
- `robot_hardware.yaml` - Axis mappings, speeds, physical constants
- `vial_positions.yaml` - Physical locations for vials/labware

**Critical**: Always validate YAML files before workflows using `check_input_file()` methods.

### 3. Workflow Structure Pattern
```python
# Standard workflow initialization
INPUT_VIAL_STATUS_FILE = "status/experiment_vials.csv"
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()  # MANDATORY validation
lash_e.nr_track.check_input_file()  # MANDATORY validation

# Move to working position
lash_e.nr_robot.move_vial_to_location("target_vial", "clamp", 0)

# Get fresh wellplate for measurements
lash_e.nr_track.get_new_wellplate()
```

### 4. Error Handling & Slack Integration
```python
# Unified error pattern via North_Base.pause_after_error()
self.pause_after_error("Error description", send_slack=True)
# Automatically logs, sends Slack notification, pauses for human intervention
```

## Key File Locations

### Primary Controllers
- `master_usdl_coordinator.py` - Lash_E orchestrator class
- `North_Safe.py` - Core robot control classes (North_Robot, North_Track, etc.)
- `biotek_new.py` - Cytation 5 plate reader interface

### Workflows Directory
- `workflows/calibration_sdl_*.py` - Pipetting parameter optimization
- `workflows/CMC_*.py` - Critical micelle concentration experiments  
- `workflows/color_*.py` - Color matching optimization
- Each workflow follows: initialization → validation → execution → analysis

### Configuration Management
- `status/` - CSV vial definitions and YAML state files
- `robot_state/` - Hardware configuration YAMLs
- `settings/` - Protocol and experimental parameters

### Analysis & Recommendations
- `analysis/` - Data processing and result extraction
- `recommenders/` - Bayesian optimization using BayBe/Ax frameworks
- Both support parameter fixing and selective optimization

## Conditional Import Pattern
```python
# Handle optional dependencies gracefully
try:
    import slack_agent
    SLACK_AVAILABLE = True
except Exception:
    slack_agent = None
    SLACK_AVAILABLE = False

try:
    import recommenders.llm_optimizer as llm_opt
    LLM_AVAILABLE = True
except ImportError:
    llm_opt = None
    LLM_AVAILABLE = False
```

## Development Practices

- Start with minimal, lean implementations focused on proof-of-concept
- ALWAYS validate status files before workflows using `check_input_file()`
- Use `simulate=True` for development - no hardware required
- Follow the Lash_E → validation → execution → analysis pattern
- Avoid creating new files until asked; extend existing workflow patterns
- Set environment variables `PIP_TIMEOUT=600` and `PIP_RETRIES=2` before installs
- Each code change should update CHANGELOG.md with semantic versioning

## Communication Style

- Use minimal emoji and special symbols
- Ask clarifying questions when needed about hardware setup or experimental parameters
- Put documentation in comment replies, not separate files unless asked
- Include direct hyperlinks with shortened (7-character) commit hashes when referencing files