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

### 1. Data Handling - CRITICAL Anti-Bug Rule
**NEVER use parallel arrays for metadata - always embed in DataFrames:**
```python
# ❌ WRONG - creates index misalignment bugs
data = filter_experimental_points(all_data)
reliability_mask = create_mask(all_data)  # DIFFERENT INDICES!

# ✅ CORRECT - metadata travels with data
data['is_reliable_ratio'] = data['ratio'] <= 1.0
data['is_reliable_turbidity'] = data['turbidity_600'] <= 0.2
```
**Why**: Filtering/reordering preserves alignment automatically, preventing catastrophic indexing bugs.

### 2. Simulation-First Development
```python
# ALWAYS support simulation mode for development/testing
SIMULATE = True  # Set in workflow config
lash_e = Lash_E(vial_file, simulate=SIMULATE)
```

### 3. YAML-Driven Configuration
All robot state and configuration stored in YAML files under `robot_state/`:
- `robot_status.yaml` - Dynamic robot state (pipet usage, gripper status)
- `track_status.yaml` - Wellplate positions and counts
- `robot_hardware.yaml` - Axis mappings, speeds, physical constants
- `vial_positions.yaml` - Physical locations for vials/labware

**Critical**: Always validate YAML files before workflows using `check_input_file()` methods.

### 4. Workflow Structure Pattern
```python
# Standard workflow initialization
INPUT_VIAL_STATUS_FILE = "status/experiment_vials.csv"
lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE)
lash_e.nr_robot.check_input_file()  # MANDATORY validation
lash_e.nr_track.check_input_file()  # MANDATORY validation
```

### 5. Error Handling & Slack Integration
```python
# Unified error pattern via North_Base.pause_after_error()
self.pause_after_error("Error description", send_slack=True)
# Automatically logs, sends Slack notification, pauses for human intervention
```

### 6. Logging Standards
**CRITICAL**: Never use Unicode characters in logging messages (μ, →, ±, etc.)
- Use "uL" not "μL"  
- Use "->" not "→"
- Use "+/-" not "±"
- Windows PowerShell cannot handle Unicode in log output and will crash with UnicodeEncodeError

## Development Practices

- Start with minimal, lean implementations focused on proof-of-concept
- ALWAYS validate status files before workflows using `check_input_file()`
- Use `simulate=True` for development - no hardware required
- Follow the Lash_E → validation → execution → analysis pattern
- Set environment variables `PIP_TIMEOUT=600` and `PIP_RETRIES=2` before installs
- Each code change should update CHANGELOG.md with semantic versioning
- **NEVER save any files in the root directory** - use appropriate subdirectories

### MANDATORY: Automatic Backup Protocol
**ALWAYS create backups before making significant code changes:**
```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "path/to/file.py" "backups/file_backup_$timestamp.py"
```

**Required for**: Files >500 lines, critical workflows, core system changes

### CRITICAL: Workflow Debugging Guidelines
**NEVER use print() statements in workflow files:**
- Use `logger.info()` or `lash_e.logger.info()` instead for debugging
- Print statements won't appear in log files and add unnecessary verbosity

## Debugging Best Practices

### Always Start with Data, Not Assumptions
- **FIRST**: Ask user to show actual data (CSV files, terminal output, logs)
- **NEVER** assume code is working as designed - verify with real examples
- **TRACE systematically**: Follow data flow from input → processing → output

### Debug Data Flow, Not Algorithms
- **Trace actual values** through the system - don't assume calculations are correct
- **Check function signatures** - are the right parameters being passed?
- **Verify data structures** - is stored data different from calculated data?

## Communication Style

- Use minimal emoji and special symbols
- Ask clarifying questions when needed about hardware setup or experimental parameters
- Put documentation in comment replies, not separate files unless asked

## Confidence and Collaboration Guidelines

### Express Uncertainty Appropriately
- **AVOID**: "This will fix it" or "The problem is definitely X"
- **USE**: "This might help" or "One possibility is..." or "Let's try..."

### Respect User Expertise
- User has real hardware, real consequences, and domain knowledge
- Ask "Does this match what you're seeing?" rather than assuming
- Suggest collaborative investigation: "Should we check..." vs prescriptive fixes

### Incremental Changes Over Confident Overhauls  
- Suggest small, reversible changes first
- Ask permission before major modifications to working systems
- "Would you rather try a simpler approach first?" for complex fixes