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

### 5. Logging Standards
**CRITICAL**: Never use Unicode characters in logging messages (μ, →, ±, etc.)
- Use "uL" not "μL"
- Use "->" not "→" 
- Use "+/-" not "±"
- Use "deg" not "°"
- Windows PowerShell cannot handle Unicode in log output and will crash with UnicodeEncodeError

```python
# CORRECT logging
logger.info(f"Volume: {volume*1000:.1f}uL, efficiency: {eff:.3f}uL/uL")
logger.info(f"Point 1: {x:.1f}uL -> {y:.1f}uL")
logger.info(f"Range: +/-{tolerance:.1f}uL")

# WRONG - will crash on Windows
logger.info(f"Volume: {volume*1000:.1f}μL, efficiency: {eff:.3f}μL/μL")
logger.info(f"Point 1: {x:.1f}μL → {y:.1f}μL")
logger.info(f"Range: ±{tolerance:.1f}μL")
```

## Key File Locations

### Primary Controllers
- `master_usdl_coordinator.py` - Lash_E orchestrator class
- `North_Safe.py` - Core robot control classes (North_Robot, North_Track, etc.)
- `biotek_new.py` - Cytation 5 plate reader interface

### Configuration Management
- `status/` - CSV vial definitions and YAML state files
- `robot_state/` - Hardware configuration YAMLs
- `settings/` - Protocol and experimental parameters

### Analysis & Recommendations
- `analysis/` - Data processing and result extraction
- `recommenders/` - Bayesian optimization using BayBe/Ax frameworks
- Both support parameter fixing and selective optimization

## Development Practices

- Start with minimal, lean implementations focused on proof-of-concept
- Use `simulate=True` for development - no hardware required
- Follow the Lash_E → validation → execution → analysis pattern
- Avoid creating new files until asked; extend existing workflow patterns
- Each code change should update CHANGELOG.md with semantic versioning
- **NEVER save any files in the root directory** - use appropriate subdirectories (`workflows/`, `analysis/`, `calibration_modular_v2/`, etc.)

### MANDATORY: Automatic Backup Protocol
**ALWAYS create backups before making significant code changes:**

```powershell
# Create timestamped backup before any major edit
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "path/to/file.py" "backups/file_backup_$timestamp.py"
```

**Required for:**
- Any `replace_string_in_file` operation on files >500 lines
- Modifying critical workflow files (calibration_sdl_*, master_usdl_*, North_Safe.py)
- Adding new functions or classes to existing files
- Any change that affects core system functionality

**Backup naming convention:**
- `backups/filename_backup_YYYYMMDD_HHMMSS.py`
- Include descriptive suffix for major changes: `_before_conditional_replication`

**Recovery protocol:**
- If changes fail or break functionality, immediately restore from most recent backup
- Test backup restoration: `Copy-Item "backups/file_backup_*.py" "original/path/file.py" -Force`
- Keep backups for at least the duration of the coding session

### CRITICAL: Workflow Debugging Guidelines
**NEVER use print() statements in workflow files:**
- Workflows are too long and print() output won't be visible in logs
- Use `logger.info()` or `lash_e.logger.info()` instead for debugging
- Print statements make workflows unnecessarily verbose without benefit
- All debugging output must go through the logging system to be captured in log files

**Correct debugging pattern:**
```python
# CORRECT - will appear in log files
logger.info(f"DEBUG: Processing {item_name} with value {value}")
lash_e.logger.info(f"DEBUG: CMC controls created: {len(controls)}")

# WRONG - invisible and adds bloat
print(f"DEBUG: Processing {item_name} with value {value}")
```

## CRITICAL: No Silent Defaults

**Silent fallbacks are a known failure mode in this codebase. They make code run but produce results that are wrong in ways that are nearly impossible to detect.**

A silent default is any pattern where missing or unavailable data is substituted with a hardcoded value instead of failing loudly. The code appears to work — no exception is raised, no warning is logged — but it is operating on fabricated inputs.

**FORBIDDEN patterns:**
```python
# WRONG - fabricates a value if the real one is unavailable
x = obj.value if obj else 0.004
x = data.get("key", 42)
x = value or "default_thing"
x = config["key"] if "key" in config else some_hardcoded_value
```

**CORRECT pattern — fail loudly instead:**
```python
# If the value must exist, retrieve it and let it raise if missing
x = obj.value                    # raises AttributeError if obj is None
x = data["key"]                  # raises KeyError if missing
x = config["key"]                # raises KeyError — caller must ensure it's present
```

**If a genuine optional with a documented default is needed, make it explicit and intentional:**
```python
# ACCEPTABLE only when the default is meaningful and documented
RETRY_LIMIT = config.get("retry_limit", 3)  # 3 is a documented architectural choice, not a guess
```

**Specific instance that caused data corruption in this codebase:**
```python
# Stage 1 pipetted using CSV-calibrated overaspirate (e.g. 0.008 mL)
# but this line fabricated the baseline for all subsequent correction math:
initial_overaspirate = parameters.overaspirate_vol if parameters else 0.004
# Result: Stages 2 and 3 optimized against the wrong anchor. Slack reports lied.
# Fix: always read from _get_optimized_parameters() — the same source Stage 1 used.
```

**Rule:** If a value is used in any calculation, report, or decision, it must come from its authoritative source. If that source is unavailable, raise an error. Do not invent a substitute.

## Debugging Best Practices

### Always Start with Data, Not Assumptions
- **FIRST**: Ask user to show actual data (CSV files, terminal output, logs)
- **NEVER** assume code is working as designed - verify with real examples
- **TRACE systematically**: Follow data flow from input → processing → output
- **Example**: "Show me the actual CSV/raw data" before analyzing code

### Debugging Best Practices

#### Always Start with Data, Not Assumptions
- **FIRST**: Ask user to show actual data (CSV files, terminal output, logs)
- **NEVER** assume code is working as designed - verify with real examples
- **TRACE systematically**: Follow data flow from input → processing → output
- **Example**: "Show me the actual CSV/raw data" before analyzing code

#### Debug Data Flow, Not Algorithms
- **Trace actual values** through the system - don't assume calculations are correct
- **Never recreate or recalculate values when you have access to the raw values**
- **Check function signatures** - are the right parameters being passed?
- **Verify data structures** - is stored data different from calculated data?

## Communication Style

- Use minimal emoji and special symbols
- Ask clarifying questions when needed about hardware setup or experimental parameters
- Put documentation in comment replies, not separate files unless asked
- Include direct hyperlinks with shortened (7-character) commit hashes when referencing files

## Confidence and Collaboration Guidelines

### Express Uncertainty Appropriately
- **AVOID**: "This will fix it" or "The problem is definitely X"
- **USE**: "This might help" or "One possibility is..." or "Let's try..."
- **CAVEAT**: Explicitly mention when suggestions are untested theories vs proven solutions

### Respect User Expertise
- User has real hardware, real consequences, and domain knowledge
- Ask "Does this match what you're seeing?" rather than assuming
- Suggest collaborative investigation: "Should we check..." vs prescriptive fixes
- When uncertain, say "You know your setup better - does this approach make sense?"

### Incremental Changes Over Confident Overhauls  
- Suggest small, reversible changes first
- Ask permission before major modifications to working systems
- Offer to help user investigate rather than confidently diagnosing
- "Would you rather try a simpler approach first?" for complex fixes