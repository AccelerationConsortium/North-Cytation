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

**CRITICAL**: `SIMULATE=True` must run every workflow step, never skip them. `Lash_E`/`North_Robot`/`North_Track`/`Biotek_Wrapper` already implement simulate mode internally - they stub out physical hardware and log `"SIMULATION MODE: Would pause for error: ..."` instead of raising. Running the real workflow body with `simulate=True` is how bugs get caught before hardware time is spent.

```python
# ❌ WRONG - never gate the automation calls themselves on SIMULATE
if SIMULATE:
    return output  # real dispensing/measuring code never runs, never gets tested
lash_e.nr_robot.dispense_from_vial_into_vial(...)
```
It IS fine to use `SIMULATE`/`lash_e.simulate` to skip file/folder creation that only matters for real runs, or to substitute synthetic instrument data when a real reader has no simulated output (e.g. `measure_wellplate()` returns `None` in simulate mode - see `simulate_dye_fluorescence()` in `workflows/surfactant_grid_ailsa.py`). It is never fine to use it to skip dispensing, vial moves, or other automation calls.

### 3. YAML-Driven Configuration
All robot state and configuration stored in YAML files under `robot_state/`:
- `robot_status.yaml` - Dynamic robot state (pipet usage, gripper status)
- `track_status.yaml` - Wellplate positions and counts
- `robot_hardware.yaml` - Axis mappings, speeds, physical constants
- `vial_positions.yaml` - Physical locations for vials/labware

**Critical**: Validate required YAML/state inputs and reviewed experiment settings.
Do not call deprecated robot/track `check_input_file()` methods: they prompt on
the terminal. Use the normal startup GUI for operator review.

### 4. Workflow Structure Pattern
```python
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

# Workflow config constants: module-level UPPERCASE constants (never a dict or
# function-local variables) so ConfigManager can auto-detect them and persist
# them to workflow_configs/<workflow_name>.yaml
SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/experiment_vials.csv"
_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE"]  # extend per workflow

def execute(config=None, show_gui=True):
    lash_e = Lash_E(workflow_globals=globals(), workflow_name="this_workflow_name",
                    config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return
    c = lash_e.workflow_config

    # Move to working position
    lash_e.nr_robot.move_vial_to_location("target_vial", "clamp", 0)

    # Get fresh wellplate for measurements
    lash_e.nr_track.get_new_wellplate()

if __name__ == "__main__":
    execute()
```

**CRITICAL: Confirmed configuration is authoritative.** ConfigManager owns
constant detection, YAML creation/loading, complete supplied-config validation
and snapshots. `Lash_E` calls it before/after startup GUI review and exposes
`lash_e.workflow_config` after confirmation. GUI edits update globals too for
legacy helpers, but do not update plans/dictionaries built earlier. Validate,
calculate, plan and execute using `lash_e.workflow_config` only AFTER `Lash_E`
returns. Do not preload configuration or cache scientific parameters in workflows.

Follow `workflows/workflow_template.py` or the small `test_vortex_scheduler.py`
example; do not add another config manager or workflow-startup helper:
- `execute()` uses GUI-confirmed YAML/global values.
- `execute(config=complete_config, show_gui=False)` uses the supplied mapping
    without workflow YAML reload/write. Require every `_CONFIG_KEYS` key; do not
    silently merge partial overrides. Pass `config`, `workflow_globals`,
    `workflow_name` and `show_gui` directly to `Lash_E`; ConfigManager selects
    the supplied path and bypasses YAML. The workflow should not branch on config.
- Reject supplied config with `show_gui=True` rather than guessing precedence.
- `show_gui=False` only skips review; it does not imply simulation. `SIMULATE`
    must be explicit in the selected config and agree with the controllers.
- With workflow config, do not redundantly pass `simulate` or a preloaded vial
    path: `Lash_E` selects both from that config before controller initialization.
    `INPUT_VIAL_STATUS_FILE=None` explicitly disables vial tracking; missing keys
    or invalid paths do not. Plain legacy `Lash_E(vial_file, simulate=...)` calls
    without workflow config continue to use their explicit arguments.
- Test GUI edits with a mocked review changing parameters and assert that the
    plan and automation calls use the changed values. Test supplied-config
    isolation with no YAML load. Never use unrestricted mocks as proof that a
    controller method exists.

### 5. Error Handling & Slack Integration
```python
# Unified error pattern via North_Base.pause_after_error()
self.pause_after_error("Error description", send_slack=True)
# Automatically logs, sends Slack notification, pauses for human intervention
```

Include workflow-level Slack start, completion and failure/interruption updates
as demonstrated by `_send_workflow_slack` in `workflows/workflow_template.py`.
Use the confirmed controller's `lash_e.simulate` to suppress all notifications
and Slack imports during simulation. Use `slack_agent.safe_send_slack_message`
for best-effort updates; log failed delivery/import as a non-fatal warning.
Attempt hardware cleanup before failure notification and preserve the original
exception. Do not label an interrupted or failed workflow completed.

### 6. Logging Standards
**CRITICAL**: Never use Unicode characters in logging messages (μ, →, ±, etc.)
- Use "uL" not "μL"  
- Use "->" not "→"
- Use "+/-" not "±"
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
- Update CHANGELOG.md only for significant, user-visible changes — not on every edit
- **NEVER save any files in the root directory** - use appropriate subdirectories
- Use git branches instead of timestamped backup copies of files

### CRITICAL: Workflow Debugging Guidelines
**NEVER use print() statements in workflow files:**
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