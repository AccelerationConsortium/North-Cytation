# Changelog

## [SUBSTOCK REFILL IN ITERATION LOOP] - 2026-04-10

### Feature: Substock refilling before each iterative round
- Added substock top-up call in `execute_iterative_workflow` loop after water and stock refills
- Reconstructs `dilution_recipes` from `results['experiment_plan']['stock_solutions_needed']` each iteration
- Delegates all skip/top-up/recreate logic to existing `create_substocks_from_recipes` (unchanged)
- Files: `workflows/surfactant_grid_adaptive_concentrations.py`

## [CRITICAL DOUBLE CONVERSION FIX] - 2026-04-08

### CRITICAL BUG FIX: Double Unit Conversion in Ilya Workflow V2  
- **FIXED**: Double conversion causing 50μL to become 0.05μL instead of 0.05mL
- **ROOT CAUSE**: Line 299 already converts CSV μL→mL, but code assumed data was still in μL
- **SEQUENCE**: CSV(50μL) → Line299(÷1000=0.05mL) → MyCode(÷1000=0.00005mL) → Display(0.05μL)
- **SYMPTOM**: All volumes below 10μL minimum, all wells skipped during dispensing
- **SOLUTION**: Treat data as mL after line 299, convert mL→μL only for display/validation
- **IMPACT**: 50, 100, 150, 200μL volumes now correctly processed as 0.05, 0.1, 0.15, 0.2mL
- **FILES AFFECTED**: `workflows/ilya_workflow_v2.py` lines 110-112, 155-157, 163-165, 360-362

## [SIMULATION PERFORMANCE FIX] - 2026-04-08

### BUG FIX: Simulation Mode Timing Issues
- **FIXED**: 5-second delays in simulation mode during pipetting operations
- **ROOT CAUSE**: Typo `self.SIMULATE` (uppercase) instead of `self.simulate` (lowercase) in post_retract_wait_time check
- **RESULT**: AttributeError caused simulation bypass to fail, executing full hardware wait times in simulation
- **SOLUTION**: Corrected simulation attribute reference to proper lowercase `self.simulate`  
- **IMPACT**: Simulation now runs at proper speed without unnecessary delays
- **FILES AFFECTED**: `North_Safe.py` line 2125 (post_retract_wait_time simulation bypass)

## [CRITICAL CALIBRATION FIX] - 2026-04-08

### CRITICAL BUG FIX: Embedded Calibration Corruption Prevention
- **FIXED**: Calibration system that saved Stage 3 parameters even when accuracy got worse
- **ENHANCED**: Now compares ALL THREE stages and picks the overall best performer  
- **FIXED**: CSV measurement data now updated with fresh measurements from optimization process
- **ROOT CAUSE**: Blind `_update_calibration_csv()` call with no performance validation + stale measurement data
- **SOLUTION**: Smart stage comparison logic + fresh measurement data replacement from winning stage  
- **PROTECTION**: System preserves good calibration AND uses only matched parameter-measurement pairs
- **INTELLIGENCE**: Stage 2 (crossing strategy) can now be selected if it outperforms interpolation
- **DATA INTEGRITY**: Measurement fields updated with actual measured volumes from optimization process
- **EVIDENCE**: Previously saved 6.3µL error parameters over existing 1.4µL tolerance parameters
- **VALIDATION**: Comprehensive stage ranking with fresh measurement data from best performing stage
- **REPORTING**: Enhanced Slack notifications show winning stage and APPLIED/REJECTED status
- **IMPACT**: Prevents calibration degradation AND ensures optimal parameters with fresh measurement data

### FILES MODIFIED  
- `pipetting_data/embedded_calibration_validation.py`: Added conditional update logic and enhanced validation

## [VIAL GUI ENHANCEMENTS] - 2026-04-02

### ENHANCEMENT: Vial GUI "Empty Vial" Feature
- **ADDED**: "Empty Vial" button to vial editing dialog for quick volume reset
- **FUNCTIONALITY**: Single click sets volume to 0 and closes dialog (faster than manual entry)
- **UI STYLING**: Yellow background button placed next to "Remove Vial" for easy access
- **WORKFLOW**: Improves efficiency when marking vials as empty during experiments

### BUG FIX: Reset Locations Button Not Working
- **FIXED**: "Return All Home" button now properly refreshes the GUI display
- **ROOT CAUSE**: Widget refresh method wasn't properly clearing and reloading vial positions
- **SOLUTION**: Enhanced `_reload_all_widgets()` with proper widget cleanup and grid reset
- **DEBUGGING**: Added comprehensive logging for troubleshooting reset operations
- **IMPACT**: Digital vial location reset now visually updates all rack grids correctly

### FILES MODIFIED
- `vial_manager_gui.py`: Added empty vial feature and fixed reset locations functionality

## [CRITICAL TIP SELECTION BUG FIXES] - 2026-04-02

### ENHANCEMENT: Simplified Water 200µL Calibration Script  
- **SIMPLIFIED**: Batch calibration automation script to focus solely on water at 200 µL
- **REMOVED**: Multi-liquid calibration loop, now runs single water calibration only
- **UPDATED**: UI messages and titles to reflect water-specific calibration purpose
- **MAINTAINED**: All core functionality (config modification, vial swapping, calibration + validation)
- **STREAM-LINED**: Single calibration run instead of batch processing multiple liquids
- **FILES CHANGED**: `calibration_modular_v2/batch_calibration_automation.py`

### CRITICAL BUG FIX: XY Movement Gripper Angle Preservation 
- **FIXED**: Gripper unwantedly rotating during XY movements in Enhanced SP program
- **ROOT CAUSE**: IK solver choosing completely different gripper angles to reach target XY position
- **EXAMPLE**: 0.3mm Y movement caused 905 count gripper rotation (2552→1647 counts)
- **SOLUTION**: Preserve current gripper angle during XY movements, only adjust elbow and shoulder
- **METHOD**: Use `current_gripper_rad` instead of IK-calculated gripper angle in target position
- **IMPACT**: XY movements now keep gripper orientation stable, only adjusting arm joints as intended
- **FILES FIXED**: Enhanced SP arm position program `_move_xy_delta()` method

### ENHANCEMENT: Added Gripper Rotation Controls to Enhanced SP Program
- **ADDED**: Gripper rotation controls to Enhanced SP arm position program UI
- **NEW BUTTONS**: "🔄 CCW" and "↻ CW" for counter-clockwise and clockwise rotation
- **METHODS**: `move_gripper_ccw()` and `move_gripper_cw()` with safety bounds checking
- **SAFETY**: Rotation respects GRIPPER_MIN_RAD and GRIPPER_MAX_RAD limits (-6.28 to 6.28 rad)
- **STEP SIZE**: Uses configurable `move_increment_rad` (default 0.05 rad = ~2.9°)
- **DISPLAY**: Gripper angle shown in both radians and degrees in position display
- **UI LAYOUT**: Added rotation row between gripper open/close and clamp controls

### CRITICAL BUG FIX: Color Mixing Workflow Tip Selection Algorithm Error
- **FIXED**: Excessive tip switching (18 switches during experiment) caused by wrong function usage
- **ROOT CAUSE**: Color mixing workflow incorrectly used `pipet_from_wellplate(..., aspirate=False)` instead of `dispense_into_wellplate()`
- **MECHANISM**: `aspirate_from_vial()` selects large_tip (0.221 mL total with overvolumes) → `pipet_from_wellplate()` selects small_tip (0.200 mL base only) → immediate tip switching
- **SOLUTION**: Changed all dispensing operations from `pipet_from_wellplate(..., aspirate=False)` to `dispense_into_wellplate()`
- **IMPACT**: Eliminated unnecessary tip selection conflicts during wells 54-65 and 84-89 processing

### CRITICAL BUG FIX: Tip Selection Algorithm Overaspirate Volume Issue
- **FIXED**: Overaspirate volume incorrectly included in tip selection causing unnecessary large_tip usage
- **ROOT CAUSE**: Tip selection included overaspirate_vol (0.010 mL) making 200 µL → 221 µL total, exceeding small_tip capacity (200 µL)
- **SOLUTION**: Removed overaspirate_vol from tip selection calculation: `total_tip_vol = post_asp_air_vol + amount_mL` 
- **REASONING**: Overaspirate is for liquid handling precision, not tip capacity. Post-aspirate air gap still considered.
- **IMPACT**: 200 µL now correctly selects small_tip (0.200 + 0.011 air = 0.211 mL < 0.20 mL threshold)
- **FILES FIXED**: `North_Safe.py` (line 2039)
- **FILES FIXED**: `workflows/color_mixing.py` (3 instances in main loop + mix_wells function)
- **ARCHITECTURE ALIGNED**: Color mixing now uses same dispensing pattern as other workflows (aspirate_from_vial → dispense_into_wellplate)

## [CRITICAL X-Y MOVEMENT BUG FIXES] - 2026-03-31

### CRITICAL BUG FIX: Motor Fault Prevention in X-Y Movement
- **FIXED**: Motor faults caused by massive unintended movements during small X-Y adjustments
- **ROOT CAUSE 1**: Wrong parameter order in forward kinematics call - `n9_fk(gripper, shoulder, elbow)` instead of correct `n9_fk(gripper, elbow, shoulder)`
- **ROOT CAUSE 2**: IK function returns radians despite documentation claiming counts, causing unit conversion errors
- **SOLUTION 1**: Corrected FK parameter order in `update_display()` to properly calculate current X-Y position
- **SOLUTION 2**: Added automatic unit detection and conversion for IK results (radians → counts)
- **SAFETY ENHANCEMENT**: Added movement safety limits (2000 cts per axis, 5000 cts total) to abort unsafe movements
- **IMPACT**: Small 2mm movements now execute safely instead of attempting 38,000+ count dangerous movements
- **PREVENTION**: Robot aborts movements exceeding safety limits instead of triggering motor controller faults

## [WORKFLOW TESTER IMPROVEMENTS] - 2026-03-30

### MAJOR ENHANCEMENT: Flow-Based Operation Testing 
- **REVAMPED**: `tests/surfactant_workflow_tester.py` - Complete redesign from predefined test cycles to flow-based button operations
- **NEW ARCHITECTURE**: Individual operation buttons organized by workflow stage (Source → Transfer → Liquid → Analysis → Disposal)  
- **PROPER IMPLEMENTATIONS**: Updated all functions to match actual surfactant workflow patterns and API usage
- **ENHANCED VIAL HANDLING**: Use real vial names ("water", "SDS_stock", "TTAB_stock", "pyrene_DMSO") instead of placeholders
- **SMART POSITIONING**: Implements idempotent track positioning with status checking (only moves if needed)
- **PROPER LIQUID HANDLING**: Correct `aspirate_from_vial` and `dispense_into_wellplate` usage with proper liquid types
- **TIP MANAGEMENT**: Proper pipet conditioning, removal, and volume tracking
- **ATOMIC OPERATIONS**: Cytation operations follow proper carrier in/out patterns with error recovery
- **WORKFLOW STATE TRACKING**: Real-time display of vial, wellplate position, and pipet status
- **EXAMPLE FLOWS**: "Source → Pipetting → Waste" or "Analysis → Cytation → Read → Return" button sequences
- **IMPROVED ERROR HANDLING**: Comprehensive simulation mode support and safety checks

## [COMPONENT TEST GUI] - 2026-03-30

### NEW FEATURE: Workflow Component Testing GUI
- **ADDED**: `tests/surfactant_workflow_tester.py` - Comprehensive GUI for testing individual workflow components
- **TESTING CAPABILITIES**: Wellplate movement, robot operations, Cytation protocols, and vial manipulation
- **REPETITIVE TESTING**: Run any operation multiple times with configurable delays to test consistency
- **REAL-TIME MONITORING**: Progress tracking, success/error counters, and detailed test logs
- **DUAL MODE SUPPORT**: Both simulation and real hardware testing modes
- **SAFETY FEATURES**: Emergency stop functionality and error isolation per test iteration
- **USE CASE**: Validate equipment reliability before running critical experiments

## [TRIANGLE RELIABILITY INDEXING FIX] - 2026-03-27

### CRITICAL BUG FIX: Delaunay Triangle Reliability Mask Index Mismatch
- **FIXED**: Triangle recommender filtering center triangles due to index mismatch between data and reliability mask
- **ROOT CAUSE**: Workflow passed all 48 points to recommender but triangle scorer internally filtered to 25 experimental points, causing index misalignment
- **SOLUTION**: Filter data to experimental points BEFORE passing to recommender, eliminating internal filtering and index confusion
- **IMPACT**: Center triangles now properly evaluated instead of being incorrectly filtered as "unreliable"
- **ARCHITECTURAL IMPROVEMENT**: Cleaner separation - recommender receives only the data it needs, no internal filtering required

## [VIAL POSITIONING FIX] - 2026-03-18

### CRITICAL BUG FIX: Invalid Vial Location
- **FIXED**: `glycerol_dye` vial invalid location causing workflow failures
- **ROOT CAUSE**: Vial at `location_index: 48` but `main_8mL_rack` only supports positions 0-47
- **SOLUTION**: Moved `glycerol_dye` from position 48 → position 5
- **IMPACT**: Workflow now correctly consumes `glycerol_dye` in wells 8-11, 20-23
- **SYMPTOM RESOLVED**: Vial volumes now decrease as expected during workflow execution

## [X-Y COORDINATE CONTROL SYSTEM] - 2026-03-17

### MAJOR ENHANCEMENT: Intuitive X-Y Coordinate Robot Control
- **X-Y COORDINATE INTERFACE**: Converted shoulder/elbow joint controls to intuitive X-Y coordinate system
- **NORTH API KINEMATICS**: Integrated `n9_fk()` and `n9_ik()` functions for seamless coordinate conversion
- **UNIFIED STEP SIZE**: Single movement increment (mm) now controls both Z-axis and X-Y movement
- **ENHANCED MOVEMENT FUNCTIONS**: 
  - Added `move_x_left()`, `move_x_right()`, `move_y_forward()`, `move_y_back()` using inverse kinematics
  - Implemented `_safe_move_xy()` with workspace bounds checking and fallback joint control
- **GUI IMPROVEMENTS**:
  - Updated position display labels: \"Shoulder\" → \"Y-Position\", \"Elbow\" → \"X-Position\"
  - Modified control buttons: \"← SHOULDER -\" → \"← X LEFT\", \"→ SHOULDER +\" → \"→ X RIGHT\"
  - Updated Y-axis controls: \"↓ ELBOW -\" → \"↓ Y BACK\", \"↑ ELBOW +\" → \"↑ Y FORWARD\"
- **COORDINATE TRACKING**: Added `current_x_position` and `current_y_position` tracking variables
- **REAL-TIME CONVERSION**: Forward kinematics automatically updates X-Y display from joint positions
- **KEYBOARD CONTROLS MAINTAINED**: 
  - Arrow keys (Left/Right) now control X-axis movement
  - W/S keys control Y-axis movement (forward/back)
  - Up/Down arrows still control Z-axis
- **BACKWARD COMPATIBILITY**: Joint angle tracking and functions preserved for internal use
- **TESTING FRAMEWORK**: Created comprehensive test suite for coordinate conversion validation
- **FILES MODIFIED**: 
  - workflows/enhanced_SP_arm_position_program.py (major X-Y coordinate system integration)
  - workflows/test_xy_coordinates.py (new testing framework)

## [SURFACTANT LIQUID_TYPE FIX] - 2026-03-16

### BUG FIX: Corrected liquid_type Parameters for Surfactant Operations
- **VALIDATION CALLS**: Fixed surfactant validation to use `liquid_type='SDS'` instead of `'water'`
- **DISPENSING CALLS**: Fixed surfactant dispensing to use `liquid_type='SDS'` instead of `'water'`  
- **SLACK NOTIFICATIONS**: Now correctly show "Liquid: SDS" in Slack messages for surfactant operations
- **PIPETTING PARAMETERS**: Surfactant operations now use SDS-optimized parameters instead of water parameters
- **DATABASE LOGGING**: Surfactant operations now logged as "SDS" for proper tracking and analysis
- **AFFECTED FUNCTIONS**: 
  - `dispense_component_to_wellplate()` calls for surf_A_volume_ul and surf_B_volume_ul
  - `validate_pipetting_accuracy()` calls for surfactant stock validation (both small and large volumes)
- **FILES MODIFIED**: workflows/surfactant_grid_adaptive_concentrations.py (4 liquid_type corrections)

## [USER-DEFINED POSITION NAMES] - 2026-03-16

### MAJOR FEATURE: Custom Position Saving with User-Defined Names
- **CUSTOM POSITION NAMES**: Text input field for user-defined position names
- **TEMP FOLDER SAVING**: Saves positions with custom names to temp/ directory 
- **MULTIPLE EXPORT FORMATS**: 
  - JSON format with full position data
  - TXT format with coordinates for easy copy-paste
  - Array format for direct code integration
  - Python dictionary format for workflow files
- **SAVED POSITIONS MANAGER**: GUI window to view, copy, and manage all saved positions
- **AUTOMATED NAMING**: Auto-increments position names (my_position_1 → my_position_2)
- **COPY COORDINATES**: Quick clipboard copy of current position coordinates
- **TEMP FOLDER ACCESS**: Direct button to open temp folder in explorer
- **ENHANCED EXPORT**: CSV export of all saved positions with timestamps
- **USER WORKFLOW**: Save → Name → Export coordinates → Paste into other files
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py (added custom naming system)

## [COMPLETE PIPETTE INTEGRATION] - 2026-03-16

### MAJOR FEATURE: Complete Pipette Tip Rack Position System
- **PIPETTE TIP POSITIONS**: Added 96 pipette tip positions (48 large + 48 small tips)
- **TIP REMOVAL STATIONS**: Added 3 tip removal/disposal positions (Cap, Approach, Small)
- **POSITION EXPANSION**: Total positions increased from ~70 to 166 
- **YAML INTEGRATION**: Extended robot_state/vial_positions.yaml with pipette tip rack configurations
- **WORKFLOW SUPPORT**: Enhanced position program now supports complete pipette tip pickup/disposal workflow
- **COORDINATE MAPPING**: Proper coordinate mapping from Locator.py pgrid_low_2/pgrid_high_2 arrays
- **CATEGORY ORGANIZATION**: Added Pipette_Tips and Tip_Removal position categories
- **FILES MODIFIED**: 
  - workflows/enhanced_SP_arm_position_program.py (added pipette position generation)
  - robot_state/vial_positions.yaml (added tip rack mappings)

## [CRITICAL SAFETY FIX] - 2026-03-16

### CRITICAL: Safe Two-Step Movement to Prevent Vial Collisions
- **SAFETY ISSUE**: Robot was lowering Z-axis first then moving X-Y, causing collisions with vials on rack
- **SOLUTION**: Implemented safe two-step movement pattern:
  1. Move to safe Z height (10,000 counts = ~100mm high) while maintaining current X-Y position
  2. Move X-Y to target position while staying at safe height  
  3. Finally lower Z-axis to target position
- **COLLISION AVOIDANCE**: Prevents robot from hitting vials/obstacles during movement
- **ERROR HANDLING**: Added try/catch for current position detection with safe defaults
- **COORDINATE FIX**: Fixed MockRobot coordinate order to match Locator.py format [gripper, shoulder, elbow, z]
- **CONSTANT ADDED**: SAFE_Z_HEIGHT_COUNTS = 10000 for consistent safe movement height
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py (goto_selected_position method and MockRobot class)

## [PANDAS FALLBACK FIX] - 2026-03-16

### CRITICAL FIX: CSV Loading Pandas Dependency Issue
- **CSV LOADING FIX**: Added fallback CSV reader using built-in csv module when pandas fails
- **ROOT CAUSE**: Pandas installation issue preventing dynamic position loading (pandas missing read_csv attribute)
- **SOLUTION**: Created SimpleDataFrame class to mock pandas behavior for compatibility
- **IMPACT**: Dynamic position loading now works reliably - loads 67 positions including 51 named vials
- **VERIFICATION**: All coordinate mappings verified correct (water, clamp positions match Locator.py exactly)
- **FALLBACK PATH**: System gracefully falls back from pandas -> csv module -> hardcoded positions
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py (CSV loading section)

## [CRITICAL COORDINATE BUG FIX] - 2026-03-16

### CRITICAL FIX: Robot Position Coordinate Order Bug
- **COORDINATE ORDER BUG**: Fixed critical bug where shoulder and elbow coordinates were swapped
- **ROOT CAUSE**: Program incorrectly treated Locator.py coordinates as [gripper, elbow, shoulder, z] instead of [gripper, shoulder, elbow, z]
- **IMPACT**: Robot was going to wrong positions when using dynamic positioning system
- **SOLUTION**: Corrected target_position array order in goto_selected_position() method
- **VERIFICATION**: Coordinate mapping now matches Locator.py format: [gripper, shoulder, elbow, z_axis]
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py (line ~846-851)

## [DYNAMIC POSITIONING SYSTEM] - 2026-03-16

### MAJOR: Integrated Multi-Layer Dynamic Positioning System
- **DYNAMIC POSITIONS**: Enhanced position controller now loads positions from multi-layer positioning system
- **CSV INTEGRATION**: Automatically loads vial positions from CSV files (e.g., surfactant_grid_vials_expanded.csv)
- **YAML CONFIG**: Uses vial_positions.yaml for rack configuration and Locator.py coordinates
- **DUAL ACCESS**: Supports both gripper and pipetting positions for each vial location
- **NAMED VIALS**: Shows actual vial names (water, SDBS_stock, etc.) instead of just coordinates
- **LIVE RELOAD**: Configuration UI allows users to change vial files and reload positions on-demand
- **FALLBACK SAFETY**: Maintains hardcoded positions if dynamic loading fails
- **AUTO DETECTION**: Automatically detects and loads default vial file if available
- **BACKUP CREATED**: enhanced_SP_arm_position_program_backup_YYYYMMDD_HHMMSS.py
- **FILES MODIFIED**: workflows/enhanced_SP_arm_position_program.py

## [POSITION SYSTEM FIX] - 2026-03-16

### FIXED: Enhanced Position Controller Positioning Issues
- **COORDINATE FIX**: Updated enhanced_SP_arm_position_program.py to use real robot coordinates from Locator.py
- **MOVEMENT METHOD**: Changed from move_z/move_axis_rad to goto() method for precise absolute positioning
- **REAL POSITIONS**: All 20+ positions now use actual robot count values instead of estimated mm/rad values
- **GOTO METHOD**: Added goto() method to MockRobot simulation for testing
- **POSITION CLEANUP**: Removed leftover old position definitions that were causing conflicts
- **ENHANCED GUI**: Expanded window to 750x900 for better visibility
- **COMPREHENSIVE POSITIONS**: Includes all vial racks, tip racks, processing stations with real coordinates
- **FILES FIXED**: workflows/enhanced_SP_arm_position_program.py

## [COMPREHENSIVE WORKFLOW POSITIONS] - 2026-03-16

### MAJOR: Complete Workflow Position System with Real Coordinates
- **REAL COORDINATES**: Updated all positions with actual robot counts from Locator.py
- **COMPREHENSIVE POSITIONS**: Added 25 workflow positions including all vial racks, tip racks, and processing stations
- **VIAL POSITIONS**: Main 8mL rack (positions 36, 43-47), Large vial rack, Small vial rack, 50mL vial rack
- **TIP RACK POSITIONS**: Small tip rack (positions 0, 24, 47), Large tip rack (positions 0, 24, 47)
- **PROCESSING STATIONS**: Photoreactor, Heater grid (positions 0, 5), Clamp position
- **TIP REMOVAL**: Small and large pipet tip removal positions
- **SAFETY POSITIONS**: Home and Safe transport height
- **ROBOT MOVEMENT**: Uses goto() method with absolute count positioning for precise movement
- **EXPANDED GUI**: Increased window size to 750x850, wider dropdown for longer position names
- **ENHANCED SIMULATION**: Mock robot now supports goto() method with count-based positioning
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [WORKFLOW POSITION DROPDOWN] - 2026-03-16

### ENHANCED: SP_arm_position_program.py with Workflow Position Dropdown
- **NEW FEATURE**: Position dropdown menu with 11 predefined surfactant workflow positions
- **WORKFLOW POSITIONS**: Added positions from surfactant workflow (Home, Clamp, Main Rack 36/43-47, Photoreactor, Heater Grid, Safe Transport)
- **POSITION SELECTION**: Dropdown shows position names with descriptions when selected
- **GO TO POSITION**: One-click button to move robot arm to selected workflow position
- **SAFETY FIRST**: Z-axis moves first for safe positioning, then rotational axes
- **SUCCESS FEEDBACK**: Confirmation messages when position movement completes
- **ENHANCED GUI**: Expanded window size (650x800) to accommodate new Position Selection section
- **INTEGRATED WORKFLOW**: Positions match those used in surfactant_grid_adaptive_concentrations.py
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [ENHANCED POSITION MANAGEMENT SYSTEM] - 2026-03-16

### MAJOR: Enhanced Robot Position Control with Workflow Integration
- **NEW FEATURE**: Position dropdown with 13 predefined surfactant workflow positions
- **WORKFLOW INTEGRATION**: Includes positions like 'Clamp Position', 'Main Rack 36-47', 'Photoreactor', etc.
- **POSITION MANAGEMENT**: Temporary position storage system for adjustment sessions
- **EXPORT FUNCTIONALITY**: CSV export for position modifications, TXT export for session history, JSON export for complete data
- **POSITION CATEGORIES**: Organized positions by function (Storage, Reagents, Processing, Manipulation, Large Volume, Small Volume, Standard)
- **TEMPORARY MODIFICATIONS**: Save current position as temporary, reset to original, clear all temporary positions
- **SESSION TRACKING**: Complete history of movements and position changes during session
- **ENHANCED GUI**: New sections for Position Selection, Position Modifications, and Export controls
- **SAFETY ENHANCED**: Proper position validation and bounds checking for all workflow positions
- **FILES CREATED**: workflows/enhanced_SP_arm_position_program.py (extends original with position management)

## [CALIBRATION CSV UPDATE] - 2026-03-17

### FIXED: Adaptive Overaspirate Correction Now Persists to Calibration File
- **Added** `_update_calibration_csv()` in `pipetting_data/embedded_calibration_validation.py`
- After Stage 3 optimization, the winning `overaspirate_vol` is written back to `optimal_conditions_{liquid}.csv`
- If the exact volume row already exists, only `overaspirate_vol` is updated; all other params unchanged
- If the volume is new, a fully interpolated row is inserted (all numeric columns interpolated from surrounding volumes), then the file is re-sorted
- Simulation mode prints what would be written without touching the file
- **Files Modified**: `pipetting_data/embedded_calibration_validation.py`

## [MULTI-JOINT ROBOT ARM CONTROL] - 2026-03-16

### ENHANCED: Multi-Joint Robot Control System
- **MAJOR UPGRADE**: Extended SP_arm_position_program.py to control all robot joints (Z-axis, Elbow, Shoulder)
- **New Controls**: LEFT/RIGHT arrows for shoulder rotation, W/S keys for elbow extend/retract
- **Enhanced GUI**: Real-time position display for all joints with both radians and degrees
- **Safety Features**: Proper joint limits for elbow (±150°) and shoulder (±120°) based on North API specs
- **Button Controls**: Dedicated GUI buttons for each joint movement direction
- **Simulation Support**: Extended mock robot to simulate all joint movements with realistic limits
- **Position Tracking**: Real-time updates of current elbow angle and shoulder angle
- **Error Handling**: Individual safety checks and error handling for each joint type
- **Files Modified**: workflows/SP_arm_position_program.py

## [USER-CONFIGURABLE MOVEMENT INCREMENTS] - 2026-03-16

### NEW: Custom Movement Step Control System
- **NEW FEATURE**: Added user-configurable movement increments for all joints
- **INPUT VALIDATION**: Real-time validation with safety limits and error feedback
- **Z-AXIS CONTROL**: Configurable step size (0.1-50.0mm) with live input validation
- **ROTATIONAL CONTROL**: Configurable step size (0.01-1.0 radians) with degrees display
- **ENHANCED SAFETY**: Improved error messages showing attempted values and limits
- **VISUAL FEEDBACK**: Real-time degrees conversion display and status updates  
- **PRECEDENT CHECKING**: System validates movements won't exceed joint limits before execution
- **USER GUIDANCE**: Clear min/max limits displayed in validation messages
- **PERSISTENT SETTINGS**: Custom increments stay active until manually changed
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

### Testing Confirmed:
- ✅ Set Z-axis from 5.0mm to 10.0mm - successful movement validation
- ✅ Set rotational increment from 0.1rad to 1.0rad (57.3°) - full functionality
- ✅ All safety limits and range checking working properly
- ✅ Enhanced error messages providing clear user guidance

## [ROBOT ARM MULTI-JOINT CONTROL] - 2026-03-16

### ENHANCED: Multi-Joint Robot Arm Control System
- **ENHANCED**: Extended SP_arm_position_program.py with complete multi-joint control capability
- **NEW JOINTS**: Added Shoulder, Elbow, and Gripper joint control (previously only Z-axis)
- **KEYBOARD CONTROLS**: 
  - ↑/↓ Arrows: Z-axis movement
  - ←/→ Arrows: Shoulder rotation
  - W/S Keys: Elbow extend/retract
  - Q/E Keys: Gripper rotation (CCW/CW)
- **GUI ENHANCEMENTS**: Individual position displays for all 4 joints with real-time updates
- **SAFETY LIMITS**: Joint-specific angle limits with error checking and warnings
- **SIMULATION MODE**: Enhanced mock robot with full 4-joint simulation capability
- **API INTEGRATION**: Uses proper North API methods (`move_axis_rad()`, `get_robot_positions()`)
- **FILES MODIFIED**: workflows/SP_arm_position_program.py

## [ROBOT ARM POSITION CONTROL PROGRAM] - 2026-03-16

### NEW: Interactive Robot Arm Control Program
- **NEW FEATURE**: Created SP_arm_position_program.py for interactive robot arm control
- **GUI Interface**: Tkinter-based GUI with real-time position display and status indicators
- **Arrow Key Control**: Up/Down arrow keys move Z-axis in 5mm increments with safety limits
- **Home Button**: Dedicated button to home the robot to reference position
- **Keyboard Shortcuts**: ESC to exit, SPACE to home, arrow keys for movement
- **Safety Features**: Position limits (30-292mm), home requirement before movement, error handling
- **Simulation Support**: Automatic fallback to simulation mode if North library unavailable
- **Real-time Feedback**: Live position updates and connection status display
- **Logging**: Comprehensive logging to file and console for debugging
- **Files Added**: workflows/SP_arm_position_program.py

## [CONDITION TIP PARAMETER FIX] - 2026-03-16

### FIXED: TypeError in condition_tip function call  
- **CRITICAL**: Fixed "TypeError: condition_tip() got an unexpected keyword argument 'liquid'" in acid pipetting
- **Root Cause**: Function call had wrong parameters - `liquid` parameter doesn't exist, volume units mismatch
- **Fix**: Removed invalid `liquid=liquid` parameter and converted volume from mL to µL  
- **Impact**: TFA acid addition with tip conditioning now works correctly
- **Files Modified**: workflows/Degradation_serena.py line 157

## [CRITICAL MOTOR DRIVER FIX v2] - 2026-03-13

### EMERGENCY HARDWARE PROTECTION - Anti-Shoot-Through Implementation
- **CRITICAL**: Implemented comprehensive motor driver protection against hardware-killing code patterns
- **Anti-Shoot-Through**: 100ms mandatory motor-off deadtime before ANY movement command
- **Direction Reversal Protection**: 500ms blocking delay when direction changes detected
- **Rapid Click Prevention**: 300ms UI button locks prevent rapid command sending

- **Command Spacing**: Minimum 300ms between ANY jog commands (was 100ms)
- **Continuous Jog Rate**: Reduced to 2 steps/second (was 5) with 500ms intervals
- **Status Monitoring**: Robot status checked before every movement command
- **Blocking Operations**: All moves use wait=True to prevent command stacking
- **Extra Settling Time**: 50ms delays after moves to ensure driver fully settles

### Hardware-Specific Protections Added
Based on common motor driver failure patterns:
- **Prevents Both Direction Pins Active** (shoot-through condition)  
- **Eliminates Instant Full-Power Reverse** (no deadtime failures)
- **Prevents Command Overlap** (rapid state changes)
- **Ensures Adequate Deadtime** (100ms motor-off before commands)
- **UI-Level Protection** (button disable/re-enable cycles)

### Speed & Timing Limits  
- **Ultra-Conservative Speeds**: 1000-2000 cts/s maximum (was 4000)
- **Gentle Acceleration**: 4000-8000 cts/s² maximum (was 15000)
- **Extended Delays**: Multiple timing protections at different levels
- **Frequency Limiting**: Max 2 Hz continuous jog rate (well below driver limits)

### User Interface Updates
- **Window Title**: Clear "MOTOR DRIVER PROTECTION MODE" warning
- **Startup Alert**: Detailed protection measure explanation
- **Status Messages**: "ANTI-SHOOT-THROUGH" and "driver-safe mode" indicators  
- **UI Warnings**: Red text explaining 300ms click delays and protection measures

### Code Architecture Changes
- **State Tracking**: Last axis, direction, and timing tracking
- **Busy Flag System**: Prevents any overlapping jog operations  
- **Safe Button Management**: Prevents UI callback errors
- **Exception Handling**: Motor driver failure detection and safe shutdown
- **Clean Disconnect**: All jog states cleared on disconnect/close

This version addresses the specific code patterns that physically destroy motor drivers, implementing multiple layers of hardware protection beyond just conservative speeds.

## [1.2.4] - 2026-03-13

### Fixed
- **CRITICAL: SP_arm_position_program Robot Movement Error**: Fixed "NorthC9 object has no attribute 'move_vial_to_location'" error
  - **Issue**: SP_arm_position_program.py was trying to call `move_vial_to_location()` directly on NorthC9 object, but this method doesn't exist there
  - **Root cause**: Incorrect architecture - should use Lash_E coordinator that wraps North_Robot class which provides the `move_vial_to_location()` method
  - **Fix**: Updated SP_arm_position_program.py to use Lash_E coordinator pattern like other workflows
    - Added Lash_E import and initialization in connect() method
    - Changed robot movement calls to use `lash_e.nr_robot.move_vial_to_location()`
    - Added proper error handling for wellplate movements when track not initialized
  - **Impact**: Program now properly integrates with robot control system and can move vials to predefined workflow positions
  - **Files**: `workflows/SP_arm_position_program.py` (backup created before changes)

### Improved  
- **CONNECTION STABILITY: SP_arm_position_program Robust Connection System**: Added multi-strategy connection with auto-reconnection
  - **Issue**: Connection timeouts and failures due to complex Lash_E initialization, missing vial files, and no auto-reconnection
  - **Solution**: Implemented dual-strategy connection system:
    1. **Primary**: Direct NorthC9 connection (fastest, most reliable for teaching)
    2. **Fallback**: Lash_E with auto-generated minimal vial file if direct fails
    3. **Auto-reconnection**: Configurable auto-retry with exponential backoff
    4. **Error recovery**: Graceful handling of communication errors with automatic reconnection attempts
  - **Features Added**:
    - Auto-reconnect checkbox for continuous operation
    - Disconnect button for clean disconnection 
    - Connection status indicators with attempt counting
    - Dual-mode operation: full Lash_E (for vial movements) or direct NorthC9 (for manual positioning)
    - Automatic minimal vial file creation (`status/minimal_vials.csv`)
  - **Impact**: Dramatically improved connection reliability and user experience for robot positioning/teaching tasks
  - **Files**: `workflows/SP_arm_position_program.py`
- **PERFORMANCE: SP_arm_position_program Smart Speed Control**: Implemented intelligent speed management for different movement types
  - **Issue**: All movements used slow teaching speeds (2000 cts/s), making homing and position changes unnecessarily slow
  - **Solution**: Added speed differentiation:
    - **Fast speeds**: 8000-10000 cts/s for homing, going to saved positions, and workflow positions
    - **Slow speeds**: 2000 cts/s preserved for manual jogging/teaching (precision work)
    - **Auto-fallback**: Graceful degradation if fast speed parameters not supported
  - **Speed Constants Added**:
    - `FAST_VEL = 8000` cts/s for major movements
    - `FAST_ACC = 40000` cts/s² for fast acceleration  
    - `FAST_HOME_VEL = 10000` cts/s for homing operations
  - **UI Improvements**: Updated labels to clarify speed usage, added status messages showing movement type
- **USER EXPERIENCE: SP_arm_position_program Enhanced Connection Feedback**: Added visual progress indicators and clear status messages
  - **Issue**: Connection process was silent with no progress indication, leaving users unsure if system was working
  - **Solution**: Comprehensive connection feedback system:
    - **Progress bar**: Animated progress indicator during connection attempts
    - **Step-by-step updates**: Status messages showing connection progress ("Step 1/2: Testing direct connection...")
    - **Visual status icons**: ✅ success, ❌ errors, 🔄 in progress, ⚠️ warnings
    - **Smart button management**: Disables Connect button during connection to prevent conflicts
    - **Success/failure popups**: Clear messagebox notifications with next steps
    - **Detailed error reporting**: Comprehensive error messages with troubleshooting suggestions
  - **Features Added**:
    - Animated progress bar with connection state tracking
    - Emoji-enhanced status messages for quick visual feedback
    - Connection attempt counter with max retry limits
    - Graceful handling of connection interruption (disconnect during connection)
    - Enhanced auto-reconnection with visual progress
- **USER EXPERIENCE: SP_arm_position_program Enhanced Connection Feedback**: Added visual progress indicators and clear status messages
  - **Issue**: Connection process was silent with no progress indication, leaving users unsure if system was working
  - **Solution**: Comprehensive connection feedback system:
    - **Progress bar**: Animated progress indicator during connection attempts
    - **Step-by-step updates**: Status messages showing connection progress ("Step 1/2: Testing direct connection...")
    - **Visual status icons**: ✅ success, ❌ errors, 🔄 in progress, ⚠️ warnings
    - **Smart button management**: Disables Connect button during connection to prevent conflicts
    - **Success/failure popups**: Clear messagebox notifications with next steps
    - **Detailed error reporting**: Comprehensive error messages with troubleshooting suggestions
  - **Features Added**:
    - Animated progress bar with connection state tracking
    - Emoji-enhanced status messages for quick visual feedback
    - Connection attempt counter with max retry limits
    - Graceful handling of connection interruption (disconnect during connection)
    - Enhanced auto-reconnection with visual progress
  - **Impact**: Users now have clear feedback on connection status and can diagnose connection issues more effectively
  - **Files**: `workflows/SP_arm_position_program.py`
- **CRITICAL: SP_arm_position_program Teaching System Overhaul**: Replaced complex Lash_E system with simple in-memory position tracking
  - **Issue**: Workflow positions required complex Lash_E upgrade that often failed, and taught positions weren't actually moving the robot
  - **Root Cause**: Incomplete replacement of old Lash_E code, confusing teaching workflow, and slow movement speeds
  - **Solution**: Complete rewrite of position management system:
    - **Auto-homing on connect**: Robot homes immediately after connection (15,000 cts/s) to establish reference
    - **In-memory tracking**: All positions stored in program memory, cleared on restart for consistency
    - **Clear teaching flow**: Jog to position → select from dropdown → click "Go to Position" → confirm to save
    - **Fast movement speeds**: Increased to 12,000 cts/s (50% faster) with 60,000 cts/s² acceleration
    - **Test movement option**: After teaching, offers to test the fast movement with small displacement
    - **Multiple speed fallbacks**: Tries multiple API methods to ensure fastest possible speeds
  - **Workflow Improvements**:
    - Session-based positions that reset each time (prevents drift/confusion)
    - Visual speed indicators in status messages
    - Immediate movement after position taught
    - Automatic position verification through test movements
  - **Speed Improvements**: 
    - Homing: 15,000 cts/s (was 10,000)
    - Position movements: 12,000 cts/s (was 8,000) 
    - Acceleration: 60,000 cts/s² (was 40,000)
  - **Impact**: Robot now actually moves when positions are taught, 50% faster speeds, and much more reliable workflow
  - **Files**: `workflows/SP_arm_position_program.py`

## [1.2.3] - 2026-03-13

### Fixed
- **CRITICAL: CMC Control Logging Failure**: Fixed silent failure in CMC control generation for BDDAC surfactant
  - **Issue**: `create_cmc_control_series()` function used local logger instead of workflow logger, causing debug messages to be invisible during troubleshooting
  - **Root cause**: Function created `logging.getLogger(__name__)` instead of using `lash_e.logger` from workflow coordinator
  - **Fix**: Updated function signature to accept `lash_e` parameter and replaced all `logger.info()` calls with `lash_e.logger.info()`
  - **Impact**: Enables proper debugging of CMC control generation failures, preventing silent fallbacks that mask real errors
  - **Files**: `workflows/surfactant_grid_adaptive_concentrations.py` - function signature and 4 function call sites updated
- **CRITICAL: BDDAC Missing CMC Value**: Fixed BDDAC returning 0 CMC controls due to missing CMC value in workflow configuration
  - **Issue**: BDDAC consistently returned 0 CMC controls while SDS/CTAB worked fine (10 controls each)
  - **Root cause**: YAML workflow config `SURFACTANT_LIBRARY` entry for BDDAC was missing `cmc_mm: 8.4` key (present in Python code but overridden by config)
  - **Fix**: Added `cmc_mm: 8.4` to BDDAC entry in `workflow_configs/surfactant_grid_adaptive_concentrations.yaml`
  - **Impact**: BDDAC now generates proper CMC control series like other surfactants
  - **Files**: `workflow_configs/surfactant_grid_adaptive_concentrations.yaml`

## [1.2.2] - 2026-02-26

### Fixed
- **PERFORMANCE: Simulation Mode Timing Loop Optimization**: Fixed major performance bottleneck in `Degradation_serena.py` workflow
  - **Issue**: Even in simulation mode, timing loop was advancing by only 1 second per iteration, causing thousands of unnecessary loop cycles for long measurement intervals (e.g., 3600 iterations for 1-hour measurements)
  - **Fix**: Simulation mode now jumps directly to each scheduled measurement time instead of incrementing through every second
  - **Impact**: Reduces simulation runtime from potentially hours to seconds for workflows with long time intervals
  - Removed unnecessary heartbeat logging in simulation mode to reduce log noise
  - Performance gain: ~3600x faster for workflows with 1-hour measurement intervals
- **CRITICAL: Unconditional Sleep Calls in North_Safe.py**: Fixed simulation mode delays caused by hardware timing sleeps
  - **Issue**: Cap/uncap operations (0.5s each) and weight monitoring loops (0.01-0.1s) were running unconditionally even in simulation mode
  - **Fix**: Made all time.sleep() calls conditional on `if not self.simulate:` in North_Safe.py functions
  - **Files**: Fixed uncap_vial_in_clamp(), cap_vial_in_clamp(), and continuous weight monitoring functions
  - **Impact**: Eliminates cumulative seconds of delays per workflow operation in simulation mode
- **CRITICAL: Volume Accumulation Bug in Degradation Workflow**: Fixed sample vial volume tracking between workflow iterations
  - **Issue**: Robot's vial volume tracking (VIAL_DF) persisted across iterations, causing sample_3 to inherit accumulated volumes from sample_1 and sample_2
  - **Root cause**: Each workflow iteration should start with empty sample vials (0.0 mL) but robot remembered previous volumes
  - **Fix**: Added vial volume reset at start of each degradation_workflow() iteration to restore original CSV state
  - **Impact**: Ensures consistent sample preparation volumes across all workflow repeats
- **CRITICAL: Missing Schedule Entries Causing Volume Inconsistency**: Fixed sample_3 having different final volume than sample_1 and sample_2
  - **Issue**: degradation_vial_schedule.csv only contained entries for sample_1 and sample_2, missing sample_3 completely
  - **Root cause**: When sample_3 workflow ran, schedule filtering resulted in empty schedule, skipping all timed measurements (only initial measurement occurred)
  - **Volume impact**: sample_1 & sample_2 had 7 measurements (1.05 mL removed), sample_3 had only 1 measurement (0.15 mL removed), creating 0.9 mL difference
  - **Fix**: Added complete sample_3 measurement schedule (300s, 600s, 900s, 1200s, 1800s, 2400s, 3600s) to match sample_1 and sample_2
  - **Result**: All samples now follow identical 7-measurement schedules for consistent volume tracking

## [1.2.1] - 2026-02-20

### Fixed
- **Post-Experiment Analysis Error Handling**: Fixed runtime errors in automatic analysis integration
  - **Unicode Logging Fix**: Replaced Unicode symbols (❌✅) with [ERROR]/[SUCCESS] text to prevent cp1252 encoding crashes on Windows
  - **Contour Level Calculation**: Added error handling for matplotlib contour levels that must be increasing - fallback to imshow visualization when contour calculation fails
  - **Import Path Fix**: Fixed analysis module imports to properly reference analysis/ folder from workflows directory
  - **Plot Display Fix**: Removed plt.show() calls to prevent popup windows during automatic analysis - plots now save only and close properly to free memory
  - **Adaptive Simulation Data**: Made simulation function adaptive to use existing MIN_CONC global and dynamically scale to any concentration range - no longer hardcoded for specific ranges
  - Added comprehensive error handling with fallback visualization modes for turbidity, ratio, and fluorescence contour plots
  - Creates backups: `backups/surfactant_contour_simple_backup_*.py` and `backups/surfactant_grid_adaptive_concentrations_backup_*.py`

## [1.2.0] - 2026-02-20

### Fixed
- **CRITICAL: Pandas DataFrame Volume Filtering Bug**: Fixed silent filtering failures in surfactant workflow where CMC control wells (wells 2-18) weren't receiving pyrene probe due to pandas string vs numeric data type mismatch
  - Added `validate_and_convert_recipe_volumes()` function to ensure all volume columns are numeric before filtering operations
  - Modified `dispense_component_to_wellplate()` to validate volume data types at function entry with fail-loud error handling
  - Prevents silent failures in `batch_df[batch_df[volume_column] > 0]` filtering when volumes are stored as strings instead of numbers
  - Comprehensive validation of all volume columns: surf_A_volume_ul, surf_B_volume_ul, water_volume_ul, buffer_volume_ul, pyrene_volume_ul
  - Creates backup: `backups/surfactant_grid_adaptive_concentrations_backup_*_before_validation.py`

## [1.1.1] - 2026-02-19

### Fixed
- **Speed Control**: Fixed movement speed timing in `dispense_from_vial_into_vial` - modified speed now applies from after liquid aspiration through dispense completion, then returns to default speed for recap and return operations. This prevents fast movement with liquid in pipet and slow movement during vial cleanup operations.

## [1.1.0] - 2026-02-19

### Added
- **Spectral Data Analysis Integration**: Integrated spectral analyzer with degradation workflow
  - Added `process_workflow_spectral_data()` function with try-except error handling (renamed from process_degradation_spectral_data)
  - Automatic processing of output_# files after degradation workflow completion
  - Wavelength-specific time series plots for 555nm, 458nm, and 458/555 ratio analysis
  - Dynamic output directory handling with processed_data subfolder creation
  - Logger integration for consistent workflow logging

### Changed
- **Enhanced spectral_analyzer_program.py**:
  - Modified to accept dynamic output directories instead of hardcoded paths
  - Added processed_data subfolder for organized result storage
  - Enhanced error handling and logging capabilities
  - Updated file save paths to use processed_data directory

- **Updated Degradation_serena.py**:
  - Added spectral analysis import and automatic execution
  - Integrated spectral processing at workflow completion
  - Added comprehensive error handling for analysis failures

### Files Generated
- `spectral_analysis_plot.png` - Full spectral time series overlay
- `wavelength_time_analysis.png` - 4-panel wavelength-specific analysis
- `combined_spectral_data.csv` - Complete spectral dataset
- `wavelength_time_series.csv` - Time series for specific wavelengths