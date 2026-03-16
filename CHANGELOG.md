# Changelog

## [1.2.3] - 2026-03-15

### Added
- **NEW WORKFLOW: Ilya Workflow V2 (Modern API)**: Created updated version of ilya_workflow.py with modern robotics API patterns
  - **File**: `workflows/ilya_workflow_v2.py` - Complete rewrite using current API standards
  - **API Updates**: Replaced index-based vial references with string names ('water', 'ethanol', etc.)
  - **Pipetting Pattern**: Separated aspirate_from_vial() and dispense_into_wellplate() calls instead of composite functions
  - **Liquid Optimization**: Added liquid= parameter for automatic pipetting parameter optimization by liquid type
  - **Validation Framework**: Embedded pipetting validation at workflow start with test volumes and error handling
  - **Configuration**: Added comprehensive configuration section with simulation mode, validation toggles, and volume limits
  - **Error Handling**: Robust validation for input files, vial availability, and recipe data consistency
  - **Logging**: Enhanced progress reporting and user confirmation patterns from modern workflows
- **VIAL STATUS: Ilya Input Vials**: Created proper CSV vial status file for Ilya workflow
  - **File**: `status/ilya_input_vials.csv` - Replaces missing .txt file with proper CSV format
  - **Vials**: Configured ethanol, ethanol_dye, water, water_dye, glycerol, glycerol_dye with proper volumes and locations
  - **Format**: Standard vial status format matching other workflow vial files

### Changed
- **WORKFLOW ARCHITECTURE: Modern Validation Patterns**: Updated workflow structure to match current best practices
  - **Validation**: Pre-flight checks for files, vials, and data consistency before execution
  - **Safety**: Minimum/maximum volume checks and pipettable range validation
  - **User Interface**: Clear step-by-step progress reporting with validation confirmations
  - **Simulation**: Full simulation mode support with automatic progression and validation skip options
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