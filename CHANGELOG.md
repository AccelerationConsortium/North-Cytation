# Changelog

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