# Changelog

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