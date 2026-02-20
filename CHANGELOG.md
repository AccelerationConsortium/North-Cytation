# Changelog

## [1.1.1] - 2026-02-19

### Fixed
- **Speed Control**: Fixed movement speed timing in `dispense_from_vial_into_vial` - modified speed now applies from after liquid aspiration through dispense completion, then returns to default speed for recap and return operations. This prevents fast movement with liquid in pipet and slow movement during vial cleanup operations.

## [1.1.0] - 2026-02-19

### Added
- **Spectral Data Analysis Integration**: Integrated spectral analyzer with degradation workflow
  - Added `process_degradation_spectral_data()` function with try-except error handling
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