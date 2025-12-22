#!/usr/bin/env python3
"""
Universal Validation System Runner
=================================

Validates calibrated parameters by testing them on specified volumes.
Works with any hardware configuration - no hardware-specific assumptions.

Usage:
    python run_validation.py

Configuration:
    Edit experiment_config.yaml validation section to configure:
    - Validation volumes
    - Number of replicates per volume
    - Calibration results file to use
    - Success criteria

Output:
    Results automatically saved to validation/ directory with:
    - Target vs measured volume plots
    - Raw measurement data (CSV)
    - Validation summary with pass/fail (CSV)
    - Statistical analysis report (JSON)
"""

import logging
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration_modular_v2 import ExperimentConfig
from calibration_modular_v2.protocol_loader import create_protocol
from calibration_modular_v2.pipetting_wizard_v2 import PipettingWizardV2


def setup_logging():
    """Configure logging for the validation run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('validation_experiment.log')
        ]
    )


class ValidationRunner:
    """Hardware-agnostic validation system using the modular protocol framework."""
    
    def __init__(self, config: ExperimentConfig, config_file_path: Path = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.protocol = None
        self.protocol_state = None
        self.pipetting_wizard = PipettingWizardV2()
        
        # Set up output directory relative to config file location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = config._config['validation']['output_directory']
        
        if config_file_path:
            # Make output directory relative to config file location
            config_dir = config_file_path.parent
            base_dir = config_dir / output_dir_name
        else:
            # Fallback to relative path from current directory
            base_dir = Path(output_dir_name)
            
        self.output_dir = base_dir / f"validation_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Validation results will be saved to: {self.output_dir}")
        
    def run_validation(self) -> Dict[str, Any]:
        """Execute the complete validation workflow."""
        try:
            self.logger.info("Starting validation workflow")
            
            # Initialize protocol
            self._initialize_protocol()
            
            # Load calibration results
            optimal_conditions = self._load_calibration_results()
            
            # Run validation measurements
            validation_results = self._run_validation_measurements(optimal_conditions)
            
            # Analyze results
            analysis = self._analyze_results(validation_results)
            
            # Generate outputs
            self._generate_outputs(validation_results, analysis)
            
            # Clean up protocol
            self._finalize_protocol()
            
            self.logger.info("Validation workflow completed successfully")
            return {
                'validation_results': validation_results,
                'analysis': analysis,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Validation workflow failed: {e}")
            if self.protocol and self.protocol_state:
                try:
                    self.protocol.wrapup(self.protocol_state)
                except Exception as cleanup_error:
                    self.logger.warning(f"Protocol cleanup failed: {cleanup_error}")
            raise
    
    def _initialize_protocol(self):
        """Initialize the measurement protocol."""
        self.logger.info("Initializing protocol...")
        
        # Create protocol instance using same system as calibration
        self.protocol = create_protocol(self.config, simulate=self.config.is_simulation())
        
        # Initialize protocol (wrapper handles config internally)
        self.protocol_state = self.protocol.initialize()
        self.logger.info("Protocol initialized successfully")
    
    def _load_calibration_results(self) -> pd.DataFrame:
        """Load optimal conditions from calibration results."""
        validation_config = self.config._config['validation']
        optimal_conditions_file = Path(validation_config['optimal_conditions_file'])
        
        # Make path absolute relative to the config file directory if it's relative
        if not optimal_conditions_file.is_absolute():
            config_dir = Path(__file__).parent  # Directory containing this script
            optimal_conditions_file = config_dir / optimal_conditions_file
        
        if not optimal_conditions_file.exists():
            raise FileNotFoundError(f"Calibration results file not found: {optimal_conditions_file}")
        
        self.logger.info(f"Loading calibration results from: {optimal_conditions_file}")
        optimal_conditions = pd.read_csv(optimal_conditions_file)
        
        self.logger.info(f"Loaded {len(optimal_conditions)} calibrated volume conditions")
        return optimal_conditions
    
    def _get_parameters_for_volume(self, target_volume_ml: float, optimal_conditions: pd.DataFrame) -> Dict[str, float]:
        """Get optimal parameters for a target volume using pipetting wizard with overvolume compensation."""
        try:
            # Apply overvolume compensation to the optimal conditions before interpolation
            # This mimics what get_pipetting_parameters() does but works with in-memory data
            compensated_conditions = optimal_conditions.copy()
            
            # Check if we have the V2 format columns needed for compensation
            required_cols = ['volume_target_ul', 'volume_measured_ml', 'calibration_overaspirate_vol']
            if all(col in compensated_conditions.columns for col in required_cols):
                self.logger.debug(f"Applying overvolume compensation for {target_volume_ml*1000:.1f}uL parameters")
                compensated_conditions = self.pipetting_wizard.apply_overvolume_compensation(compensated_conditions)
                compensation_note = " (with overvolume compensation)"
            else:
                self.logger.warning(f"V2 format columns not available for overvolume compensation: missing {[c for c in required_cols if c not in compensated_conditions.columns]}")
                compensation_note = " (no compensation - missing V2 columns)"
            
            # Use pipetting wizard to interpolate from the compensated data
            parameters = self.pipetting_wizard.interpolate_parameters(compensated_conditions, target_volume_ml)
            
            if parameters:
                self.logger.debug(f"Parameters for {target_volume_ml*1000:.1f}uL{compensation_note}: {parameters}")
                return parameters
            else:
                self.logger.warning(f"No parameters found for volume {target_volume_ml*1000:.1f}uL, using minimal defaults")
                return self._get_minimal_default_parameters(optimal_conditions)
                
        except Exception as e:
            self.logger.warning(f"Error getting parameters for {target_volume_ml*1000:.1f}uL: {e}")
            return self._get_minimal_default_parameters(optimal_conditions)
    
    def _get_minimal_default_parameters(self, optimal_conditions: pd.DataFrame) -> Dict[str, float]:
        """Get zero defaults for all parameters when interpolation fails."""
        if len(optimal_conditions) == 0:
            self.logger.warning("No calibration data available - cannot run validation")
            return {}
        
        # Use pipetting wizard to discover available parameter names
        try:
            parameter_names = self.pipetting_wizard.discover_hardware_parameters(optimal_conditions)
            
            # Set ALL parameters to zero - let the protocol handle what's safe
            defaults = {param_name: 0.0 for param_name in parameter_names}
            
            self.logger.warning(f"Parameter interpolation failed - using zero defaults for all {len(defaults)} parameters")
            self.logger.warning(f"Zero defaults set for: {list(defaults.keys())}")
            self.logger.warning("This may result in failed measurements - check calibration data coverage")
            
            return defaults
            
        except Exception as e:
            self.logger.error(f"Could not discover parameter names: {e}")
            return {}
    
    def _run_validation_measurements(self, optimal_conditions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Execute validation measurements for all specified volumes."""
        validation_config = self.config._config['validation']
        volumes_ml = validation_config['volumes_ml']
        replicates = validation_config['replicates_per_volume']
        
        self.logger.info(f"Running validation on {len(volumes_ml)} volumes with {replicates} replicates each")
        
        all_results = []
        
        for i, volume_ml in enumerate(volumes_ml):
            self.logger.info(f"Testing volume {i+1}/{len(volumes_ml)}: {volume_ml*1000:.1f}uL")
            
            # Get optimal parameters for this volume
            parameters = self._get_parameters_for_volume(volume_ml, optimal_conditions)
            
            # Run replicates for this volume
            volume_results = []
            for rep in range(replicates):
                self.logger.info(f"  Replicate {rep+1}/{replicates}...")
                
                try:
                    # Perform measurement using protocol
                    measurement_results = self.protocol.measure(
                        self.protocol_state,
                        volume_ml,
                        parameters,
                        replicates=1
                    )
                    
                    if measurement_results and len(measurement_results) > 0:
                        result = measurement_results[0]
                        
                        # Store measurement with metadata
                        measurement = {
                            'volume_target_ml': volume_ml,
                            'volume_target_ul': volume_ml * 1000,
                            'volume_measured_ml': result['volume'],
                            'volume_measured_ul': result['volume'] * 1000,
                            'duration_s': result['elapsed_s'],
                            'replicate': rep + 1,
                            'timestamp': datetime.now().isoformat(),
                            'parameters_used': parameters,
                            'protocol_result': result  # Include full protocol result
                        }
                        
                        volume_results.append(measurement)
                        
                        self.logger.info(f"    Measured: {result['volume']*1000:.2f}uL in {result['elapsed_s']:.2f}s")
                    
                    else:
                        raise RuntimeError("Protocol returned no measurement results")
                        
                except Exception as e:
                    self.logger.error(f"  Measurement failed: {e}")
                    # Record failed measurement
                    measurement = {
                        'volume_target_ml': volume_ml,
                        'volume_target_ul': volume_ml * 1000,
                        'volume_measured_ml': np.nan,
                        'volume_measured_ul': np.nan,
                        'duration_s': np.nan,
                        'replicate': rep + 1,
                        'timestamp': datetime.now().isoformat(),
                        'parameters_used': parameters,
                        'error': str(e)
                    }
                    volume_results.append(measurement)
            
            all_results.extend(volume_results)
        
        return all_results
    
    def _get_volume_tolerance(self, volume_ml: float) -> float:
        """Get volume-dependent tolerance from config tolerances.volume_ranges."""
        volume_ul = volume_ml * 1000
        
        # Check if we should use volume-dependent tolerances
        validation_config = self.config._config['validation']
        success_criteria = validation_config['success_criteria']
        
        if not success_criteria.get('use_volume_tolerances', False):
            # Fall back to fixed tolerance if not using volume tolerances
            return success_criteria.get('max_cv_pct', 10.0)
        
        # Get volume ranges from config
        tolerances_config = self.config._config.get('tolerances', {})
        volume_ranges = tolerances_config.get('volume_ranges', [])
        
        # Find matching volume range
        for range_config in volume_ranges:
            vol_min = range_config.get('volume_min_ul', 0)
            vol_max = range_config.get('volume_max_ul', float('inf'))
            
            if vol_min <= volume_ul <= vol_max:
                tolerance = range_config.get('tolerance_pct', 10.0)
                self.logger.debug(f"Volume {volume_ul:.1f}uL -> {tolerance}% tolerance for both accuracy and precision")
                return tolerance
        
        # Default fallback if no range matches
        self.logger.warning(f"No tolerance range found for volume {volume_ul:.1f}uL, using 10% default")
        return 10.0
    
    def _analyze_results(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze validation results and calculate statistics."""
        self.logger.info("Analyzing validation results...")
        
        validation_config = self.config._config['validation']
        success_criteria = validation_config['success_criteria']
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(validation_results)
        
        # Filter out failed measurements for statistics
        valid_df = df.dropna(subset=['volume_measured_ml'])
        
        if len(valid_df) == 0:
            self.logger.warning("No valid measurements for analysis")
            return {'error': 'No valid measurements'}
        
        # Group by target volume and calculate statistics
        volume_stats = []
        for volume_ml in sorted(valid_df['volume_target_ml'].unique()):
            volume_data = valid_df[valid_df['volume_target_ml'] == volume_ml]
            
            if len(volume_data) == 0:
                continue
                
            target_ml = volume_ml
            measured_ml = volume_data['volume_measured_ml'].values
            
            # Calculate statistics
            mean_measured = np.mean(measured_ml)
            std_measured = np.std(measured_ml)
            cv_pct = (std_measured / mean_measured * 100) if mean_measured > 0 else 0
            
            # Accuracy metrics
            deviation_pct = (mean_measured - target_ml) / target_ml * 100
            absolute_deviation_pct = abs(deviation_pct)
            
            # Individual errors
            individual_errors_pct = [abs(m - target_ml) / target_ml * 100 for m in measured_ml]
            mean_absolute_error_pct = np.mean(individual_errors_pct)
            
            # Get volume-dependent tolerance for both accuracy and precision evaluation
            volume_tolerance_pct = self._get_volume_tolerance(target_ml)
            
            # Success evaluation - use same tolerance for both accuracy and precision
            accuracy_pass = absolute_deviation_pct <= volume_tolerance_pct
            precision_pass = cv_pct <= volume_tolerance_pct  # Use same tolerance instead of separate max_cv_pct
            overall_pass = accuracy_pass and precision_pass
            
            volume_stat = {
                'volume_target_ml': target_ml,
                'volume_target_ul': target_ml * 1000,
                'volume_measured_mean_ml': mean_measured,
                'volume_measured_mean_ul': mean_measured * 1000,
                'volume_measured_std_ml': std_measured,
                'volume_measured_std_ul': std_measured * 1000,
                'cv_pct': cv_pct,
                'deviation_pct': deviation_pct,
                'absolute_deviation_pct': absolute_deviation_pct,
                'mean_absolute_error_pct': mean_absolute_error_pct,
                'tolerance_pct': volume_tolerance_pct,  # Same tolerance used for both accuracy and precision
                'n_measurements': len(volume_data),
                'accuracy_pass': accuracy_pass,
                'precision_pass': precision_pass,
                'overall_pass': overall_pass
            }
            
            volume_stats.append(volume_stat)
        
        # Overall validation summary
        total_volumes = len(volume_stats)
        passed_volumes = sum(1 for v in volume_stats if v['overall_pass'])
        overall_pass_rate = passed_volumes / total_volumes if total_volumes > 0 else 0
        
        analysis = {
            'volume_statistics': volume_stats,
            'summary': {
                'total_volumes_tested': total_volumes,
                'volumes_passed': passed_volumes,
                'volumes_failed': total_volumes - passed_volumes,
                'overall_pass_rate': overall_pass_rate,
                'success_criteria': success_criteria
            },
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Validation analysis complete: {passed_volumes}/{total_volumes} volumes passed")
        return analysis
    
    def _generate_outputs(self, validation_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Generate validation output files and plots."""
        self.logger.info("Generating validation outputs...")
        
        # Save raw data
        df = pd.DataFrame(validation_results)
        df.to_csv(self.output_dir / "raw_validation_data.csv", index=False)
        
        # Save volume statistics
        if 'volume_statistics' in analysis:
            stats_df = pd.DataFrame(analysis['volume_statistics'])
            stats_df.to_csv(self.output_dir / "validation_summary.csv", index=False)
        
        # Save analysis report (convert numpy types to native Python types for JSON)
        analysis_for_json = self._convert_numpy_types(analysis)
        
        # Add simulation flag and config info to analysis
        analysis_for_json['simulation_mode'] = self.config.is_simulation()
        analysis_for_json['config_info'] = {
            'experiment_name': self.config.get_experiment_name(),
            'target_volumes': self.config.get_target_volumes_ml(),
            'simulation_mode': self.config.is_simulation()
        }
        
        with open(self.output_dir / "validation_analysis.json", 'w') as f:
            json.dump(analysis_for_json, f, indent=2)
        
        # Save the exact config file that was used for this validation
        try:
            import shutil
            original_config_path = Path(__file__).parent / "experiment_config.yaml"
            config_backup_path = self.output_dir / "experiment_config_used.yaml"
            if original_config_path.exists():
                shutil.copy2(original_config_path, config_backup_path)
                self.logger.info(f"Saved config file used for this validation to {config_backup_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save config file: {e}")
        
        # Generate plots if requested
        validation_config = self.config._config['validation']
        if validation_config.get('generate_plots', True):
            self._generate_validation_plots(validation_results, analysis)
        
        # Generate text report
        self._generate_text_report(analysis)
        
        self.logger.info(f"Validation outputs saved to: {self.output_dir}")
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_validation_plots(self, validation_results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Generate target vs measured volume plots."""
        if 'volume_statistics' not in analysis:
            return
            
        stats_df = pd.DataFrame(analysis['volume_statistics'])
        
        if len(stats_df) == 0:
            return
        
        # Create target vs measured plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Target vs Measured Volume
        target_ul = stats_df['volume_target_ul']
        measured_ul = stats_df['volume_measured_mean_ul']
        error_ul = stats_df['volume_measured_std_ul']
        
        ax1.errorbar(target_ul, measured_ul, yerr=error_ul, 
                    fmt='o', capsize=5, capthick=2, label='Measured')
        
        # Perfect accuracy line
        min_vol = min(target_ul.min(), measured_ul.min())
        max_vol = max(target_ul.max(), measured_ul.max())
        ax1.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect Accuracy')
        
        ax1.set_xlabel('Target Volume (uL)')
        ax1.set_ylabel('Measured Volume (uL)')
        ax1.set_title('Validation: Target vs Measured Volume')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy and Precision by Volume
        ax2_twin = ax2.twinx()
        
        width = 0.35
        x_pos = np.arange(len(stats_df))
        
        bars1 = ax2.bar(x_pos - width/2, stats_df['absolute_deviation_pct'], width, 
                       label='Accuracy (% deviation)', alpha=0.7, color='blue')
        bars2 = ax2_twin.bar(x_pos + width/2, stats_df['cv_pct'], width,
                            label='Precision (% CV)', alpha=0.7, color='orange')
        
        ax2.set_xlabel('Target Volume (uL)')
        ax2.set_ylabel('Accuracy (% Deviation)', color='blue')
        ax2_twin.set_ylabel('Precision (% CV)', color='orange')
        ax2.set_title('Accuracy and Precision by Volume')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{vol:.0f}" for vol in target_ul])
        
        # Add success criteria lines
        success_criteria = analysis['summary']['success_criteria']
        
        # Add volume-dependent accuracy thresholds (stepped line)
        if 'tolerance_pct' in stats_df.columns:
            # Use individual volume tolerances for both accuracy and precision
            for i, tolerance in enumerate(stats_df['tolerance_pct']):
                # Accuracy threshold (blue)
                ax2.plot([i-0.4, i+0.4], [tolerance, tolerance], 
                        color='blue', linestyle='--', alpha=0.7, linewidth=2)
                # Precision threshold (orange) - same value
                ax2_twin.plot([i-0.4, i+0.4], [tolerance, tolerance], 
                            color='orange', linestyle='--', alpha=0.7, linewidth=2)
        else:
            # Fallback to fixed thresholds if available
            if 'max_deviation_pct' in success_criteria:
                ax2.axhline(y=success_criteria['max_deviation_pct'], color='blue', linestyle='--', alpha=0.5)
            if 'max_cv_pct' in success_criteria:
                ax2_twin.axhline(y=success_criteria['max_cv_pct'], color='orange', linestyle='--', alpha=0.5)
        
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "validation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Validation plots generated")
    
    def _generate_text_report(self, analysis: Dict[str, Any]):
        """Generate human-readable validation report."""
        lines = [
            "Validation Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment: {self.config.get_experiment_name()}",
            f"Liquid: {self.config.get_liquid_name()}",
            f"Protocol: {'Simulation' if self.config.is_simulation() else 'Hardware'}",
            ""
        ]
        
        if 'summary' in analysis:
            summary = analysis['summary']
            lines.extend([
                "OVERALL RESULTS:",
                f"  Volumes tested: {summary['total_volumes_tested']}",
                f"  Volumes passed: {summary['volumes_passed']}",
                f"  Volumes failed: {summary['volumes_failed']}",
                f"  Pass rate: {summary['overall_pass_rate']:.1%}",
                "",
                "SUCCESS CRITERIA:",
                f"  Volume-dependent tolerances: Used for both accuracy and precision",
                ""
            ])
        
        if 'volume_statistics' in analysis:
            lines.append("DETAILED RESULTS:")
            for stat in analysis['volume_statistics']:
                status = "PASS" if stat['overall_pass'] else "FAIL"
                tolerance = stat.get('tolerance_pct', 'N/A')
                lines.extend([
                    f"  {stat['volume_target_ul']:.0f}uL: {status} (tolerance: {tolerance}%)",
                    f"    Measured: {stat['volume_measured_mean_ul']:.1f} +/- {stat['volume_measured_std_ul']:.1f}uL",
                    f"    Accuracy: {stat['absolute_deviation_pct']:.1f}% deviation",
                    f"    Precision: {stat['cv_pct']:.1f}% CV",
                    f"    Measurements: {stat['n_measurements']}",
                    ""
                ])
        
        report_text = "\\n".join(lines)
        
        with open(self.output_dir / "validation_report.txt", 'w') as f:
            f.write(report_text)
        
        # Also print to console
        self.logger.info("\\n" + report_text)
    
    def _finalize_protocol(self):
        """Clean up protocol resources."""
        if self.protocol and self.protocol_state:
            try:
                self.protocol.wrapup(self.protocol_state)
                self.logger.info("Protocol cleanup completed")
            except Exception as e:
                self.logger.warning(f"Protocol cleanup failed: {e}")


def main():
    """Run validation based on configuration."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Universal Validation System")
    
    try:
        # Load configuration
        config_path = Path(__file__).parent / "experiment_config.yaml"
        logger.info(f"Loading configuration from {config_path}")
        
        config = ExperimentConfig.from_yaml(str(config_path))
        
        # Validate configuration has validation section
        if 'validation' not in config._config:
            raise ValueError("Configuration file missing 'validation' section")
        
        validation_config = config._config['validation']
        logger.info(f"Validation volumes: {validation_config['volumes_ml']} mL")
        logger.info(f"Replicates per volume: {validation_config['replicates_per_volume']}")
        logger.info(f"Execution mode: {'Simulation' if config.is_simulation() else 'Hardware'}")
        
        # Create and run validation
        validator = ValidationRunner(config, config_path)
        results = validator.run_validation()
        
        # Display summary
        analysis = results['analysis']
        if 'summary' in analysis:
            summary = analysis['summary']
            logger.info("=" * 60)
            logger.info("VALIDATION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Pass rate: {summary['overall_pass_rate']:.1%} ({summary['volumes_passed']}/{summary['total_volumes_tested']} volumes)")
            logger.info(f"Results saved to: {results['output_dir']}")
            
            # Just report results - no arbitrary pass/fail threshold
            if summary['overall_pass_rate'] == 1.0:
                logger.info("PERFECT: All volumes meet validation criteria!")
            elif summary['overall_pass_rate'] >= 0.8:
                logger.info("GOOD: Most volumes meet validation criteria")
            elif summary['overall_pass_rate'] >= 0.5:
                logger.info("WARNING: Mixed validation - some volumes need improvement")
            else:
                logger.info("POOR: Validation failed - calibration needs refinement")
            
            # Return success if any volumes passed (for script exit code)
            return summary['volumes_passed'] > 0
        else:
            logger.error("Validation completed but analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error("Check configuration and try again")
        raise


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)