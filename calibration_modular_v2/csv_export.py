"""
Hardware-agnostic CSV export module for calibration experiments.

This module provides clean, readable CSV output that flattens nested parameter
dictionaries into individual columns, making the data easy to analyze in Excel
or other tools. Works with any parameter set automatically.

Key Features:
- Flattens nested parameter structures into individual columns
- Creates clean, readable CSV files
- Maintains backward compatibility with existing data formats
- Hardware/parameter agnostic design

Usage:
    exporter = CleanCSVExporter(output_dir)
    exporter.export_all_data(trial_results, optimal_conditions, raw_measurements)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CleanCSVExporter:
    """Hardware-agnostic clean CSV export for calibration experiments."""
    
    def __init__(self, output_dir: str):
        """Initialize exporter with output directory."""
        self.output_dir = Path(output_dir)
        
    def export_all_data(self, trial_results: List[Dict], optimal_conditions: List[Dict], 
                       raw_measurements: List[Dict]) -> None:
        """Export all calibration data in clean CSV format."""
        logger.info("Exporting clean CSV files...")
        
        try:
            # Export trial results
            if trial_results:
                self._export_trial_results(trial_results)
                
            # Export optimal conditions  
            if optimal_conditions:
                self._export_optimal_conditions(optimal_conditions)
                
            # Export raw measurements
            if raw_measurements:
                self._export_raw_measurements(raw_measurements)
                
            logger.info("✅ Clean CSV files exported successfully")
            
        except Exception as e:
            logger.error(f"Error exporting CSV files: {e}")
    
    def _export_trial_results(self, trial_results: List[Dict]) -> None:
        """Export trial results to clean CSV format."""
        rows = []
        
        for trial in trial_results:
            # Base trial information
            row = {
                'trial_id': trial.get('trial_id', ''),
                'target_volume_ml': trial.get('target_volume_ml', 0),
                'target_volume_ul': trial.get('target_volume_ml', 0) * 1000,
                'measured_volume_ml': trial.get('analysis', {}).get('mean_volume_ml', 0),
                'measured_volume_ul': trial.get('analysis', {}).get('mean_volume_ml', 0) * 1000,
                'deviation_pct': trial.get('analysis', {}).get('absolute_deviation_pct', 0),
                'cv_pct': trial.get('analysis', {}).get('cv_volume_pct', 0),
                'mean_time_s': trial.get('analysis', {}).get('mean_duration_s', 0),
                'composite_score': trial.get('composite_score', 0),
                'overall_quality': trial.get('quality', {}).get('overall_quality', ''),
                'measurements_count': trial.get('analysis', {}).get('measurement_count', 0),
            }
            
            # Add quality metrics
            quality = trial.get('quality', {})
            row.update({
                'accuracy_quality': quality.get('accuracy_quality', ''),
                'precision_quality': quality.get('precision_quality', ''),
                'time_quality': quality.get('time_quality', ''),
                'meets_accuracy_threshold': quality.get('meets_accuracy_threshold', False),
                'meets_precision_threshold': quality.get('meets_precision_threshold', False),
            })
            
            # Flatten parameters (hardware-agnostic)
            parameters = trial.get('parameters', {})
            flattened_params = self._flatten_parameters(parameters)
            row.update(flattened_params)
            
            # Add tolerances used
            tolerances = trial.get('tolerances_used', {})
            if tolerances:
                row.update({
                    'tolerance_accuracy_pct': tolerances.get('accuracy_tolerance_pct', 0),
                    'tolerance_precision_pct': tolerances.get('precision_tolerance_pct', 0),
                })
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "trial_results_clean.csv", index=False)
        logger.info(f"Exported {len(df)} trial results to trial_results_clean.csv")
    
    def _export_optimal_conditions(self, optimal_conditions: List[Dict]) -> None:
        """Export optimal conditions to clean CSV format."""
        rows = []
        
        for condition in optimal_conditions:
            # Base condition information
            row = {
                'volume_ul': condition.get('volume_ul', 0),
                'volume_ml': condition.get('volume_ul', 0) / 1000,
                'measured_volume_ul': condition.get('measured_volume_ul', 0),
                'measured_volume_ml': condition.get('measured_volume_ul', 0) / 1000,
                'deviation_pct': condition.get('deviation_pct', 0),
                'cv_pct': condition.get('cv_pct', 0),
                'time_s': condition.get('time_s', 0),
                'trials_used': condition.get('trials_used', 0),
                'status': condition.get('status', ''),
                'measurements_count': condition.get('measurements_count', 0),
            }
            
            # Flatten optimal parameters
            parameters = condition.get('parameters', {})
            flattened_params = self._flatten_parameters(parameters)
            row.update(flattened_params)
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "optimal_conditions_clean.csv", index=False)
        logger.info(f"Exported {len(df)} optimal conditions to optimal_conditions_clean.csv")
    
    def _export_raw_measurements(self, raw_measurements: List[Dict]) -> None:
        """Export raw measurements to clean CSV format."""
        rows = []
        
        for measurement in raw_measurements:
            # Base measurement information
            row = {
                'measurement_id': measurement.get('measurement_id', ''),
                'trial_id': measurement.get('trial_id', ''),
                'target_volume_ml': measurement.get('target_volume_ml', 0),
                'target_volume_ul': measurement.get('target_volume_ml', 0) * 1000,
                'actual_volume_ml': measurement.get('actual_volume_ml', 0),
                'actual_volume_ul': measurement.get('actual_volume_ml', 0) * 1000,
                'duration_s': measurement.get('duration_s', 0),
                'timestamp': measurement.get('timestamp', ''),
                'replicate_id': measurement.get('replicate_id', ''),
            }
            
            # Calculate deviation for this individual measurement
            if row['target_volume_ml'] > 0:
                row['individual_deviation_pct'] = abs(row['actual_volume_ml'] - row['target_volume_ml']) / row['target_volume_ml'] * 100
            else:
                row['individual_deviation_pct'] = 0
            
            # Flatten parameters
            parameters = measurement.get('parameters', {})
            flattened_params = self._flatten_parameters(parameters)
            row.update(flattened_params)
            
            # Add metadata if available
            metadata = measurement.get('metadata', {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        row[f'metadata_{key}'] = value
            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "raw_measurements_clean.csv", index=False)
        logger.info(f"Exported {len(df)} raw measurements to raw_measurements_clean.csv")
    
    def _flatten_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested parameter dictionaries into individual columns.
        
        This method handles any parameter structure automatically, making it
        hardware-agnostic.
        """
        flattened = {}
        
        if not isinstance(parameters, dict):
            return flattened
        
        def _flatten_recursive(obj: Any, prefix: str = '') -> None:
            """Recursively flatten nested dictionaries."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten_recursive(value, new_key)
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        # Convert numpy types to native Python types
                        if hasattr(value, 'item'):
                            value = value.item()
                        flattened[new_key] = value
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        # Handle simple lists/tuples of scalars
                        if all(isinstance(x, (int, float)) for x in value):
                            flattened[f"{new_key}_mean"] = np.mean(value)
                            if len(value) > 1:
                                flattened[f"{new_key}_std"] = np.std(value)
            elif isinstance(obj, (int, float, str, bool, type(None))):
                if hasattr(obj, 'item'):
                    obj = obj.item()
                key = prefix if prefix else 'value'
                flattened[key] = obj
        
        _flatten_recursive(parameters)
        return flattened
    
    def create_experiment_summary(self, trial_results: List[Dict], 
                                optimal_conditions: List[Dict]) -> None:
        """Create a high-level experiment summary CSV."""
        if not optimal_conditions:
            return
            
        # Calculate summary statistics
        summary_data = {
            'experiment_timestamp': pd.Timestamp.now().isoformat(),
            'total_trials': len(trial_results),
            'volumes_calibrated': len(optimal_conditions),
            'mean_deviation_pct': np.mean([c.get('deviation_pct', 0) for c in optimal_conditions]),
            'best_deviation_pct': np.min([c.get('deviation_pct', 100) for c in optimal_conditions]),
            'worst_deviation_pct': np.max([c.get('deviation_pct', 0) for c in optimal_conditions]),
            'mean_cv_pct': np.mean([c.get('cv_pct', 0) for c in optimal_conditions]),
            'mean_time_s': np.mean([c.get('time_s', 0) for c in optimal_conditions]),
            'total_measurements': sum([c.get('measurements_count', 0) for c in optimal_conditions]),
            'successful_volumes': len([c for c in optimal_conditions if c.get('deviation_pct', 100) <= 10]),
            'efficiency_trials_per_volume': len(trial_results) / len(optimal_conditions) if optimal_conditions else 0,
        }
        
        # Add volume-specific data
        for i, condition in enumerate(optimal_conditions):
            vol_ul = condition.get('volume_ul', 0)
            summary_data[f'volume_{i+1}_ul'] = vol_ul
            summary_data[f'volume_{i+1}_deviation_pct'] = condition.get('deviation_pct', 0)
            summary_data[f'volume_{i+1}_trials'] = condition.get('trials_used', 0)
        
        # Save summary
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(self.output_dir / "experiment_summary_clean.csv", index=False)
        logger.info("Exported experiment summary to experiment_summary_clean.csv")

def export_clean_csvs(trial_results: List[Dict], optimal_conditions: List[Dict], 
                     raw_measurements: List[Dict], output_dir: str) -> None:
    """
    Convenience function to export all clean CSV files.
    
    Args:
        trial_results: List of trial result dictionaries
        optimal_conditions: List of optimal condition dictionaries
        raw_measurements: List of raw measurement dictionaries
        output_dir: Directory to save CSV files
    """
    exporter = CleanCSVExporter(output_dir)
    exporter.export_all_data(trial_results, optimal_conditions, raw_measurements)
    exporter.create_experiment_summary(trial_results, optimal_conditions)