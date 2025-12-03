"""
Hardware-agnostic visualization module for calibration experiments.

This module provides streamlined plotting functionality focused on the most 
essential and informative visualizations for calibration analysis.

Final plot selection:
- Time vs Deviation scatter plot (color-coded by volume) - Shows optimization efficiency  
- Measured Volume Over Time (with target lines) - Shows pipetting accuracy over measurements
- Improvement Summary - Shows overall optimization performance metrics

Usage:
    visualizer = CalibrationVisualizer(output_dir)
    visualizer.generate_all_plots(trial_results, optimal_conditions, raw_measurements)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

class CalibrationVisualizer:
    """Hardware-agnostic visualization matching original calibration_sdl_simplified plots."""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.ioff()
        
    def generate_all_plots(self, trial_results: List[Dict], optimal_conditions: List[Dict], 
                          raw_measurements: List[Dict]) -> None:
        """Generate all visualization plots matching original system."""
        logger.info("Generating calibration visualization plots...")
        
        try:
            # Convert to DataFrames for plotting
            trials_df = self._prepare_trials_dataframe(trial_results)
            raw_df = self._prepare_raw_measurements_dataframe(raw_measurements)
            
            if trials_df.empty:
                logger.warning("No trial data available for plotting")
                return
                
            # Generate only the preferred plots
            self._plot_time_vs_deviation(trials_df)
            self._plot_measured_volume_over_time(raw_df)
            self._plot_improvement_summary(trials_df)
            
            logger.info(f"[SUCCESS] All plots saved to: {self.plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _prepare_trials_dataframe(self, trial_results: List[Dict]) -> pd.DataFrame:
        """Convert trial results to DataFrame matching original format."""
        if not trial_results:
            return pd.DataFrame()
            
        rows = []
        trial_index = 0
        
        for trial in trial_results:
            trial_index += 1
            
            # Extract basic information
            target_volume_ml = trial.get('target_volume_ml', 0)
            analysis = trial.get('analysis', {})
            
            row = {
                'trial_index': trial_index,
                'volume': target_volume_ml,
                'deviation': analysis.get('absolute_deviation_pct', 0),
                'time': analysis.get('mean_duration_s', 0),
                'variability': analysis.get('cv_volume_pct', 0),
                'composite_score': trial.get('composite_score', 0),
                'quality': trial.get('quality', {}).get('overall_quality', 'unknown'),
            }
            
            # Flatten all parameters (hardware-agnostic)
            parameters = trial.get('parameters', {})
            self._flatten_parameters_to_row(parameters, row)
            
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def _prepare_raw_measurements_dataframe(self, raw_measurements: List[Dict]) -> pd.DataFrame:
        """Convert raw measurements to DataFrame matching original format."""
        if not raw_measurements:
            return pd.DataFrame()
            
        rows = []
        for measurement in raw_measurements:
            row = {
                'volume': measurement.get('target_volume_ml', 0),
                'calculated_volume': measurement.get('measured_volume_ml', 0),
                'time': measurement.get('duration_s', 0),
                'measurement_id': measurement.get('measurement_id', ''),
                'trial_id': measurement.get('trial_id', ''),
            }
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def _plot_time_vs_deviation(self, trials_df: pd.DataFrame) -> None:
        """
        Recreate the original Time vs Deviation scatter plot.
        This is the classic plot showing optimization efficiency.
        """
        if trials_df.empty or 'time' not in trials_df.columns or 'deviation' not in trials_df.columns:
            logger.warning("Cannot create time vs deviation plot - missing data")
            return
            
        plt.figure(figsize=(10, 6))
        
        volumes = sorted(trials_df['volume'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        
        logger.info(f"Creating Time vs Deviation scatter for {len(volumes)} volumes: {[v*1000 for v in volumes]} uL")
        
        for i, vol in enumerate(volumes):
            df_sub = trials_df[trials_df['volume'] == vol].copy()
            label = f"{vol*1000:.0f}uL"
            
            plt.scatter(
                df_sub["time"],
                df_sub["deviation"],
                color=colors[i],
                label=label,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                s=60
            )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Deviation (%)")
        plt.title("Time vs Deviation (%) - Optimization Efficiency")
        plt.legend(title="Volume")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.plots_dir / "time_vs_deviation_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Time vs deviation scatter plot saved")
    
    def _plot_measured_volume_over_time(self, raw_df: pd.DataFrame) -> None:
        """
        Recreate the original Measured Volume Over Time plot.
        This shows measurement accuracy with target lines - often the most informative plot!
        """
        if raw_df.empty:
            logger.warning("No raw measurement data for volume over time plot")
            return

        raw_df = raw_df.copy()
        
        # Handle different volume column names
        if 'calculated_volume' in raw_df.columns:
            raw_df['measured_volume_ul'] = raw_df['calculated_volume'] * 1000
            source = 'calculated_volume'
        else:
            logger.warning("No calculated_volume column found for measured volume plot")
            return

        plt.figure(figsize=(12, 8))

        if 'volume' in raw_df.columns:
            volumes = sorted(raw_df['volume'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
            
            for i, vol in enumerate(volumes):
                vol_data = raw_df[raw_df['volume'] == vol].reset_index(drop=True)
                target_ul = vol * 1000
                
                # Scatter plot of measured volumes
                plt.scatter(
                    range(len(vol_data)),
                    vol_data['measured_volume_ul'],
                    color=colors[i],
                    alpha=0.7,
                    label=f'{target_ul:.0f}uL target',
                    s=50
                )
                
                # Target line
                plt.axhline(
                    y=target_ul,
                    color=colors[i],
                    linestyle='--',
                    alpha=0.8,
                    linewidth=2
                )
        else:
            plt.scatter(range(len(raw_df)), raw_df['measured_volume_ul'], alpha=0.7, s=50)

        plt.xlabel('Measurement Number')
        plt.ylabel('Measured Volume (uL)')
        plt.title(f'Measured Volume Over Time (source: {source})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.plots_dir / 'measured_volume_over_time.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Measured volume over time plot saved")
    
    def _plot_improvement_summary(self, trials_df: pd.DataFrame) -> None:
        """Recreate the original improvement summary."""
        if trials_df.empty:
            return
            
        metrics = ['deviation', 'time', 'variability']
        summary = []
        
        for vol in sorted(trials_df['volume'].unique()):
            sub = trials_df[trials_df['volume'] == vol].sort_values('trial_index')
            n = len(sub)
            if n < 4:
                continue
                
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                    
                first_half = sub.iloc[:n//2][metric].mean()
                second_half = sub.iloc[n//2:][metric].mean()
                
                if first_half != 0:
                    improvement = (first_half - second_half) / first_half * 100
                else:
                    improvement = 0
                
                summary.append({
                    'Volume': f'{vol*1000:.0f}uL',
                    'Metric': metric.capitalize(),
                    'Improvement (%)': improvement
                })
        
        if not summary:
            return
            
        summary_df = pd.DataFrame(summary)
        
        # Pivot for plotting
        pivot_df = summary_df.pivot(index='Volume', columns='Metric', values='Improvement (%)')
        
        plt.figure(figsize=(10, 6))
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('Optimization Improvement Summary')
        plt.xlabel('Volume')
        plt.ylabel('Improvement (%)')
        plt.legend(title='Metric')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = self.plots_dir / 'improvement_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Improvement summary saved")
    
    def _flatten_parameters_to_row(self, parameters: Dict[str, Any], row: Dict[str, Any]) -> None:
        """Flatten nested parameters into row dictionary."""
        def _flatten_recursive(obj: Any, prefix: str = '') -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten_recursive(value, new_key)
                    elif isinstance(value, (int, float)):
                        if hasattr(value, 'item'):
                            value = value.item()
                        row[new_key] = value
            elif isinstance(obj, (int, float)):
                if hasattr(obj, 'item'):
                    obj = obj.item()
                key = prefix if prefix else 'value'
                row[key] = obj
        
        _flatten_recursive(parameters)

def generate_calibration_plots(trial_results: List[Dict], optimal_conditions: List[Dict], 
                             raw_measurements: List[Dict], output_dir: str) -> None:
    """
    Convenience function to generate all calibration plots matching original system.
    
    Args:
        trial_results: List of trial result dictionaries
        optimal_conditions: List of optimal condition dictionaries  
        raw_measurements: List of raw measurement dictionaries
        output_dir: Directory to save plots
    """
    visualizer = CalibrationVisualizer(output_dir)
    visualizer.generate_all_plots(trial_results, optimal_conditions, raw_measurements)