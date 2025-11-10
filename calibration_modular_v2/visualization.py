"""
Hardware-agnostic visualization module for calibration experiments.

This module provides parameter-agnostic plotting functions that work with any
calibration parameters and hardware configurations. All plots are generated
dynamically based on the actual parameters present in the data.

Key Features:
- Parameter scatter plots (accuracy vs any parameter)
- Optimization convergence plots
- Parameter importance analysis
- Volume comparison plots
- Trial quality distribution

Usage:
    visualizer = CalibrationVisualizer(output_dir)
    visualizer.generate_all_plots(trial_results, optimal_conditions)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

class CalibrationVisualizer:
    """Hardware-agnostic visualization for calibration experiments."""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.ioff()
        
    def generate_all_plots(self, trial_results: List[Dict], optimal_conditions: List[Dict]) -> None:
        """Generate all visualization plots for calibration results."""
        logger.info("Generating calibration visualization plots...")
        
        try:
            # Convert to DataFrames for easier plotting
            trials_df = self._prepare_trials_dataframe(trial_results)
            optimal_df = self._prepare_optimal_dataframe(optimal_conditions)
            
            if trials_df.empty:
                logger.warning("No trial data available for plotting")
                return
                
            # Generate individual plots
            self._plot_optimization_convergence(trials_df)
            self._plot_parameter_scatter_matrix(trials_df)
            self._plot_volume_comparison(optimal_df)
            self._plot_quality_distribution(trials_df)
            self._plot_parameter_vs_accuracy(trials_df)
            self._plot_trial_timeline(trials_df)
            
            # Generate summary plot
            self._plot_experiment_summary(trials_df, optimal_df)
            
            logger.info(f"✅ All plots saved to: {self.plots_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _prepare_trials_dataframe(self, trial_results: List[Dict]) -> pd.DataFrame:
        """Convert trial results to DataFrame with flattened parameters."""
        if not trial_results:
            return pd.DataFrame()
            
        rows = []
        for trial in trial_results:
            row = {
                'trial_id': trial.get('trial_id', 'unknown'),
                'target_volume_ml': trial.get('target_volume_ml', 0),
                'target_volume_ul': trial.get('target_volume_ml', 0) * 1000,
                'measured_volume_ml': trial.get('analysis', {}).get('mean_volume_ml', 0),
                'measured_volume_ul': trial.get('analysis', {}).get('mean_volume_ml', 0) * 1000,
                'deviation_pct': trial.get('analysis', {}).get('absolute_deviation_pct', 0),
                'cv_pct': trial.get('analysis', {}).get('cv_volume_pct', 0),
                'mean_time_s': trial.get('analysis', {}).get('mean_duration_s', 0),
                'composite_score': trial.get('composite_score', 0),
                'quality': trial.get('quality', {}).get('overall_quality', 'unknown'),
                'trial_number': len(rows) + 1  # Sequential trial number
            }
            
            # Flatten all parameters (hardware-agnostic)
            parameters = trial.get('parameters', {})
            if isinstance(parameters, dict):
                # Handle nested parameter structure
                for category, params in parameters.items():
                    if isinstance(params, dict):
                        for param_name, param_value in params.items():
                            if isinstance(param_value, (int, float)):
                                row[f'{param_name}'] = param_value
                    elif isinstance(params, (int, float)):
                        row[f'{category}'] = params
            
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def _prepare_optimal_dataframe(self, optimal_conditions: List[Dict]) -> pd.DataFrame:
        """Convert optimal conditions to DataFrame."""
        if not optimal_conditions:
            return pd.DataFrame()
            
        rows = []
        for condition in optimal_conditions:
            row = {
                'volume_ul': condition.get('volume_ul', 0),
                'measured_volume_ul': condition.get('measured_volume_ul', 0),
                'deviation_pct': condition.get('deviation_pct', 0),
                'cv_pct': condition.get('cv_pct', 0),
                'time_s': condition.get('time_s', 0),
                'trials_used': condition.get('trials_used', 0),
                'status': condition.get('status', 'unknown')
            }
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def _plot_optimization_convergence(self, trials_df: pd.DataFrame) -> None:
        """Plot how optimization improved over time."""
        if trials_df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy convergence
        ax1.plot(trials_df['trial_number'], trials_df['deviation_pct'], 'o-', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Deviation from Target (%)')
        ax1.set_title('Accuracy Convergence Over Trials')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(trials_df) > 1:
            z = np.polyfit(trials_df['trial_number'], trials_df['deviation_pct'], 1)
            p = np.poly1d(z)
            ax1.plot(trials_df['trial_number'], p(trials_df['trial_number']), "--", alpha=0.8, color='red')
        
        # Plot 2: Composite score convergence
        ax2.plot(trials_df['trial_number'], trials_df['composite_score'], 'o-', alpha=0.7, color='green')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Composite Score (lower = better)')
        ax2.set_title('Overall Quality Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "optimization_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_scatter_matrix(self, trials_df: pd.DataFrame) -> None:
        """Plot scatter matrix of key parameters vs performance."""
        if trials_df.empty:
            return
            
        # Find numeric parameter columns (hardware-agnostic)
        param_cols = [col for col in trials_df.columns 
                     if col not in ['trial_id', 'quality', 'trial_number', 'target_volume_ml', 'target_volume_ul', 
                                   'measured_volume_ml', 'measured_volume_ul', 'deviation_pct', 'cv_pct', 
                                   'mean_time_s', 'composite_score'] 
                     and trials_df[col].dtype in ['float64', 'int64']]
        
        if not param_cols:
            logger.warning("No numeric parameters found for scatter matrix")
            return
        
        # Limit to top 6 most varying parameters to keep plot readable
        param_variance = trials_df[param_cols].var().sort_values(ascending=False)
        top_params = param_variance.head(6).index.tolist()
        
        # Create scatter plots of parameters vs accuracy
        n_params = len(top_params)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(top_params):
            if i >= 6:
                break
                
            ax = axes[i]
            scatter = ax.scatter(trials_df[param], trials_df['deviation_pct'], 
                               c=trials_df['composite_score'], cmap='viridis', alpha=0.7)
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Deviation (%)')
            ax.set_title(f'{param.replace("_", " ").title()} vs Accuracy')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for first plot
            if i == 0:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Composite Score')
        
        # Hide unused subplots
        for i in range(n_params, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "parameter_scatter_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_volume_comparison(self, optimal_df: pd.DataFrame) -> None:
        """Plot comparison across different target volumes."""
        if optimal_df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy by volume
        volumes = optimal_df['volume_ul'].values
        deviations = optimal_df['deviation_pct'].values
        
        bars1 = ax1.bar(range(len(volumes)), deviations, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Target Volume (μL)')
        ax1.set_ylabel('Deviation (%)')
        ax1.set_title('Calibration Accuracy by Volume')
        ax1.set_xticks(range(len(volumes)))
        ax1.set_xticklabels([f'{v:.0f}' for v in volumes])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, dev in zip(bars1, deviations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{dev:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Trials used by volume  
        trials_used = optimal_df['trials_used'].values
        bars2 = ax2.bar(range(len(volumes)), trials_used, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Target Volume (μL)')
        ax2.set_ylabel('Trials Required')
        ax2.set_title('Optimization Efficiency by Volume')
        ax2.set_xticks(range(len(volumes)))
        ax2.set_xticklabels([f'{v:.0f}' for v in volumes])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, trials in zip(bars2, trials_used):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{trials:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "volume_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_distribution(self, trials_df: pd.DataFrame) -> None:
        """Plot distribution of trial qualities."""
        if trials_df.empty:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Quality categories
        quality_counts = trials_df['quality'].value_counts()
        ax1.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Trial Quality Distribution')
        
        # Plot 2: Deviation histogram
        ax2.hist(trials_df['deviation_pct'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Deviation (%)')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Accuracy Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_dev = trials_df['deviation_pct'].mean()
        ax2.axvline(mean_dev, color='red', linestyle='--', 
                   label=f'Mean: {mean_dev:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "quality_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_vs_accuracy(self, trials_df: pd.DataFrame) -> None:
        """Plot how each parameter affects accuracy."""
        if trials_df.empty:
            return
            
        # Find numeric parameter columns
        param_cols = [col for col in trials_df.columns 
                     if col not in ['trial_id', 'quality', 'trial_number', 'target_volume_ml', 'target_volume_ul',
                                   'measured_volume_ml', 'measured_volume_ul', 'deviation_pct', 'cv_pct',
                                   'mean_time_s', 'composite_score']
                     and trials_df[col].dtype in ['float64', 'int64']]
        
        if not param_cols:
            return
            
        # Calculate correlation with accuracy
        correlations = []
        for param in param_cols:
            corr = trials_df[param].corr(trials_df['deviation_pct'])
            if not np.isnan(corr):
                correlations.append((param, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        if not correlations:
            return
        
        # Plot top correlations
        fig, ax = plt.subplots(figsize=(12, 8))
        
        params, corrs = zip(*correlations[:10])  # Top 10
        y_pos = np.arange(len(params))
        
        bars = ax.barh(y_pos, corrs, alpha=0.7, color='orange')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
        ax.set_xlabel('Correlation with Deviation (absolute)')
        ax.set_title('Parameter Influence on Accuracy')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, corr in zip(bars, corrs):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{corr:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "parameter_influence.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trial_timeline(self, trials_df: pd.DataFrame) -> None:
        """Plot trial performance over time."""
        if trials_df.empty:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color points by quality
        quality_colors = {'excellent': 'green', 'good': 'blue', 'poor': 'red', 'unknown': 'gray'}
        
        for quality in trials_df['quality'].unique():
            mask = trials_df['quality'] == quality
            if mask.any():
                ax.scatter(trials_df.loc[mask, 'trial_number'], 
                          trials_df.loc[mask, 'deviation_pct'],
                          label=quality.title(), alpha=0.7, 
                          color=quality_colors.get(quality, 'gray'))
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Deviation (%)')
        ax.set_title('Trial Performance Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "trial_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_experiment_summary(self, trials_df: pd.DataFrame, optimal_df: pd.DataFrame) -> None:
        """Create comprehensive summary plot."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Optimization convergence
        ax1 = fig.add_subplot(gs[0, :2])
        if not trials_df.empty:
            ax1.plot(trials_df['trial_number'], trials_df['deviation_pct'], 'o-', alpha=0.7)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Deviation (%)')
            ax1.set_title('Optimization Convergence')
            ax1.grid(True, alpha=0.3)
        
        # 2. Quality distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        if not trials_df.empty:
            quality_counts = trials_df['quality'].value_counts()
            ax2.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
            ax2.set_title('Trial Quality Distribution')
        
        # 3. Volume comparison
        ax3 = fig.add_subplot(gs[1, :2])
        if not optimal_df.empty:
            volumes = optimal_df['volume_ul'].values
            deviations = optimal_df['deviation_pct'].values
            ax3.bar(range(len(volumes)), deviations, alpha=0.7)
            ax3.set_xlabel('Target Volume (μL)')
            ax3.set_ylabel('Deviation (%)')
            ax3.set_title('Final Accuracy by Volume')
            ax3.set_xticks(range(len(volumes)))
            ax3.set_xticklabels([f'{v:.0f}' for v in volumes])
        
        # 4. Efficiency comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        if not optimal_df.empty:
            trials_used = optimal_df['trials_used'].values
            ax4.bar(range(len(volumes)), trials_used, alpha=0.7, color='lightcoral')
            ax4.set_xlabel('Target Volume (μL)')
            ax4.set_ylabel('Trials Required')
            ax4.set_title('Optimization Efficiency')
            ax4.set_xticks(range(len(volumes)))
            ax4.set_xticklabels([f'{v:.0f}' for v in volumes])
        
        # 5. Summary statistics (text)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate summary stats
        if not trials_df.empty and not optimal_df.empty:
            total_trials = len(trials_df)
            mean_deviation = optimal_df['deviation_pct'].mean()
            best_deviation = optimal_df['deviation_pct'].min()
            total_volumes = len(optimal_df)
            
            summary_text = f"""
EXPERIMENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Trials: {total_trials}
• Volumes Calibrated: {total_volumes}
• Mean Accuracy: {mean_deviation:.2f}% deviation
• Best Accuracy: {best_deviation:.2f}% deviation
• Average Trials per Volume: {total_trials/total_volumes:.1f}
• Success Rate: {len(optimal_df[optimal_df['deviation_pct'] <= 10])}/{total_volumes} volumes ≤10% deviation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Calibration Experiment Summary', fontsize=16, fontweight='bold')
        plt.savefig(self.plots_dir / "experiment_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_calibration_plots(trial_results: List[Dict], optimal_conditions: List[Dict], 
                             output_dir: str) -> None:
    """
    Convenience function to generate all calibration plots.
    
    Args:
        trial_results: List of trial result dictionaries
        optimal_conditions: List of optimal condition dictionaries  
        output_dir: Directory to save plots
    """
    visualizer = CalibrationVisualizer(output_dir)
    visualizer.generate_all_plots(trial_results, optimal_conditions)