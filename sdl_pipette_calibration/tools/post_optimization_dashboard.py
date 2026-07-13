"""
Post-Optimization Dashboard

Creates a comprehensive visualization dashboard showing:
1. Deviation vs Time scatter plot with optimal conditions highlighted
2. Optimal parameter list display
3. SHAP parameter importance analysis

Usage:
    python post_optimization_dashboard.py output/run_1234567890
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import argparse
import logging
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostOptimizationDashboard:
    """Creates comprehensive post-optimization visualization dashboard."""
    
    def __init__(self, run_directory: str):
        """Initialize dashboard with optimization run directory."""
        self.run_dir = Path(run_directory)
        
        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Data storage
        self.trial_results_df = None
        self.optimal_conditions_df = None
        self.insights_data = None
        
    def load_data(self) -> bool:
        """Load optimization data from run directory."""
        try:
            # Load trial results
            trial_results_path = self.run_dir / "trial_results.csv"
            if trial_results_path.exists():
                self.trial_results_df = pd.read_csv(trial_results_path)
                logger.info(f"Loaded {len(self.trial_results_df)} trial results")
            else:
                logger.error(f"Trial results not found: {trial_results_path}")
                return False
            
            # Load optimal conditions - check for different naming patterns
            optimal_paths = [
                self.run_dir / "optimal_conditions.csv",
                list(self.run_dir.glob("optimal_conditions_*.csv"))
            ]
            
            optimal_path = None
            for path_option in optimal_paths:
                if isinstance(path_option, list) and path_option:
                    optimal_path = path_option[0]
                    break
                elif isinstance(path_option, Path) and path_option.exists():
                    optimal_path = path_option
                    break
            
            if optimal_path and optimal_path.exists():
                self.optimal_conditions_df = pd.read_csv(optimal_path)
                logger.info(f"Loaded {len(self.optimal_conditions_df)} optimal conditions from {optimal_path.name}")
            else:
                logger.warning("No optimal conditions file found - will show all data without highlighting")
            
            # Try to load insights for SHAP analysis
            insights_path = self.run_dir / "experiment_insights.json"
            if insights_path.exists():
                import json
                with open(insights_path, 'r') as f:
                    self.insights_data = json.load(f)
                logger.info("Loaded experiment insights for SHAP analysis")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_dashboard(self, show_plot: bool = True, save_plot: bool = True) -> None:
        """Create the complete optimization dashboard."""
        if self.trial_results_df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main deviation vs time plot (left side, takes up 60% width)
        ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        
        # Parameter importance plot (top right)
        ax_shap = plt.subplot2grid((2, 3), (0, 2))
        
        # Optimal parameters text (bottom right) 
        ax_params = plt.subplot2grid((2, 3), (1, 2))
        
        # Generate main plot
        self._plot_deviation_vs_time_highlighted(ax_main)
        
        # Generate SHAP plot if available
        if self.insights_data:
            self._plot_shap_importance_compact(ax_shap)
        else:
            ax_shap.text(0.5, 0.5, 'SHAP analysis\\nnot available', 
                        ha='center', va='center', transform=ax_shap.transAxes,
                        fontsize=12, style='italic')
            ax_shap.set_xticks([])
            ax_shap.set_yticks([])
        
        # Show optimal parameters
        self._display_optimal_parameters(ax_params)
        
        # Overall title
        fig.suptitle(f'Optimization Results Dashboard - {self.run_dir.name}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        
        if save_plot:
            save_path = self.run_dir / "optimization_dashboard.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Dashboard saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _plot_deviation_vs_time_highlighted(self, ax) -> None:
        """Create main deviation vs time plot with optimal conditions highlighted."""
        df = self.trial_results_df.copy()
        
        # Prepare data
        if 'volume_target_ml' in df.columns:
            df['volume'] = df['volume_target_ml']
        elif 'volume' not in df.columns:
            df['volume'] = 0.05  # Default volume if not specified
        
        # Get unique volumes for color coding
        volumes = sorted(df['volume'].unique())
        volume_colors = plt.cm.tab10(np.linspace(0, 1, len(volumes)))
        
        # Plot all trials first
        for i, vol in enumerate(volumes):
            vol_data = df[df['volume'] == vol]
            ax.scatter(vol_data['duration_s'], vol_data['deviation_pct'], 
                      c=[volume_colors[i]], label=f'{vol*1000:.0f}μL', 
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Highlight optimal conditions if available
        if self.optimal_conditions_df is not None:
            optimal_points_plotted = 0
            
            # For each optimal condition, find matching trials and highlight them
            for _, optimal in self.optimal_conditions_df.iterrows():
                vol = optimal.get('volume_target_ml', 0)
                deviation = optimal.get('deviation_pct', 0) 
                duration = optimal.get('duration_s', 0)
                
                if deviation > 0 and duration > 0:  # Valid data point
                    # Plot with special highlighting
                    ax.scatter(duration, deviation, c='gold', s=200, 
                              marker='*', edgecolors='red', linewidth=2,
                              label='Optimal Conditions' if optimal_points_plotted == 0 else "",
                              zorder=10)
                    optimal_points_plotted += 1
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Deviation (%)', fontsize=12)
        ax.set_title('Deviation vs Time - Optimization Progress', fontsize=14, fontweight='bold')
        ax.legend(title='Volume', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add performance zone indicators
        if self.optimal_conditions_df is not None:
            # Add a light background rectangle for "good" zone
            ax.axhspan(0, 10, alpha=0.1, color='green', label='Excellent (<10%)')
            ax.axhspan(10, 25, alpha=0.1, color='yellow', label='Good (10-25%)')
    
    def _plot_shap_importance_compact(self, ax) -> None:
        """Create compact SHAP importance plot."""
        sensitivity = self.insights_data.get('parameter_sensitivity', {})
        shap_imp = sensitivity.get('shap_importance', {})
        
        if not shap_imp:
            ax.text(0.5, 0.5, 'SHAP data\\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get accuracy importance (most relevant for calibration)
        if 'accuracy' in shap_imp:
            importance_data = shap_imp['accuracy']
        else:
            # Fallback to first available target
            importance_data = list(shap_imp.values())[0]
        
        # Get top 6 parameters for compact display
        sorted_params = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if sorted_params:
            params, values = zip(*sorted_params)
            
            # Create horizontal bar chart for compact display
            y_pos = np.arange(len(params))
            bars = ax.barh(y_pos, values, color='steelblue', alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([p.replace('_', ' ').title() for p in params], fontsize=10)
            ax.set_xlabel('SHAP Importance', fontsize=10)
            ax.set_title('Parameter Importance\\n(Top 6)', fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='center', va='center', fontsize=9, color='white')
        
        ax.grid(True, alpha=0.3, axis='x')
    
    def _display_optimal_parameters(self, ax) -> None:
        """Display optimal parameters in text format."""
        ax.axis('off')  # Turn off axes for text display
        
        if self.optimal_conditions_df is None or len(self.optimal_conditions_df) == 0:
            ax.text(0.5, 0.5, 'No optimal\\nconditions found', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            return
        
        # Get the best condition (first row, should be best)
        best_condition = self.optimal_conditions_df.iloc[0]
        
        # Create parameter text
        param_text = "**OPTIMAL PARAMETERS**\\n\\n"
        
        # Performance metrics first
        param_text += "Performance:\\n"
        param_text += f"• Deviation: {best_condition.get('deviation_pct', 0):.2f}%\\n"
        param_text += f"• Precision: {best_condition.get('precision_cv_pct', 0):.2f}% CV\\n"
        param_text += f"• Time: {best_condition.get('duration_s', 0):.1f}s\\n"
        param_text += f"• Volume: {best_condition.get('volume_target_ml', 0)*1000:.0f}μL\\n\\n"
        
        # Hardware parameters
        param_text += "Hardware Settings:\\n"
        
        # Find parameter columns (exclude metadata columns)
        metadata_cols = {'volume_target_ml', 'volume_target_ul', 'volume_measured_ml', 
                        'volume_measured_ul', 'deviation_pct', 'precision_cv_pct', 
                        'duration_s', 'trials_count', 'status', 'measurement_count'}
        
        param_cols = [col for col in best_condition.index if col not in metadata_cols]
        
        # Display most important parameters
        for param in param_cols[:8]:  # Limit to 8 parameters for space
            value = best_condition[param]
            if pd.notna(value):
                # Format parameter name nicely
                param_name = param.replace('_', ' ').title()
                
                # Format value based on type
                if isinstance(value, (int, float)):
                    if abs(value) < 0.01:
                        param_text += f"• {param_name}: {value:.4f}\\n"
                    else:
                        param_text += f"• {param_name}: {value:.3f}\\n"
                else:
                    param_text += f"• {param_name}: {value}\\n"
        
        # Add text to axes
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))


def main():
    """Main function to run the dashboard from command line."""
    parser = argparse.ArgumentParser(description="Generate post-optimization dashboard")
    parser.add_argument("run_directory", help="Path to optimization run directory")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot (just save)")
    parser.add_argument("--no-save", action="store_true", help="Don't save plot (just show)")
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = PostOptimizationDashboard(args.run_directory)
    
    if not dashboard.load_data():
        logger.error("Failed to load optimization data")
        sys.exit(1)
    
    # Generate dashboard
    dashboard.create_dashboard(
        show_plot=not args.no_show,
        save_plot=not args.no_save
    )
    
    logger.info("Dashboard generation complete!")


if __name__ == "__main__":
    main()