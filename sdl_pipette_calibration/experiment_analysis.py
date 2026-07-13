"""
Hardware-agnostic analysis module for calibration experiments.

This module provides statistical analysis and insights that work with any
parameter set automatically. No hardcoded parameter names or hardware-specific
assumptions.

Key Features:
- Parameter sensitivity analysis (works with any parameter set)
- Statistical summaries and quality metrics
- Performance trends and optimization insights
- Hardware/parameter agnostic design

Usage:
    analyzer = CalibrationAnalyzer()
    insights = analyzer.analyze_experiment(trial_results, optimal_conditions)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# SHAP for feature importance analysis - skipped gracefully if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class CalibrationAnalyzer:
    """Hardware-agnostic analysis for calibration experiments."""
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def analyze_experiment(self, trial_results: List[Dict], 
                         optimal_conditions: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of calibration experiment.
        
        Args:
            trial_results: List of trial result dictionaries
            optimal_conditions: List of optimal condition dictionaries
            
        Returns:
            Dictionary containing analysis results and insights
        """
        logger.info("Performing calibration experiment analysis...")
        
        insights = {
            'experiment_overview': self._analyze_experiment_overview(trial_results, optimal_conditions),
            'optimization_performance': self._analyze_optimization_performance(trial_results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(trial_results),
            'volume_scaling': self._analyze_volume_scaling(optimal_conditions),
            'quality_trends': self._analyze_quality_trends(trial_results),
            'efficiency_metrics': self._analyze_efficiency_metrics(trial_results, optimal_conditions),
            'recommendations': self._generate_recommendations(trial_results, optimal_conditions)
        }
        
        logger.info("[SUCCESS] Experiment analysis completed")
        return insights
    
    def _analyze_experiment_overview(self, trial_results: List[Dict], 
                                   optimal_conditions: List[Dict]) -> Dict[str, Any]:
        """Generate high-level experiment overview."""
        overview = {}
        
        if trial_results:
            # Basic counts
            overview['total_trials'] = len(trial_results)
            overview['volumes_calibrated'] = len(optimal_conditions)
            
            # Accuracy metrics
            deviations = [r.get('analysis', {}).get('absolute_deviation_pct', 0) for r in trial_results]
            overview['mean_deviation_pct'] = np.mean(deviations)
            overview['median_deviation_pct'] = np.median(deviations)
            overview['best_deviation_pct'] = np.min(deviations)
            overview['worst_deviation_pct'] = np.max(deviations)
            
            # Success rates
            excellent_trials = len([r for r in trial_results if r.get('quality', {}).get('overall_quality') == 'excellent'])
            good_trials = len([r for r in trial_results if r.get('quality', {}).get('overall_quality') == 'good'])
            overview['excellent_rate_pct'] = (excellent_trials / len(trial_results)) * 100
            overview['good_or_better_rate_pct'] = ((excellent_trials + good_trials) / len(trial_results)) * 100
            
            # Timing
            times = [r.get('analysis', {}).get('mean_duration_s', 0) for r in trial_results]
            overview['mean_time_s'] = np.mean(times)
            overview['total_experiment_time_s'] = np.sum(times)
            
        return overview
    
    def _analyze_optimization_performance(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Analyze how optimization performance improved over trials."""
        performance = {}
        
        if not trial_results:
            return performance
        
        # Sort trials by execution order (assuming trial_id or timestamp ordering)
        sorted_trials = sorted(trial_results, key=lambda x: x.get('trial_id', ''))
        
        deviations = [r.get('analysis', {}).get('absolute_deviation_pct', 0) for r in sorted_trials]
        scores = [r.get('composite_score', 0) for r in sorted_trials]
        
        if len(deviations) > 1:
            # Calculate improvement trends
            early_deviation = np.mean(deviations[:len(deviations)//3]) if len(deviations) >= 3 else deviations[0]
            late_deviation = np.mean(deviations[-len(deviations)//3:]) if len(deviations) >= 3 else deviations[-1]
            
            performance['early_vs_late_improvement_pct'] = ((early_deviation - late_deviation) / early_deviation * 100) if early_deviation > 0 else 0
            
            # Calculate correlation with trial number
            trial_numbers = list(range(1, len(deviations) + 1))
            if len(set(deviations)) > 1:  # Check for variation
                correlation, p_value = stats.pearsonr(trial_numbers, deviations)
                performance['convergence_correlation'] = correlation
                performance['convergence_p_value'] = p_value
                performance['is_converging'] = correlation < -0.3 and p_value < 0.1
            
            # Find best consecutive improvement
            improvements = []
            for i in range(1, len(deviations)):
                if deviations[i] < deviations[i-1]:
                    improvements.append(deviations[i-1] - deviations[i])
                else:
                    improvements.append(0)
            
            performance['max_single_improvement_pct'] = max(improvements) if improvements else 0
            performance['total_improvements'] = sum(1 for imp in improvements if imp > 0)
        
        # Quality progression
        qualities = [r.get('quality', {}).get('overall_quality', '') for r in sorted_trials]
        quality_scores = {'excellent': 3, 'good': 2, 'poor': 1, '': 0}
        quality_nums = [quality_scores.get(q, 0) for q in qualities]
        
        if len(quality_nums) > 1:
            performance['quality_trend_slope'] = np.polyfit(range(len(quality_nums)), quality_nums, 1)[0]
        
        return performance
    
    def _analyze_parameter_sensitivity(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Analyze which parameters most affect calibration performance."""
        sensitivity = {}
        
        if not trial_results:
            return sensitivity
        
        # Extract parameter data (hardware-agnostic)
        param_data = []
        deviations = []
        cv_pcts = []       # precision - raw, before penalty filtering
        times = []         # mean duration
        
        for trial in trial_results:
            analysis = trial.get('analysis', {})
            deviation = analysis.get('absolute_deviation_pct', 0)
            cv_pct = analysis.get('cv_volume_pct', 0)
            duration = analysis.get('mean_duration_s', 0)
            parameters = trial.get('parameters', {})
            
            # Flatten parameters
            flat_params = self._flatten_parameters_for_analysis(parameters)
            if flat_params:  # Only include if we have parameter data
                param_data.append(flat_params)
                deviations.append(deviation)
                cv_pcts.append(cv_pct)
                times.append(duration)
        
        if len(param_data) < 2:
            return sensitivity
        
        # Convert to DataFrame
        param_df = pd.DataFrame(param_data)
        
        # Find numeric columns only
        numeric_cols = param_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return sensitivity
        
        # Calculate correlations with accuracy
        correlations = {}
        for col in numeric_cols:
            if param_df[col].var() > 0:  # Only analyze parameters that vary
                try:
                    corr, p_val = stats.pearsonr(param_df[col], deviations)
                    if not np.isnan(corr):
                        correlations[col] = {
                            'correlation': corr,
                            'abs_correlation': abs(corr),
                            'p_value': p_val,
                            'is_significant': p_val < 0.05,
                            'effect_direction': 'increases_error' if corr > 0 else 'decreases_error'
                        }
                except:
                    continue
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: x[1]['abs_correlation'], 
                                   reverse=True)
        
        sensitivity['parameter_correlations'] = dict(sorted_correlations)
        sensitivity['most_influential_parameters'] = [item[0] for item in sorted_correlations[:5]]
        
        # SHAP feature importance for 3 targets (requires shap + enough data)
        if SHAP_AVAILABLE and len(param_data) >= 10 and len(numeric_cols) >= 2:
            try:
                X = param_df[numeric_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

                shap_targets = {
                    'accuracy': (np.array(deviations), list(range(len(deviations)))),
                    # Precision: filter out 100% penalty rows (penalized by accuracy failure, not real precision)
                    'precision': (
                        np.array([v for v in cv_pcts if v < 99.0]),
                        [i for i, v in enumerate(cv_pcts) if v < 99.0]
                    ),
                    'time': (np.array(times), list(range(len(times)))),
                }

                shap_importance = {}
                for target_name, (y_vals, row_idx) in shap_targets.items():
                    if len(y_vals) < 5 or len(set(y_vals)) < 2:
                        logger.debug(f"Skipping SHAP for {target_name}: insufficient data variation")
                        continue
                    try:
                        X_subset = X_scaled_df.iloc[row_idx]
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                        rf.fit(X_subset, y_vals)
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_subset)
                        mean_abs_shap = np.abs(shap_values).mean(axis=0)
                        importance_dict = dict(zip(numeric_cols, mean_abs_shap))
                        # Sort descending by importance
                        shap_importance[target_name] = dict(
                            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        )
                        logger.debug(f"SHAP computed for {target_name} ({len(y_vals)} trials)")
                    except Exception as e:
                        logger.debug(f"SHAP failed for {target_name}: {e}")

                if shap_importance:
                    sensitivity['shap_importance'] = shap_importance
                    # Top features across all targets (union, ranked by accuracy importance)
                    accuracy_imp = shap_importance.get('accuracy', {})
                    sensitivity['top_important_features'] = list(accuracy_imp.keys())[:5]
                    logger.info(f"SHAP analysis complete for targets: {list(shap_importance.keys())}")

            except Exception as e:
                logger.debug(f"SHAP analysis failed: {e}")

        elif not SHAP_AVAILABLE and len(param_data) >= 10 and len(numeric_cols) >= 2:
            # Fallback: RF feature importance (accuracy only) when shap not installed
            try:
                X = param_df[numeric_cols].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                rf.fit(X_scaled, np.array(deviations))
                importance_scores = dict(zip(numeric_cols, rf.feature_importances_))
                sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                sensitivity['feature_importance'] = dict(sorted_importance)
                sensitivity['top_important_features'] = [item[0] for item in sorted_importance[:5]]
            except Exception as e:
                logger.debug(f"RF feature importance fallback failed: {e}")

        return sensitivity
    
    def _analyze_volume_scaling(self, optimal_conditions: List[Dict]) -> Dict[str, Any]:
        """Analyze how calibration performance scales with volume."""
        scaling = {}
        
        if len(optimal_conditions) < 2:
            return scaling
        
        # Sort by volume
        sorted_conditions = sorted(optimal_conditions, key=lambda x: x.get('volume_ul', 0))
        
        volumes = [c.get('volume_ul', 0) for c in sorted_conditions]
        deviations = [c.get('deviation_pct', 0) for c in sorted_conditions]
        times = [c.get('time_s', 0) for c in sorted_conditions]
        trials_used = [c.get('trials_used', 0) for c in sorted_conditions]
        
        # Volume range analysis
        scaling['volume_range_ul'] = {'min': min(volumes), 'max': max(volumes)}
        scaling['volume_range_ratio'] = max(volumes) / min(volumes) if min(volumes) > 0 else 0
        
        # Accuracy vs volume relationship
        if len(set(volumes)) > 1 and len(set(deviations)) > 1:
            try:
                corr, p_val = stats.pearsonr(volumes, deviations)
                scaling['accuracy_volume_correlation'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05,
                    'interpretation': self._interpret_volume_correlation(corr, p_val)
                }
            except:
                pass
        
        # Efficiency vs volume relationship
        if len(set(volumes)) > 1 and len(set(trials_used)) > 1:
            try:
                corr, p_val = stats.pearsonr(volumes, trials_used)
                scaling['efficiency_volume_correlation'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05
                }
            except:
                pass
        
        # Best and worst volumes
        best_idx = np.argmin(deviations)
        worst_idx = np.argmax(deviations)
        
        scaling['best_volume'] = {
            'volume_ul': volumes[best_idx],
            'deviation_pct': deviations[best_idx],
            'trials_used': trials_used[best_idx]
        }
        
        scaling['worst_volume'] = {
            'volume_ul': volumes[worst_idx],
            'deviation_pct': deviations[worst_idx],
            'trials_used': trials_used[worst_idx]
        }
        
        return scaling
    
    def _analyze_quality_trends(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Analyze quality trends and patterns."""
        trends = {}
        
        if not trial_results:
            return trends
        
        # Quality distribution
        qualities = [r.get('quality', {}).get('overall_quality', '') for r in trial_results]
        quality_counts = pd.Series(qualities).value_counts().to_dict()
        trends['quality_distribution'] = quality_counts
        
        # Score statistics
        scores = [r.get('composite_score', 0) for r in trial_results]
        trends['score_statistics'] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
        
        # Accuracy vs precision trade-offs
        deviations = [r.get('analysis', {}).get('absolute_deviation_pct', 0) for r in trial_results]
        cvs = [r.get('analysis', {}).get('cv_volume_pct', 0) for r in trial_results]
        
        if len(set(deviations)) > 1 and len(set(cvs)) > 1:
            try:
                corr, p_val = stats.pearsonr(deviations, cvs)
                trends['accuracy_precision_tradeoff'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'has_tradeoff': corr > 0.3 and p_val < 0.05
                }
            except:
                pass
        
        return trends
    
    def _analyze_efficiency_metrics(self, trial_results: List[Dict], 
                                  optimal_conditions: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization efficiency metrics."""
        efficiency = {}
        
        if not trial_results or not optimal_conditions:
            return efficiency
        
        # Basic efficiency metrics
        total_trials = len(trial_results)
        volumes_calibrated = len(optimal_conditions)
        
        efficiency['trials_per_volume'] = total_trials / volumes_calibrated if volumes_calibrated > 0 else 0
        
        # Calculate total measurements
        total_measurements = sum([c.get('measurements_count', 0) for c in optimal_conditions])
        efficiency['measurements_per_volume'] = total_measurements / volumes_calibrated if volumes_calibrated > 0 else 0
        
        # Success efficiency (trials needed to achieve good results)
        excellent_trials = [r for r in trial_results if r.get('quality', {}).get('overall_quality') == 'excellent']
        good_trials = [r for r in trial_results if r.get('quality', {}).get('overall_quality') == 'good']
        
        efficiency['excellent_trial_rate'] = len(excellent_trials) / total_trials if total_trials > 0 else 0
        efficiency['good_or_better_rate'] = (len(excellent_trials) + len(good_trials)) / total_trials if total_trials > 0 else 0
        
        # Time efficiency
        total_time = sum([r.get('analysis', {}).get('mean_duration_s', 0) for r in trial_results])
        efficiency['total_experiment_time_min'] = total_time / 60
        efficiency['time_per_volume_min'] = total_time / (60 * volumes_calibrated) if volumes_calibrated > 0 else 0
        
        return efficiency
    
    def _generate_recommendations(self, trial_results: List[Dict], 
                                optimal_conditions: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not trial_results:
            return recommendations
        
        # Analyze success rates
        excellent_rate = len([r for r in trial_results if r.get('quality', {}).get('overall_quality') == 'excellent']) / len(trial_results)
        
        if excellent_rate < 0.3:
            recommendations.append("Low excellent trial rate (<30%). Consider tightening parameter bounds or increasing measurement budget.")
        
        # Analyze convergence
        deviations = [r.get('analysis', {}).get('absolute_deviation_pct', 0) for r in trial_results]
        if len(deviations) > 5:
            early_avg = np.mean(deviations[:len(deviations)//3])
            late_avg = np.mean(deviations[-len(deviations)//3:])
            
            if late_avg >= early_avg:
                recommendations.append("No clear optimization convergence detected. Consider different parameter bounds or optimization strategy.")
        
        # Analyze volume scaling
        if len(optimal_conditions) > 1:
            sorted_conditions = sorted(optimal_conditions, key=lambda x: x.get('volume_ul', 0))
            volumes = [c.get('volume_ul', 0) for c in sorted_conditions]
            vol_deviations = [c.get('deviation_pct', 0) for c in sorted_conditions]
            
            if max(vol_deviations) > 10:
                worst_vol = volumes[np.argmax(vol_deviations)]
                recommendations.append(f"Volume {worst_vol}uL shows high deviation (>{max(vol_deviations):.1f}%). Consider additional trials or parameter adjustment for this volume.")
        
        # Analyze parameter sensitivity
        param_data = []
        for trial in trial_results:
            parameters = trial.get('parameters', {})
            flat_params = self._flatten_parameters_for_analysis(parameters)
            if flat_params:
                param_data.append(flat_params)
        
        if param_data:
            param_df = pd.DataFrame(param_data)
            numeric_cols = param_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Check for parameters with low variation
            low_variation_params = []
            for col in numeric_cols:
                if param_df[col].var() < (param_df[col].mean() * 0.1) ** 2:  # Less than 10% relative variation
                    low_variation_params.append(col)
            
            if low_variation_params:
                recommendations.append(f"Parameters with low variation detected: {', '.join(low_variation_params[:3])}. Consider expanding search ranges.")
        
        # Time efficiency recommendations
        times = [r.get('analysis', {}).get('mean_duration_s', 0) for r in trial_results]
        if np.std(times) > np.mean(times) * 0.5:  # High time variability
            recommendations.append("High variability in measurement times detected. Consider investigating timing parameters or measurement consistency.")
        
        return recommendations
    
    def _flatten_parameters_for_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested parameters for analysis."""
        flattened = {}
        
        def _flatten_recursive(obj: Any, prefix: str = '') -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, dict):
                        _flatten_recursive(value, new_key)
                    elif isinstance(value, (int, float)):
                        # Convert numpy types to native Python types
                        if hasattr(value, 'item'):
                            value = value.item()
                        flattened[new_key] = value
            elif isinstance(obj, (int, float)):
                if hasattr(obj, 'item'):
                    obj = obj.item()
                key = prefix if prefix else 'value'
                flattened[key] = obj
        
        _flatten_recursive(parameters)
        return flattened
    
    def _interpret_volume_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret volume-accuracy correlation."""
        if p_value >= 0.05:
            return "No significant relationship between volume and accuracy"
        
        if correlation > 0.5:
            return "Larger volumes tend to have worse accuracy"
        elif correlation < -0.5:
            return "Larger volumes tend to have better accuracy"
        elif correlation > 0.3:
            return "Weak trend: larger volumes slightly less accurate"
        elif correlation < -0.3:
            return "Weak trend: larger volumes slightly more accurate"
        else:
            return "No clear relationship between volume and accuracy"
    
    def save_analysis_report(self, insights: Dict[str, Any], output_file: str) -> None:
        """Save analysis insights to a formatted text report."""
        with open(output_file, 'w') as f:
            f.write("CALIBRATION EXPERIMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Experiment Overview
            overview = insights.get('experiment_overview', {})
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Trials: {overview.get('total_trials', 0)}\n")
            f.write(f"Volumes Calibrated: {overview.get('volumes_calibrated', 0)}\n")
            f.write(f"Mean Deviation: {overview.get('mean_deviation_pct', 0):.2f}%\n")
            f.write(f"Best Deviation: {overview.get('best_deviation_pct', 0):.2f}%\n")
            f.write(f"Excellent Trial Rate: {overview.get('excellent_rate_pct', 0):.1f}%\n")
            f.write(f"Total Experiment Time: {overview.get('total_experiment_time_s', 0)/60:.1f} minutes\n\n")
            
            # Parameter Sensitivity
            sensitivity = insights.get('parameter_sensitivity', {})

            # SHAP importance (preferred - shows direction and covers 3 targets)
            shap_imp = sensitivity.get('shap_importance', {})
            if shap_imp:
                f.write("PARAMETER IMPORTANCE (SHAP - mean |effect| on each target)\n")
                f.write("-" * 55 + "\n")
                # Collect all params across targets for aligned table
                all_params = list(dict.fromkeys(
                    p for target_dict in shap_imp.values() for p in target_dict
                ))
                targets_present = list(shap_imp.keys())
                header = f"{'Parameter':<30}" + "".join(f"{t.capitalize():>12}" for t in targets_present)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for param in all_params[:10]:  # top 10
                    row = f"{param:<30}"
                    for t in targets_present:
                        val = shap_imp[t].get(param, 0.0)
                        row += f"{val:>12.4f}"
                    f.write(row + "\n")
                f.write("\n")
            elif 'most_influential_parameters' in sensitivity:
                # Fallback: Pearson correlation (shap not available)
                f.write("MOST INFLUENTIAL PARAMETERS (Pearson correlation, accuracy only)\n")
                f.write("-" * 50 + "\n")
                for param in sensitivity['most_influential_parameters'][:5]:
                    corr_data = sensitivity.get('parameter_correlations', {}).get(param, {})
                    corr = corr_data.get('correlation', 0)
                    f.write(f"{param}: {corr:.3f} correlation\n")
                f.write("\n")
            
            # Recommendations
            recommendations = insights.get('recommendations', [])
            if recommendations:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 15 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
        
        logger.info(f"Analysis report saved to: {output_file}")

def analyze_calibration_experiment(trial_results: List[Dict], optimal_conditions: List[Dict], 
                                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze calibration experiment.
    
    Args:
        trial_results: List of trial result dictionaries
        optimal_conditions: List of optimal condition dictionaries
        output_dir: Optional directory to save analysis report
        
    Returns:
        Dictionary containing analysis insights
    """
    analyzer = CalibrationAnalyzer()
    insights = analyzer.analyze_experiment(trial_results, optimal_conditions)
    
    if output_dir:
        report_file = Path(output_dir) / "analysis_report.txt"
        analyzer.save_analysis_report(insights, str(report_file))
    
    return insights