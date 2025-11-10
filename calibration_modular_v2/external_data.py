"""
External Data Integration for Calibration System
===============================================

This module handles loading and processing external calibration data from CSV files.
When external data is available, it replaces the screening phase by providing
initial parameter suggestions based on historical calibration results.

Key Features:
- CSV data loading with validation
- Volume and liquid filtering
- Parameter extraction and conversion
- Integration with existing trial workflow
- Fallback to normal screening if data unavailable

Expected CSV Format:
    volume_ml, aspirate_speed, dispense_speed, aspirate_wait_time_s, 
    dispense_wait_time_s, retract_speed, blowout_vol_ml, post_asp_air_vol_ml,
    overaspirate_vol_ml, deviation_pct, variability_pct, duration_s

Example Usage:
    loader = ExternalDataLoader(config)
    if loader.has_valid_data():
        screening_trials = loader.generate_screening_trials(target_volume_ml=0.05)
    else:
        # Fall back to normal screening
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from config_manager import ExperimentConfig
from data_structures import PipettingParameters, TrialResult, RawMeasurement, AdaptiveMeasurementResult, QualityEvaluation, VolumeTolerances

logger = logging.getLogger(__name__)


class ExternalDataLoader:
    """
    Loads and processes external calibration data for screening replacement.
    
    Handles CSV loading, data validation, filtering, and conversion to
    trial results that can be used by the optimization system.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize with experiment configuration."""
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.is_loaded = False
        
        if config.is_external_data_enabled():
            self._load_external_data()
    
    def _load_external_data(self):
        """Load external data from configured CSV file."""
        data_path = self.config.get_external_data_path()
        if not data_path:
            logger.warning("External data enabled but no data_path specified")
            return
        
        path = Path(data_path)
        if not path.exists():
            logger.warning(f"External data file not found: {data_path}")
            return
        
        try:
            # Load CSV data
            self.data = pd.read_csv(path)
            logger.info(f"Loaded external data: {len(self.data)} rows from {data_path}")
            
            # Validate required columns
            required_columns = self.config.get_external_data_required_columns()
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in external data: {missing_columns}")
                self.data = None
                return
            
            # Apply filters
            self._apply_filters()
            
            # Validate data quality
            if len(self.data) == 0:
                logger.warning("No data remaining after filtering")
                self.data = None
                return
            
            self.is_loaded = True
            logger.info(f"External data loaded successfully: {len(self.data)} rows after filtering")
            
        except Exception as e:
            logger.error(f"Failed to load external data: {e}")
            self.data = None
    
    def _apply_filters(self):
        """Apply volume and liquid filters to the data."""
        if self.data is None:
            return
        
        original_count = len(self.data)
        
        # Apply volume filter
        volume_filter = self.config.get_external_data_volume_filter()
        if volume_filter is not None:
            tolerance = 0.001  # 1μL tolerance for volume matching
            volume_mask = abs(self.data['volume_ml'] - volume_filter) <= tolerance
            self.data = self.data[volume_mask]
            logger.info(f"Volume filter ({volume_filter} mL): {original_count} → {len(self.data)} rows")
        
        # Apply liquid filter
        liquid_filter = self.config.get_external_data_liquid_filter()
        if liquid_filter is not None and 'liquid' in self.data.columns:
            liquid_mask = self.data['liquid'].str.lower() == liquid_filter.lower()
            self.data = self.data[liquid_mask]
            logger.info(f"Liquid filter ({liquid_filter}): filtered to {len(self.data)} rows")
        
        # Remove rows with missing critical data
        critical_columns = ['aspirate_speed', 'dispense_speed', 'deviation_pct']
        for col in critical_columns:
            if col in self.data.columns:
                self.data = self.data.dropna(subset=[col])
        
        logger.info(f"Data filtering complete: {original_count} → {len(self.data)} rows")
    
    def has_valid_data(self) -> bool:
        """Check if valid external data is available."""
        return self.is_loaded and self.data is not None and len(self.data) > 0
    
    def generate_screening_trials(self, target_volume_ml: float, 
                                 max_trials: int = 5) -> List[TrialResult]:
        """
        Generate screening trials from external data.
        
        Args:
            target_volume_ml: Target volume for screening
            max_trials: Maximum number of trials to generate
            
        Returns:
            List[TrialResult]: Generated trials from external data
        """
        if not self.has_valid_data():
            logger.warning("No valid external data available for screening")
            return []
        
        # Filter data for target volume (with tolerance)
        volume_tolerance = 0.001  # 1μL tolerance
        volume_mask = abs(self.data['volume_ml'] - target_volume_ml) <= volume_tolerance
        volume_data = self.data[volume_mask]
        
        if len(volume_data) == 0:
            logger.warning(f"No external data available for volume {target_volume_ml} mL")
            return []
        
        # Select best performing entries
        # Sort by deviation (accuracy) first, then by variability
        volume_data = volume_data.sort_values(['deviation_pct', 'variability_pct'], 
                                            key=lambda x: abs(x) if x.name == 'deviation_pct' else x)
        
        # Take up to max_trials best entries
        selected_data = volume_data.head(max_trials)
        
        logger.info(f"Selected {len(selected_data)} external data entries for screening")
        
        # Convert to trial results
        trials = []
        for idx, row in selected_data.iterrows():
            trial = self._convert_row_to_trial(row, target_volume_ml, f"external_{idx}")
            if trial:
                trials.append(trial)
        
        return trials
    
    def _convert_row_to_trial(self, row: pd.Series, target_volume_ml: float, 
                             trial_id: str) -> Optional[TrialResult]:
        """Convert external data row to TrialResult."""
        try:
            # Extract parameters
            parameters = PipettingParameters(
                aspirate_speed=float(row['aspirate_speed']),
                dispense_speed=float(row['dispense_speed']),
                aspirate_wait_time_s=float(row.get('aspirate_wait_time_s', 12.0)),
                dispense_wait_time_s=float(row.get('dispense_wait_time_s', 12.0)),
                retract_speed=float(row.get('retract_speed', 8.0)),
                blowout_vol_ml=float(row.get('blowout_vol_ml', 0.07)),
                post_asp_air_vol_ml=float(row.get('post_asp_air_vol_ml', 0.05)),
                overaspirate_vol_ml=float(row.get('overaspirate_vol_ml', 0.004))
            )
            
            # Apply volume constraints
            parameters = self.config.apply_volume_constraints(parameters, target_volume_ml)
            
            # Create synthetic measurement
            deviation_pct = float(row['deviation_pct'])
            variability_pct = float(row.get('variability_pct', 5.0))
            duration_s = float(row.get('duration_s', 10.0))
            
            # Calculate actual volume from deviation
            actual_volume_ml = target_volume_ml * (1 + deviation_pct / 100.0)
            
            # Create raw measurement
            measurement = RawMeasurement(
                measurement_id=f"{trial_id}_external",
                parameters=parameters,
                target_volume_ml=target_volume_ml,
                actual_volume_ml=actual_volume_ml,
                duration_s=duration_s,
                replicate_id=0,
                metadata={'source': 'external_data', 'original_index': str(row.name)}
            )
            
            # Create analysis result
            analysis = AdaptiveMeasurementResult(
                target_volume_ml=target_volume_ml,
                num_replicates=1,
                mean_volume_ml=actual_volume_ml,
                stdev_volume_ml=0.0,  # Unknown for external data
                cv_volume_pct=variability_pct,
                deviation_ml=actual_volume_ml - target_volume_ml,
                deviation_pct=deviation_pct,
                absolute_deviation_pct=abs(deviation_pct),
                mean_duration_s=duration_s,
                stdev_duration_s=0.0,
                min_volume_ml=actual_volume_ml,
                max_volume_ml=actual_volume_ml,
                median_volume_ml=actual_volume_ml
            )
            
            # Evaluate quality
            tolerances = self.config.calculate_tolerances_for_volume(target_volume_ml)
            deviation_ul = abs(analysis.deviation_ml * 1000)
            
            quality = QualityEvaluation(
                accuracy_good=deviation_ul <= tolerances.accuracy_tolerance_ul,
                precision_good=variability_pct <= tolerances.precision_tolerance_pct,
                time_good=duration_s <= tolerances.time_tolerance_s,
                overall_quality="good",  # Mark external data as good quality
                accuracy_tolerance_ul=tolerances.accuracy_tolerance_ul,
                precision_tolerance_pct=tolerances.precision_tolerance_pct,
                time_tolerance_s=tolerances.time_tolerance_s,
                measured_accuracy_ul=deviation_ul,
                measured_precision_pct=variability_pct,
                measured_time_s=duration_s
            )
            
            # Calculate simple composite score
            # External data doesn't get adaptive measurement logic
            accuracy_score = abs(deviation_pct) / 10.0  # Normalize to ~0-1 range
            precision_score = variability_pct / 10.0
            time_score = duration_s / 30.0
            
            weights = self.config.get_objective_weights()
            composite_score = (weights.accuracy_weight * accuracy_score +
                             weights.precision_weight * precision_score +
                             weights.time_weight * time_score)
            
            trial = TrialResult(
                parameters=parameters,
                target_volume_ml=target_volume_ml,
                measurements=[measurement],
                analysis=analysis,
                quality=quality,
                composite_score=composite_score,
                tolerances_used=tolerances,
                needs_additional_replicates=False,  # External data doesn't get replicates
                metadata={'source': 'external_data', 'trial_id': trial_id}
            )
            
            return trial
            
        except Exception as e:
            logger.error(f"Failed to convert external data row to trial: {e}")
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded external data."""
        if not self.has_valid_data():
            return {'loaded': False, 'message': 'No valid external data'}
        
        summary = {
            'loaded': True,
            'total_rows': len(self.data),
            'volumes': sorted(self.data['volume_ml'].unique()) if 'volume_ml' in self.data.columns else [],
            'liquids': sorted(self.data['liquid'].unique()) if 'liquid' in self.data.columns else [],
            'deviation_range': [self.data['deviation_pct'].min(), self.data['deviation_pct'].max()] if 'deviation_pct' in self.data.columns else [],
            'columns': list(self.data.columns)
        }
        
        return summary