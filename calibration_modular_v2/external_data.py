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
The CSV should contain individual measurement columns (target_volume_ml, measured_volume_ml, measurement_time_s) 
plus any hardware-specific parameters defined in your configuration.

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

from .config_manager import ExperimentConfig
from data_structures import PipettingParameters, TrialResult, RawMeasurement, AdaptiveMeasurementResult, QualityEvaluation, VolumeTolerances, HardwareParameters, CalibrationParameters

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
        
        logger.info("=== EXTERNAL DATA LOADER INIT DEBUG ===")
        enabled = config.is_external_data_enabled()
        logger.info(f"External data enabled check: {enabled}")
        
        if enabled:
            logger.info("External data is enabled, calling _load_external_data()")
            self._load_external_data()
        else:
            logger.warning("External data is DISABLED - no loading will occur")
        logger.info("=== EXTERNAL DATA LOADER INIT COMPLETE ===")
    
    def _load_external_data(self):
        """Load external data from configured CSV file."""
        logger.info("=== _load_external_data() START ===")
        data_path = self.config.get_external_data_path()
        logger.info(f"Config returned data_path: '{data_path}'")
        
        if not data_path:
            logger.warning("External data enabled but no data_path specified")
            return
        
        path = Path(data_path)
        logger.info(f"Created Path object: {path}")
        logger.info(f"Path exists check: {path.exists()}")
        logger.info(f"Current working directory: {Path.cwd()}")
        
        if not path.exists():
            logger.warning(f"External data file not found: {data_path}")
            logger.info(f"Tried absolute path: {path.resolve()}")
            return
        
        try:
            # Load CSV data
            self.data = pd.read_csv(path)
            logger.info(f"Loaded external data: {len(self.data)} rows from {data_path}")
            
            # Validate required columns - ONLY the truly essential ones
            required_columns = self.config.get_external_data_required_columns()
            
            # Always require basic measurement data (universal for any calibration)
            universal_required = ['target_volume_ml', 'measured_volume_ml', 'measurement_time_s']
            
            # If config specifies additional requirements, add them
            all_required = universal_required + (required_columns or [])
            
            missing_columns = [col for col in all_required if col not in self.data.columns]
            
            if missing_columns:
                logger.error(f"Missing essential columns in external data: {missing_columns}")
                logger.info(f"Available columns: {list(self.data.columns)}")
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
        
        # Apply volume filter (using target_volume_ml for raw measurements)
        volume_filter = self.config.get_external_data_volume_filter()
        if volume_filter is not None:
            tolerance = 0.001  # 1uL tolerance for volume matching
            if 'target_volume_ml' in self.data.columns:
                volume_mask = abs(self.data['target_volume_ml'] - volume_filter) <= tolerance
                self.data = self.data[volume_mask]
                logger.info(f"Volume filter ({volume_filter} mL): {original_count} -> {len(self.data)} rows")
        
        # Apply liquid filter (using liquid_type for raw measurements)
        liquid_filter = self.config.get_external_data_liquid_filter()
        if liquid_filter is not None and 'liquid_type' in self.data.columns:
            liquid_mask = self.data['liquid_type'].str.lower() == liquid_filter.lower()
            self.data = self.data[liquid_mask]
            logger.info(f"Liquid filter ({liquid_filter}): filtered to {len(self.data)} rows")
        
        # Remove rows with missing critical data (raw measurement columns)
        critical_columns = []  # No hardcoded critical columns
        
        # Only check for columns that actually exist in the data
        if 'measured_volume_ml' in self.data.columns:
            critical_columns.append('measured_volume_ml')
        if 'target_volume_ml' in self.data.columns:
            critical_columns.append('target_volume_ml')
            
        for col in critical_columns:
            self.data = self.data.dropna(subset=[col])
        
        logger.info(f"Data filtering complete: {original_count} -> {len(self.data)} rows")
    
    def has_valid_data(self) -> bool:
        """Check if valid external data is available."""
        result = self.is_loaded and self.data is not None and len(self.data) > 0
        logger.info(f"=== has_valid_data() DEBUG ===")
        logger.info(f"is_loaded: {self.is_loaded}")
        logger.info(f"data is not None: {self.data is not None}")
        if self.data is not None:
            logger.info(f"data length: {len(self.data)}")
        logger.info(f"has_valid_data result: {result}")
        logger.info(f"=== has_valid_data() END ===")
        return result
    
    def generate_screening_trials(self, target_volume_ml: float, 
                                 max_trials: int = 5) -> List[TrialResult]:
        """
        Load individual measurements from external data and group into trials.
        This works exactly like SOBOL trials: individual measurements -> TrialResult objects.
        
        Args:
            target_volume_ml: Target volume for screening
            max_trials: Maximum number of trials to generate
            
        Returns:
            List[TrialResult]: Generated trials from external measurements
        """
        if not self.has_valid_data():
            logger.warning("No valid external data available for screening")
            return []
        
        logger.info(f"Loading individual measurements from external data")
        logger.info(f"External data columns: {list(self.data.columns)}")
        logger.info(f"External data shape: {self.data.shape}")
        
        # Convert each row to a RawMeasurement object
        measurements = []
        for idx, row in self.data.iterrows():
            try:
                measurement = self._convert_row_to_measurement(row, target_volume_ml, f"external_{idx}")
                if measurement:
                    measurements.append(measurement)
                    logger.debug(f"Created measurement {measurement.measurement_id}")
            except Exception as e:
                logger.warning(f"Failed to convert row {idx} to measurement: {e}")
                continue
        
        if not measurements:
            logger.warning("No valid measurements created from external data")
            return []
            
        logger.info(f"Created {len(measurements)} individual measurements from external data")
        
        # Group measurements by parameter signature (like the optimizer does)
        parameter_groups = {}
        for measurement in measurements:
            # Create parameter signature for grouping
            param_signature = self._get_parameter_signature(measurement.parameters)
            
            if param_signature not in parameter_groups:
                parameter_groups[param_signature] = []
            parameter_groups[param_signature].append(measurement)
        
        logger.info(f"Grouped measurements into {len(parameter_groups)} parameter sets")
        
        # Convert each parameter group to a TrialResult (like experiment.py does)
        trials = []
        for group_idx, (param_signature, group_measurements) in enumerate(parameter_groups.items()):
            if len(group_measurements) == 0:
                continue
                
            try:
                trial_result = self._convert_measurements_to_trial(
                    group_measurements, 
                    target_volume_ml
                )
                if trial_result:
                    trials.append(trial_result)
                    logger.info(f"Created trial with {len(group_measurements)} measurements")
                    
            except Exception as e:
                logger.warning(f"Failed to convert measurement group to trial: {e}")
                continue
        
        logger.info(f"Generated {len(trials)} screening trials from external data")
        return trials[:max_trials]  # Limit to max_trials
    
    def _convert_row_to_measurement(self, row: pd.Series, target_volume_ml: float, 
                                   measurement_id: str) -> Optional[RawMeasurement]:
        """Convert a CSV row (individual measurement) to RawMeasurement object."""
        try:
            # Build hardware parameters from available columns
            hardware_params = {}
            config_params = self.config.get_optimization_parameters()
            
            for param_name in config_params.keys():
                if param_name in row and pd.notna(row[param_name]):
                    param_type = self.config.get_parameter_type(param_name)
                    if param_type == "integer":
                        hardware_params[param_name] = int(round(float(row[param_name])))
                    else:
                        hardware_params[param_name] = float(row[param_name])
            
            hardware = HardwareParameters(parameters=hardware_params)
            
            # Handle overaspirate_vol (calibration parameter)
            overaspirate_vol = 0.004  # Default
            if 'overaspirate_vol' in row and pd.notna(row['overaspirate_vol']):
                overaspirate_vol = float(row['overaspirate_vol'])
            
            calibration = CalibrationParameters(overaspirate_vol=overaspirate_vol)
            
            # Create combined parameters
            parameters = PipettingParameters(
                hardware=hardware,
                calibration=calibration
            )
            
            # Apply volume constraints
            parameters = self.config.apply_volume_constraints(parameters, target_volume_ml)
            
            # Extract measurement data from CSV (new format)
            measured_volume_ml = float(row['measured_volume_ml'])
            duration_s = float(row['measurement_time_s'])
            target_volume_ml = float(row.get('target_volume_ml', target_volume_ml))  # Use parameter if missing
            replicate_id = int(row.get('replicate_number', 0))
            
            # Create RawMeasurement object
            measurement = RawMeasurement(
                measurement_id=measurement_id,
                parameters=parameters,
                target_volume_ml=target_volume_ml,
                measured_volume_ml=measured_volume_ml,
                duration_s=duration_s,
                replicate_id=replicate_id,
                metadata={
                    'source': 'external_data', 
                    'original_timestamp': str(row.get('timestamp', '')),
                    'liquid_type': str(row.get('liquid_type', 'water'))
                }
            )
            
            return measurement
            
        except Exception as e:
            logger.error(f"Failed to convert CSV row to RawMeasurement: {e}")
            return None
    
    def _get_parameter_signature(self, parameters: PipettingParameters) -> str:
        """Create parameter signature for grouping measurements."""
        hardware_dict = parameters.hardware.parameters
        calibration_dict = {'overaspirate_vol': parameters.calibration.overaspirate_vol}
        
        # Combine all parameters
        all_params = {**hardware_dict, **calibration_dict}
        
        # Create sorted signature
        signature_parts = []
        for key in sorted(all_params.keys()):
            signature_parts.append(f"{key}:{all_params[key]}")
        
        return "|".join(signature_parts)
    
    def _convert_measurements_to_trial(self, measurements: List[RawMeasurement], 
                                     target_volume_ml: float) -> Optional[TrialResult]:
        """Convert a group of measurements with same parameters to TrialResult."""
        if not measurements:
            return None
            
        try:
            # Use the parameters from the first measurement (all should be identical)
            parameters = measurements[0].parameters
            
            # Calculate statistics from actual measurements
            measured_volumes = [m.measured_volume_ml for m in measurements]
            durations = [m.duration_s for m in measurements]
            
            mean_volume_ml = np.mean(measured_volumes)
            stdev_volume_ml = np.std(measured_volumes, ddof=1) if len(measured_volumes) > 1 else 0.0
            cv_volume_pct = (stdev_volume_ml / mean_volume_ml) * 100 if mean_volume_ml > 0 else 0.0
            
            deviation_ml = mean_volume_ml - target_volume_ml
            deviation_pct = (deviation_ml / target_volume_ml) * 100
            
            # Create analysis result
            analysis = AdaptiveMeasurementResult(
                target_volume_ml=target_volume_ml,
                num_replicates=len(measurements),
                mean_volume_ml=mean_volume_ml,
                stdev_volume_ml=stdev_volume_ml,
                cv_volume_pct=cv_volume_pct,
                deviation_ml=deviation_ml,
                deviation_pct=deviation_pct,
                absolute_deviation_pct=abs(deviation_pct),
                mean_duration_s=np.mean(durations),
                stdev_duration_s=np.std(durations, ddof=1) if len(durations) > 1 else 0.0,
                min_volume_ml=min(measured_volumes),
                max_volume_ml=max(measured_volumes),
                median_volume_ml=np.median(measured_volumes)
            )
            
            # Evaluate quality
            tolerances = self.config.calculate_tolerances_for_volume(target_volume_ml)
            deviation_ul = abs(analysis.deviation_ml * 1000)
            
            accuracy_good = deviation_ul <= tolerances.accuracy_tolerance_ul
            precision_good = cv_volume_pct <= tolerances.precision_tolerance_pct
            
            if accuracy_good and precision_good:
                overall_quality = "within_tolerance"
            elif accuracy_good or precision_good:
                overall_quality = "partial_tolerance" 
            else:
                overall_quality = "outside_tolerance"
            
            quality = QualityEvaluation(
                accuracy_good=accuracy_good,
                precision_good=precision_good,
                overall_quality=overall_quality,
                accuracy_tolerance_ul=tolerances.accuracy_tolerance_ul,
                precision_tolerance_pct=tolerances.precision_tolerance_pct,
                measured_accuracy_ul=deviation_ul,
                measured_precision_pct=cv_volume_pct,
                measured_time_s=analysis.mean_duration_s
            )
            
            # Calculate composite score
            weights = self.config.get_objective_weights()
            accuracy_score = abs(deviation_pct) / 10.0
            precision_score = cv_volume_pct / 10.0
            time_score = analysis.mean_duration_s / 30.0
            
            composite_score = (weights.accuracy_weight * accuracy_score +
                             weights.precision_weight * precision_score +
                             weights.time_weight * time_score)
            
            # Create TrialResult
            trial = TrialResult(
                parameters=parameters,
                target_volume_ml=target_volume_ml,
                measurements=measurements,
                analysis=analysis,
                quality=quality,
                composite_score=composite_score,
                tolerances_used=tolerances,
                strategy="screening",  # Mark as screening data
                needs_additional_replicates=False,
                metadata={'source': 'external_data'}
            )
            
            return trial
            
        except Exception as e:
            logger.error(f"Failed to convert measurements to trial: {e}")
            return None