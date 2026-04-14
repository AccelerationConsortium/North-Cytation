#!/usr/bin/env python3
"""
Calibration Test GUI - PySide6
==============================

Standalone GUI for testing North Robot pipetting parameters.
Provides real-time feedback on pipetting accuracy and precision.

Features:
- Parameter testing with min/max ranges
- Multiple replicates for precision analysis
- Real-time mass-time plotting
- Volume accuracy and CV calculations
- Integration with North Robot calibration protocols

Usage:
    python calibration_test_gui.py
"""

import sys
import os
import csv
import time
import logging
import yaml  # TODO: Will need this when optimization config format is fixed
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback

# Add parent directory to path for North Robot imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QGridLayout, QPushButton, QLabel, QMessageBox,
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QGroupBox, QFrame, QScrollArea, QFormLayout,
    QSplitter, QProgressBar, QTextEdit, QCheckBox,
    QDialog, QDialogButtonBox, QInputDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QFileSystemWatcher
# Note: Qt.ConnectionType is available through Qt import
from PySide6.QtGui import QFont, QColor, QPalette

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')  # Use Qt backend
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, using placeholder plots")

# Import pandas for CSV reading
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available for mass data loading")

# Import calibration system components
try:
    from calibration_protocol_northrobot import HardwareCalibrationProtocol, LIQUIDS
    from config_manager import ExperimentConfig
    from data_structures import PipettingParameters, CalibrationParameters, HardwareParameters
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Calibration components not available: {e}")
    CALIBRATION_AVAILABLE = False

# Default pipetting parameters with common ranges
DEFAULT_PARAMETERS = {
    'overaspirate_vol': {'value': 0.000, 'min': 0.0, 'max': 0.050, 'unit': 'mL', 'type': 'float'},
    'aspirate_speed': {'value': 10, 'min': 1, 'max': 40, 'unit': '', 'type': 'int'},
    'dispense_speed': {'value': 10, 'min': 1, 'max': 40, 'unit': '', 'type': 'int'},
    'aspirate_wait_time': {'value': 0.0, 'min': 0.0, 'max': 30.0, 'unit': 's', 'type': 'float'},
    'dispense_wait_time': {'value': 0.0, 'min': 0.0, 'max': 30.0, 'unit': 's', 'type': 'float'},
    'pre_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.500, 'unit': 'mL', 'type': 'float'},
    'post_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.100, 'unit': 'mL', 'type': 'float'},
    'blowout_vol': {'value': 0.0, 'min': 0.0, 'max': 1.000, 'unit': 'mL', 'type': 'float'},
    'retract_speed': {'value': 5, 'min': 1, 'max': 50, 'unit': '', 'type': 'int'},
    'post_retract_wait_time': {'value': 0.0, 'min': 0.0, 'max': 10.0, 'unit': 's', 'type': 'float'}
}

# Available vials - loaded from status file
AVAILABLE_VIALS = []

def load_available_vials():
    """Load actual vial names from robot status file."""
    try:
        import pandas as pd
        vial_file = "status/calibration_vials_short.csv"
        if Path(vial_file).exists():
            df = pd.read_csv(vial_file)
            vial_names = df['vial_name'].dropna().tolist()
            print(f"Loaded {len(vial_names)} vials from {vial_file}: {vial_names[:5]}...")
            return vial_names
    except Exception as e:
        print(f"Warning: Could not load vials from status file: {e}")
    
    # Fallback vials if file can't be loaded
    return None

# Load vials at startup
AVAILABLE_VIALS = load_available_vials()

class ParameterWidget(QFrame):
    """Widget for editing a single parameter with value/min/max."""
    
    def __init__(self, name: str, config: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.name = name
        self.config = config
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()  # Changed to vertical layout to accommodate speed reminder
        
        # Main parameter row
        param_layout = QHBoxLayout()
        
        # Parameter name label
        name_label = QLabel(f"{self.name}:")
        name_label.setMinimumWidth(150)
        param_layout.addWidget(name_label)
        
        # Determine if this is an integer parameter
        is_int = self.config.get('type', 'float') == 'int'
        unit_suffix = f" {self.config['unit']}" if self.config['unit'] else ""
        
        # Value spinbox
        if is_int:
            self.value_spinbox = QSpinBox()
        else:
            self.value_spinbox = QDoubleSpinBox()
            self.value_spinbox.setDecimals(4)
        
        self.value_spinbox.setMinimum(int(self.config['min']) if is_int else self.config['min'])
        self.value_spinbox.setMaximum(int(self.config['max']) if is_int else self.config['max'])
        self.value_spinbox.setValue(int(self.config['value']) if is_int else self.config['value'])
        if unit_suffix:
            self.value_spinbox.setSuffix(unit_suffix)
        param_layout.addWidget(self.value_spinbox)
        
        # Min value label (changed from editable spinbox to read-only label)
        min_label = QLabel("Min:")
        param_layout.addWidget(min_label)
        
        min_value = int(self.config['min']) if is_int else self.config['min']
        min_display_text = f"{min_value}{unit_suffix}"
        self.min_display = QLabel(min_display_text)
        self.min_display.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 3px; }")
        self.min_display.setMinimumWidth(60)
        param_layout.addWidget(self.min_display)
        
        # Max value label (changed from editable spinbox to read-only label)
        max_label = QLabel("Max:")
        param_layout.addWidget(max_label)
        
        max_value = int(self.config['max']) if is_int else self.config['max']
        max_display_text = f"{max_value}{unit_suffix}"
        self.max_display = QLabel(max_display_text)
        self.max_display.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 3px; }")
        self.max_display.setMinimumWidth(60)
        param_layout.addWidget(self.max_display)
        
        # Current value from optimizer (colorful, read-only)
        current_label = QLabel("Current:")
        param_layout.addWidget(current_label)
        
        self.current_display = QLabel("--")
        self.current_display.setStyleSheet("""
            QLabel { 
                background-color: #FFF4E6; 
                border: 1px solid #D2691E; 
                padding: 3px; 
                font-weight: bold; 
                color: #8B4513;
                border-radius: 3px;
            }
        """)
        self.current_display.setMinimumWidth(80)
        self.current_display.setToolTip("Current value being tested by optimizer")
        param_layout.addWidget(self.current_display)
        
        layout.addLayout(param_layout)
        
        # Add speed inversion reminder only for aspirate/dispense speeds (not retract)
        if self.name.lower() in ['aspirate_speed', 'dispense_speed']:
            speed_reminder = QLabel("⚠ Speed Scale: 1 = Fast, 40 = Slow")
            speed_reminder.setStyleSheet("QLabel { color: #666; font-size: 11px; font-style: italic; margin-left: 155px; }")
            layout.addWidget(speed_reminder)
        
        self.setLayout(layout)
    
    def get_values(self) -> Tuple[float, float, float]:
        """Return (value, min, max)."""
        return (
            self.value_spinbox.value(),
            self.config['min'],  # Return original config values since min/max are now read-only
            self.config['max']   # Return original config values since min/max are now read-only
        )
    
    def get_parameter_value(self) -> float:
        """Get just the current parameter value."""
        return self.value_spinbox.value()
    
    def update_current_value(self, value: float):
        """Update the current value display with optimizer value."""
        unit_suffix = f" {self.config['unit']}" if self.config['unit'] else ""
        if self.config.get('type', 'float') == 'int':
            display_text = f"{int(value)}{unit_suffix}"
        else:
            display_text = f"{value:.4f}{unit_suffix}"
        self.current_display.setText(display_text)

class AddParameterDialog(QDialog):
    """Dialog for adding custom parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Parameter")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Form layout for parameter details
        form_layout = QFormLayout()
        
        # Parameter name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., custom_wait_time")
        form_layout.addRow("Name:", self.name_edit)
        
        # Parameter type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["float", "int"])
        form_layout.addRow("Type:", self.type_combo)
        
        # Value
        self.value_spinbox = QDoubleSpinBox()
        self.value_spinbox.setDecimals(4)
        self.value_spinbox.setRange(-999999, 999999)
        form_layout.addRow("Value:", self.value_spinbox)
        
        # Minimum
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setDecimals(4)
        self.min_spinbox.setRange(-999999, 999999)
        form_layout.addRow("Minimum:", self.min_spinbox)
        
        # Maximum  
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setDecimals(4)
        self.max_spinbox.setRange(-999999, 999999)
        self.max_spinbox.setValue(10.0)
        form_layout.addRow("Maximum:", self.max_spinbox)
        
        # Unit
        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("e.g., mL, s, cycles (leave empty for unitless)")
        form_layout.addRow("Unit:", self.unit_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_parameter_info(self):
        """Get parameter name and config dict."""
        name = self.name_edit.text().strip()
        if not name:
            return None, None
            
        config = {
            'value': self.value_spinbox.value(),
            'min': self.min_spinbox.value(),
            'max': self.max_spinbox.value(),
            'unit': self.unit_edit.text().strip(),
            'type': self.type_combo.currentText()
        }
        
        return name, config

class OptimizationPlotWidget(QWidget):
    """Widget for displaying optimization progress in real-time."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.optimization_data = []  # List of measurement dictionaries
        self.strategy_colors = {
            'screening': '#1f77b4',      # Blue
            'optimization': '#ff7f0e',   # Orange  
            'calibration': '#2ca02c',    # Green
            'validation': '#d62728',     # Red
            'other': '#9467bd'           # Purple
        }
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Optimization Progress")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(8, 4))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            placeholder = QLabel("Matplotlib not available - optimization plot disabled")
            placeholder.setStyleSheet("color: red; text-align: center;")
            layout.addWidget(placeholder)
            
        self.setLayout(layout)
        
    def clear(self):
        """Clear optimization data and plot."""
        self.optimization_data = []
        if MATPLOTLIB_AVAILABLE:
            self.figure.clear()
            self.canvas.draw()
            
    def add_measurement(self, measurement_data: dict):
        """Add new measurement data and update plot."""
        # DEBUG: Log what time data we're receiving
        duration_s = measurement_data.get('duration_s', 0)
        elapsed_s = measurement_data.get('elapsed_s', 0)
        print(f"DEBUG OptimizationPlot: Adding measurement with duration_s={duration_s}, elapsed_s={elapsed_s}")
        print(f"DEBUG OptimizationPlot: Full measurement_data keys: {list(measurement_data.keys())}")
        
        self.optimization_data.append(measurement_data)
        self.update_plot()
        
    def update_plot(self):
        """Update the optimization plot with current data."""
        if not MATPLOTLIB_AVAILABLE or not self.optimization_data:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Group data by parameter combinations instead of measurement count
        parameter_sets = self._group_by_parameter_combinations()
        
        # Create dual Y-axis plot
        ax2 = ax.twinx()  # Create second Y-axis for time data
        
        # Plot volume data (left Y-axis)
        volume_handles = []
        for strategy in self.strategy_colors.keys():
            x_values = []
            y_values = []
            
            for param_set_idx, measurements in parameter_sets.items():
                strategy_measurements = [m for m in measurements if m.get('strategy', 'other') == strategy]
                for measurement in strategy_measurements:
                    x_values.append(param_set_idx)
                    y_values.append(measurement['measured_volume_ul'])
            
            if x_values:  # Only plot if we have data for this strategy
                color = self.strategy_colors.get(strategy, self.strategy_colors['other'])
                handle = ax.scatter(x_values, y_values, c=color, label=f'{strategy.title()} (Volume)', 
                                  alpha=0.7, s=50, marker='o')
                volume_handles.append(handle)
        
        # Plot time data (right Y-axis) - use different markers and X-offset for visibility
        time_handles = []
        for strategy in self.strategy_colors.keys():
            x_values = []
            time_values = []
            
            for param_set_idx, measurements in parameter_sets.items():
                strategy_measurements = [m for m in measurements if m.get('strategy', 'other') == strategy]
                for measurement in strategy_measurements:
                    # Slight X-offset so time points don't overlay volume points exactly
                    x_values.append(param_set_idx + 0.1)  # Offset time points slightly right
                    # Get time from duration_s or elapsed_s
                    time_s = measurement.get('duration_s', measurement.get('elapsed_s', 0))
                    print(f"DEBUG: Processing measurement for plotting - duration_s: {measurement.get('duration_s')}, elapsed_s: {measurement.get('elapsed_s')}, final time_s: {time_s}")
                    time_values.append(time_s)
            
            if x_values and time_values:  # Only plot if we have data for this strategy
                color = self.strategy_colors.get(strategy, self.strategy_colors['other'])
                print(f"DEBUG: Plotting time values for {strategy}: {time_values}")
                # Use squares with different color and size to make them clearly distinct
                handle = ax2.scatter(x_values, time_values, c='red', label=f'{strategy.title()} (Time)', 
                                   alpha=0.8, s=60, marker='s', edgecolors='darkred', linewidth=1.5)
                time_handles.append(handle)
        
        # Add target line for volume (left Y-axis)
        if self.optimization_data:
            target_ul = self.optimization_data[0]['target_volume_ml'] * 1000
            ax.axhline(y=target_ul, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Target: {target_ul:.0f} μL')
        
        # Configure axes
        ax.set_xlabel('Parameter Combination')
        ax.set_ylabel('Volume Measured (μL)', color='black')
        ax2.set_ylabel('Time (seconds)', color='#666666')
        ax.set_title('Optimization Progress - Volume & Time vs Parameter Set')
        
        # Dynamic Y-axis limits based on actual data (not fixed multipliers)
        if self.optimization_data:
            all_volumes = [d['measured_volume_ul'] for d in self.optimization_data]
            all_times = [d.get('duration_s', d.get('elapsed_s', 0)) for d in self.optimization_data]
            
            # Volume axis - dynamic with 10% padding
            vol_range = max(all_volumes) - min(all_volumes)
            vol_padding = vol_range * 0.1 if vol_range > 0 else 5  # 5 μL minimum padding
            ax.set_ylim(min(all_volumes) - vol_padding, max(all_volumes) + vol_padding)
            
            # Time axis - dynamic with 10% padding
            if all_times and max(all_times) > 0:
                time_range = max(all_times) - min(all_times)
                time_padding = time_range * 0.1 if time_range > 0 else 1  # 1 second minimum padding
                ax2.set_ylim(min(all_times) - time_padding, max(all_times) + time_padding)
            
        # Set x-axis to show integer parameter set numbers
        if parameter_sets:
            ax.set_xlim(0.5, max(parameter_sets.keys()) + 0.5)
            ax.set_xticks(list(parameter_sets.keys()))
            
        # Color the right Y-axis labels
        ax2.tick_params(axis='y', labelcolor='#666666')
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1))
        
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _group_by_parameter_combinations(self):
        """Group measurements by unique parameter combinations.
        
        Returns:
            Dict[int, List[dict]]: Parameter set number -> list of measurements with those parameters
        """
        if not self.optimization_data:
            return {}
            
        parameter_groups = {}
        current_set = 1
        
        for measurement in self.optimization_data:
            # Create parameter signature from measurement data 
            # Look for parameter columns that would be in the CSV after column 7
            param_keys = ['overaspirate_vol', 'aspirate_speed', 'dispense_speed', 'air_gap_vol', 'pre_dispense_vol']
            
            # Create a signature tuple from available parameters
            param_signature = tuple([
                measurement.get(key, 0) for key in param_keys
            ])
            
            # Find existing group with same parameters or create new one
            found_group = None
            for group_num, measurements in parameter_groups.items():
                if measurements and len(measurements) > 0:
                    # Compare with first measurement in this group
                    first_measurement = measurements[0]
                    first_signature = tuple([
                        first_measurement.get(key, 0) for key in param_keys  
                    ])
                    if param_signature == first_signature:
                        found_group = group_num
                        break
            
            if found_group is not None:
                parameter_groups[found_group].append(measurement)
            else:
                parameter_groups[current_set] = [measurement]
                current_set += 1
                
        return parameter_groups

class OptimizationWorker(QThread):
    """Worker thread for running optimization and monitoring progress."""
    
    optimization_started = Signal(str)  # output_dir
    measurement_update = Signal(dict)    # measurement data
    optimization_complete = Signal(dict) # final results
    optimization_error = Signal(str)     # error message
    debug_message = Signal(str)          # debug messages for GUI status area
    
    def __init__(self, config_dict: dict, parent=None, test_mode=False):
        super().__init__(parent)
        self.config_dict = config_dict
        self.output_dir = None
        self.emergency_file = None
        self.last_row_count = 0
        self.process = None
        self.file_watcher = None
        self.check_timer = None
        self.test_mode = test_mode  # NEW: Simple threading test mode
        
    def run(self):
        """Run optimization in background and monitor progress."""
        try:
            # TEST MODE: Create mock data for debugging threading
            if self.test_mode:
                self.run_test_mode()
                return
                
            # Normal mode continues...
            # Generate config file
            config_path = self.create_config_file()
            
            # Start calibration process
            self.start_calibration_process(config_path)
            
            # Monitor for output directory creation
            self.wait_for_output_dir()
            
            # Setup file monitoring
            # NOTE: File monitoring now handled in main GUI thread via on_optimization_started()
            # Don't setup worker thread monitoring here
            
            # SAFETY: Monitor process with reasonable timeout for calibration campaigns
            start_time = time.time()
            max_runtime = 4 * 60 * 60  # 4 hours - safe for 100+ measurements
            timeout_check_interval = 30  # Check every 30 seconds
            
            self.debug_message.emit(f"[DEBUG] Calibration started with {max_runtime//3600} hour timeout")
            
            # Wait for process to complete with timeout monitoring  
            while self.process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > max_runtime:
                    self.debug_message.emit(f"[DEBUG] Process timeout after {elapsed//3600:.1f} hours - terminating")
                    self.process.terminate()
                    time.sleep(5)  # Give more time for graceful termination
                    if self.process.poll() is None:
                        self.debug_message.emit("[DEBUG] Process didn't terminate gracefully, killing it")
                        self.process.kill()
                    self.optimization_error.emit(f"Calibration process timed out after {max_runtime//3600} hours")
                    return
                time.sleep(timeout_check_interval)  # Check every 30 seconds instead of 2
            
            if self.process.returncode == 0:
                self.optimization_complete.emit({"status": "success"})
            else:
                # Since we're not capturing output, just show return code
                error_msg = f"Calibration process failed with return code {self.process.returncode}\n"
                error_msg += "Check terminal output above for error details."
                self.optimization_error.emit(error_msg)
                
        except Exception as e:
            self.optimization_error.emit(f"Error during optimization: {str(e)}")
    
    def run_test_mode(self):
        """TEST MODE: Simulate optimization for debugging threading."""
        import random
        
        self.debug_message.emit("[TEST] Starting test mode - no subprocess")
        
        # Create mock output directory
        test_output = Path("calibration_modular_v2/output/test_run_debug")
        test_output.mkdir(parents=True, exist_ok=True)
        self.output_dir = test_output
        
        # Create mock emergency file with header
        self.emergency_file = test_output / "emergency_raw_measurements.csv"
        with open(self.emergency_file, 'w') as f:
            f.write("timestamp,liquid_type,volume_ml,measured_volume_ul,deviation_pct,overaspirate_vol,strategy,total_measurement_count\n")
        
        self.debug_message.emit(f"[TEST] Created mock files in: {test_output}")
        
        # Emit optimization_started signal to test connection
        self.debug_message.emit("[TEST] About to emit optimization_started signal...")
        self.optimization_started.emit(str(self.output_dir))
        self.debug_message.emit("[TEST] Signal emitted!")
        
        # Simulate measurements being added over time
        for i in range(5):
            time.sleep(3)  # Wait 3 seconds between measurements
            
            # Add fake measurement to CSV
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            measured_ul = 200 + random.uniform(-10, 10)
            deviation = abs(measured_ul - 200) / 200 * 100
            
            with open(self.emergency_file, 'a') as f:
                f.write(f"{timestamp},PVA_water,0.200,{measured_ul:.1f},{deviation:.2f},0.008,test_strategy,{i+1}\n")
            
            self.debug_message.emit(f"[TEST] Added fake measurement {i+1}: {measured_ul:.1f}uL")
        
        self.debug_message.emit("[TEST] Test mode complete!")
        self.optimization_complete.emit({"status": "test_success"})
            
    def create_config_file(self) -> str:
        """Create experiment_config.yaml from GUI parameters."""
        try:
            config_path = "calibration_modular_v2/experiment_config_gui.yaml"
            
            # Create config structure based on GUI parameters
            config = {
                'experiment': {
                    'liquid': self.config_dict['liquid'],
                    'volume_targets_ml': [self.config_dict['target_volume_ml']],
                    'simulate': self.config_dict.get('simulate', False),
                    'max_measurements_per_volume': 50,  # Reasonable default
                    'max_total_measurements': 200,     # Reasonable default
                },
                'hardware_parameters': {},
                'optimization': {
                    'enabled': True,
                    'strategy': 'qNEHVI',
                    'min_good_trials': 3
                }
            }
            
            # Add hardware parameters from GUI
            for param_name, value in self.config_dict['parameters'].items():
                if param_name == 'overaspirate_vol':
                    continue  # This goes in calibration section
                config['hardware_parameters'][param_name] = {
                    'bounds': [value * 0.5, value * 1.5],  # Simple bounds around current value
                    'type': 'continuous' if isinstance(value, float) else 'discrete'
                }
                
            # Write config file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            return config_path
        except Exception as e:
            raise
        
    def start_calibration_process(self, config_path: str):
        """Start the calibration subprocess."""
        try:
            # For now, use the existing experiment_config.yaml instead of GUI-generated config
            # TODO: Later improve GUI config generation to include all required sections
            cmd = [sys.executable, "calibration_modular_v2/run_calibration.py"]
            cwd = str(Path(__file__).parent.parent)
            
            # Remove PIPE so output shows in terminal
            self.process = subprocess.Popen(cmd, cwd=cwd)
        except Exception as e:
            raise
        
    def wait_for_output_dir(self):
        """Wait for NEW output directory to be created by the subprocess."""
        # Monitor calibration_modular_v2/output for run directories
        output_base = Path(__file__).parent / "output"
        self.debug_message.emit(f"[DEBUG] Looking for NEW output directory in: {output_base}")
        
        # Record current time as baseline - only accept directories created AFTER this
        start_time = time.time()
        self.debug_message.emit(f"[DEBUG] Subprocess started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        
        # Wait up to 60 seconds for NEW directory creation (increased from 30)
        for i in range(60):
            self.debug_message.emit(f"[DEBUG] Checking for NEW output directory, attempt {i+1}/60")
            if output_base.exists():
                # Find run directories created AFTER subprocess start
                dirs = [d for d in output_base.iterdir() 
                       if (d.is_dir() and 
                           d.name.startswith("run_") and 
                           d.stat().st_ctime > start_time)]
                           
                self.debug_message.emit(f"[DEBUG] Found {len(dirs)} NEW run directories created after {time.strftime('%H:%M:%S', time.localtime(start_time))}")
                
                if dirs:
                    # Pick newest of the NEW directories
                    self.output_dir = max(dirs, key=lambda x: x.stat().st_ctime) 
                    self.emergency_file = self.output_dir / "emergency_raw_measurements.csv"
                    created_time = time.strftime('%H:%M:%S', time.localtime(self.output_dir.stat().st_ctime))
                    self.debug_message.emit(f"[DEBUG] Selected NEW output dir: {self.output_dir.name} (created at {created_time})")
                    self.debug_message.emit(f"[DEBUG] Emergency file path: {self.emergency_file.name}")
                    self.debug_message.emit(f"[SIGNAL] About to emit optimization_started signal with: {str(self.output_dir)}")
                    self.optimization_started.emit(str(self.output_dir))
                    self.debug_message.emit(f"[SIGNAL] Signal emitted successfully")
                    return
            else:
                self.debug_message.emit(f"[DEBUG] Output base doesn't exist yet")
            time.sleep(1)
            
        raise Exception("NEW output directory not created within timeout")
        
    def setup_file_monitoring(self):
        """Setup file watcher for emergency measurements file."""
        self.debug_message.emit(f"[DEBUG] Setting up file monitoring for: {self.emergency_file.name}")
        self.file_watcher = QFileSystemWatcher()
        
        # Watch the directory (file might not exist yet)
        self.debug_message.emit(f"[DEBUG] Adding directory to file watcher: {self.output_dir.name}")
        self.file_watcher.directoryChanged.connect(self.check_file_updates)
        self.file_watcher.addPath(str(self.output_dir))
        
        # Start periodic checks as backup
        self.debug_message.emit("[DEBUG] Starting periodic file check timer (every 2 seconds)")
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_file_updates)
        self.check_timer.start(2000)  # Check every 2 seconds
        
        # Test timer immediately
        self.debug_message.emit(f"[DEBUG] FULL PATH being monitored: {self.emergency_file}")
        
        # Check if file exists right now
        if self.emergency_file.exists():
            lines = len(open(self.emergency_file).readlines())
            self.debug_message.emit(f"[DEBUG] File EXISTS with {lines} lines!")
        else:
            self.debug_message.emit(f"[DEBUG] File does NOT exist at: {self.emergency_file}")
        
    def check_file_updates(self):
        """Check for updates to emergency measurements file."""
        # Always announce we're checking (to confirm timer works)
        check_time = time.strftime("%H:%M:%S")
        self.debug_message.emit(f"[DEBUG] File check at {check_time}")
        
        if not self.emergency_file:
            self.debug_message.emit("[DEBUG] No emergency_file set yet")
            return
            
        if not self.emergency_file.exists():
            self.debug_message.emit(f"[DEBUG] Emergency file doesn't exist yet: {self.emergency_file}")
            return
            
        self.debug_message.emit("[DEBUG] Emergency file EXISTS! Checking for new data...")
        
        try:
            # Simple approach: count lines to detect new rows
            with open(self.emergency_file, 'r') as f:
                lines = f.readlines()
                
            current_row_count = len(lines) - 1  # Subtract header
            
            if current_row_count > self.last_row_count:
                new_count = current_row_count - self.last_row_count
                self.debug_message.emit(f"[DEBUG] Found {new_count} new measurements! Processing...")
                
                # New measurement(s) added
                new_rows = lines[self.last_row_count + 1:]  # Skip header and previously processed
                
                for i, row in enumerate(new_rows):
                    if row.strip():  # Skip empty lines
                        measurement_data = self.parse_csv_row(row.strip())
                        if measurement_data:
                            vol_ul = measurement_data.get('measured_volume_ul', 0)
                            count = measurement_data.get('total_measurement_count', 0)
                            self.debug_message.emit(f"[DEBUG] Processing measurement #{count}: {vol_ul:.1f}uL")
                            self.measurement_update.emit(measurement_data)
                        else:
                            self.debug_message.emit(f"[DEBUG] Failed to parse CSV row")
                            
                self.last_row_count = current_row_count
                
        except Exception as e:
            self.debug_message.emit(f"[DEBUG] Error reading file: {str(e)}")
            
    def parse_csv_row(self, row: str) -> Optional[dict]:
        """Parse a CSV row into measurement data dictionary."""
        try:
            # Split by comma (simple parsing - assumes no commas in values)
            parts = row.split(',')
            if len(parts) < 8:
                return None
                
            # Map to expected columns (based on emergency file format)
            return {
                'timestamp': parts[0],
                'target_volume_ml': float(parts[1]),
                'measured_volume_ml': float(parts[2]),
                'measured_volume_ul': float(parts[3]),
                'deviation_pct': float(parts[4]),
                'duration_s': float(parts[5]),
                'strategy': parts[6],
                'total_measurement_count': int(parts[7])
            }
        except (ValueError, IndexError):
            return None
    
    def cleanup_and_terminate(self):
        """Clean up resources and terminate subprocess."""
        self.debug_message.emit("[DEBUG] Starting OptimizationWorker cleanup...")
        
        # Stop file monitoring timer
        if self.check_timer:
            self.check_timer.stop()
            self.debug_message.emit("[DEBUG] Stopped file monitoring timer")
        
        # Stop file watcher
        if self.file_watcher:
            self.file_watcher.deleteLater()
            self.file_watcher = None
            self.debug_message.emit("[DEBUG] Stopped file watcher")
        
        # Terminate subprocess if still running
        if self.process and self.process.poll() is None:
            self.debug_message.emit("[DEBUG] Terminating calibration subprocess...")
            self.process.terminate()
            # Give it 3 seconds to terminate gracefully
            try:
                self.process.wait(timeout=3)
                self.debug_message.emit("[DEBUG] Subprocess terminated gracefully")
            except subprocess.TimeoutExpired:
                self.debug_message.emit("[DEBUG] Subprocess timeout - killing it")
                self.process.kill()
                self.process.wait()
        
        # Stop this thread
        if self.isRunning():
            self.quit()
            self.wait(2000)  # Wait up to 2 seconds for thread to finish
            self.debug_message.emit("[DEBUG] OptimizationWorker thread stopped")

class MeasurementWorker(QThread):
    """Worker thread for running measurements to avoid GUI freezing."""
    
    measurement_complete = Signal(dict)  # Results
    measurement_error = Signal(str)     # Error message
    progress_update = Signal(str)       # Status message
    replicate_complete = Signal(int, str, dict)  # replicate_num, vial_name, measurement_data
    
    def __init__(self, vial_name: str, volume_ml: float, replicates: int, 
                 parameters: Dict[str, float], protocol, protocol_state, parent=None):
        super().__init__(parent)
        self.vial_name = vial_name
        self.volume_ml = volume_ml
        self.replicates = replicates
        self.parameters = parameters
        self.protocol = protocol  # Use existing initialized protocol
        self.protocol_state = protocol_state  # Use existing state
    
    def run(self):
        """Execute measurements in background thread."""
        try:
            self.progress_update.emit("Using initialized protocol for measurements...")
            
            if not CALIBRATION_AVAILABLE:
                raise Exception("Calibration system not available")
            
            if not self.protocol or not self.protocol_state:
                raise Exception("Protocol not properly initialized")
            
            results = []
            for rep in range(self.replicates):
                self.progress_update.emit(f"Running replicate {rep+1}/{self.replicates}...")
                
                # Create parameters structure expected by protocol
                # Convert asp_disp_cycles to integer (must be int for range() function)
                params_for_protocol = {k: v for k, v in self.parameters.items() if k != 'overaspirate_vol'}
                if 'asp_disp_cycles' in params_for_protocol:
                    params_for_protocol['asp_disp_cycles'] = int(params_for_protocol['asp_disp_cycles'])
                
                pipetting_params = {
                    'overaspirate_vol': self.parameters.get('overaspirate_vol', 0.004),
                    'parameters': params_for_protocol
                }
                
                # Execute measurement using existing protocol (no re-initialization)
                measurement_list = self.protocol.measure(self.protocol_state, self.volume_ml, pipetting_params, replicates=1)
                measurement = measurement_list[0]  # Single replicate
                
                results.append(measurement)
                
                # Signal that replicate is complete for immediate data collection
                self.replicate_complete.emit(rep + 1, self.vial_name, measurement)
                
                self.progress_update.emit(f"Completed replicate {rep+1}: {measurement['volume']*1000:.1f} μL")
            
            # Calculate statistics
            volumes_ml = [r['volume'] for r in results]
            times_s = [r['elapsed_s'] for r in results]
            
            mean_volume = np.mean(volumes_ml)
            mean_time = np.mean(times_s)
            
            # Calculate accuracy (% deviation from target)
            accuracy_pct = ((mean_volume - self.volume_ml) / self.volume_ml) * 100
            
            # Calculate precision (coefficient of variation)
            cv_pct = (np.std(volumes_ml) / mean_volume) * 100 if mean_volume > 0 else 0
            
            # Package results
            summary = {
                'raw_measurements': results,
                'mean_volume_ml': mean_volume,
                'mean_time_s': mean_time,
                'accuracy_pct': accuracy_pct,
                'cv_pct': cv_pct,
                'target_volume_ml': self.volume_ml,
                'volumes_ml': volumes_ml,
                'times_s': times_s
            }
            
            self.measurement_complete.emit(summary)
            
        except Exception as e:
            error_msg = f"Measurement failed: {str(e)}\n{traceback.format_exc()}"
            self.measurement_error.emit(error_msg)

class PlotWidget(QFrame):
    """Matplotlib plot widget for embedding in Qt."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.setFrameStyle(QFrame.Box)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(6, 4))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            # Placeholder when matplotlib not available
            placeholder = QLabel(f"Plot Placeholder: {self.title}")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: lightgray; border: 1px dashed gray;")
            placeholder.setMinimumHeight(200)
            layout.addWidget(placeholder)
        
        self.setLayout(layout)
    
    def plot_mass_time(self, results: Dict[str, Any]):
        """Plot mass vs time for all replicates (placeholder for now)."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Placeholder - simulate mass-time data
        for i, measurement in enumerate(results['raw_measurements']):
            time_points = np.linspace(0, measurement['elapsed_s'], 50)
            # Simulate mass increasing over time
            mass_points = np.random.normal(0, 0.001, 50) + time_points * (measurement['volume'] / measurement['elapsed_s'])
            ax.plot(time_points, mass_points, label=f'Replicate {i+1}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mass (g)')
        ax.set_title('Mass vs Time (Simulated)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_individual_measurements(self, results: Dict[str, Any]):
        """Plot individual measurement points as strip plot."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        volumes_ul = np.array(results['volumes_ml']) * 1000  # Convert to μL
        target_ul = results['target_volume_ml'] * 1000
        
        # Create strip plot - each measurement as individual point
        n_measurements = len(volumes_ul)
        y_positions = np.ones(n_measurements)  # All points at y=1
        
        # Add small random jitter to prevent overlapping points
        np.random.seed(42)  # Consistent jitter
        jitter = np.random.normal(0, 0.02, n_measurements)
        y_jitter = y_positions + jitter
        
        # Plot individual measurements
        scatter = ax.scatter(volumes_ul, y_jitter, s=80, alpha=0.7, 
                           c=range(n_measurements), cmap='viridis', 
                           edgecolors='black', linewidth=1, 
                           label=f'Measurements (n={n_measurements})')
        
        # Add target line
        ax.axvline(target_ul, color='red', linestyle='--', linewidth=2, 
                  label=f'Target: {target_ul:.1f} μL')
        
        # Add mean line
        mean_ul = np.mean(volumes_ul)
        ax.axvline(mean_ul, color='green', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean_ul:.1f} μL')
        
        ax.set_xlabel('Volume (μL)')
        ax.set_ylabel('')
        ax.set_ylim(0.5, 1.8)
        ax.set_yticks([])  # Hide y-axis ticks since they're not meaningful
        ax.set_title('Individual Measurement Values')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')  # Only x-axis grid
        
        self.canvas.draw()

class CalibrationTestMainWindow(QMainWindow):
    """Main window for calibration testing GUI."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize attributes BEFORE setup_ui() since they're needed during setup
        self.parameter_widgets = {}
        self.measurement_worker = None
        self.optimization_worker = None
        self.current_results = None
        
        # Protocol state - persistent between measurements
        self.protocol = None
        self.protocol_state = None
        self.protocol_initialized = False
        
        # Session management for data persistence
        self.session_folder = None
        self.raw_data_csv = None
        self.summary_data_csv = None
        self.current_session_id = None
        
        # Progressive data for optimization tracking
        self.optimization_measurements = []     # All measurements from optimization
        self.current_trial_measurements = {}    # Group measurements by trial_id
        
        # Progressive data for optimization tracking
        self.optimization_measurements = []     # All measurements from optimization
        self.current_trial_measurements = {}    # Group measurements by trial_id
        
        # Progressive mass data collection
        self.mass_data_replicates = []  # List of DataFrames, one per replicate
        self.volume_data_replicates = []  # List of volumes as they come in
        self.current_target_volume = 0.1  # mL
        
        # Progressive data for optimization tracking
        self.optimization_measurements = []     # All measurements from optimization
        self.current_trial_measurements = {}    # Group measurements by strategy
        
        self.setup_ui()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_session_folder(self):
        """Create unique session folder and initialize CSV files."""
        try:
            # Create unique session folder in main output directory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_session_id = f"calibration_session_{timestamp}"
            
            # Use main output directory (not calibration_modular_v2 subfolder)
            main_output_dir = Path(__file__).parent.parent / "output"
            self.session_folder = main_output_dir / self.current_session_id
            
            # Create session directory
            self.session_folder.mkdir(parents=True, exist_ok=True)
            
            # Initialize CSV file paths
            self.raw_data_csv = self.session_folder / "raw_measurements.csv"
            self.summary_data_csv = self.session_folder / "parameter_sets.csv"
            
            # Create raw data CSV with headers
            self.create_raw_data_csv()
            
            # Create summary data CSV with headers
            self.create_summary_data_csv()
            
            self.logger.info(f"Session folder created: {self.session_folder}")
            
        except Exception as e:
            self.logger.error(f"Error creating session folder: {e}")
            self.session_folder = None
    
    def create_raw_data_csv(self):
        """Create raw measurements CSV with headers."""
        headers = [
            'timestamp', 'measurement_id', 'replicate_number',
            'target_volume_ml', 'measured_volume_ml', 'measured_mass_mg', 'density_g_ml',
            'liquid_type', 'measurement_time_s',
            # Pipetting parameters (using actual GUI parameter names)
            'overaspirate_vol', 'aspirate_speed', 'dispense_speed', 'retract_speed',
            'aspirate_wait_time', 'dispense_wait_time', 'pre_asp_air_vol', 'post_asp_air_vol',
            'blowout_vol', 'post_retract_wait_time',
            # Environmental conditions at time of measurement
            'temperature_c', 'humidity_pct', 'pressure_pa', 'env_timestamp_age_minutes'
        ]
        
        with open(self.raw_data_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def create_summary_data_csv(self):
        """Create parameter sets summary CSV with headers."""
        headers = [
            'timestamp', 'parameter_set_id', 'replicate_count',
            'target_volume_ml', 'mean_volume_ml', 'accuracy_ul', 'accuracy_pct', 
            'precision_ul', 'cv_pct', 'mean_time_s',
            'liquid_type',
            # Pipetting parameters for this set (using actual GUI parameter names)
            'overaspirate_vol', 'aspirate_speed', 'dispense_speed', 'retract_speed',
            'aspirate_wait_time', 'dispense_wait_time', 'pre_asp_air_vol', 'post_asp_air_vol',
            'blowout_vol', 'post_retract_wait_time',
            # Environmental conditions (average during parameter set)
            'avg_temperature_c', 'avg_humidity_pct', 'avg_pressure_pa'
        ]
        
        with open(self.summary_data_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def get_current_environmental_data(self):
        """Get current environmental data for CSV logging."""
        env_data = {
            'temperature_c': None,
            'humidity_pct': None, 
            'pressure_pa': None,
            'env_timestamp_age_minutes': None
        }
        
        try:
            if not PANDAS_AVAILABLE:
                return env_data
                
            mqtt_log_file = "C:\\Users\\Imaging Controller\\Desktop\\m5stack\\mqtt_log.csv"
            if not os.path.exists(mqtt_log_file):
                return env_data
                
            df = pd.read_csv(mqtt_log_file)
            if len(df) == 0:
                return env_data
                
            latest = df.iloc[-1]
            timestamp = pd.to_datetime(latest["Timestamp"])
            age_minutes = (datetime.now() - timestamp.to_pydatetime()).total_seconds() / 60
            
            env_data = {
                'temperature_c': float(latest["sht_temp_c"]) if pd.notna(latest["sht_temp_c"]) else None,
                'humidity_pct': float(latest["sht_rh"]) if pd.notna(latest["sht_rh"]) else None,
                'pressure_pa': float(latest["bmp_pa"]) if pd.notna(latest["bmp_pa"]) else None,
                'env_timestamp_age_minutes': age_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Error reading environmental data: {e}")
            
        return env_data
    
    def get_current_parameters(self):
        """Get current GUI parameter values."""
        params = {}
        for name, widget in self.parameter_widgets.items():
            params[name] = widget.get_parameter_value()
        
        # Get liquid type
        params['liquid_type'] = self.liquid_type_combo.currentText()
        
        return params
    
    def save_measurement_data(self, results: Dict[str, Any]):
        """Save both raw measurements and summary data to CSV files."""
        if not self.session_folder or not results:
            return
            
        try:
            current_params = self.get_current_parameters()
            env_data = self.get_current_environmental_data()
            timestamp = datetime.now().isoformat()
            
            # Generate unique measurement ID
            measurement_id = f"meas_{int(time.time())}"
            
            # Save raw data for each replicate
            if 'volumes_ml' in results and 'raw_measurements' in results:
                volumes = results['volumes_ml']
                raw_measurements = results['raw_measurements']
                replicate_count = len(volumes)
                
                # Build rows for each replicate
                env_rows = []
                for i in range(replicate_count):
                    # Get mass from raw measurement if available
                    mass_mg = None
                    if i < len(raw_measurements) and 'mass' in raw_measurements[i]:
                        mass_mg = raw_measurements[i]['mass'] * 1000  # Convert g to mg
                    
                    # Get individual measurement time
                    individual_time_s = 0
                    if i < len(raw_measurements) and 'elapsed_s' in raw_measurements[i]:
                        individual_time_s = raw_measurements[i]['elapsed_s']
                    
                    raw_row = [
                        timestamp, measurement_id, i + 1,
                        results.get('target_volume_ml', 0.1),
                        volumes[i] if i < len(volumes) else None,
                        mass_mg,
                        results.get('density', 1.0),
                        current_params.get('liquid_type', 'water'),
                        individual_time_s,
                        # Parameters (using correct names)
                        current_params.get('overaspirate_vol', 0),
                        current_params.get('aspirate_speed', 20),
                        current_params.get('dispense_speed', 20), 
                        current_params.get('retract_speed', 20),
                        current_params.get('aspirate_wait_time', 0),
                        current_params.get('dispense_wait_time', 0),
                        current_params.get('pre_asp_air_vol', 0),
                        current_params.get('post_asp_air_vol', 0),
                        current_params.get('blowout_vol', 0),
                        current_params.get('post_retract_wait_time', 0),
                        # Environmental
                        env_data['temperature_c'],
                        env_data['humidity_pct'],
                        env_data['pressure_pa'],
                        env_data['env_timestamp_age_minutes']
                    ]
                    env_rows.append(raw_row)
                
                # Append raw data rows
                with open(self.raw_data_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(env_rows)
            
            # Save summary data for this parameter set
            summary_row = [
                timestamp, measurement_id, len(results.get('volumes_ml', [])),
                results.get('target_volume_ml', 0.1),
                results.get('mean_volume_ml', 0),
                abs(results.get('mean_volume_ml', 0) - results.get('target_volume_ml', 0.1)) * 1000,  # accuracy_ul
                results.get('accuracy_pct', 0),
                results.get('std_volume_ml', 0) * 1000,  # precision_ul
                results.get('cv_pct', 0),
                results.get('mean_time_s', 0),
                current_params.get('liquid_type', 'water'),
                # Parameters (using correct names)
                current_params.get('overaspirate_vol', 0),
                current_params.get('aspirate_speed', 20),
                current_params.get('dispense_speed', 20),
                current_params.get('retract_speed', 20), 
                current_params.get('aspirate_wait_time', 0),
                current_params.get('dispense_wait_time', 0),
                current_params.get('pre_asp_air_vol', 0),
                current_params.get('post_asp_air_vol', 0),
                current_params.get('blowout_vol', 0),
                current_params.get('post_retract_wait_time', 0),
                # Environmental (same as individual measurements since they're taken together)
                env_data['temperature_c'],
                env_data['humidity_pct'],
                env_data['pressure_pa']
            ]
            
            with open(self.summary_data_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(summary_row)
                
            self.logger.info(f"Measurement data saved to session {self.current_session_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving measurement data: {e}")
            self.add_status_message(f"Warning: Could not save measurement data - {e}")
    
    def setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("North Robot Calibration Test GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Central widget with splitter for two columns
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Column A - Input
        input_widget = self.create_input_column()
        splitter.addWidget(input_widget)
        
        # Column B - Output  
        output_widget = self.create_output_column()
        splitter.addWidget(output_widget)
        
        # Set equal sizes for both columns
        splitter.setSizes([700, 700])
    
    def create_input_column(self) -> QWidget:
        """Create the input column with settings and parameters."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input Settings Group
        settings_group = QGroupBox("Input Settings")
        settings_layout = QFormLayout()
        
        # Liquid (vial) selection
        self.liquid_combo = QComboBox()
        self.liquid_combo.addItems(AVAILABLE_VIALS)
        if AVAILABLE_VIALS:  # Set first vial as default
            self.liquid_combo.setCurrentText(AVAILABLE_VIALS[0])
        settings_layout.addRow("Source Vial:", self.liquid_combo)
        
        # Liquid type selection (for density calculations)
        self.liquid_type_combo = QComboBox()
        liquid_types = ["water", "ethanol", "toluene", "heptane", "2MeTHF", "isopropanol", 
                       "DMSO", "acetone", "glycerol", "PEG_Water", "4%_hyaluronic_acid_water", 
                       "agar_water", "agar_water_refill", "TFA", "6M_HCl", "6M_TFA", 
                       "6M_p_TSA", "6M_Citric_Acid", "6M_H2SO4", "6M_H3PO4", "PVA_water"]
        self.liquid_type_combo.addItems(liquid_types)
        self.liquid_type_combo.setCurrentText("water")  # Default to water
        settings_layout.addRow("Liquid Type:", self.liquid_type_combo)
        
        # Volume
        self.volume_spinbox = QDoubleSpinBox()
        self.volume_spinbox.setDecimals(3)
        self.volume_spinbox.setMinimum(0.001)
        self.volume_spinbox.setMaximum(1.000)
        self.volume_spinbox.setValue(0.100)
        self.volume_spinbox.setSuffix(" mL")
        settings_layout.addRow("Target Volume:", self.volume_spinbox)
        
        # Replicates
        self.replicates_spinbox = QSpinBox()
        self.replicates_spinbox.setMinimum(1)
        self.replicates_spinbox.setMaximum(10)
        self.replicates_spinbox.setValue(3)
        settings_layout.addRow("Replicates:", self.replicates_spinbox)
        
        # Simulation mode checkbox
        self.simulate_checkbox = QCheckBox()
        self.simulate_checkbox.setChecked(False)  # Default to hardware mode
        self.simulate_checkbox.setToolTip("Run in simulation mode (no hardware required)")
        settings_layout.addRow("Simulation Mode:", self.simulate_checkbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Pipetting Parameters Group
        self.parameters_group = QGroupBox("Pipetting Parameters")
        self.parameters_layout = QVBoxLayout()
        self.parameters_group.setLayout(self.parameters_layout)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidget(self.parameters_group)
        scroll.setWidgetResizable(True)
        # Remove height constraint to use all available space
        layout.addWidget(scroll)
        
        # Add default parameters
        self.populate_default_parameters()
        
        # Parameter control buttons
        param_button_layout = QHBoxLayout()
        add_param_btn = QPushButton("Add Parameter")
        add_param_btn.clicked.connect(self.show_add_parameter_dialog)
        remove_param_btn = QPushButton("Remove Parameter") 
        remove_param_btn.clicked.connect(self.show_remove_parameter_dialog)
        param_button_layout.addWidget(add_param_btn)
        param_button_layout.addWidget(remove_param_btn)
        layout.addLayout(param_button_layout)
        
        # Robot Control buttons
        robot_button_layout = QHBoxLayout()
        
        self.initialize_btn = QPushButton("INITIALIZE ROBOT")
        self.initialize_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.initialize_btn.clicked.connect(self.initialize_protocol)
        
        self.cleanup_btn = QPushButton("CLEANUP")
        self.cleanup_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.cleanup_btn.clicked.connect(self.cleanup_protocol)
        self.cleanup_btn.setEnabled(False)  # Disabled until initialized
        
        robot_button_layout.addWidget(self.initialize_btn)
        robot_button_layout.addWidget(self.cleanup_btn)
        layout.addLayout(robot_button_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.measure_btn = QPushButton("MEASURE")
        self.measure_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.measure_btn.clicked.connect(self.run_measurement)
        self.measure_btn.setEnabled(False)  # Disabled until robot initialized
        self.measure_btn.setToolTip("Initialize robot first before measuring")
        
        self.optimize_btn = QPushButton("RUN OPTIMIZATION")
        self.optimize_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 10px; }")
        self.optimize_btn.setEnabled(True)  # Now enabled!
        self.optimize_btn.setToolTip("Run Bayesian optimization to find optimal pipetting parameters")
        self.optimize_btn.clicked.connect(self.run_optimization)
        
        # STATUS CHECK button for debugging
        self.status_check_btn = QPushButton("STATUS CHECK")
        self.status_check_btn.setStyleSheet("QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 10px; }")
        self.status_check_btn.clicked.connect(self.check_system_status)
        
        # TEST THREADING button for simple threading debug
        self.test_thread_btn = QPushButton("TEST THREADING")
        self.test_thread_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 10px; }")
        self.test_thread_btn.clicked.connect(self.test_threading)
        
        button_layout.addWidget(self.measure_btn)
        button_layout.addWidget(self.optimize_btn) 
        button_layout.addWidget(self.status_check_btn)
        button_layout.addWidget(self.test_thread_btn)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlaceholderText("Status messages will appear here...")
        layout.addWidget(self.status_text)
        
        # Remove addStretch() to allow parameters section to expand
        return widget
    
    def create_output_column(self) -> QWidget:
        """Create the output column with plots and statistics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Row 1 - Plots
        plots_layout = QHBoxLayout()
        
        self.mass_time_plot = PlotWidget("Mass vs Time")
        self.volume_replicate_plot = PlotWidget("Individual Measurements")
        
        plots_layout.addWidget(self.mass_time_plot)
        plots_layout.addWidget(self.volume_replicate_plot)
        layout.addLayout(plots_layout)
        
        # Row 1b - Optimization Plot (full width)
        self.optimization_plot = OptimizationPlotWidget()
        layout.addWidget(self.optimization_plot)
        
        # Row 2 - Statistics and Environmental Data (horizontal layout)
        summary_layout = QHBoxLayout()
        
        # Results Summary (left side)
        stats_group = QGroupBox("Results Summary")
        stats_layout = QFormLayout()
        
        self.time_label = QLabel("--")
        self.accuracy_label = QLabel("--")
        self.cv_label = QLabel("--")
        
        stats_layout.addRow("Average Time:", self.time_label)
        stats_layout.addRow("Accuracy (% dev):", self.accuracy_label)
        stats_layout.addRow("Precision (CV%):", self.cv_label)
        
        stats_group.setLayout(stats_layout)
        summary_layout.addWidget(stats_group)
        
        # Environmental Monitoring Group (right side)
        env_group = QGroupBox("Environmental Conditions")
        env_layout = QFormLayout()
        
        self.temp_label = QLabel("--°C")
        self.humidity_label = QLabel("--%")  
        self.pressure_label = QLabel("-- Pa")
        self.env_timestamp_label = QLabel("--")
        
        env_layout.addRow("Temperature:", self.temp_label)
        env_layout.addRow("Humidity:", self.humidity_label)
        env_layout.addRow("Pressure:", self.pressure_label)
        env_layout.addRow("Last Updated:", self.env_timestamp_label)
        
        env_group.setLayout(env_layout)
        summary_layout.addWidget(env_group)
        
        # Add the horizontal layout to main layout
        layout.addLayout(summary_layout)
        
        # Start environmental monitoring timer
        self.env_timer = QTimer()
        self.env_timer.timeout.connect(self.update_environmental_data)
        self.env_timer.start(30000)  # Update every 30 seconds
        self.update_environmental_data()  # Initial update
        
        layout.addStretch()  # Push everything to top
        return widget
    
    def populate_default_parameters(self):
        """Add all default parameters to the UI."""
        for param_name, config in DEFAULT_PARAMETERS.items():
            param_widget = ParameterWidget(param_name, config)
            self.parameter_widgets[param_name] = param_widget
            self.parameters_layout.addWidget(param_widget)
    
    def show_add_parameter_dialog(self):
        """Show dialog to add custom parameter."""
        # Create dialog to get parameter info
        dialog = AddParameterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name, config = dialog.get_parameter_info()
            if name and name not in self.parameter_widgets:
                # Create new parameter widget
                param_widget = ParameterWidget(name, config)
                self.parameter_widgets[name] = param_widget
                
                # Add to layout
                self.parameters_layout.addWidget(param_widget)
                
                self.add_status_message(f"Added parameter: {name}")
            elif name in self.parameter_widgets:
                QMessageBox.warning(self, "Parameter Exists", f"Parameter '{name}' already exists!")
    
    def show_remove_parameter_dialog(self):
        """Show dialog to remove parameter."""
        if not self.parameter_widgets:
            QMessageBox.information(self, "No Parameters", "No parameters to remove.")
            return
            
        # Get list of all parameters (allow removal of any parameter)
        all_params = list(self.parameter_widgets.keys())
        
        # Create selection dialog
        param_name, ok = QInputDialog.getItem(self, "Remove Parameter", "Select parameter to remove:", all_params, 0, False)
        
        if ok and param_name:
            # Remove widget from layout and dict
            widget = self.parameter_widgets[param_name]
            self.parameters_layout.removeWidget(widget)
            widget.deleteLater()
            del self.parameter_widgets[param_name]
            
            self.add_status_message(f"Removed parameter: {param_name}")
    
    def run_measurement(self):
        """Run measurement with current parameters."""
        try:
            # Check if protocol is initialized
            if not self.protocol_initialized or not self.protocol or not self.protocol_state:
                self.add_status_message("ERROR: Robot not initialized. Click 'INITIALIZE ROBOT' first.")
                return
            
            # Collect parameters
            parameters = {}
            for param_name, widget in self.parameter_widgets.items():
                parameters[param_name] = widget.get_parameter_value()
            
            vial_name = self.liquid_combo.currentText()
            volume_ml = self.volume_spinbox.value()
            replicates = self.replicates_spinbox.value()
            
            # Disable UI during measurement
            self.measure_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Clear previous results
            self.clear_measurement_data()
            
            # Start measurement in background thread with existing protocol
            self.measurement_worker = MeasurementWorker(vial_name, volume_ml, replicates, parameters, 
                                                      self.protocol, self.protocol_state)
            self.measurement_worker.measurement_complete.connect(self.on_measurement_complete)
            self.measurement_worker.measurement_error.connect(self.on_measurement_error)
            self.measurement_worker.progress_update.connect(self.on_progress_update)
            self.measurement_worker.replicate_complete.connect(self.on_replicate_complete)
            
            # Store target volume for progressive calculations
            self.current_target_volume = volume_ml
            
            self.measurement_worker.start()
            
        except Exception as e:
            self.on_measurement_error(f"Error starting measurement: {str(e)}")
    
    def initialize_protocol(self):
        """Initialize robot protocol and home axes - done once at start of session."""
        try:
            if not CALIBRATION_AVAILABLE:
                self.add_status_message("ERROR: Calibration system not available")
                return
            
            self.add_status_message("Initializing robot protocol...")
            self.initialize_btn.setEnabled(False)
            
            # Create config for protocol initialization
            vial_name = self.liquid_combo.currentText()
            simulate = self.simulate_checkbox.isChecked()
            
            config = {
                'experiment': {
                    'liquid': self.liquid_type_combo.currentText(),  # Use selected liquid type
                    'simulate': simulate,
                    'source_vial': vial_name,
                    'measurement_vial': vial_name
                }
            }
            
            # Initialize protocol (this handles homing and setup)
            self.protocol = HardwareCalibrationProtocol()
            self.protocol_state = self.protocol.initialize(config)
            
            # Create session folder and CSV files for data persistence
            self.create_session_folder()
            
            # Mark as initialized and enable measurement
            self.protocol_initialized = True
            self.measure_btn.setEnabled(True)
            self.measure_btn.setToolTip("")  # Clear tooltip
            self.cleanup_btn.setEnabled(True)
            self.initialize_btn.setEnabled(False)  # Prevent double-initialization
            
            self.add_status_message("✓ Robot initialized successfully - ready for measurements")
            self.add_status_message(f"✓ Session data will be saved to: {self.session_folder}")
            
        except Exception as e:
            self.add_status_message(f"ERROR initializing robot: {str(e)}")
            self.initialize_btn.setEnabled(True)  # Re-enable on error
    
    def cleanup_protocol(self):
        """Clean up robot protocol and return vials home."""
        try:
            if not self.protocol_initialized or not self.protocol:
                self.add_status_message("No protocol to clean up")
                return
            
            self.add_status_message("Cleaning up robot protocol...")
            self.cleanup_btn.setEnabled(False)
            
            # Call cleanup method if available
            if hasattr(self.protocol, 'wrapup') and self.protocol_state:
                self.protocol.wrapup(self.protocol_state)
            
            # Reset protocol state
            self.protocol = None
            self.protocol_state = None
            self.protocol_initialized = False
            
            # Reset UI state
            self.measure_btn.setEnabled(False)
            self.measure_btn.setToolTip("Initialize robot first before measuring")
            self.initialize_btn.setEnabled(True)
            self.cleanup_btn.setEnabled(False)
            
            self.add_status_message("✓ Robot cleanup completed - vials returned home")
            
        except Exception as e:
            self.add_status_message(f"ERROR during cleanup: {str(e)}")
            # Still reset state even if cleanup failed
            self.protocol = None
            self.protocol_state = None
            self.protocol_initialized = False
            self.initialize_btn.setEnabled(True)
            self.cleanup_btn.setEnabled(False)

    def run_optimization(self):
        """Run Bayesian optimization with real-time progress monitoring."""
        try:
            # SAFETY: Kill any existing calibration processes first
            self.add_status_message("[DEBUG] Checking for zombie calibration processes...")
            self.kill_existing_calibration_processes()
            
            # Collect current GUI parameters
            parameters = {}
            for param_name, widget in self.parameter_widgets.items():
                parameters[param_name] = widget.get_parameter_value()
            
            liquid = self.liquid_type_combo.currentText()
            target_volume_ml = self.volume_spinbox.value()
            simulate = self.simulate_checkbox.isChecked()
            
            # Create configuration for optimization
            config_dict = {
                'liquid': liquid,
                'target_volume_ml': target_volume_ml,
                'parameters': parameters,
                'simulate': simulate
            }
            
            # Clear previous optimization data
            self.optimization_plot.clear()
            
            # Disable UI during optimization
            self.optimize_btn.setEnabled(False)
            self.measure_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Start optimization in background
            self.optimization_worker = OptimizationWorker(config_dict)
            
            # Connect signals with explicit queued connections for thread safety
            self.optimization_worker.optimization_started.connect(
                self.on_optimization_started,
                type=Qt.ConnectionType.QueuedConnection
            )
            self.optimization_worker.measurement_update.connect(
                self.on_optimization_measurement,
                type=Qt.ConnectionType.QueuedConnection
            )
            self.optimization_worker.optimization_complete.connect(
                self.on_optimization_complete,
                type=Qt.ConnectionType.QueuedConnection
            )
            self.optimization_worker.optimization_error.connect(
                self.on_optimization_error,
                type=Qt.ConnectionType.QueuedConnection
            )
            self.optimization_worker.debug_message.connect(
                self.on_debug_message,
                type=Qt.ConnectionType.QueuedConnection
            )
            
            self.add_status_message("[DEBUG] Worker signals connected with QueuedConnection")
            
            self.optimization_worker.start()
            self.add_status_message("Starting Bayesian optimization...")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.add_status_message(f"Error in run_optimization: {error_details}")
            self.on_optimization_error(f"Error starting optimization: {str(e)}")
    
    def on_optimization_started(self, output_dir: str):
        """Handle optimization process started - setup main thread monitoring."""
        try:
            self.add_status_message(f"[SIGNAL] on_optimization_started called with: {output_dir}")
            self.add_status_message(f"Optimization started - monitoring: {output_dir}")
            
            # Clear previous measurement data (same as measure button does)
            self.clear_measurement_data()
            self.add_status_message("[DEBUG] Cleared previous measurement data for fresh optimization")
            
            # Initialize replicate tracking for transient plots
            self.current_replicate_count = 0
            self.replicates_per_set = self.replicates_spinbox.value()  # Use GUI setting
            self.add_status_message(f"[DEBUG] Using {self.replicates_per_set} replicates per parameter set")
            
            # Setup file monitoring in MAIN GUI THREAD (not worker thread)  
            self.output_dir_path = Path(output_dir)
            self.emergency_file_path = self.output_dir_path / "emergency_raw_measurements.csv"
            self.last_row_count = 0
            
            self.add_status_message(f"[DEBUG] Main thread will monitor: {self.emergency_file_path.name}")
            self.add_status_message(f"[DEBUG] File path set to: {str(self.emergency_file_path)}")
            
            # Start timer in main GUI thread (where QTimer works properly)
            if not hasattr(self, 'main_file_timer'):
                self.main_file_timer = QTimer(self)
                self.main_file_timer.timeout.connect(self.main_thread_file_check)
                self.add_status_message("[DEBUG] Created new main_file_timer")
            else:
                self.add_status_message("[DEBUG] Reusing existing main_file_timer")
            
            # CRITICAL: Verify timer setup
            self.add_status_message(f"[DEBUG] Timer active before start: {self.main_file_timer.isActive()}")
            self.main_file_timer.start(2000)
            self.add_status_message(f"[DEBUG] Timer active after start: {self.main_file_timer.isActive()}")
            self.add_status_message("[DEBUG] Main thread timer started - this should work!")
            
            # IMMEDIATE file check to see if it exists
            if self.emergency_file_path.exists():
                lines = len(open(self.emergency_file_path).readlines())
                self.add_status_message(f"[IMMEDIATE] File already exists with {lines} lines!")
            else:
                self.add_status_message(f"[IMMEDIATE] File doesn't exist yet, will wait...")
            
            # FORCE immediate test of file checking
            self.add_status_message("[DEBUG] Testing file check immediately...")
            self.main_thread_file_check()
            
        except Exception as e:
            import traceback
            self.add_status_message(f"[ERROR] Exception in on_optimization_started: {str(e)}")
            self.add_status_message(f"[ERROR] Traceback: {traceback.format_exc()}")

    def main_thread_file_check(self):
        """File check in main GUI thread."""
        try:
            check_time = time.strftime("%H:%M:%S")
            self.add_status_message(f"[MAIN] File check at {check_time}")
            
            if not hasattr(self, 'emergency_file_path'):
                self.add_status_message("[MAIN] No emergency_file_path attribute")
                return
                
            if not self.emergency_file_path.exists():
                self.add_status_message(f"[MAIN] File not ready yet: {self.emergency_file_path}")
                return
            
            with open(self.emergency_file_path, 'r') as f:
                lines = f.readlines()
                
            current_row_count = len(lines) - 1
            self.add_status_message(f"[MAIN] File has {current_row_count} data rows (last_row_count: {self.last_row_count})")
            
            if current_row_count > self.last_row_count:
                new_count = current_row_count - self.last_row_count
                self.add_status_message(f"[MAIN] Found {new_count} new measurements!")
                
                # UPDATE row count BEFORE processing to prevent infinite retry on errors
                self.last_row_count = current_row_count
                
                new_rows = lines[self.last_row_count - new_count + 1:]  # Adjust for updated count
                for i, row in enumerate(new_rows):
                    if row.strip():
                        parts = row.strip().split(',')
                        self.add_status_message(f"[MAIN] Processing row {i}: {len(parts)} parts")
                        if len(parts) >= 8:
                            # Use GUI target volume instead of CSV value for consistency
                            gui_target_volume = self.volume_spinbox.value() if hasattr(self, 'volume_spinbox') else 0.1
                            
                            measurement_data = {
                                'liquid_type': parts[1],  # Extract vial name from CSV 
                                'target_volume_ml': gui_target_volume,  # Use GUI value, not CSV
                                'measured_volume_ul': float(parts[3]),
                                'deviation_pct': float(parts[4]),
                                'duration_s': float(parts[5]) if len(parts) > 5 else 2.5,  # Real elapsed time from CSV
                                'strategy': parts[6],
                                'total_measurement_count': int(parts[7]),
                                # ADD missing fields expected by plots
                                'volume': float(parts[3]) / 1000,  # Convert uL to mL for plots
                                'elapsed_s': float(parts[5]) if len(parts) > 5 else 2.5  # Real elapsed time for plots
                            }
                            
                            # Extract parameter values for parameter grouping (if available)
                            if len(parts) >= 18:  # Full parameter set available
                                measurement_data.update({
                                    'aspirate_speed': float(parts[9]) if len(parts) > 9 else 0,
                                    'dispense_speed': float(parts[12]) if len(parts) > 12 else 0,  
                                    'overaspirate_vol': float(parts[14]) if len(parts) > 14 else 0,
                                    'air_gap_vol': float(parts[15]) if len(parts) > 15 else 0,  # post_asp_air_vol
                                    'pre_dispense_vol': float(parts[17]) if len(parts) > 17 else 0  # pre_asp_air_vol as proxy
                                })
                            self.add_status_message(f"[MAIN] Calling on_optimization_measurement with: {measurement_data}")
                            # Direct main thread call with proper test data
                            self.on_optimization_measurement(measurement_data)
                                
                        else:
                            self.add_status_message(f"[MAIN] Row too short: {parts}")
            else:
                self.add_status_message("[MAIN] No new data to process")
        
        except Exception as e:
            import traceback
            self.add_status_message(f"[MAIN] File check error: {str(e)}")
            self.add_status_message(f"[MAIN] Traceback: {traceback.format_exc()}")
            # CRITICAL: Don't update last_row_count on error to retry next time
        
    def on_optimization_measurement(self, measurement_data: dict):
        """Handle new measurement from optimization process - same as measure button."""
        
        # Extract data from emergency CSV (same fields we parse from CSV)
        measured_volume_ul = measurement_data.get('measured_volume_ul', 0)
        measured_volume_ml = measured_volume_ul / 1000  # Convert uL to mL
        strategy = measurement_data.get('strategy', 'unknown')
        count = measurement_data.get('total_measurement_count', 0)
        deviation = measurement_data.get('deviation_pct', 0)
        
        # Get vial name from CSV data - extract from emergency CSV liquid_type field
        vial_name = measurement_data.get('liquid_type', 'PVA_water')  # Extract from CSV, fallback to default
        self.add_status_message(f"[OPT] Using vial name from CSV: '{vial_name}'")
        
        self.add_status_message(f"[{strategy.upper()}] Measurement #{count}: {measured_volume_ul:.1f}uL ({deviation:+.1f}% dev)")
        
        # *** DO EXACTLY WHAT MEASURE BUTTON DOES (with replicate reset logic) ***
        
        # Track replicates for transient plot behavior
        if not hasattr(self, 'current_replicate_count'):
            self.current_replicate_count = 0
        if not hasattr(self, 'replicates_per_set'):
            self.replicates_per_set = 3  # Default fallback
        if not hasattr(self, 'current_measurement_times'):
            self.current_measurement_times = []  # Track elapsed times for average calculation
        
        self.current_replicate_count += 1
        
        # 1. Store volume data (same as on_replicate_complete)
        self.volume_data_replicates.append(measured_volume_ml)
        
        # 2. Store elapsed time for statistics
        elapsed_time = measurement_data.get('duration_s', measurement_data.get('elapsed_s', 0))
        self.current_measurement_times.append(elapsed_time)
        
        # 2. Try to find and load real mass data (same as on_replicate_complete) 
        if not self.simulate_checkbox.isChecked():
            self.add_status_message(f"[OPT] Looking for mass data file for vial: '{vial_name}'")
            mass_file = self.find_latest_mass_file(vial_name)
            if mass_file:
                self.add_status_message(f"[OPT] Found mass file: {mass_file}")
                mass_data = self.load_mass_data(mass_file)
                if mass_data is not None:
                    self.mass_data_replicates.append(mass_data)
                    self.add_status_message(f"[OPT] Loaded real mass data for measurement {count}")
                else:
                    self.mass_data_replicates.append(None)
                    self.add_status_message(f"[OPT] Failed to load mass data for measurement {count}")
            else:
                self.mass_data_replicates.append(None)
                self.add_status_message(f"[OPT] No mass file found for vial '{vial_name}' - checking what files exist...")
                # Debug: Check what mass files actually exist
                import glob
                all_mass_files = glob.glob("output/mass_measurements/*/mass_data_*.csv")
                self.add_status_message(f"[OPT] Available mass files: {len(all_mass_files)} found")
                if all_mass_files:
                    for f in all_mass_files[:3]:  # Show first 3
                        self.add_status_message(f"[OPT]   - {f}")
                else:
                    # Also try from current directory  
                    import os
                    self.add_status_message(f"[OPT] Current working directory: {os.getcwd()}")
                    self.add_status_message(f"[OPT] Try from calibration_modular_v2/output/mass_measurements/")
        else:
            # In simulation mode, add None for mass data
            self.mass_data_replicates.append(None)
            self.add_status_message(f"[OPT] Simulation: measurement {count} complete")
        
        # 3. Update both top plots and stats with current replicate set
        self.update_plots_progressive()
        self.update_statistics_progressive()
        
        # 4. Always update optimization plot (cumulative, never resets)
        self.optimization_plot.add_measurement(measurement_data)
        
        # 5. Check if we completed a replicate set - RESET top graphs for next parameter set
        if self.current_replicate_count >= self.replicates_per_set:
            self.add_status_message(f"[OPT] Completed replicate set ({self.replicates_per_set} measurements) - resetting top graphs")
            # Clear data for top graphs only (bottom optimization plot keeps accumulating)
            self.mass_data_replicates = []
            self.volume_data_replicates = []
            self.current_measurement_times = []  # Reset elapsed times for next parameter set
            self.current_replicate_count = 0
            # Note: optimization_plot keeps its data forever
        
        # Store current target volume for calculations - use GUI setting, not CSV data
        self.current_target_volume = self.volume_spinbox.value()  # Always use GUI value
        self.add_status_message(f"[OPT] Using target volume from GUI: {self.current_target_volume*1000:.1f}uL")
        
        # Update parameter current values from emergency CSV
        self.update_parameter_values_from_emergency_csv()
        
    def update_parameter_values_from_emergency_csv(self):
        """Extract parameter values from emergency CSV and update parameter widget displays."""
        try:
            if not hasattr(self, 'emergency_file_path') or not self.emergency_file_path.exists():
                return
                
            # Read CSV file as lines and extract latest row
            with open(self.emergency_file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:  # Only header or empty
                return
                
            # Get the latest row (most recent optimization parameters)
            latest_line = lines[-1].strip()
            if not latest_line:
                return
                
            parts = latest_line.split(',')
            if len(parts) < 18:  # Not enough columns
                self.add_status_message(f"[PARAM] CSV row too short: {len(parts)} columns")
                return
            
            # Map CSV column indices to parameter names
            # Based on CSV header: timestamp,target_volume_ml,measured_volume_ml,measured_volume_ul,deviation_pct,duration_s,strategy,total_measurement_count,asp_disp_cycles,aspirate_speed,aspirate_wait_time,blowout_vol,dispense_speed,dispense_wait_time,overaspirate_vol,post_asp_air_vol,post_retract_wait_time,pre_asp_air_vol,retract_speed
            csv_column_mapping = {
                'overaspirate_vol': 14,
                'aspirate_speed': 9, 
                'dispense_speed': 12,
                'aspirate_wait_time': 10,
                'dispense_wait_time': 13,
                'pre_asp_air_vol': 17,
                'post_asp_air_vol': 15,
                'blowout_vol': 11,
                'retract_speed': 18 if len(parts) > 18 else None
            }
            
            # Extract values from latest row and update widgets
            updated_params = []
            for param_name, column_idx in csv_column_mapping.items():
                if column_idx is not None and column_idx < len(parts) and param_name in self.parameter_widgets:
                    try:
                        value = float(parts[column_idx])
                        widget = self.parameter_widgets[param_name]
                        widget.update_current_value(value)
                        updated_params.append(param_name)
                    except (ValueError, IndexError) as e:
                        self.add_status_message(f"[PARAM] Error extracting {param_name}: {e}")
                        
            self.add_status_message(f"[PARAM] Updated {len(updated_params)} parameters: {updated_params}")
            
        except Exception as e:
            self.add_status_message(f"[PARAM] Error updating parameters: {e}")
    
    def on_optimization_complete(self, results: dict):
        """Handle optimization completion."""
        # Stop main thread file monitoring
        if hasattr(self, 'main_file_timer'):
            self.main_file_timer.stop()
            self.add_status_message("[DEBUG] Stopped main thread file monitoring")
            
        # Re-enable UI
        self.optimize_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message("Optimization completed successfully!")
        QMessageBox.information(self, "Optimization Complete", 
                              "Bayesian optimization finished. Check the optimization plot for results.")
        
    def on_optimization_error(self, error_msg: str):
        """Handle optimization error."""
        # Re-enable UI
        self.optimize_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message(f"Optimization error: {error_msg}")
        QMessageBox.critical(self, "Optimization Error", f"Optimization failed:\n{error_msg}")
    
    def on_debug_message(self, message: str):
        """Handle debug messages from optimization worker - show in GUI status."""
        self.add_status_message(message)
    
    def check_system_status(self):
        """STATUS CHECK: Get immediate info on what's actually happening."""
        self.add_status_message("=== SYSTEM STATUS CHECK ===")
        
        # Check if main thread monitoring is set up
        if hasattr(self, 'main_file_timer'):
            if self.main_file_timer.isActive():
                self.add_status_message("✅ Main thread timer is RUNNING")
            else:
                self.add_status_message("❌ Main thread timer is STOPPED")
        else:
            self.add_status_message("❌ Main thread timer NOT SET UP")
        
        # Check worker thread monitoring 
        if hasattr(self, 'optimization_worker') and self.optimization_worker:
            self.add_status_message("✅ Worker thread exists")
        else:
            self.add_status_message("❌ No worker thread")
            
        # Check file paths
        if hasattr(self, 'emergency_file_path'):
            self.add_status_message(f"📁 Main thread path: {self.emergency_file_path}")
            if self.emergency_file_path.exists():
                try:
                    lines = len(open(self.emergency_file_path).readlines())
                    self.add_status_message(f"✅ File EXISTS with {lines} lines")
                    
                    # Show recent data
                    with open(self.emergency_file_path) as f:
                        all_lines = f.readlines()
                        if len(all_lines) > 2:  # Header + at least 1 data row
                            latest = all_lines[-1].strip()
                            parts = latest.split(',')
                            if len(parts) >= 8:
                                vol = parts[3]
                                count = parts[7]
                                self.add_status_message(f"📊 Latest: #{count}, {vol}uL")
                        
                except Exception as e:
                    self.add_status_message(f"❌ File read error: {e}")
            else:
                self.add_status_message("❌ File does NOT exist")
        else:
            self.add_status_message("❌ No main thread file path set")
        
        # Check if subprocess is actually writing files  
        output_base = Path(__file__).parent / "output"
        if output_base.exists():
            run_dirs = [d for d in output_base.iterdir() if d.name.startswith('run_')]
            if run_dirs:
                newest = max(run_dirs, key=lambda x: x.stat().st_ctime)
                emergency_file = newest / "emergency_raw_measurements.csv"
                self.add_status_message(f"📂 Newest run: {newest.name}")
                if emergency_file.exists():
                    lines = len(open(emergency_file).readlines())
                    self.add_status_message(f"✅ Subprocess file: {lines} lines")
                else:
                    self.add_status_message("❌ Subprocess file missing")
        
        self.add_status_message("=== END STATUS CHECK ===")
    
    def test_threading(self):
        """Test threading and signal connections without subprocess."""
        try:
            self.add_status_message("=== THREADING TEST STARTED ===")
            
            # Disable buttons during test
            self.test_thread_btn.setEnabled(False)
            self.optimize_btn.setEnabled(False)
            
            # Clear any existing worker
            if hasattr(self, 'optimization_worker') and self.optimization_worker:
                if self.optimization_worker.isRunning():
                    self.optimization_worker.terminate()
                    self.optimization_worker.wait()
            
            # Create test worker with test_mode=True
            dummy_config = {'liquid': 'PVA_water', 'target_volume_ml': 0.200, 'parameters': {}}
            self.optimization_worker = OptimizationWorker(dummy_config, test_mode=True)
            
            # Connect signals with explicit queued connections for thread safety
            self.optimization_worker.optimization_started.connect(
                self.on_optimization_started, 
                type=Qt.ConnectionType.QueuedConnection
            )
            self.optimization_worker.optimization_complete.connect(
                self.on_optimization_complete,
                type=Qt.ConnectionType.QueuedConnection  
            )
            self.optimization_worker.debug_message.connect(
                self.on_debug_message,
                type=Qt.ConnectionType.QueuedConnection
            )
            
            self.add_status_message("[TEST] Worker created, signals connected with QueuedConnection")
            self.add_status_message("[TEST] Starting test worker thread...")
            
            # Start the test worker
            self.optimization_worker.start()
            
        except Exception as e:
            self.add_status_message(f"[TEST ERROR] {str(e)}")
            self.test_thread_btn.setEnabled(True)
            self.optimize_btn.setEnabled(True)
    
    def on_optimization_complete(self, results):
        """Handle test completion."""
        self.add_status_message(f"[TEST] Optimization complete: {results}")
        
        # Re-enable buttons
        self.test_thread_btn.setEnabled(True)
        self.optimize_btn.setEnabled(True)
        
        if results.get('status') == 'test_success':
            self.add_status_message("=== THREADING TEST PASSED! ===")
        else:
            self.add_status_message("=== THREADING TEST COMPLETED ===")
    
    def on_measurement_complete(self, results: Dict[str, Any]):
        """Handle completed measurement (all replicates done)."""
        self.current_results = results
        
        # Final update to plots (should already be current from progressive updates)
        if not hasattr(self, 'mass_data_replicates') or len(self.mass_data_replicates) == 0:
            # Fallback: use old plotting if progressive didn't work
            self.mass_time_plot.plot_mass_time(results)
            self.volume_replicate_plot.plot_individual_measurements(results)
        
        # Update final statistics with complete timing info
        self.time_label.setText(f"{results['mean_time_s']:.2f} seconds")
        
        # Re-enable UI
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message(f"Measurement complete: {results['mean_volume_ml']*1000:.1f} μL " +
                               f"({results['accuracy_pct']:+.1f}%, CV={results['cv_pct']:.1f}%)")
        
        # Save measurement data to session CSV files
        if self.session_folder:
            self.save_measurement_data(results)
    
    def on_measurement_error(self, error_msg: str):
        """Handle measurement error."""
        QMessageBox.critical(self, "Measurement Error", error_msg)
        
        # Re-enable UI
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message(f"Error: {error_msg}")
    
    def on_progress_update(self, message: str):
        """Handle progress update."""
        self.add_status_message(message)
    
    def update_environmental_data(self):
        """Update environmental conditions display."""
        try:
            if not PANDAS_AVAILABLE:
                self.temp_label.setText("pandas N/A")
                self.humidity_label.setText("pandas N/A")
                self.pressure_label.setText("pandas N/A")
                self.env_timestamp_label.setText("pandas required")
                return
                
            mqtt_log_file = "C:\\Users\\Imaging Controller\\Desktop\\m5stack\\mqtt_log.csv"
            
            if not os.path.exists(mqtt_log_file):
                self.temp_label.setText("File not found")
                self.humidity_label.setText("File not found")
                self.pressure_label.setText("File not found")
                self.env_timestamp_label.setText("MQTT log missing")
                return
                
            # Read the latest environmental data
            df = pd.read_csv(mqtt_log_file)
            if len(df) == 0:
                self.temp_label.setText("No data")
                self.humidity_label.setText("No data")
                self.pressure_label.setText("No data")
                self.env_timestamp_label.setText("Empty log file")
                return
                
            latest = df.iloc[-1]  # Most recent row
            timestamp = pd.to_datetime(latest["Timestamp"])
            
            # Update labels with color coding based on data freshness
            age_minutes = (datetime.now() - timestamp.to_pydatetime()).total_seconds() / 60
            
            if age_minutes > 60:  # Data older than 1 hour
                color_style = "color: red;"
            elif age_minutes > 10:  # Data older than 10 minutes
                color_style = "color: orange;"
            else:  # Fresh data
                color_style = "color: green;"
            
            # Update temperature
            temp_c = float(latest["sht_temp_c"]) if pd.notna(latest["sht_temp_c"]) else None
            if temp_c is not None:
                self.temp_label.setText(f"{temp_c:.1f}°C")
                self.temp_label.setStyleSheet(color_style)
            else:
                self.temp_label.setText("--°C")
                self.temp_label.setStyleSheet("color: gray;")
            
            # Update humidity
            humidity = float(latest["sht_rh"]) if pd.notna(latest["sht_rh"]) else None
            if humidity is not None:
                self.humidity_label.setText(f"{humidity:.1f}%")
                self.humidity_label.setStyleSheet(color_style)
            else:
                self.humidity_label.setText("--%")
                self.humidity_label.setStyleSheet("color: gray;")
            
            # Update pressure
            pressure = float(latest["bmp_pa"]) if pd.notna(latest["bmp_pa"]) else None
            if pressure is not None:
                self.pressure_label.setText(f"{pressure:.0f} Pa")
                self.pressure_label.setStyleSheet(color_style)
            else:
                self.pressure_label.setText("-- Pa")
                self.pressure_label.setStyleSheet("color: gray;")
            
            # Update timestamp
            time_str = timestamp.strftime("%H:%M:%S")
            self.env_timestamp_label.setText(f"{time_str} ({age_minutes:.0f}m ago)")
            self.env_timestamp_label.setStyleSheet(color_style)
            
        except Exception as e:
            # Handle errors gracefully
            error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
            self.temp_label.setText("Error")
            self.humidity_label.setText("Error") 
            self.pressure_label.setText("Error")
            self.env_timestamp_label.setText(error_msg)
            
            # Set error styling
            for label in [self.temp_label, self.humidity_label, self.pressure_label, self.env_timestamp_label]:
                label.setStyleSheet("color: red;")

    def add_status_message(self, message: str):
        """Add message to status text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_text.append(formatted_message)
        
    def closeEvent(self, event):
        """Handle application close event - clean up resources."""
        self.add_status_message("[DEBUG] GUI closing, starting cleanup...")
        
        # Stop optimization if running
        if hasattr(self, 'optimization_worker') and self.optimization_worker:
            self.add_status_message("[DEBUG] Cleaning up optimization worker...")
            self.optimization_worker.cleanup_and_terminate()
            self.optimization_worker = None
        
        # Stop measurement worker if running  
        if hasattr(self, 'measurement_worker') and self.measurement_worker:
            self.add_status_message("[DEBUG] Stopping measurement worker...")
            if self.measurement_worker.isRunning():
                self.measurement_worker.quit()
                self.measurement_worker.wait(2000)
            self.measurement_worker = None
        
        # Robot cleanup if protocol is initialized
        if hasattr(self, 'protocol') and self.protocol:
            try:
                self.add_status_message("[DEBUG] Performing robot cleanup...")
                self.protocol.wrapup()
                self.add_status_message("[DEBUG] Robot cleanup completed")
            except Exception as e:
                self.add_status_message(f"[DEBUG] Robot cleanup error: {e}")
                # Continue with close even if robot cleanup fails
        
        self.add_status_message("[DEBUG] GUI cleanup completed")
        event.accept()  # Allow the window to close
        
    def kill_existing_calibration_processes(self):
        """SAFETY: Kill any existing calibration processes before starting new ones."""
        try:
            import subprocess
            
            # Find Python processes running calibration
            result = subprocess.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'],
                capture_output=True, text=True, timeout=10
            )
            
            killed_count = 0
            for line in result.stdout.split('\n'):
                if 'run_calibration.py' in line and 'ProcessId' not in line:
                    # Extract PID
                    parts = line.strip().split()
                    if parts:
                        try:
                            pid = int(parts[-1])  # PID is usually last
                            subprocess.run(['taskkill', '/PID', str(pid), '/F'], 
                                         capture_output=True, timeout=5)
                            killed_count += 1
                            self.add_status_message(f"[DEBUG] Killed zombie process PID {pid}")
                        except (ValueError, subprocess.TimeoutExpired):
                            pass
            
            if killed_count > 0:
                self.add_status_message(f"[DEBUG] Killed {killed_count} zombie processes")
                time.sleep(1)  # Wait for cleanup
        except Exception as e:
            self.add_status_message(f"[DEBUG] Process cleanup error: {e}")
    
    
    def clear_results(self):
        """Clear current results and plots."""
        self.current_results = None
        self.time_label.setText("Not available")
        self.accuracy_label.setText("Not available")
        self.cv_label.setText("Not available")
        
        # Clear plots
        self.mass_time_plot.clear()
        self.volume_replicate_plot.clear()
        
    def on_optimization_complete(self, results: dict):
        """Handle optimization completion."""
        # Re-enable UI
        self.optimize_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message("Optimization completed successfully!")
        QMessageBox.information(self, "Optimization Complete", 
                              "Bayesian optimization finished. Check the optimization plot for results.")
        
    def on_optimization_error(self, error_msg: str):
        """Handle optimization error."""
        # Re-enable UI
        self.optimize_btn.setEnabled(True)
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message(f"Optimization error: {error_msg}")
        QMessageBox.critical(self, "Optimization Error", f"Optimization failed:\n{error_msg}")
    
    def clear_measurement_data(self):
        """Clear all progressive measurement data when starting new measurement set."""
        self.mass_data_replicates = []
        self.volume_data_replicates = []
        
        # Clear both plots
        if MATPLOTLIB_AVAILABLE:
            self.mass_time_plot.figure.clear()
            self.mass_time_plot.canvas.draw()
            self.volume_replicate_plot.figure.clear()
            self.volume_replicate_plot.canvas.draw()
    
    def find_latest_mass_file(self, vial_name):
        """Find the most recent mass measurement CSV file for a vial."""
        try:
            import glob
            from pathlib import Path
            import time
            
            # Wait a moment for file to be written
            time.sleep(0.5)
            
            # Updated search pattern to match actual file naming: mass_data_liquid_source_X_TIMESTAMP
            # Try multiple possible locations
            patterns = [
                "output/mass_measurements/*/mass_data_*.csv",
                "calibration_modular_v2/output/mass_measurements/*/mass_data_*.csv", 
                "../output/mass_measurements/*/mass_data_*.csv"
            ]
            
            files = []
            for pattern in patterns:
                found_files = glob.glob(pattern)
                files.extend(found_files)
                if found_files:
                    print(f"Found {len(found_files)} files with pattern: {pattern}")
                    break
            
            if not files:
                print(f"No mass data files found with any pattern. Tried: {patterns}")
                return None
            
            # Sort by modification time, get most recent
            files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            latest_file = files[0]
            print(f"Found latest mass file: {latest_file}")
            return latest_file
            
        except Exception as e:
            print(f"Error finding mass file: {e}")
            return None
    
    def load_mass_data(self, csv_file):
        """Load mass data from CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} mass data points from {csv_file}")
            return df
        except Exception as e:
            print(f"Error loading mass data: {e}")
            return None
    
    def on_replicate_complete(self, replicate_num: int, vial_name: str, measurement_data: dict):
        """Handle completion of individual replicate - collect data and update plots."""
        # Store volume data immediately
        measured_volume_ml = measurement_data['volume']
        self.volume_data_replicates.append(measured_volume_ml)
        
        # Try to find and load real mass data (only if not simulating)
        if not self.simulate_checkbox.isChecked():
            mass_file = self.find_latest_mass_file(vial_name)
            if mass_file:
                mass_data = self.load_mass_data(mass_file)
                if mass_data is not None:
                    self.mass_data_replicates.append(mass_data)
                    self.add_status_message(f"Loaded real mass data for replicate {replicate_num}")
                else:
                    self.mass_data_replicates.append(None)
                    self.add_status_message(f"Failed to load mass data for replicate {replicate_num}")
            else:
                self.mass_data_replicates.append(None)
                self.add_status_message(f"No mass file found for replicate {replicate_num}")
        else:
            # In simulation mode, add None for mass data
            self.mass_data_replicates.append(None)
            self.add_status_message(f"Simulation: replicate {replicate_num} complete")
        
        # Update both plots progressively
        self.update_plots_progressive()
        
        # Update statistics with current data
        self.update_statistics_progressive()
    
    def update_plots_progressive(self):
        """Update both plots with all data collected so far."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Update mass-time plot
        self.plot_mass_time_progressive()
        
        # Update individual measurements plot
        self.plot_individual_measurements_progressive()
    
    def plot_mass_time_progressive(self):
        """Plot mass vs time with all replicates collected so far."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.mass_time_plot.figure.clear()
        ax = self.mass_time_plot.figure.add_subplot(111)
        
        if not self.simulate_checkbox.isChecked() and any(data is not None for data in self.mass_data_replicates):
            # Plot real mass data
            colors = ['blue', 'orange', 'green', 'red', 'purple']
            
            for i, mass_data in enumerate(self.mass_data_replicates):
                if mass_data is not None and len(mass_data) > 0:
                    color = colors[i % len(colors)]
                    ax.plot(mass_data['time_relative'], mass_data['mass_g'], 
                           color=color, linewidth=2, label=f'Replicate {i+1}')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mass (g)')
            ax.set_title(f'Real Mass vs Time ({len(self.mass_data_replicates)} replicates)')
            if len(self.mass_data_replicates) > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Fallback to simulated data for simulation mode or if no real data available
            colors = ['blue', 'orange', 'green', 'red', 'purple']
            
            for i, volume_ml in enumerate(self.volume_data_replicates):
                # Create simulated mass-time data for this replicate
                time_points = np.linspace(0, 0.5, 50)  # 0.5 second measurement
                color = colors[i % len(colors)]
                
                # Simulate increasing mass over time
                mass_points = np.random.normal(0, 0.0001, 50) + time_points * (volume_ml * 2)  # ~2g/mL density
                ax.plot(time_points, mass_points, 
                       color=color, linewidth=2, label=f'Replicate {i+1} (sim)')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Mass (g)')
            ax.set_title(f'Mass vs Time - Simulated ({len(self.volume_data_replicates)} replicates)')
            if len(self.volume_data_replicates) > 0:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        self.mass_time_plot.canvas.draw()
    
    def plot_individual_measurements_progressive(self):
        """Plot individual measurements as strip plot with all volumes collected so far."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        if len(self.volume_data_replicates) == 0:
            return
            
        self.volume_replicate_plot.figure.clear()
        ax = self.volume_replicate_plot.figure.add_subplot(111)
        
        volumes_ul = np.array(self.volume_data_replicates) * 1000  # Convert to μL
        target_ul = self.current_target_volume * 1000
        n_measurements = len(volumes_ul)
        
        # Create strip plot positions
        y_positions = np.ones(n_measurements) 
        
        # Add small jitter to prevent overlapping
        np.random.seed(42)  # Consistent jitter
        jitter = np.random.normal(0, 0.03, n_measurements)
        y_jitter = y_positions + jitter
        
        # Plot measurements with different colors per replicate
        colors = plt.cm.Set3(np.linspace(0, 1, n_measurements))
        scatter = ax.scatter(volumes_ul, y_jitter, s=100, alpha=0.8, 
                           c=colors, edgecolors='black', linewidth=1.5)
        
        # Add target line
        ax.axvline(target_ul, color='red', linestyle='--', linewidth=3, 
                  label=f'Target: {target_ul:.1f} μL', alpha=0.8)
        
        # Add mean line if multiple measurements
        if n_measurements > 1:
            mean_ul = np.mean(volumes_ul)
            ax.axvline(mean_ul, color='darkgreen', linestyle='-', linewidth=3, 
                      label=f'Mean: {mean_ul:.1f} μL', alpha=0.8)
            
            # Add standard deviation shading
            std_ul = np.std(volumes_ul)
            ax.axvspan(mean_ul - std_ul, mean_ul + std_ul, alpha=0.1, 
                      color='green', label=f'±1σ: {std_ul:.1f} μL')
        
        # Styling
        ax.set_xlabel('Volume (μL)', fontsize=12, weight='bold')
        ax.set_ylabel('')
        ax.set_ylim(0.4, 2.0)
        ax.set_yticks([])  # Hide meaningless y-axis
        ax.set_title(f'Individual Measurements (n={n_measurements})', fontsize=14, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')
        
        self.volume_replicate_plot.canvas.draw()
    
    def update_statistics_progressive(self):
        """Update statistics display with current data."""
        if len(self.volume_data_replicates) == 0:
            return
            
        volumes_ml = np.array(self.volume_data_replicates)
        target_ml = self.current_target_volume
        
        # Calculate current statistics
        mean_volume = np.mean(volumes_ml)
        accuracy_pct = ((mean_volume - target_ml) / target_ml) * 100
        
        if len(volumes_ml) > 1:
            cv_pct = (np.std(volumes_ml) / mean_volume) * 100
        else:
            cv_pct = 0.0
        
        # Update labels
        self.accuracy_label.setText(f"{accuracy_pct:+.1f}% ({len(volumes_ml)} reps)")
        self.cv_label.setText(f"{cv_pct:.1f}%")
        
        # Calculate average elapsed time from stored measurement data
        if hasattr(self, 'current_measurement_times') and self.current_measurement_times:
            mean_time = np.mean(self.current_measurement_times)
            self.time_label.setText(f"{mean_time:.1f} seconds")
        else:
            self.time_label.setText("In progress...")

    def update_plots_from_optimization_data(self, trial_measurements: List[dict]):
        """Update mass-time plot, histogram, and stats from optimization measurements."""
        if not trial_measurements:
            return
            
        # Convert optimization data to format expected by existing plot methods
        target_volume_ml = trial_measurements[0].get('target_volume_ml', 0.1)
        
        # Create fake measurement results in the format expected by existing methods
        volumes_ml = [m.get('measured_volume_ml', 0) for m in trial_measurements]
        times_s = [m.get('duration_s', 0) for m in trial_measurements]
        
        # Create raw_measurements list for plots
        raw_measurements = []
        for i, m in enumerate(trial_measurements):
            raw_measurements.append({
                'replicate': i + 1,
                'volume': m.get('measured_volume_ml', 0),
                'elapsed_s': m.get('duration_s', 0),
                'start_time': m.get('timestamp', ''),
                'end_time': m.get('timestamp', '')
            })
        
        # Calculate statistics
        mean_volume = sum(volumes_ml) / len(volumes_ml) if volumes_ml else 0
        mean_time = sum(times_s) / len(times_s) if times_s else 0
        
        # Calculate accuracy and precision
        if len(volumes_ml) > 1:
            accuracy_pct = ((mean_volume - target_volume_ml) / target_volume_ml) * 100 if target_volume_ml > 0 else 0
            cv_pct = (np.std(volumes_ml) / mean_volume) * 100 if mean_volume > 0 else 0
        else:
            accuracy_pct = 0
            cv_pct = 0
        
        # Package as results dict
        results = {
            'raw_measurements': raw_measurements,
            'mean_volume_ml': mean_volume,
            'mean_time_s': mean_time,
            'accuracy_pct': accuracy_pct,
            'cv_pct': cv_pct,
            'target_volume_ml': target_volume_ml,
            'volumes_ml': volumes_ml,
            'times_s': times_s
        }
        
        # Update existing plots with this strategy's data
        if hasattr(self, 'mass_time_plot'):
            self.mass_time_plot.plot_mass_time(results)
        if hasattr(self, 'volume_replicate_plot'):
            self.volume_replicate_plot.plot_individual_measurements(results)
        
        # Update statistics display
        if hasattr(self, 'time_label'):
            self.time_label.setText(f"{mean_time:.2f} seconds")
        if hasattr(self, 'accuracy_label'):
            self.accuracy_label.setText(f"{accuracy_pct:+.1f}%")
        if hasattr(self, 'cv_label'):
            self.cv_label.setText(f"{cv_pct:.1f}%")

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = CalibrationTestMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()