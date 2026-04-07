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
# import yaml  # TODO: Will need this when optimization config format is fixed
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
    'overaspirate_vol': {'value': 0.004, 'min': 0.0, 'max': 0.050, 'unit': 'mL', 'type': 'float'},
    'aspirate_speed': {'value': 10, 'min': 1, 'max': 50, 'unit': '', 'type': 'int'},
    'dispense_speed': {'value': 10, 'min': 1, 'max': 50, 'unit': '', 'type': 'int'},
    'aspirate_wait_time': {'value': 0.0, 'min': 0.0, 'max': 10.0, 'unit': 's', 'type': 'float'},
    'dispense_wait_time': {'value': 1.5, 'min': 0.0, 'max': 10.0, 'unit': 's', 'type': 'float'},
    'pre_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.020, 'unit': 'mL', 'type': 'float'},
    'post_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.020, 'unit': 'mL', 'type': 'float'},
    'blowout_vol': {'value': 0.0, 'min': 0.0, 'max': 0.010, 'unit': 'mL', 'type': 'float'},
    'retract_speed': {'value': 5, 'min': 1, 'max': 20, 'unit': '', 'type': 'int'},
    'post_retract_wait_time': {'value': 0.0, 'min': 0.0, 'max': 5.0, 'unit': 's', 'type': 'float'},
    'asp_disp_cycles': {'value': 0, 'min': 0, 'max': 5, 'unit': 'cycles', 'type': 'int'}
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
    return ['6M_TFA', '6M_p_TSA', '6M_Citric_Acid', '6M_H2SO4', '6M_H3PO4']

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
        layout = QHBoxLayout()
        
        # Parameter name label
        name_label = QLabel(f"{self.name}:")
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)
        
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
        layout.addWidget(self.value_spinbox)
        
        # Min value spinbox
        min_label = QLabel("Min:")
        layout.addWidget(min_label)
        if is_int:
            self.min_spinbox = QSpinBox()
        else:
            self.min_spinbox = QDoubleSpinBox()
            self.min_spinbox.setDecimals(4)
        self.min_spinbox.setValue(int(self.config['min']) if is_int else self.config['min'])
        if unit_suffix:
            self.min_spinbox.setSuffix(unit_suffix)
        layout.addWidget(self.min_spinbox)
        
        # Max value spinbox
        max_label = QLabel("Max:")
        layout.addWidget(max_label)
        if is_int:
            self.max_spinbox = QSpinBox()
        else:
            self.max_spinbox = QDoubleSpinBox()
        if not is_int:
            self.max_spinbox.setDecimals(4)
        self.max_spinbox.setValue(int(self.config['max']) if is_int else self.config['max'])
        if unit_suffix:
            self.max_spinbox.setSuffix(unit_suffix)
        layout.addWidget(self.max_spinbox)
        
        self.setLayout(layout)
        self.max_spinbox.setSuffix(unit_suffix)
        layout.addWidget(self.max_spinbox)
        
        self.setLayout(layout)
    
    def get_values(self) -> Tuple[float, float, float]:
        """Return (value, min, max)."""
        return (
            self.value_spinbox.value(),
            self.min_spinbox.value(), 
            self.max_spinbox.value()
        )
    
    def get_parameter_value(self) -> float:
        """Get just the current parameter value."""
        return self.value_spinbox.value()

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
        self.optimization_data.append(measurement_data)
        self.update_plot()
        
    def update_plot(self):
        """Update the optimization plot with current data."""
        if not MATPLOTLIB_AVAILABLE or not self.optimization_data:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Group data by strategy
        strategy_data = {}
        for data in self.optimization_data:
            strategy = data.get('strategy', 'other')
            if strategy not in strategy_data:
                strategy_data[strategy] = {'x': [], 'y': []}
            strategy_data[strategy]['x'].append(data['total_measurement_count'])
            strategy_data[strategy]['y'].append(data['measured_volume_ul'])
        
        # Plot each strategy with different colors
        for strategy, data in strategy_data.items():
            color = self.strategy_colors.get(strategy, self.strategy_colors['other'])
            ax.scatter(data['x'], data['y'], c=color, label=strategy.title(), alpha=0.7, s=50)
        
        # Add target line if we have data
        if self.optimization_data:
            target_ul = self.optimization_data[0]['target_volume_ml'] * 1000
            ax.axhline(y=target_ul, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Target: {target_ul:.0f} μL')
        
        ax.set_xlabel('Measurement Number')
        ax.set_ylabel('Volume Measured (μL)')
        ax.set_title('Optimization Progress - Volume vs Measurement Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits based on target
        if self.optimization_data:
            target = self.optimization_data[0]['target_volume_ml'] * 1000
            all_volumes = [d['measured_volume_ul'] for d in self.optimization_data]
            y_min = min(min(all_volumes), target * 0.7)
            y_max = max(max(all_volumes), target * 1.3)
            ax.set_ylim(y_min, y_max)
        
        self.canvas.draw()

class OptimizationWorker(QThread):
    """Worker thread for running optimization and monitoring progress."""
    
    optimization_started = Signal(str)  # output_dir
    measurement_update = Signal(dict)    # measurement data
    optimization_complete = Signal(dict) # final results
    optimization_error = Signal(str)     # error message
    
    def __init__(self, config_dict: dict, parent=None):
        super().__init__(parent)
        self.config_dict = config_dict
        self.output_dir = None
        self.emergency_file = None
        self.last_row_count = 0
        self.process = None
        self.file_watcher = None
        
    def run(self):
        """Run optimization in background and monitor progress."""
        try:
            # Generate config file
            config_path = self.create_config_file()
            
            # Start calibration process
            self.start_calibration_process(config_path)
            
            # Monitor for output directory creation
            self.wait_for_output_dir()
            
            # Setup file monitoring
            self.setup_file_monitoring()
            
            # Wait for process to complete
            self.process.wait()
            
            if self.process.returncode == 0:
                self.optimization_complete.emit({"status": "success"})
            else:
                self.optimization_error.emit(f"Calibration process failed with return code {self.process.returncode}")
                
        except Exception as e:
            self.optimization_error.emit(f"Error during optimization: {str(e)}")
            
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
            cmd = [sys.executable, "calibration_modular_v2/run_calibration.py"]
            cwd = str(Path(__file__).parent.parent)
            
            self.process = subprocess.Popen(cmd, 
                                          cwd=cwd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        except Exception as e:
            raise
        
    def wait_for_output_dir(self):
        """Wait for output directory to be created."""
        output_base = Path(__file__).parent.parent / "output" / "calibration_v2_runs"
        
        # Wait up to 30 seconds for directory creation
        for _ in range(30):
            if output_base.exists():
                # Find newest directory
                dirs = [d for d in output_base.iterdir() if d.is_dir()]
                if dirs:
                    self.output_dir = max(dirs, key=lambda x: x.stat().st_ctime)
                    self.emergency_file = self.output_dir / "emergency_raw_measurements.csv"
                    self.optimization_started.emit(str(self.output_dir))
                    return
            time.sleep(1)
            
        raise Exception("Output directory not created within timeout")
        
    def setup_file_monitoring(self):
        """Setup file watcher for emergency measurements file."""
        self.file_watcher = QFileSystemWatcher()
        
        # Watch the directory (file might not exist yet)
        self.file_watcher.directoryChanged.connect(self.check_file_updates)
        self.file_watcher.addPath(str(self.output_dir))
        
        # Start periodic checks as backup
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_file_updates)
        self.check_timer.start(2000)  # Check every 2 seconds
        
    def check_file_updates(self):
        """Check for updates to emergency measurements file."""
        if not self.emergency_file or not self.emergency_file.exists():
            return
            
        try:
            # Simple approach: count lines to detect new rows
            with open(self.emergency_file, 'r') as f:
                lines = f.readlines()
                
            current_row_count = len(lines) - 1  # Subtract header
            
            if current_row_count > self.last_row_count:
                # New measurement(s) added
                new_rows = lines[self.last_row_count + 1:]  # Skip header and previously processed
                
                for row in new_rows:
                    if row.strip():  # Skip empty lines
                        measurement_data = self.parse_csv_row(row.strip())
                        if measurement_data:
                            self.measurement_update.emit(measurement_data)
                            
                self.last_row_count = current_row_count
                
        except Exception as e:
            # File might be temporarily locked during writing
            pass
            
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

class MeasurementWorker(QThread):
    """Worker thread for running measurements to avoid GUI freezing."""
    
    measurement_complete = Signal(dict)  # Results
    measurement_error = Signal(str)     # Error message
    progress_update = Signal(str)       # Status message
    replicate_complete = Signal(int, str, dict)  # replicate_num, vial_name, measurement_data
    
    def __init__(self, vial_name: str, volume_ml: float, replicates: int, 
                 parameters: Dict[str, float], simulate: bool = True, parent=None):
        super().__init__(parent)
        self.vial_name = vial_name
        self.volume_ml = volume_ml
        self.replicates = replicates
        self.parameters = parameters
        self.simulate = simulate
    
    def run(self):
        """Execute measurements in background thread."""
        try:
            self.progress_update.emit("Initializing North Robot protocol...")
            
            if not CALIBRATION_AVAILABLE:
                raise Exception("Calibration system not available")
            
            # Create minimal config for protocol initialization
            config = {
                'experiment': {
                    'liquid': 'water',  # Default liquid type
                    'volume_targets_ml': [self.volume_ml],
                    'simulate': self.simulate,  # Use simulation mode from GUI
                    'source_vial': self.vial_name,      # Pass selected vial from GUI
                    'measurement_vial': self.vial_name   # Use same vial for both (SINGLE_VIAL mode)
                }
            }
            
            # Initialize protocol (vial names now passed in config)
            protocol = HardwareCalibrationProtocol()
            state = protocol.initialize(config)
            
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
                
                # Execute measurement
                measurement_list = protocol.measure(state, self.volume_ml, pipetting_params, replicates=1)
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
    
    def plot_volume_histogram(self, results: Dict[str, Any]):
        """Plot histogram of measured volumes."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        volumes_ul = np.array(results['volumes_ml']) * 1000  # Convert to μL
        target_ul = results['target_volume_ml'] * 1000
        
        ax.hist(volumes_ul, bins=max(3, len(volumes_ul)//2), alpha=0.7, edgecolor='black')
        ax.axvline(target_ul, color='red', linestyle='--', label=f'Target: {target_ul:.1f} μL')
        ax.axvline(np.mean(volumes_ul), color='green', linestyle='-', label=f'Mean: {np.mean(volumes_ul):.1f} μL')
        
        ax.set_xlabel('Volume (μL)')
        ax.set_ylabel('Frequency')
        ax.set_title('Volume Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
        self.simulate_checkbox.setChecked(True)  # Default to simulation
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
        scroll.setMaximumHeight(400)
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
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.measure_btn = QPushButton("MEASURE")
        self.measure_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.measure_btn.clicked.connect(self.run_measurement)
        
        self.optimize_btn = QPushButton("RUN OPTIMIZATION")
        self.optimize_btn.setStyleSheet("QPushButton { background-color: #cccccc; color: #666666; font-weight: bold; padding: 10px; }")
        self.optimize_btn.setEnabled(False)
        self.optimize_btn.setToolTip("Optimization temporarily disabled - config format needs to match actual calibration system")
        self.optimize_btn.clicked.connect(self.run_optimization)
        
        button_layout.addWidget(self.measure_btn)
        button_layout.addWidget(self.optimize_btn)
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
        
        layout.addStretch()  # Push everything to top
        return widget
    
    def create_output_column(self) -> QWidget:
        """Create the output column with plots and statistics."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Row 1 - Plots
        plots_layout = QHBoxLayout()
        
        self.mass_time_plot = PlotWidget("Mass vs Time")
        self.volume_histogram_plot = PlotWidget("Volume Histogram")
        
        plots_layout.addWidget(self.mass_time_plot)
        plots_layout.addWidget(self.volume_histogram_plot)
        layout.addLayout(plots_layout)
        
        # Row 1b - Optimization Plot (full width)
        self.optimization_plot = OptimizationPlotWidget()
        layout.addWidget(self.optimization_plot)
        
        # Row 2 - Statistics
        stats_group = QGroupBox("Results Summary")
        stats_layout = QFormLayout()
        
        self.time_label = QLabel("--")
        self.accuracy_label = QLabel("--")
        self.cv_label = QLabel("--")
        
        stats_layout.addRow("Average Time:", self.time_label)
        stats_layout.addRow("Accuracy (% dev):", self.accuracy_label)
        stats_layout.addRow("Precision (CV%):", self.cv_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
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
            # Collect parameters
            parameters = {}
            for param_name, widget in self.parameter_widgets.items():
                parameters[param_name] = widget.get_parameter_value()
            
            vial_name = self.liquid_combo.currentText()
            volume_ml = self.volume_spinbox.value()
            replicates = self.replicates_spinbox.value()
            simulate = self.simulate_checkbox.isChecked()
            
            # Disable UI during measurement
            self.measure_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Clear previous results
            self.clear_measurement_data()
            
            # Start measurement in background thread
            self.measurement_worker = MeasurementWorker(vial_name, volume_ml, replicates, parameters, simulate)
            self.measurement_worker.measurement_complete.connect(self.on_measurement_complete)
            self.measurement_worker.measurement_error.connect(self.on_measurement_error)
            self.measurement_worker.progress_update.connect(self.on_progress_update)
            self.measurement_worker.replicate_complete.connect(self.on_replicate_complete)
            
            # Store target volume for progressive calculations
            self.current_target_volume = volume_ml
            
            self.measurement_worker.start()
            
        except Exception as e:
            self.on_measurement_error(f"Error starting measurement: {str(e)}")
    
    def run_optimization(self):
        """Run Bayesian optimization with real-time progress monitoring."""
        try:
            # Collect current GUI parameters
            parameters = {}
            for param_name, widget in self.parameter_widgets.items():
                parameters[param_name] = widget.get_parameter_value()
            
            liquid = self.liquid_combo.currentText()
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
            self.optimization_worker.optimization_started.connect(self.on_optimization_started)
            self.optimization_worker.measurement_update.connect(self.on_optimization_measurement)
            self.optimization_worker.optimization_complete.connect(self.on_optimization_complete)
            self.optimization_worker.optimization_error.connect(self.on_optimization_error)
            
            self.optimization_worker.start()
            self.add_status_message("Starting Bayesian optimization...")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.add_status_message(f"Error in run_optimization: {error_details}")
            self.on_optimization_error(f"Error starting optimization: {str(e)}")
    
    def on_optimization_started(self, output_dir: str):
        """Handle optimization process started."""
        self.add_status_message(f"Optimization started - monitoring: {output_dir}")
        
    def on_optimization_measurement(self, measurement_data: dict):
        """Handle new measurement from optimization process."""
        # Add measurement to optimization plot
        self.optimization_plot.add_measurement(measurement_data)
        
        # Store measurement for unified updates
        self.optimization_measurements.append(measurement_data)
        
        # Group by strategy for replicate-like behavior
        strategy = measurement_data.get('strategy', 'unknown')
        
        if strategy not in self.current_trial_measurements:
            self.current_trial_measurements[strategy] = []
        self.current_trial_measurements[strategy].append(measurement_data)
        
        # Update all plots with current strategy data (treat as replicates)
        current_trial_data = self.current_trial_measurements[strategy]
        self.update_plots_from_optimization_data(current_trial_data)
        
        # Update status
        volume_ul = measurement_data.get('measured_volume_ul', 0)
        deviation = measurement_data.get('deviation_pct', 0)
        count = measurement_data.get('total_measurement_count', 0)
        
        self.add_status_message(f"[{strategy.upper()}] Measurement #{count}: {volume_ul:.1f} uL ({deviation:+.1f}% dev)")
        
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
    
    def on_measurement_complete(self, results: Dict[str, Any]):
        """Handle completed measurement (all replicates done)."""
        self.current_results = results
        
        # Final update to plots (should already be current from progressive updates)
        if not hasattr(self, 'mass_data_replicates') or len(self.mass_data_replicates) == 0:
            # Fallback: use old plotting if progressive didn't work
            self.mass_time_plot.plot_mass_time(results)
            self.volume_histogram_plot.plot_volume_histogram(results)
        
        # Update final statistics with complete timing info
        self.time_label.setText(f"{results['mean_time_s']:.2f} seconds")
        
        # Re-enable UI
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.add_status_message(f"Measurement complete: {results['mean_volume_ml']*1000:.1f} μL " +
                               f"({results['accuracy_pct']:+.1f}%, CV={results['cv_pct']:.1f}%)")
    
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
    
    def add_status_message(self, message: str):
        """Add message to status text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.status_text.append(formatted_message)
    
    def clear_results(self):
        """Clear current results and plots."""
        self.current_results = None
        self.time_label.setText("Not available")
        self.accuracy_label.setText("Not available")
        self.cv_label.setText("Not available")
        
        # Clear plots
        self.mass_time_plot.clear()
        self.volume_histogram_plot.clear()
        
    def clear_measurement_data(self):
        """Clear progressive measurement data at start of new measurement."""
        if hasattr(self, 'mass_data_replicates'):
            self.mass_data_replicates = []
        if hasattr(self, 'volumes_progressive'):
            self.volumes_progressive = []
        
        # Clear plots
        self.mass_time_plot.clear()
        self.volume_histogram_plot.clear()
        
    def on_optimization_started(self, output_dir: str):
        """Handle optimization process started."""
        self.add_status_message(f"Optimization started - monitoring: {output_dir}")
        
    def on_optimization_measurement(self, measurement_data: dict):
        """Handle new measurement from optimization process."""
        # Add measurement to optimization plot
        self.optimization_plot.add_measurement(measurement_data)
        
        # Update status
        strategy = measurement_data.get('strategy', 'unknown')
        volume_ul = measurement_data.get('measured_volume_ul', 0)
        deviation = measurement_data.get('deviation_pct', 0)
        count = measurement_data.get('total_measurement_count', 0)
        
        self.add_status_message(f"[{strategy.upper()}] Measurement #{count}: {volume_ul:.1f} μL ({deviation:+.1f}% dev)")
        
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
            self.volume_histogram_plot.figure.clear()
            self.volume_histogram_plot.canvas.draw()
    
    def find_latest_mass_file(self, vial_name):
        """Find the most recent mass measurement CSV file for a vial."""
        try:
            import glob
            from pathlib import Path
            import time
            
            # Wait a moment for file to be written
            time.sleep(0.5)
            
            # Search pattern for mass files
            pattern = f"output/mass_measurements/*/mass_data_{vial_name}_*.csv"
            files = glob.glob(pattern)
            
            if not files:
                print(f"No mass data files found for vial {vial_name}")
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
        
        # Update volume histogram
        self.plot_volume_histogram_progressive()
    
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
    
    def plot_volume_histogram_progressive(self):
        """Plot volume histogram with all volumes collected so far."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        if len(self.volume_data_replicates) == 0:
            return
            
        self.volume_histogram_plot.figure.clear()
        ax = self.volume_histogram_plot.figure.add_subplot(111)
        
        volumes_ul = np.array(self.volume_data_replicates) * 1000  # Convert to μL
        target_ul = self.current_target_volume * 1000
        
        # For single measurement, show as bar; for multiple, show as histogram
        if len(volumes_ul) == 1:
            ax.bar([volumes_ul[0]], [1], width=2, alpha=0.7, edgecolor='black', label='Measurement')
        else:
            bins = max(3, len(volumes_ul)//2 + 1)
            ax.hist(volumes_ul, bins=bins, alpha=0.7, edgecolor='black')
        
        # Add target and mean lines
        ax.axvline(target_ul, color='red', linestyle='--', linewidth=2, label=f'Target: {target_ul:.1f} μL')
        ax.axvline(np.mean(volumes_ul), color='green', linestyle='-', linewidth=2, 
                  label=f'Mean: {np.mean(volumes_ul):.1f} μL')
        
        ax.set_xlabel('Volume (μL)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Volume Distribution ({len(volumes_ul)} measurements)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.volume_histogram_plot.canvas.draw()
    
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
        
        # Time will be updated when all measurements complete
        if hasattr(self, 'current_results') and self.current_results:
            self.time_label.setText(f"{self.current_results['mean_time_s']:.2f} seconds")
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
        if hasattr(self, 'volume_histogram_plot'):
            self.volume_histogram_plot.plot_volume_histogram(results)
        
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