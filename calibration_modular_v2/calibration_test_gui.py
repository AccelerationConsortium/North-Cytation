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
    QSplitter, QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, pyqtSignal
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
    'overaspirate_vol': {'value': 0.004, 'min': 0.0, 'max': 0.050, 'unit': 'mL'},
    'aspirate_speed': {'value': 10.0, 'min': 1.0, 'max': 50.0, 'unit': 'mm/s'},
    'dispense_speed': {'value': 10.0, 'min': 1.0, 'max': 50.0, 'unit': 'mm/s'},
    'aspirate_wait_time': {'value': 0.0, 'min': 0.0, 'max': 10.0, 'unit': 's'},
    'dispense_wait_time': {'value': 1.5, 'min': 0.0, 'max': 10.0, 'unit': 's'},
    'pre_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.020, 'unit': 'mL'},
    'post_asp_air_vol': {'value': 0.0, 'min': 0.0, 'max': 0.020, 'unit': 'mL'},
    'blowout_vol': {'value': 0.0, 'min': 0.0, 'max': 0.010, 'unit': 'mL'},
    'retract_speed': {'value': 5.0, 'min': 1.0, 'max': 20.0, 'unit': 'mm/s'},
    'post_retract_wait_time': {'value': 0.0, 'min': 0.0, 'max': 5.0, 'unit': 's'},
    'asp_disp_cycles': {'value': 0, 'min': 0, 'max': 5, 'unit': 'cycles'}
}

# Available vials (hardcoded for now - could be read from status file)
AVAILABLE_VIALS = [
    'water_source', 'ethanol_source', 'isopropanol_source', 'DMSO_source',
    'measurement_vial_1', 'measurement_vial_2', 'waste_vial'
]

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
        
        # Value spinbox
        self.value_spinbox = QDoubleSpinBox()
        self.value_spinbox.setDecimals(4)
        self.value_spinbox.setMinimum(self.config['min'])
        self.value_spinbox.setMaximum(self.config['max'])
        self.value_spinbox.setValue(self.config['value'])
        self.value_spinbox.setSuffix(f" {self.config['unit']}")
        layout.addWidget(self.value_spinbox)
        
        # Min value spinbox
        min_label = QLabel("Min:")
        layout.addWidget(min_label)
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setDecimals(4)
        self.min_spinbox.setValue(self.config['min'])
        self.min_spinbox.setSuffix(f" {self.config['unit']}")
        layout.addWidget(self.min_spinbox)
        
        # Max value spinbox
        max_label = QLabel("Max:")
        layout.addWidget(max_label)
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setDecimals(4)
        self.max_spinbox.setValue(self.config['max'])
        self.max_spinbox.setSuffix(f" {self.config['unit']}")
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

class MeasurementWorker(QThread):
    """Worker thread for running measurements to avoid GUI freezing."""
    
    measurement_complete = pyqtSignal(dict)  # Results
    measurement_error = pyqtSignal(str)     # Error message
    progress_update = pyqtSignal(str)       # Status message
    
    def __init__(self, vial_name: str, volume_ml: float, replicates: int, 
                 parameters: Dict[str, float], parent=None):
        super().__init__(parent)
        self.vial_name = vial_name
        self.volume_ml = volume_ml
        self.replicates = replicates
        self.parameters = parameters
    
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
                    'simulate': False  # Use real hardware
                }
            }
            
            # Initialize protocol
            protocol = HardwareCalibrationProtocol()
            state = protocol.initialize(config)
            
            # Override source vial based on GUI selection
            state['source_vial'] = self.vial_name
            
            results = []
            for rep in range(self.replicates):
                self.progress_update.emit(f"Running replicate {rep+1}/{self.replicates}...")
                
                # Create parameters structure expected by protocol
                pipetting_params = {
                    'overaspirate_vol': self.parameters.get('overaspirate_vol', 0.004),
                    'parameters': {k: v for k, v in self.parameters.items() if k != 'overaspirate_vol'}
                }
                
                # Execute measurement
                measurement_list = protocol.measure(state, self.volume_ml, pipetting_params, replicates=1)
                measurement = measurement_list[0]  # Single replicate
                
                results.append(measurement)
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
        self.setup_ui()
        self.parameter_widgets = {}
        self.measurement_worker = None
        self.current_results = None
        
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
        self.liquid_combo.setCurrentText('water_source')
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
        self.optimize_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.optimize_btn.clicked.connect(self.run_optimization)
        self.optimize_btn.setEnabled(False)  # Placeholder
        
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
        """Show dialog to add custom parameter (placeholder)."""
        QMessageBox.information(self, "Add Parameter", "Add parameter functionality coming soon!")
    
    def show_remove_parameter_dialog(self):
        """Show dialog to remove parameter (placeholder)."""
        QMessageBox.information(self, "Remove Parameter", "Remove parameter functionality coming soon!")
    
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
            
            # Disable UI during measurement
            self.measure_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Clear previous results
            self.clear_results()
            
            # Start measurement in background thread
            self.measurement_worker = MeasurementWorker(vial_name, volume_ml, replicates, parameters)
            self.measurement_worker.measurement_complete.connect(self.on_measurement_complete)
            self.measurement_worker.measurement_error.connect(self.on_measurement_error)
            self.measurement_worker.progress_update.connect(self.on_progress_update)
            self.measurement_worker.start()
            
        except Exception as e:
            self.on_measurement_error(f"Error starting measurement: {str(e)}")
    
    def run_optimization(self):
        """Run Bayesian optimization (placeholder)."""
        QMessageBox.information(self, "Optimization", "Bayesian optimization functionality coming soon!")
    
    def on_measurement_complete(self, results: Dict[str, Any]):
        """Handle completed measurement."""
        self.current_results = results
        
        # Update plots
        self.mass_time_plot.plot_mass_time(results)
        self.volume_histogram_plot.plot_volume_histogram(results)
        
        # Update statistics
        self.time_label.setText(f"{results['mean_time_s']:.2f} seconds")
        self.accuracy_label.setText(f"{results['accuracy_pct']:+.1f}%")
        self.cv_label.setText(f"{results['cv_pct']:.1f}%")
        
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
        """Clear all results displays."""
        self.time_label.setText("--")
        self.accuracy_label.setText("--")
        self.cv_label.setText("--")
        
        if MATPLOTLIB_AVAILABLE:
            self.mass_time_plot.figure.clear()
            self.mass_time_plot.canvas.draw()
            self.volume_histogram_plot.figure.clear()
            self.volume_histogram_plot.canvas.draw()

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