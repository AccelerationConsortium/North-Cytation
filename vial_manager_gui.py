#!/usr/bin/env python3
"""
Visual Vial Management GUI - PySide6

A visual interface for managing robot vial status files.
Shows vials as clickable grid arranged by racks, with editing capabilities.

Usage:
    python vial_manager_gui.py [status_file.csv]

Features:
- Visual vial grid organized by location/rack
- Click vials to edit properties (name, volume, location, etc.)
- Color-coded status indicators (volume, capped, etc.)
- Save/load CSV status files
- Backup on save
- Drag & drop vial moving between locations
"""

import sys
import os
import csv
import yaml
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QGridLayout, QPushButton, QLabel,
    QMessageBox, QFileDialog, QDialog, QFormLayout,
    QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QStatusBar, QMenuBar, QToolBar, QScrollArea, QFrame,
    QSplitter, QGroupBox, QTextEdit, QDialogButtonBox, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize, QMimeData, QTimer
from PySide6.QtGui import (
    QAction, QPalette, QPainter, QFont, QPixmap, 
    QDrag, QColor, QBrush, QPen
)

# Import ConfigManager for workflow config handling
try:
    from config_manager import ConfigManager
except ImportError:
    ConfigManager = None


class ClickableLabel(QLabel):
    '''Clickable QLabel for empty vial slots.'''
    clicked = Signal()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class VialWidget(QFrame):
    """Visual representation of a single vial that can be clicked and edited."""
    
    vial_clicked = Signal(dict)  # Emits vial data when clicked
    vial_moved = Signal(dict, str, int)  # Emits (vial_data, new_location, new_index)
    
    def __init__(self, vial_data: Dict, parent=None):
        super().__init__(parent)
        self.vial_data = vial_data.copy()
        self.setFixedSize(150, 100)  # Even wider vial widget for longer names
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Vial name label
        self.name_label = QLabel(self.vial_data.get('vial_name', 'Unknown'))
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(8)
        font.setBold(True)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)
        
        # Volume label
        volume = self.vial_data.get('vial_volume', 0)
        try:
            volume_val = float(volume)
            volume_text = f"{volume_val:.1f}mL"
        except (ValueError, TypeError):
            volume_text = "?.?mL"
        
        self.volume_label = QLabel(volume_text)
        self.volume_label.setAlignment(Qt.AlignCenter)
        font.setPointSize(7)
        font.setBold(False)
        self.volume_label.setFont(font)
        layout.addWidget(self.volume_label)
        
        # Status indicator
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        self._update_appearance()
    
    def _update_appearance(self):
        """Update visual appearance based on vial properties."""
        # Color coding based on volume and status
        volume = self.vial_data.get('vial_volume', 0)
        try:
            volume_val = float(volume)
        except (ValueError, TypeError):
            volume_val = 0
            
        capped = self.vial_data.get('capped', False)
        if isinstance(capped, str):
            capped = capped.lower() in ('true', '1', 'yes')
        
        # Color based on volume level (keep colors, remove text labels)
        if volume_val <= 0:
            color = "#ffcccc"  # Light red - empty
        elif volume_val < 1.0:
            color = "#ffffcc"  # Light yellow - low
        elif volume_val < 3.0:
            color = "#ccffcc"  # Light green - medium
        else:
            color = "#cccfff"  # Light blue - full
        
        # Show clear cap status text
        cap_type = self.vial_data.get('cap_type', 'open')
        if capped:
            if cap_type == 'open':
                status = "Open Cap"  # Capped but accessible for pipetting
            elif cap_type and cap_type not in ['closed', 'capped', '']:
                status = f"Closed Cap ({cap_type})"  # Show specific cap type like screw, snap
            else:
                status = "Closed Cap"  # Generic closed cap
        else:
            status = "Open"  # No cap at all
        
        self.status_label.setText(status)
        
        # Simple consistent border for all vials
        self.setStyleSheet(f"""
            VialWidget {{
                background-color: {color};
                border: 2px solid #666;
                border-radius: 8px;
            }}
            VialWidget:hover {{
                border-color: #0078d4;
                border-width: 3px;
            }}
        """)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks to emit vial data."""
        if event.button() == Qt.LeftButton:
            self.vial_clicked.emit(self.vial_data.copy())
        super().mousePressEvent(event)
    
    def update_vial_data(self, new_data: Dict):
        """Update vial data and refresh display."""
        self.vial_data.update(new_data)
        
        # Update labels
        self.name_label.setText(self.vial_data.get('vial_name', 'Unknown'))
        
        volume = self.vial_data.get('vial_volume', 0)
        try:
            volume_val = float(volume)
            volume_text = f"{volume_val:.1f}mL"
        except (ValueError, TypeError):
            volume_text = "?.?mL"
        self.volume_label.setText(volume_text)
        
        self._update_appearance()
    
    def get_vial_data(self) -> Dict:
        """Get current vial data."""
        return self.vial_data.copy()


class VialEditDialog(QDialog):
    """Dialog for editing vial properties."""
    
    def __init__(self, vial_data: Dict, parent=None):
        super().__init__(parent)
        self.vial_data = vial_data.copy()
        self.setWindowTitle(f"Edit Vial: {vial_data.get('vial_name', 'Unknown')}")
        self.setModal(True)
        self.resize(500, 625)  # 25% larger than 400x500
        
        self._setup_ui()
        self._populate_fields()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Form layout for vial properties
        form_layout = QFormLayout()
        
        # Basic properties
        self.name_edit = QLineEdit()
        form_layout.addRow("Vial Name:", self.name_edit)
        
        self.location_edit = QLineEdit()
        form_layout.addRow("Location:", self.location_edit)
        
        self.location_index_spin = QSpinBox()
        self.location_index_spin.setRange(0, 99)
        form_layout.addRow("Location Index:", self.location_index_spin)
        
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.0, 50.0)
        self.volume_spin.setDecimals(3)
        self.volume_spin.setSuffix(" mL")
        form_layout.addRow("Volume:", self.volume_spin)
        
        self.capped_check = QCheckBox()
        form_layout.addRow("Capped:", self.capped_check)
        
        self.cap_type_edit = QComboBox()
        self.cap_type_edit.addItems(["closed", "open"])
        form_layout.addRow("Cap Type:", self.cap_type_edit)
        
        self.vial_type_combo = QComboBox()
        self.vial_type_combo.addItems(["8_mL", "20_mL", "50_mL", "custom"])
        self.vial_type_combo.setEditable(True)
        form_layout.addRow("Vial Type:", self.vial_type_combo)
        
        # Home location
        self.home_location_edit = QLineEdit()
        form_layout.addRow("Home Location:", self.home_location_edit)
        
        self.home_index_spin = QSpinBox()
        self.home_index_spin.setRange(0, 99)
        form_layout.addRow("Home Index:", self.home_index_spin)
        
        # Notes field (optional - for additional information)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setPlaceholderText("Optional notes about this vial...")
        form_layout.addRow("Notes (optional):", self.notes_edit)
        
        layout.addLayout(form_layout)
        
        # Button box with Remove option
        button_layout = QHBoxLayout()
        
        self.remove_button = QPushButton("Remove Vial")
        self.remove_button.setStyleSheet("QPushButton { background-color: #ffcccc; }")
        button_layout.addWidget(self.remove_button)
        
        button_layout.addStretch()
        
        ok_button = QPushButton("Save Changes")
        cancel_button = QPushButton("Cancel")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        self.remove_button.clicked.connect(self.remove_vial)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def remove_vial(self):
        """Mark vial for removal."""
        reply = QMessageBox.question(
            self, "Remove Vial", 
            f"Remove vial '{self.vial_data.get('vial_name', 'Unknown')}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Set a special flag to indicate removal
            self.vial_data['_remove'] = True
            self.accept()
    
    def _populate_fields(self):
        """Populate form fields with vial data."""
        self.name_edit.setText(str(self.vial_data.get('vial_name', '')))
        self.location_edit.setText(str(self.vial_data.get('location', '')))
        
        try:
            loc_index = int(self.vial_data.get('location_index', 0))
        except (ValueError, TypeError):
            loc_index = 0
        self.location_index_spin.setValue(loc_index)
        
        try:
            volume = float(self.vial_data.get('vial_volume', 0))
        except (ValueError, TypeError):
            volume = 0.0
        self.volume_spin.setValue(volume)
        
        capped = self.vial_data.get('capped', False)
        if isinstance(capped, str):
            capped = capped.lower() in ('true', '1', 'yes')
        self.capped_check.setChecked(bool(capped))
        
        cap_type = str(self.vial_data.get('cap_type', 'closed'))
        index = self.cap_type_edit.findText(cap_type)
        if index >= 0:
            self.cap_type_edit.setCurrentIndex(index)
        else:
            self.cap_type_edit.setCurrentIndex(0)  # Default to "closed"
        
        vial_type = str(self.vial_data.get('vial_type', '8_mL'))
        index = self.vial_type_combo.findText(vial_type)
        if index >= 0:
            self.vial_type_combo.setCurrentIndex(index)
        else:
            self.vial_type_combo.setCurrentText(vial_type)
        
        self.home_location_edit.setText(str(self.vial_data.get('home_location', '')))
        
        try:
            home_index = int(self.vial_data.get('home_location_index', 0))
        except (ValueError, TypeError):
            home_index = 0
        self.home_index_spin.setValue(home_index)
        
        self.notes_edit.setPlainText(str(self.vial_data.get('notes', '')))
    
    def get_vial_data(self) -> Dict:
        """Get updated vial data from form."""
        return {
            'vial_name': self.name_edit.text(),
            'location': self.location_edit.text(),
            'location_index': self.location_index_spin.value(),
            'vial_volume': self.volume_spin.value(),
            'capped': self.capped_check.isChecked(),
            'cap_type': self.cap_type_edit.currentText(),
            'vial_type': self.vial_type_combo.currentText(),
            'home_location': self.home_location_edit.text(),
            'home_location_index': self.home_index_spin.value(),
            'notes': self.notes_edit.toPlainText(),
            # Preserve other fields that might exist
            **{k: v for k, v in self.vial_data.items() 
               if k not in ['vial_name', 'location', 'location_index', 'vial_volume',
                           'capped', 'cap_type', 'vial_type', 'home_location', 
                           'home_location_index', 'notes']}
        }


class LogVialsWidget(QWidget):
    """Widget to display vials found in the latest log file with status indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_vials = set()
        self.csv_vials = set()
        self.errors_detected = False
        print("[DEBUG] LogVialsWidget initialized")
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the log vials display UI."""
        layout = QVBoxLayout(self)
        
        # Header label
        header = QLabel("Log File Vials")
        header.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
        layout.addWidget(header)
        
        # Manual load button - below the label
        self.load_button = QPushButton("Load Log File")
        self.load_button.setFixedSize(100, 25)
        self.load_button.setToolTip("Load log file manually")
        self.load_button.clicked.connect(self._load_log_file_manually)
        layout.addWidget(self.load_button)
        
        # Scrollable list - much narrower but taller
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(120)  # Much narrower
        scroll.setMaximumHeight(500)  # Taller to use more vertical space
        
        self.vials_list_widget = QWidget()
        self.vials_list_layout = QVBoxLayout(self.vials_list_widget)
        self.vials_list_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.vials_list_widget)
        layout.addWidget(scroll)
        
        # Error warning label (hidden by default)
        self.error_warning = QLabel("⚠️ Errors Detected")
        self.error_warning.setStyleSheet("""
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            border-radius: 3px;
            padding: 3px;
            font-size: 9px;
            font-weight: bold;
        """)
        self.error_warning.hide()
        layout.addWidget(self.error_warning)
    
    def update_log_vials(self):
        """Find and parse the latest log file for vial names."""
        try:
            # Find the most recent log file
            log_pattern = os.path.join("logs", "experiment_log*.log")
            log_files = glob.glob(log_pattern)
            
            if not log_files:
                print("[DEBUG] No log files found")
                return
            
            # Sort by modification time, get the most recent
            latest_log = max(log_files, key=os.path.getmtime)
            log_name = os.path.basename(latest_log)
            
            # Parse the log file for vial names and errors
            vial_names = set()
            errors_found = False
            pattern = r'Pipetting from vial\\s+([^,]+),'
            
            with open(latest_log, 'r', encoding='utf-8') as f:
                for line in f:
                    # Check for errors
                    if '- ERROR -' in line:
                        errors_found = True
                    
                    # Check for vial names
                    matches = re.findall(pattern, line)
                    for match in matches:
                        vial_name = match.strip()
                        if vial_name:
                            vial_names.add(vial_name)
            
            self.log_vials = vial_names
            self.errors_detected = errors_found
            print(f"[DEBUG] Loaded from: {log_name} ({len(vial_names)} vials)")
            self._update_display()
            
        except Exception as e:
            print(f"[DEBUG] Error reading logs: {str(e)}")
    
    def update_csv_vials(self, csv_vials):
        """Update the list of vials from the loaded CSV file."""
        print(f"[DEBUG] Updating CSV vials: {csv_vials}")
        self.csv_vials = set(csv_vials) if csv_vials else set()
        print(f"[DEBUG] CSV vials set: {self.csv_vials}")
        self._update_display()
    
    def _load_log_file_manually(self):
        """Allow user to manually select a log file."""
        print("[DEBUG] Manual log file selection requested")
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Log File",
            "logs",
            "Log Files (*.log);;All Files (*)"
        )
        
        if file_path:
            print(f"[DEBUG] User selected log file: {file_path}")
            self._parse_specific_log_file(file_path)
    
    def _parse_specific_log_file(self, log_file_path):
        """Parse a specific log file for vial names."""
        print(f"[DEBUG] Parsing specific log file: {log_file_path}")
        try:
            if not os.path.exists(log_file_path):
                print(f"[DEBUG] ERROR: Log file does not exist: {log_file_path}")
                return
                
            file_size = os.path.getsize(log_file_path)
            print(f"[DEBUG] Log file size: {file_size} bytes")
            
            # Parse the log file for vial names and errors
            vial_names = set()
            errors_found = False
            patterns_to_try = [
                r'Pipetting from vial\s+([^,]+),',
                r'Pipetting from vial ([^,]+),',
                r'from vial\s+([^,]+),',
                r'from vial ([^,]+),'
            ]
            
            line_count = 0
            total_matches_found = 0
            
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    
                    # Show first few lines for debugging
                    if line_num <= 3:
                        print(f"[DEBUG] Line {line_num}: {line.strip()[:80]}...")
                    
                    # Check for errors
                    if '- ERROR -' in line:
                        errors_found = True
                        print(f"[DEBUG] Error detected on line {line_num}")
                    
                    # Try each pattern
                    for i, pattern in enumerate(patterns_to_try):
                        matches = re.findall(pattern, line)
                        if matches:
                            total_matches_found += 1
                            for match in matches:
                                vial_name = match.strip()
                                if vial_name:
                                    vial_names.add(vial_name)
                                    print(f"[DEBUG] Pattern {i+1} found vial: '{vial_name}' (line {line_num})")
            
            print(f"[DEBUG] Processed {line_count} lines, found {total_matches_found} pipetting matches")
            print(f"[DEBUG] Unique vials found: {sorted(vial_names)} (total: {len(vial_names)})")
            
            self.log_vials = vial_names
            self.errors_detected = errors_found
            log_name = os.path.basename(log_file_path)
            if errors_found:
                print(f"[DEBUG] Manual load complete: {log_name} ({len(vial_names)} vials, ERRORS DETECTED)")
            else:
                print(f"[DEBUG] Manual load complete: {log_name} ({len(vial_names)} vials)")
            self._update_display()
            
        except Exception as e:
            import traceback
            error_msg = f"Error parsing log: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
    
    def _update_display(self):
        """Update the visual display of log vials with status indicators."""
        print(f"[DEBUG] Updating display with {len(self.log_vials)} log vials and {len(self.csv_vials)} CSV vials")
        
        # Clear existing items
        for i in reversed(range(self.vials_list_layout.count())):
            child = self.vials_list_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add vials with status indicators
        for vial_name in sorted(self.log_vials):
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(1, 1, 1, 1)  # Smaller margins
            
            # Status indicator
            status_label = QLabel()
            if vial_name in self.csv_vials:
                status_label.setText("✅")
                status_label.setToolTip(f"{vial_name} - Found in CSV")
                item_widget.setStyleSheet("background-color: #e8f5e8; border-radius: 2px;")
                print(f"[DEBUG] ✅ {vial_name} - found in CSV")
            else:
                status_label.setText("❌")
                status_label.setToolTip(f"{vial_name} - Missing from CSV")
                item_widget.setStyleSheet("background-color: #ffe8e8; border-radius: 2px;")
                print(f"[DEBUG] ❌ {vial_name} - missing from CSV")
            
            status_label.setFixedWidth(16)  # Smaller icon
            item_layout.addWidget(status_label)
            
            # Vial name - smaller text
            name_label = QLabel(vial_name)
            name_label.setStyleSheet("font-size: 8px; padding: 1px;")  # Smaller font
            name_label.setWordWrap(True)
            item_layout.addWidget(name_label)
            
            self.vials_list_layout.addWidget(item_widget)
        
        # Add stretch to push items to top
        self.vials_list_layout.addStretch()
        
        # Update error warning visibility
        if self.errors_detected:
            self.error_warning.show()
            print(f"[DEBUG] Display update complete - ERRORS DETECTED")
        else:
            self.error_warning.hide()
            print(f"[DEBUG] Display update complete")


class VialRackWidget(QScrollArea):
    """Widget showing vials arranged in a rack grid."""
    
    vial_edited = Signal(dict)  # Emits updated vial data
    vial_added = Signal(dict)   # Emits new vial data
    
    def __init__(self, location_name: str, parent=None):
        super().__init__(parent)
        self.location_name = location_name
        self.vials = {}  # location_index -> VialWidget
        self.vial_data_list = []
        self.empty_slots = {}  # location_index -> QLabel (for clickable empty slots)
        
        # Create scroll area content
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        self.setWidgetResizable(True)
        
        # Grid layout for vials (8x6 = 48 positions for main_8mL_rack)
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setSpacing(5)
        
        # Configure grid size based on location
        if "8mL" in location_name or "main" in location_name:
            self.grid_rows = 6
            self.grid_cols = 8
        elif "large" in location_name:
            self.grid_rows = 2  # Keep as 2x2 for better layout
            self.grid_cols = 2
        elif "50mL" in location_name:
            self.grid_rows = 1
            self.grid_cols = 2
        else:
            self.grid_rows = 4  # Default
            self.grid_cols = 4
        
        self._setup_grid()
    
    def _setup_grid(self):
        """Set up the empty vial grid."""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate position number to match our column-major right-to-left layout
                column_number = (self.grid_cols - 1) - col  # Rightmost col = 0
                position = column_number * self.grid_rows + row
                
                # Create clickable empty slot placeholder
                placeholder = ClickableLabel(f"{position}")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("""
                    ClickableLabel {
                        border: 1px dashed #ccc;
                        border-radius: 8px;
                        color: #999;
                        background-color: #f9f9f9;
                    }
                    ClickableLabel:hover {
                        background-color: #e8f4fd;
                        border-color: #0078d4;
                    }
                """)
                placeholder.setFixedSize(80, 100)
                placeholder.position = position
                placeholder.location_name = self.location_name
                placeholder.clicked.connect(self._on_empty_slot_clicked)
                
                self.grid_layout.addWidget(placeholder, row, col)
                self.empty_slots[position] = placeholder
    
    def add_vials(self, vials_data: List[Dict]):
        """Add vials to the rack."""
        self.vial_data_list = [v for v in vials_data if v.get('location') == self.location_name]
        
        for vial_data in self.vial_data_list:
            try:
                location_index = int(vial_data.get('location_index', 0))
            except (ValueError, TypeError):
                continue
                
            if location_index >= self.grid_rows * self.grid_cols:
                continue  # Skip if position is out of bounds
            
            # Calculate grid position: 0=top-right, 47=bottom-left
            # Column-major ordering going right-to-left, top-to-bottom
            column_number = location_index // self.grid_rows  # Which column (0 for positions 0-5, 1 for 6-11, etc.)
            row = location_index % self.grid_rows              # Which row within that column
            col = (self.grid_cols - 1) - column_number         # Flip horizontally (rightmost column = 0)
            
            # Remove placeholder and add vial widget
            old_item = self.grid_layout.itemAtPosition(row, col)
            if old_item:
                old_widget = old_item.widget()
                if old_widget:
                    old_widget.deleteLater()
            
            # Create vial widget
            vial_widget = VialWidget(vial_data)
            vial_widget.vial_clicked.connect(self._on_vial_clicked)
            
            self.grid_layout.addWidget(vial_widget, row, col)
            self.vials[location_index] = vial_widget
    
    def _on_vial_clicked(self, vial_data: Dict):
        """Handle vial click to open edit dialog."""
        dialog = VialEditDialog(vial_data, self)
        if dialog.exec() == QDialog.Accepted:
            updated_data = dialog.get_vial_data()
            
            # Get location index for grid operations
            location_index = int(vial_data.get('location_index', 0))
            
            # Check for vial removal
            if updated_data.get('_remove'):
                # Remove vial from grid and data
                if location_index in self.vials:
                    self.vials[location_index].deleteLater()
                    del self.vials[location_index]
                
                # Remove from data list
                self.vial_data_list = [v for v in self.vial_data_list if v.get('vial_index') != vial_data.get('vial_index')]
                
                # Restore empty placeholder
                row = location_index % self.grid_rows
                column_number = (self.grid_cols - 1) - (location_index // self.grid_rows)
                col = column_number if column_number >= 0 else 0
                
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    placeholder = ClickableLabel(f"{location_index}")
                    placeholder.setAlignment(Qt.AlignCenter)
                    placeholder.setStyleSheet("""
                        ClickableLabel {
                            border: 1px dashed #ccc;
                            border-radius: 8px;
                            color: #999;
                            background-color: #f9f9f9;
                        }
                        ClickableLabel:hover {
                            background-color: #e8f4fd;
                            border-color: #0078d4;
                        }
                    """)
                    placeholder.setFixedSize(80, 100)
                    placeholder.position = location_index
                    placeholder.location_name = self.location_name
                    placeholder.clicked.connect(self._on_empty_slot_clicked)
                    
                    self.grid_layout.addWidget(placeholder, row, col)
                    self.empty_slots[location_index] = placeholder
                
                self.vial_edited.emit(updated_data)  # Notify of removal
            else:
                # Update the vial widget
                if location_index in self.vials:
                    self.vials[location_index].update_vial_data(updated_data)
                
                # Update our data list - use original vial_data to find the entry
                for i, vial in enumerate(self.vial_data_list):
                    if vial.get('vial_index') == vial_data.get('vial_index'):
                        self.vial_data_list[i] = updated_data
                        break
                
                self.vial_edited.emit(updated_data)
    
    def _on_empty_slot_clicked(self):
        """Handle clicking on empty slot to add new vial."""
        sender = self.sender()
        position = getattr(sender, 'position', 0)
        
        # Create new vial data template
        new_vial_data = {
            'vial_index': self._get_next_vial_index(),
            'vial_name': f'new_vial_{position}',
            'location': self.location_name,
            'location_index': position,
            'vial_volume': 0.0,
            'capped': False,
            'cap_type': 'open',
            'vial_type': '8_mL',
            'home_location': self.location_name,
            'home_location_index': position
        }
        
        dialog = VialEditDialog(new_vial_data, self)
        dialog.setWindowTitle('Add New Vial')
        if dialog.exec() == QDialog.Accepted:
            updated_data = dialog.get_vial_data()
            
            # Remove the placeholder
            if position in self.empty_slots:
                self.empty_slots[position].deleteLater()
                del self.empty_slots[position]
            
            # Add to data list
            self.vial_data_list.append(updated_data)
            
            # Create and add vial widget
            row = position % self.grid_rows
            column_number = (self.grid_cols - 1) - (position // self.grid_rows)
            col = column_number if column_number >= 0 else 0
            
            vial_widget = VialWidget(updated_data)
            vial_widget.vial_clicked.connect(self._on_vial_clicked)
            
            self.grid_layout.addWidget(vial_widget, row, col)
            self.vials[position] = vial_widget
            
            self.vial_added.emit(updated_data)
    
    def _get_next_vial_index(self) -> int:
        """Get the next available vial index globally across all racks."""
        existing_indices = set()
        
        # Check all vials from all racks via parent window
        parent_window = self.parent()
        while parent_window and not hasattr(parent_window, 'rack_widgets'):
            parent_window = parent_window.parent()
        
        if parent_window and hasattr(parent_window, 'rack_widgets'):
            # Collect indices from all rack widgets
            for widget in parent_window.rack_widgets.values():
                try:
                    widget_vials = widget.get_vials_data()
                    for vial in widget_vials:
                        try:
                            existing_indices.add(int(vial.get('vial_index', 0)))
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass
        else:
            # Fallback: only check local vials if parent not found
            for vial in self.vial_data_list:
                try:
                    existing_indices.add(int(vial.get('vial_index', 0)))
                except (ValueError, TypeError):
                    pass
        
        # Find first available index starting from 0
        next_index = 0
        while next_index in existing_indices:
            next_index += 1
        return next_index
    
    def get_vials_data(self) -> List[Dict]:
        """Get current vials data for this rack."""
        return self.vial_data_list.copy()


class CombinedRackWidget(QScrollArea):
    """Widget showing multiple small racks (large_vial_rack, photoreactor, clamp) in one tab."""
    
    vial_edited = Signal(dict)  # Emits updated vial data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vials = {}  # (location, location_index) -> VialWidget
        self.vial_data_list = []
        
        # Create scroll area content
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        self.setWidgetResizable(True)
        
        # Main vertical layout
        self.main_layout = QVBoxLayout(self.content_widget)
        
        # Create sections for each rack type
        self._setup_sections()
    
    def _setup_sections(self):
        """Set up sections for different rack types."""
        # Large Vial Rack section (2x2 grid)
        large_vial_group = QGroupBox("Large Vial Rack (20mL)")
        large_vial_layout = QGridLayout(large_vial_group)
        self.large_vial_grid = large_vial_layout
        self.large_vial_placeholders = {}
        
        for row in range(2):
            for col in range(2):
                pos = row * 2 + col
                placeholder = ClickableLabel(f"LV-{pos}")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("""
                    ClickableLabel {
                        border: 1px dashed #ccc;
                        border-radius: 8px;
                        color: #999;
                        background-color: #f9f9f9;
                        min-height: 100px;
                        min-width: 80px;
                    }
                    ClickableLabel:hover {
                        background-color: #e8f4fd;
                        border-color: #0078d4;
                    }
                """)
                placeholder.location = 'large_vial_rack'
                placeholder.location_index = pos
                placeholder.clicked.connect(self._on_empty_slot_clicked)
                # Ensure consistent sizing in grid
                placeholder.setFixedSize(150, 100)  # Match new wider size
                placeholder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                large_vial_layout.addWidget(placeholder, row, col, Qt.AlignCenter)
                self.large_vial_placeholders[('large_vial_rack', pos)] = (placeholder, row, col)
        
        self.main_layout.addWidget(large_vial_group)
        
        # Photoreactor Array section (single position)
        photoreactor_group = QGroupBox("Photoreactor Array")
        photoreactor_layout = QGridLayout(photoreactor_group)
        self.photoreactor_grid = photoreactor_layout
        self.photoreactor_placeholders = {}
        
        placeholder = ClickableLabel("PR-0")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            ClickableLabel {
                border: 1px dashed #ccc;
                border-radius: 8px;
                color: #999;
                background-color: #fff9e6;
                min-height: 60px;
                min-width: 80px;
            }
            ClickableLabel:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
        """)
        placeholder.location = 'photoreactor_array'
        placeholder.location_index = 0
        placeholder.clicked.connect(self._on_empty_slot_clicked)
        photoreactor_layout.addWidget(placeholder, 0, 0)
        self.photoreactor_placeholders[('photoreactor_array', 0)] = (placeholder, 0, 0)
        
        self.main_layout.addWidget(photoreactor_group)
        
        # Clamp section (single position)
        clamp_group = QGroupBox("Clamp Position")
        clamp_layout = QGridLayout(clamp_group)
        self.clamp_grid = clamp_layout
        self.clamp_placeholders = {}
        
        placeholder = ClickableLabel("CL-0")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            ClickableLabel {
                border: 1px dashed #ccc;
                border-radius: 8px;
                color: #999;
                background-color: #f0f8ff;
                min-height: 60px;
                min-width: 80px;
            }
            ClickableLabel:hover {
                background-color: #e8f4fd;
                border-color: #0078d4;
            }
        """)
        placeholder.location = 'clamp'
        placeholder.location_index = 0
        placeholder.clicked.connect(self._on_empty_slot_clicked)
        clamp_layout.addWidget(placeholder, 0, 0)
        self.clamp_placeholders[('clamp', 0)] = (placeholder, 0, 0)
        
        self.main_layout.addWidget(clamp_group)
        
        # Add stretch to push everything to top
        self.main_layout.addStretch()
    
    def add_vials(self, vials_data: List[Dict]):
        """Add vials to the appropriate sections."""
        aux_locations = {'large_vial_rack', 'photoreactor_array', 'clamp'}
        self.vial_data_list = [v for v in vials_data if v.get('location') in aux_locations]
        
        for vial_data in self.vial_data_list:
            location = vial_data.get('location')
            
            try:
                location_index = int(vial_data.get('location_index', 0))
            except (ValueError, TypeError):
                continue
            
            # Determine which grid and position
            grid_info = None
            if location == 'large_vial_rack' and location_index < 4:
                grid_info = self.large_vial_placeholders.get(('large_vial_rack', location_index))
                grid_layout = self.large_vial_grid
            elif location == 'photoreactor_array' and location_index == 0:
                grid_info = self.photoreactor_placeholders.get(('photoreactor_array', 0))
                grid_layout = self.photoreactor_grid
            elif location == 'clamp' and location_index == 0:
                grid_info = self.clamp_placeholders.get(('clamp', 0))
                grid_layout = self.clamp_grid
            
            if grid_info:
                placeholder, row, col = grid_info
                
                # Remove placeholder and add vial widget
                placeholder.deleteLater()
                
                # Create vial widget
                vial_widget = VialWidget(vial_data)
                vial_widget.vial_clicked.connect(self._on_vial_clicked)
                # Ensure consistent sizing in grid
                vial_widget.setFixedSize(150, 100)  # Match new wider size
                vial_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                
                grid_layout.addWidget(vial_widget, row, col, Qt.AlignCenter)
                self.vials[(location, location_index)] = vial_widget
    
    def _on_vial_clicked(self, vial_data: Dict):
        """Handle vial click to open edit dialog."""
        dialog = VialEditDialog(vial_data, self)
        if dialog.exec() == QDialog.Accepted:
            updated_data = dialog.get_vial_data()
            
            # Get keys for operations
            location = vial_data.get('location')
            location_index = int(vial_data.get('location_index', 0))
            key = (location, location_index)
            
            # Check for vial removal
            if updated_data.get('_remove'):
                # Remove vial from grid and data
                if key in self.vials:
                    self.vials[key].deleteLater()
                    del self.vials[key]
                
                # Remove from data list
                self.vial_data_list = [v for v in self.vial_data_list if v.get('vial_index') != vial_data.get('vial_index')]
                
                # Restore empty placeholder
                self._restore_placeholder(location, location_index)
                
                self.vial_edited.emit(updated_data)  # Notify of removal
            else:
                # Update the vial widget
                if key in self.vials:
                    self.vials[key].update_vial_data(updated_data)
                
                # Update our data list - use original vial_data to find the entry
                for i, vial in enumerate(self.vial_data_list):
                    if vial.get('vial_index') == vial_data.get('vial_index'):
                        self.vial_data_list[i] = updated_data
                        break
                
                self.vial_edited.emit(updated_data)
    
    def _restore_placeholder(self, location: str, location_index: int):
        """Restore empty placeholder after vial removal."""
        if location == 'large_vial_rack' and location_index < 4:
            row = location_index // 2
            col = location_index % 2
            placeholder = ClickableLabel(f"LV-{location_index}")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                ClickableLabel {
                    border: 1px dashed #ccc;
                    border-radius: 8px;
                    color: #999;
                    background-color: #f9f9f9;
                    min-height: 100px;
                    min-width: 80px;
                }
                ClickableLabel:hover {
                    background-color: #e8f4fd;
                    border-color: #0078d4;
                }
            """)
            placeholder.location = location
            placeholder.location_index = location_index
            placeholder.clicked.connect(self._on_empty_slot_clicked)
            # Ensure consistent sizing in grid
            placeholder.setFixedSize(150, 100)  # Match new wider size
            placeholder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            self.large_vial_grid.addWidget(placeholder, row, col, Qt.AlignCenter)
            self.large_vial_placeholders[(location, location_index)] = (placeholder, row, col)
            
        elif location == 'photoreactor_array' and location_index == 0:
            placeholder = ClickableLabel("PR-0")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                ClickableLabel {
                    border: 1px dashed #ccc;
                    border-radius: 8px;
                    color: #999;
                    background-color: #fff9e6;
                    min-height: 60px;
                    min-width: 80px;
                }
                ClickableLabel:hover {
                    background-color: #e8f4fd;
                    border-color: #0078d4;
                }
            """)
            placeholder.location = location
            placeholder.location_index = location_index
            placeholder.clicked.connect(self._on_empty_slot_clicked)
            
            self.photoreactor_grid.addWidget(placeholder, 0, 0)
            self.photoreactor_placeholders[(location, location_index)] = (placeholder, 0, 0)
            
        elif location == 'clamp' and location_index == 0:
            placeholder = ClickableLabel("CL-0")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                ClickableLabel {
                    border: 1px dashed #ccc;
                    border-radius: 8px;
                    color: #999;
                    background-color: #f0f8ff;
                    min-height: 60px;
                    min-width: 80px;
                }
                ClickableLabel:hover {
                    background-color: #e8f4fd;
                    border-color: #0078d4;
                }
            """)
            placeholder.location = location
            placeholder.location_index = location_index
            placeholder.clicked.connect(self._on_empty_slot_clicked)
            
            self.clamp_grid.addWidget(placeholder, 0, 0)
            self.clamp_placeholders[(location, location_index)] = (placeholder, 0, 0)
    
    def _on_empty_slot_clicked(self):
        """Handle clicking on empty slot to add new vial."""
        sender = self.sender()
        location = getattr(sender, 'location', '')
        location_index = getattr(sender, 'location_index', 0)
        
        # Create new vial data template
        new_vial_data = {
            'vial_index': self._get_next_vial_index(),
            'vial_name': f'new_vial_{location}_{location_index}',
            'location': location,
            'location_index': location_index,
            'vial_volume': 0.0,
            'capped': False,
            'cap_type': 'open',
            'vial_type': '20_mL' if location == 'large_vial_rack' else '8_mL',
            'home_location': location,
            'home_location_index': location_index
        }
        
        dialog = VialEditDialog(new_vial_data, self)
        dialog.setWindowTitle('Add New Vial')
        if dialog.exec() == QDialog.Accepted:
            updated_data = dialog.get_vial_data()
            
            # Remove the placeholder from the appropriate dict
            key = (location, location_index)
            if key in self.large_vial_placeholders:
                self.large_vial_placeholders[key][0].deleteLater()
                del self.large_vial_placeholders[key]
            elif key in self.photoreactor_placeholders:
                self.photoreactor_placeholders[key][0].deleteLater()
                del self.photoreactor_placeholders[key]
            elif key in self.clamp_placeholders:
                self.clamp_placeholders[key][0].deleteLater()
                del self.clamp_placeholders[key]
            
            # Add to data list
            self.vial_data_list.append(updated_data)
            
            # Create and add vial widget
            if location == 'large_vial_rack':
                row = location_index // 2
                col = location_index % 2
                grid_layout = self.large_vial_grid
            elif location == 'photoreactor_array':
                row, col = 0, 0
                grid_layout = self.photoreactor_grid
            elif location == 'clamp':
                row, col = 0, 0
                grid_layout = self.clamp_grid
            else:
                return  # Unknown location
            
            vial_widget = VialWidget(updated_data)
            vial_widget.vial_clicked.connect(self._on_vial_clicked)
            
            grid_layout.addWidget(vial_widget, row, col)
            self.vials[key] = vial_widget
            
            self.vial_edited.emit(updated_data)
    
    def _get_next_vial_index(self) -> int:
        """Get the next available vial index globally across all racks."""
        existing_indices = set()
        
        # Check all vials from all racks via parent window
        parent_window = self.parent()
        while parent_window and not hasattr(parent_window, 'rack_widgets'):
            parent_window = parent_window.parent()
        
        if parent_window and hasattr(parent_window, 'rack_widgets'):
            # Collect indices from all rack widgets
            for widget in parent_window.rack_widgets.values():
                try:
                    widget_vials = widget.get_vials_data()
                    for vial in widget_vials:
                        try:
                            existing_indices.add(int(vial.get('vial_index', 0)))
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass
        else:
            # Fallback: only check local vials if parent not found
            for vial in self.vial_data_list:
                try:
                    existing_indices.add(int(vial.get('vial_index', 0)))
                except (ValueError, TypeError):
                    pass
        
        # Find first available index starting from 0
        next_index = 0
        while next_index in existing_indices:
            next_index += 1
        return next_index
    
    def get_vials_data(self) -> List[Dict]:
        """Get current vials data for all auxiliary racks."""
        return self.vial_data_list.copy()


class TrackStatusWidget(QWidget):
    """Widget for editing track status YAML file."""
    
    def __init__(self):
        super().__init__()
        self.track_data = {}
        self.track_file_path = os.path.join("robot_state", "track_status.yaml")
        self._setup_ui()
        self._load_track_status()
    
    def _setup_ui(self):
        """Setup the track status editing UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Track Status Configuration")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header_label)
        
        # Form layout
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        
        # Active wellplate position
        self.active_position_edit = QLineEdit()
        self.active_position_edit.setPlaceholderText("null or position number")
        form_layout.addRow("Active Wellplate Position:", self.active_position_edit)
        
        # Number in source
        self.num_source_spin = QSpinBox()
        self.num_source_spin.setRange(0, 100)
        form_layout.addRow("Number in Source:", self.num_source_spin)
        
        # Number in waste
        self.num_waste_spin = QSpinBox()
        self.num_waste_spin.setRange(0, 100)
        form_layout.addRow("Number in Waste:", self.num_waste_spin)
        
        # Wellplate type
        self.wellplate_type_combo = QComboBox()
        self.wellplate_type_combo.addItems(["96 WELL PLATE", "quartz", "48 WELL PLATE"])
        form_layout.addRow("Wellplate Type:", self.wellplate_type_combo)
        
        layout.addWidget(form_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def _load_track_status(self):
        """Load track status from YAML file."""
        try:
            with open(self.track_file_path, 'r') as file:
                self.track_data = yaml.safe_load(file) or {}
            
            self._populate_form()
            
        except FileNotFoundError:
            QMessageBox.warning(self, "File Not Found", 
                              f"Track status file not found: {self.track_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load track status: {str(e)}")
    
    def _populate_form(self):
        """Populate form with loaded data."""
        # Active wellplate position
        active_pos = self.track_data.get('active_wellplate_position')
        if active_pos is None:
            self.active_position_edit.setText("")
        else:
            self.active_position_edit.setText(str(active_pos))
        
        # Numbers
        self.num_source_spin.setValue(self.track_data.get('num_in_source', 0))
        self.num_waste_spin.setValue(self.track_data.get('num_in_waste', 0))
        
        # Wellplate type
        wellplate_type = self.track_data.get('wellplate_type', '96 WELL PLATE')
        index = self.wellplate_type_combo.findText(wellplate_type)
        if index >= 0:
            self.wellplate_type_combo.setCurrentIndex(index)
    
    def _save_track_status(self, silent=False):
        """Save track status to YAML file."""
        try:
            # Update data from form
            active_pos_text = self.active_position_edit.text().strip()
            if active_pos_text == "" or active_pos_text.lower() == "null":
                active_pos = None
            else:
                try:
                    active_pos = int(active_pos_text)
                except ValueError:
                    active_pos = active_pos_text
            
            self.track_data = {
                'active_wellplate_position': active_pos,
                'num_in_source': self.num_source_spin.value(),
                'num_in_waste': self.num_waste_spin.value(),
                'wellplate_type': self.wellplate_type_combo.currentText()
            }
            
            # Save to file
            with open(self.track_file_path, 'w') as file:
                yaml.dump(self.track_data, file, default_flow_style=False)
            
            if not silent:
                QMessageBox.information(self, "Success", "Track status saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save track status: {str(e)}")


class RobotStatusWidget(QWidget):
    """Widget for editing robot status YAML file."""
    
    def __init__(self):
        super().__init__()
        self.robot_data = {}
        self.robot_file_path = os.path.join("robot_state", "robot_status.yaml")
        self._setup_ui()
        self._load_robot_status()
    
    def _setup_ui(self):
        """Setup the robot status editing UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Robot Status Configuration")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header_label)
        
        # Scroll area for the form
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Gripper section
        gripper_group = QGroupBox("Gripper Status")
        gripper_layout = QFormLayout(gripper_group)
        
        self.gripper_status_edit = QLineEdit()
        self.gripper_status_edit.setPlaceholderText("null or status")
        gripper_layout.addRow("Gripper Status:", self.gripper_status_edit)
        
        self.gripper_vial_index_edit = QLineEdit()
        self.gripper_vial_index_edit.setPlaceholderText("null or vial index")
        gripper_layout.addRow("Gripper Vial Index:", self.gripper_vial_index_edit)
        
        scroll_layout.addWidget(gripper_group)
        
        # Pipet section
        pipet_group = QGroupBox("Pipet Status")
        pipet_layout = QFormLayout(pipet_group)
        
        self.held_pipet_type_edit = QLineEdit()
        self.held_pipet_type_edit.setPlaceholderText("null or pipet type")
        pipet_layout.addRow("Held Pipet Type:", self.held_pipet_type_edit)
        
        self.pipet_fluid_vial_index_edit = QLineEdit()
        self.pipet_fluid_vial_index_edit.setPlaceholderText("null or vial index")
        pipet_layout.addRow("Pipet Fluid Vial Index:", self.pipet_fluid_vial_index_edit)
        
        self.pipet_fluid_volume_spin = QDoubleSpinBox()
        self.pipet_fluid_volume_spin.setRange(0.0, 1000.0)
        self.pipet_fluid_volume_spin.setDecimals(1)
        pipet_layout.addRow("Pipet Fluid Volume:", self.pipet_fluid_volume_spin)
        
        scroll_layout.addWidget(pipet_group)
        
        # Pipets used section
        pipets_group = QGroupBox("Pipets Used")
        pipets_layout = QFormLayout(pipets_group)
        
        self.large_tip_rack_1_spin = QSpinBox()
        self.large_tip_rack_1_spin.setRange(0, 1000)
        pipets_layout.addRow("Large Tip Rack 1:", self.large_tip_rack_1_spin)
        
        self.large_tip_rack_2_spin = QSpinBox()
        self.large_tip_rack_2_spin.setRange(0, 1000)
        pipets_layout.addRow("Large Tip Rack 2:", self.large_tip_rack_2_spin)
        
        self.small_tip_rack_1_spin = QSpinBox()
        self.small_tip_rack_1_spin.setRange(0, 1000)
        pipets_layout.addRow("Small Tip Rack 1:", self.small_tip_rack_1_spin)
        
        self.small_tip_rack_2_spin = QSpinBox()
        self.small_tip_rack_2_spin.setRange(0, 1000)
        pipets_layout.addRow("Small Tip Rack 2:", self.small_tip_rack_2_spin)
        
        scroll_layout.addWidget(pipets_group)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def _load_robot_status(self):
        """Load robot status from YAML file."""
        try:
            with open(self.robot_file_path, 'r') as file:
                self.robot_data = yaml.safe_load(file) or {}
            
            self._populate_form()
            
        except FileNotFoundError:
            QMessageBox.warning(self, "File Not Found", 
                              f"Robot status file not found: {self.robot_file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load robot status: {str(e)}")
    
    def _populate_form(self):
        """Populate form with loaded data."""
        # Gripper status
        gripper_status = self.robot_data.get('gripper_status')
        self.gripper_status_edit.setText(str(gripper_status) if gripper_status is not None else "")
        
        gripper_vial_index = self.robot_data.get('gripper_vial_index')
        self.gripper_vial_index_edit.setText(str(gripper_vial_index) if gripper_vial_index is not None else "")
        
        # Pipet status
        held_pipet_type = self.robot_data.get('held_pipet_type')
        self.held_pipet_type_edit.setText(str(held_pipet_type) if held_pipet_type is not None else "")
        
        pipet_fluid_vial_index = self.robot_data.get('pipet_fluid_vial_index')
        self.pipet_fluid_vial_index_edit.setText(str(pipet_fluid_vial_index) if pipet_fluid_vial_index is not None else "")
        
        self.pipet_fluid_volume_spin.setValue(self.robot_data.get('pipet_fluid_volume', 0.0))
        
        # Pipets used
        pipets_used = self.robot_data.get('pipets_used', {})
        self.large_tip_rack_1_spin.setValue(pipets_used.get('large_tip_rack_1', 0))
        self.large_tip_rack_2_spin.setValue(pipets_used.get('large_tip_rack_2', 0))
        self.small_tip_rack_1_spin.setValue(pipets_used.get('small_tip_rack_1', 0))
        self.small_tip_rack_2_spin.setValue(pipets_used.get('small_tip_rack_2', 0))
    
    def _save_robot_status(self, silent=False):
        """Save robot status to YAML file."""
        try:
            # Helper function to parse nullable fields
            def parse_nullable_field(text):
                text = text.strip()
                if text == "" or text.lower() == "null":
                    return None
                try:
                    return int(text)
                except ValueError:
                    return text
            
            # Update data from form
            self.robot_data = {
                'gripper_status': parse_nullable_field(self.gripper_status_edit.text()),
                'gripper_vial_index': parse_nullable_field(self.gripper_vial_index_edit.text()),
                'held_pipet_type': parse_nullable_field(self.held_pipet_type_edit.text()),
                'pipet_fluid_vial_index': parse_nullable_field(self.pipet_fluid_vial_index_edit.text()),
                'pipet_fluid_volume': self.pipet_fluid_volume_spin.value(),
                'pipets_used': {
                    'large_tip_rack_1': self.large_tip_rack_1_spin.value(),
                    'large_tip_rack_2': self.large_tip_rack_2_spin.value(),
                    'small_tip_rack_1': self.small_tip_rack_1_spin.value(),
                    'small_tip_rack_2': self.small_tip_rack_2_spin.value()
                }
            }
            
            # Save to file
            with open(self.robot_file_path, 'w') as file:
                yaml.dump(self.robot_data, file, default_flow_style=False)
            
            if not silent:
                QMessageBox.information(self, "Success", "Robot status saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save robot status: {str(e)}")


class ConfigEditor(QWidget):
    """Widget for editing workflow configuration YAML files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_file = None
        self.config_data = {}
        self.config_widgets = {}  # key -> widget mapping
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the configuration editor interface."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        self.config_info_label = QLabel("No configuration loaded")
        self.config_info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        header_layout.addWidget(self.config_info_label)
        
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Scrollable config area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.config_widget = QWidget()
        self.config_layout = QVBoxLayout(self.config_widget)
        scroll_area.setWidget(self.config_widget)
        
        layout.addWidget(scroll_area)
        
        # Status
        self.config_status_label = QLabel("Ready to load configuration")
        self.config_status_label.setStyleSheet("padding: 5px; background: #f0f0f0;")
        layout.addWidget(self.config_status_label)
    
    def load_workflow_config(self, workflow_name):
        """Load configuration for the specified workflow."""
        if not ConfigManager:
            self.config_status_label.setText("ConfigManager not available")
            return False
        
        try:
            config_file = os.path.join("workflow_configs", f"{workflow_name}.yaml")
            self.config_file = config_file
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                self.config_info_label.setText(f"Config: {workflow_name}.yaml")
                self.config_status_label.setText(f"Loaded {len(self.config_data)} configuration items")
            else:
                self.config_data = {}
                self.config_info_label.setText(f"Config: {workflow_name}.yaml (not found)")
                self.config_status_label.setText("Configuration file not found - will be created on save")
            
            self._populate_config_editor()
            # Config changed - could add visual indicator here if needed
            pass
            return True
            
        except Exception as e:
            self.config_status_label.setText(f"Error loading config: {str(e)}")
            return False
    
    def _populate_config_editor(self):
        """Populate the configuration editor with current config values."""
        # Clear existing widgets
        for i in reversed(range(self.config_layout.count())):
            child = self.config_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.config_widgets.clear()
        
        if not self.config_data:
            no_config_label = QLabel("No configuration data available")
            no_config_label.setStyleSheet("color: gray; font-style: italic; padding: 20px;")
            self.config_layout.addWidget(no_config_label)
            return
        
        # Create form for each config item
        for key, value in self.config_data.items():
            if key.startswith('_'):  # Skip internal keys like _USE_FILE_ONLY
                continue
                
            group_box = QGroupBox(key)
            group_layout = QFormLayout(group_box)
            
            # Create appropriate widget based on value type
            widget = self._create_config_widget(key, value)
            if widget:
                self.config_widgets[key] = widget
                group_layout.addRow("Value:", widget)
                
                # Add type info
                type_label = QLabel(f"Type: {type(value).__name__}")
                type_label.setStyleSheet("color: gray; font-size: 10px;")
                group_layout.addRow("Type:", type_label)
            
            self.config_layout.addWidget(group_box)
        
        self.config_layout.addStretch()
    
    def _create_config_widget(self, key, value):
        """Create appropriate widget for config value based on its type."""
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            return widget
        
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            return widget
        
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(6)
            widget.setValue(value)
            return widget
        
        elif isinstance(value, str):
            widget = QLineEdit()
            widget.setText(value)
            return widget
        
        elif isinstance(value, list):
            # For lists, show as comma-separated text
            widget = QLineEdit()
            if all(isinstance(item, str) for item in value):
                widget.setText(", ".join(value))
            else:
                widget.setText(str(value))
            widget.setToolTip("List items (comma-separated for strings)")
            return widget
        
        elif isinstance(value, dict):
            # For complex dicts, show as read-only YAML text
            widget = QTextEdit()
            widget.setPlainText(yaml.dump(value, default_flow_style=False))
            widget.setMaximumHeight(200)
            widget.setToolTip("Complex dictionary - edit as YAML")
            return widget
        
        else:
            # Fallback - show as text
            widget = QLineEdit()
            widget.setText(str(value))
            return widget
    
    def _save_config(self, silent=False):
        """Save current configuration values to file."""
        if not self.config_file:
            return
        
        try:
            # Update config_data with current widget values
            for key, widget in self.config_widgets.items():
                original_value = self.config_data.get(key)
                
                if isinstance(widget, QCheckBox):
                    self.config_data[key] = widget.isChecked()
                elif isinstance(widget, QSpinBox):
                    self.config_data[key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    self.config_data[key] = widget.value()
                elif isinstance(widget, QLineEdit):
                    text = widget.text()
                    if isinstance(original_value, list) and isinstance(original_value[0] if original_value else "", str):
                        # Parse comma-separated list
                        self.config_data[key] = [item.strip() for item in text.split(',') if item.strip()]
                    else:
                        self.config_data[key] = text
                elif isinstance(widget, QTextEdit):
                    # Parse YAML text for complex structures
                    try:
                        yaml_text = widget.toPlainText()
                        self.config_data[key] = yaml.safe_load(yaml_text)
                    except yaml.YAMLError as e:
                        QMessageBox.warning(self, "YAML Error", f"Invalid YAML for {key}: {e}")
                        return

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                # Write header
                f.write("# Workflow Configuration File\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Workflow: {Path(self.config_file).stem}\n")
                f.write("#\n")
                f.write("# Edit this file to customize workflow behavior.\n")
                f.write("# Changes will be loaded automatically on next run.\n")
                f.write("#\n\n")
                
                # Write config data
                yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False, indent=2)
            
            self.config_status_label.setText(f"Config saved successfully at {datetime.now().strftime('%H:%M:%S')}")
            if not silent:
                QMessageBox.information(self, "Success", "Configuration saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
            self.config_status_label.setText(f"Save failed: {str(e)}")
    
    def _reload_config(self):
        """Reload configuration from file."""
        if not self.config_file:
            return
        
        reply = QMessageBox.question(
            self, "Reload Config", 
            "Reload configuration from file? Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            workflow_name = Path(self.config_file).stem
            self.load_workflow_config(workflow_name)


class VialManagerMainWindow(QMainWindow):
    """Main window for visual vial management."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Vial Manager")
        self.setGeometry(100, 100, 1500, 1000)  # 25% larger than 1200x800
        
        self.status_file_path = None
        self.original_vials_data = []
        self.rack_widgets = {}  # location_name -> VialRackWidget
        
        # Workflow mode attributes
        self._workflow_mode = False
        self._workflow_continue = True
        self._lash_e_instance = None
        
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        
        # Load default file if provided via command line
        if len(sys.argv) > 1:
            self.load_status_file(sys.argv[1])
    
    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # File info label
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.file_info_label)
        
        # Tab widget for different racks/locations
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Button bar
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Status File")
        self.load_button.clicked.connect(self._load_file_dialog)
        button_layout.addWidget(self.load_button)
        
        self.reload_button = QPushButton("🔄 Reload All")
        self.reload_button.clicked.connect(self._reload_all)
        self.reload_button.setEnabled(False)
        button_layout.addWidget(self.reload_button)
        
        self.save_all_button = QPushButton("💾 Save All")
        self.save_all_button.clicked.connect(self._save_all)
        self.save_all_button.setStyleSheet(
            "QPushButton { background-color: #FF6B35; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #E55A2B; }"
        )
        button_layout.addWidget(self.save_all_button)
        
        button_layout.addStretch()
        
        # Workflow control buttons (hidden by default)
        self.run_workflow_button = QPushButton("🚀 Run Workflow")
        self.run_workflow_button.clicked.connect(self._run_workflow)
        self.run_workflow_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.run_workflow_button.hide()  # Hidden by default
        button_layout.addWidget(self.run_workflow_button)
        
        self.abort_workflow_button = QPushButton("❌ Abort Workflow")
        self.abort_workflow_button.clicked.connect(self._abort_workflow)
        self.abort_workflow_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        self.abort_workflow_button.hide()  # Hidden by default
        button_layout.addWidget(self.abort_workflow_button)
        
        # Stats label
        self.stats_label = QLabel("No vials loaded")
        button_layout.addWidget(self.stats_label)
        
        layout.addLayout(button_layout)
    
    def _setup_menu_bar(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        if not self._workflow_mode:
            # Standard mode - include Load Status File
            load_action = QAction("&Load Status File...", self)
            load_action.setShortcut("Ctrl+O")
            load_action.triggered.connect(self._load_file_dialog)
            file_menu.addAction(load_action)
        
        save_action = QAction("&Save Changes", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
        reload_action = QAction("&Reload from File", self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self._reload_file)
        file_menu.addAction(reload_action)
        
        if self._workflow_mode:
            # Workflow mode - add workflow controls
            file_menu.addSeparator()
            
            run_action = QAction("&Run Workflow", self)
            run_action.setShortcut("Ctrl+G")
            run_action.triggered.connect(self._run_workflow)
            file_menu.addAction(run_action)
            
            abort_action = QAction("&Abort Workflow", self)
            abort_action.setShortcut("Ctrl+Q")
            abort_action.triggered.connect(self._abort_workflow)
            file_menu.addAction(abort_action)
        else:
            # Standard mode - include Exit
            file_menu.addSeparator()
            
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
    
    def _setup_status_bar(self):
        """Set up status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a vial status file to begin")
    
    def load_status_file(self, file_path: str) -> bool:
        """Load vials from CSV status file."""
        try:
            vials_data = []
            original_fieldnames = None  # Store original field order
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                # Auto-detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                original_fieldnames = reader.fieldnames  # Preserve original order
                for row in reader:
                    vials_data.append(row)
            
            if not vials_data:
                QMessageBox.warning(self, "Empty File", "No vial data found in file.")
                return False
            
            self.original_vials_data = vials_data.copy()
            self.original_fieldnames = original_fieldnames  # Store field order
            self.status_file_path = file_path
            
            self._populate_racks(vials_data)
            self._update_ui_state()
            
            # Update log vials widget if it exists
            if hasattr(self, 'log_vials_widget'):
                current_vial_names = [v.get('vial_name', '') for v in vials_data if v.get('vial_name')]
                self.log_vials_widget.update_csv_vials(current_vial_names)
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", 
                               f"Failed to load {file_path}:\n{str(e)}")
            return False
    
    def _populate_racks(self, vials_data: List[Dict]):
        """Populate rack widgets with vial data."""
        # Preserve config editor if it exists
        config_editor = None
        if hasattr(self, 'config_editor') and self.config_editor:
            config_editor = self.config_editor
        
        # Clear existing tabs
        self.tab_widget.clear()
        self.rack_widgets.clear()
        
        # Group vials by location
        locations = set(vial.get('location', 'unknown') for vial in vials_data)
        
        # Create special combined view for auxiliary locations
        aux_locations = {'large_vial_rack', 'photoreactor_array', 'clamp'}
        
        # First add main_8mL_rack tab with log vials display (ensure it's first)
        main_locations = {loc for loc in locations if '8mL' in loc or 'main' in loc}
        first_main_processed = False
        for location in sorted(main_locations):
            if location and location != 'unknown':
                if not first_main_processed:
                    # Create horizontal layout for rack + log vials (first main tab only)
                    main_tab_widget = QWidget()
                    main_tab_layout = QHBoxLayout(main_tab_widget)
                    
                    # Add the rack widget to the left
                    rack_widget = VialRackWidget(location)
                    rack_widget.vial_edited.connect(self._on_vial_edited)
                    rack_widget.vial_added.connect(self._on_vial_edited)  # Handle new vials
                    rack_widget.add_vials(vials_data)
                    main_tab_layout.addWidget(rack_widget)
                    
                    # Add log vials widget to the right - narrow sidebar
                    self.log_vials_widget = LogVialsWidget()
                    self.log_vials_widget.setMaximumWidth(130)  # Keep it narrow
                    main_tab_layout.addWidget(self.log_vials_widget)
                    
                    # CRITICAL: Actually call the update method!
                    print("[DEBUG] About to call update_log_vials()")
                    self.log_vials_widget.update_log_vials()
                    print("[DEBUG] Called update_log_vials()")
                    
                    # Update with current CSV vials
                    current_vial_names = [v.get('vial_name', '') for v in vials_data if v.get('vial_name')]
                    self.log_vials_widget.update_csv_vials(current_vial_names)
                    print(f"[DEBUG] CSV vials loaded: {current_vial_names}")
                    
                    self.tab_widget.addTab(main_tab_widget, location)
                    self.rack_widgets[location] = rack_widget
                    first_main_processed = True
                else:
                    # Regular rack widget for additional main locations
                    rack_widget = VialRackWidget(location)
                    rack_widget.vial_edited.connect(self._on_vial_edited)
                    rack_widget.vial_added.connect(self._on_vial_edited)  # Handle new vials
                    
                    self.tab_widget.addTab(rack_widget, location)
                    self.rack_widgets[location] = rack_widget
                    rack_widget.add_vials(vials_data)
        
        # Then add auxiliary racks tab (always show, even if empty)        
        # Create combined auxiliary rack widget
        combined_widget = CombinedRackWidget()
        combined_widget.vial_edited.connect(self._on_vial_edited)
        self.tab_widget.addTab(combined_widget, "Auxiliary Racks")
        self.rack_widgets['_combined_aux'] = combined_widget
        combined_widget.add_vials(vials_data)
        
        # Finally add other location tabs (excluding auxiliary and main ones)
        other_locations = {loc for loc in locations if loc and loc != 'unknown' and loc not in aux_locations and '8mL' not in loc and 'main' not in loc}
        for location in sorted(other_locations):
            rack_widget = VialRackWidget(location)
            rack_widget.vial_edited.connect(self._on_vial_edited)
            rack_widget.vial_added.connect(self._on_vial_edited)  # Handle new vials
            
            self.tab_widget.addTab(rack_widget, location)
            self.rack_widgets[location] = rack_widget
            rack_widget.add_vials(vials_data)
        
        # Add YAML editing tabs
        self._add_yaml_tabs()
        
        # Re-add config editor if it was previously present
        if config_editor:
            self.config_editor = config_editor
            self.tab_widget.addTab(self.config_editor, "Workflow Config")
    
    def _on_vial_edited(self, updated_vial_data: Dict):
        """Handle vial edit from any rack."""
        # Handle removal or addition/edit
        vial_index = updated_vial_data.get('vial_index')
        
        if updated_vial_data.get('_remove'):
            # PRECISE DELETE - only remove exact matches
            vial_name_to_remove = updated_vial_data.get('vial_name', '')
            vial_index_to_remove = str(updated_vial_data.get('vial_index', ''))
            
            if vial_index_to_remove and vial_name_to_remove:
                print(f"DELETING: '{vial_name_to_remove}' (index: {vial_index_to_remove})")
                
                # Remove from master data list - must match BOTH name AND index
                original_count = len(self.original_vials_data)
                self.original_vials_data = [v for v in self.original_vials_data 
                                          if not (str(v.get('vial_index', '')) == vial_index_to_remove 
                                                 and v.get('vial_name', '') == vial_name_to_remove)]
                print(f"Removed {original_count - len(self.original_vials_data)} from master list")
                
                # CRITICAL: Also remove from ALL widget data lists to prevent ghost copies
                for widget_name, rack_widget in self.rack_widgets.items():
                    if hasattr(rack_widget, 'vial_data_list'):
                        before_count = len(rack_widget.vial_data_list)
                        rack_widget.vial_data_list = [v for v in rack_widget.vial_data_list 
                                                    if not (str(v.get('vial_index', '')) == vial_index_to_remove 
                                                           and v.get('vial_name', '') == vial_name_to_remove)]
                        after_count = len(rack_widget.vial_data_list)
                        if before_count != after_count:
                            print(f"Cleaned {before_count - after_count} ghost vials from {widget_name}")
                
            # Update log vials widget after removal
            if hasattr(self, 'log_vials_widget'):
                current_vial_names = [v.get('vial_name', '') for v in self.original_vials_data if v.get('vial_name')]
                self.log_vials_widget.update_csv_vials(current_vial_names)
                
            self.status_bar.showMessage("Vial removed - remember to save changes")
        else:
            # Handle vial edit or addition
            if vial_index is not None:
                # Check if this is a new vial or existing one
                existing_vial_found = False
                for i, vial in enumerate(self.original_vials_data):
                    if str(vial.get('vial_index', '')) == str(vial_index):
                        # Update existing vial (remove _remove flag)
                        clean_data = {k: v for k, v in updated_vial_data.items() if k != '_remove'}
                        self.original_vials_data[i].update(clean_data)
                        existing_vial_found = True
                        break
                
                if not existing_vial_found:
                    # Add new vial (remove _remove flag)
                    clean_data = {k: v for k, v in updated_vial_data.items() if k != '_remove'}
                    self.original_vials_data.append(clean_data)
            
            # Update log vials widget if it exists
            if hasattr(self, 'log_vials_widget'):
                current_vial_names = [v.get('vial_name', '') for v in self.original_vials_data if v.get('vial_name')]
                self.log_vials_widget.update_csv_vials(current_vial_names)
            
            self.status_bar.showMessage("Vial data modified - remember to save changes")
        
        self._update_ui_state()
    
    def _add_yaml_tabs(self):
        """Add YAML editing tabs for track and robot status."""
        # Track status tab
        self.track_status_widget = TrackStatusWidget()
        self.tab_widget.addTab(self.track_status_widget, "Track Status")
        
        # Robot status tab
        self.robot_status_widget = RobotStatusWidget()
        self.tab_widget.addTab(self.robot_status_widget, "Robot Status")
    
    def _setup_workflow_mode(self, vial_file_path, lash_e_instance, workflow_name=None):
        """Configure GUI for workflow mode with optional config editing."""
        self._workflow_mode = True
        self._lash_e_instance = lash_e_instance
        
        # Handle vial file loading
        if vial_file_path is None:
            # No vial file - show blank, uneditable interface
            self.status_file_path = None
            self.original_vials_data = []
            self.file_info_label.setText("No vial file specified - interface is read-only")
            self._set_interface_readonly(True)
        else:
            # Vial file specified - try to load or create
            if os.path.exists(vial_file_path):
                # File exists - load it
                self.load_status_file(vial_file_path)
            else:
                # File doesn't exist - create empty file for user to populate
                self.status_file_path = vial_file_path
                self.original_vials_data = []
                self.file_info_label.setText(f"File: {os.path.basename(vial_file_path)} (will be created)")
                
                # Create empty CSV file with headers
                try:
                    with open(vial_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['vial_index', 'vial_name', 'location', 'location_index', 
                                       'vial_volume', 'capped', 'cap_type', 'vial_type', 
                                       'home_location', 'home_location_index', 'notes'])
                    self._lash_e_instance.logger.info(f"Created empty vial status file: {vial_file_path}")
                except Exception as e:
                    QMessageBox.warning(self, "File Creation Error", 
                                      f"Could not create vial file: {e}")
        
        # Populate the interface
        self._populate_racks(self.original_vials_data)
        
        # Update button visibility for workflow mode
        self._update_button_visibility()
        
        # Update status message
        if self._workflow_mode:
            self.status_bar.showMessage("Workflow Mode: Review robot status, then Run or Abort workflow")
        
        # Add config editor tab if workflow name is provided
        if workflow_name and ConfigManager:
            self.config_editor = ConfigEditor()
            self.tab_widget.addTab(self.config_editor, "Workflow Config")
            
            # Load the workflow config
            config_loaded = self.config_editor.load_workflow_config(workflow_name)
            if config_loaded:
                self._lash_e_instance.logger.info(f"Config editor loaded for workflow: {workflow_name}")
            else:
                self._lash_e_instance.logger.warning(f"Failed to load config for workflow: {workflow_name}")
        elif workflow_name:
            self._lash_e_instance.logger.warning("ConfigManager not available - config editing disabled")
        else:
            self._lash_e_instance.logger.info("No workflow name provided - config editing not available")
    
    def _update_button_visibility(self):
        """Update button visibility based on workflow mode."""
        if self._workflow_mode:
            # Workflow mode - hide load button, show workflow buttons
            self.load_button.hide()
            self.run_workflow_button.show()
            self.abort_workflow_button.show()
        else:
            # Standard mode - show load button, hide workflow buttons
            self.load_button.show()
            self.run_workflow_button.hide()
            self.abort_workflow_button.hide()
    
    def _set_interface_readonly(self, readonly=True):
        """Enable or disable interface editing."""
        
        self.save_button.setEnabled(not readonly)
        self.reload_button.setEnabled(not readonly and self.status_file_path is not None)
        # Note: Individual vial editing will be handled by the widgets themselves
    
    def _run_workflow(self):
        """Continue with workflow - close GUI and proceed."""
        
        # FIRST: Check for duplicate vial names - block workflow if found
        try:
            all_vials_data = []
            print("=== CURRENT WIDGET DATA ===")
            for widget_name, rack_widget in self.rack_widgets.items():
                widget_data = rack_widget.get_vials_data()
                all_vials_data.extend(widget_data)
                print(f"{widget_name}: {len(widget_data)} vials")
                for vial in widget_data:
                    print(f"  - '{vial.get('vial_name', 'UNNAMED')}' (idx:{vial.get('vial_index', 'NONE')}) loc:{vial.get('location', 'NONE')}")
            
            print(f"\nTotal collected: {len(all_vials_data)} vials")
            
            # Check for duplicates
            seen_names = set()
            duplicate_names = []
            for vial in all_vials_data:
                vial_name = vial.get('vial_name', '').strip()
                if vial_name:
                    if vial_name in seen_names:
                        if vial_name not in duplicate_names:
                            duplicate_names.append(vial_name)
                    else:
                        seen_names.add(vial_name)
            
            if duplicate_names:
                QMessageBox.critical(self, "Cannot Start Workflow", 
                    f"Duplicate vial names found:\\n\\n{chr(10).join(duplicate_names)}\\n\\n"
                    f"Check the console output to see which widgets have the duplicates.")
                return  # STOP - GUI stays open
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Could not validate vial data:\n{str(e)}")
            return
        
        # Check for unsaved changes before running
        if self._has_unsaved_changes():
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Save before running workflow?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                # Try to save all - if it fails, don't continue
                try:
                    self._save_all()
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Could not save changes:\n{str(e)}")
                    return  # STOP here - don't continue to close GUI
                # If save succeeded, continue to close GUI
            elif reply == QMessageBox.Cancel:
                return  # Don't run workflow
            elif reply == QMessageBox.Discard:
                pass  # Continue without saving
        
        # Only reach here if validation passed AND save succeeded OR user chose discard OR no unsaved changes
        if self._workflow_mode:
            self._workflow_continue = True
            if self._lash_e_instance:
                self._lash_e_instance.logger.info("User selected: Run Workflow")
        self.close()
    
    def _reload_all(self):
        """Reload all data from files."""
        reply = QMessageBox.question(
            self, "Reload All", 
            "Reload all data from files? Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.status_file_path:
                self.load_status_file(self.status_file_path)
            
            # Reload other components
            if hasattr(self, 'track_status_widget'):
                self.track_status_widget._load_track_status()
            if hasattr(self, 'robot_status_widget'):
                self.robot_status_widget._load_robot_status()
            if hasattr(self, 'config_editor'):
                self.config_editor._reload_config()
            
            self.status_bar.showMessage("All data reloaded from files")

    def _save_all(self):
        """Save all modified data (vials, track, robot, config)."""
        success_count = 0
        errors = []
        
        # Save vial status file if loaded
        if self.status_file_path:
            try:
                self._save_file()
                success_count += 1
            except Exception as e:
                errors.append(f"Vials: {str(e)}")
        
        # Save track status
        try:
            if hasattr(self, 'track_status_widget'):
                self.track_status_widget._save_track_status(silent=True)
                success_count += 1
        except Exception as e:
            errors.append(f"Track: {str(e)}")
        
        # Save robot status
        try:
            if hasattr(self, 'robot_status_widget'):
                self.robot_status_widget._save_robot_status(silent=True)
                success_count += 1
        except Exception as e:
            errors.append(f"Robot: {str(e)}")
            
        # Save config
        try:
            if hasattr(self, 'config_editor'):
                self.config_editor._save_config(silent=True)
                success_count += 1
        except Exception as e:
            errors.append(f"Config: {str(e)}")
        
        # Simple feedback message
        if errors:
            self.status_bar.showMessage(f"Saved {success_count} items with some errors")
        else:
            self.status_bar.showMessage(f"All {success_count} items saved successfully")
    
    def _has_unsaved_changes(self) -> bool:
        """Check if there are any unsaved changes across all tabs."""
        # For now, assume there might be changes - user can choose to save or not
        # Could add more sophisticated change tracking later if needed
        return True

    def _abort_workflow(self):
        """Abort workflow - close GUI and cancel."""
        if self._workflow_mode:
            self._workflow_continue = False
            if self._lash_e_instance:
                self._lash_e_instance.logger.info("User selected: Abort Workflow")
            # Immediately terminate the workflow
            print("🛑 Workflow aborted by user")
            sys.exit(1)  # Force immediate workflow termination
        self.close()
    
    def _update_ui_state(self):
        """Update UI state based on loaded data."""
        has_file = self.status_file_path is not None
        
        self.reload_button.setEnabled(has_file)
        
        if has_file:
            file_name = Path(self.status_file_path).name
            self.file_info_label.setText(f"File: {file_name}")
            
            # Update stats
            total_vials = len(self.original_vials_data)
            locations = set(vial.get('location', 'unknown') for vial in self.original_vials_data)
            self.stats_label.setText(f"{total_vials} vials in {len(locations)} locations")
        else:
            self.file_info_label.setText("No file loaded")
            self.stats_label.setText("No vials loaded")
    
    def _load_file_dialog(self):
        """Open file dialog to load status file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Vial Status File", 
            "status", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.load_status_file(file_path)
    
    def _save_file(self):
        """Save current vial data back to CSV file."""
        if not self.status_file_path:
            return
            
        try:
            # Collect all current vial data from rack widgets ONLY
            all_vials_data = []
            for rack_widget in self.rack_widgets.values():
                try:
                    widget_data = rack_widget.get_vials_data()
                    all_vials_data.extend(widget_data)
                except Exception as e:
                    QMessageBox.critical(self, "Data Collection Error", 
                                       f"Failed to collect vial data from {type(rack_widget).__name__}:\n{str(e)}")
                    return
            
            # Check for duplicate vial names - BLOCK the save if found
            seen_names = set()
            duplicate_names = []
            
            for vial in all_vials_data:
                vial_name = vial.get('vial_name', '').strip()
                if vial_name:  # Only check non-empty names
                    if vial_name in seen_names:
                        if vial_name not in duplicate_names:
                            duplicate_names.append(vial_name)
                    else:
                        seen_names.add(vial_name)
            
            # Basic duplicate check for save operation
            if duplicate_names:
                error_msg = f"Cannot save: Duplicate vial names found: {', '.join(duplicate_names)}"
                QMessageBox.critical(self, "Duplicate Vial Names", error_msg)
                raise ValueError(error_msg)
            
            # Keep original vial indices - DO NOT auto-assign new ones!
            # The workflow depends on specific vial indices staying stable
            
            # Sort by vial_index if present
            try:
                all_vials_data.sort(key=lambda x: int(x.get('vial_index', 999)))
            except (ValueError, TypeError):
                pass  # Keep original order if sorting fails
            
            # Write to CSV
            if all_vials_data:
                # Use original field order if available, otherwise use standard order
                if hasattr(self, 'original_fieldnames') and self.original_fieldnames:
                    # Start with original field order
                    fieldnames = list(self.original_fieldnames)
                    
                    # Add any new fields that weren't in original (append at end)
                    all_fieldnames = set()
                    for vial in all_vials_data:
                        all_fieldnames.update(vial.keys())
                    
                    for field in all_fieldnames:
                        if field not in fieldnames:
                            fieldnames.append(field)
                else:
                    # Fallback to preferred standard order
                    preferred_order = ['vial_index', 'vial_name', 'location', 'location_index', 
                                     'vial_volume', 'capped', 'cap_type', 'vial_type', 
                                     'home_location', 'home_location_index', 'notes']
                    
                    all_fieldnames = set()
                    for vial in all_vials_data:
                        all_fieldnames.update(vial.keys())
                    
                    # Use preferred order for known fields, then add any extras
                    fieldnames = [f for f in preferred_order if f in all_fieldnames]
                    for field in sorted(all_fieldnames):
                        if field not in fieldnames:
                            fieldnames.append(field)
                
                with open(self.status_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_vials_data)
                
                self.status_bar.showMessage("File saved successfully")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                               f"Failed to save file:\n{str(e)}")
    
    def _reload_file(self):
        """Reload file from disk."""
        if self.status_file_path:
            reply = QMessageBox.question(
                self, "Reload File", 
                "Reload file from disk? Any unsaved changes will be lost.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.load_status_file(self.status_file_path)




def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Visual Vial Manager")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = VialManagerMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()