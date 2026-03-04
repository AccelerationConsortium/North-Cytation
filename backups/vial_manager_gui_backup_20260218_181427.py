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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QGridLayout, QPushButton, QLabel,
    QMessageBox, QFileDialog, QDialog, QFormLayout,
    QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QStatusBar, QMenuBar, QToolBar, QScrollArea, QFrame,
    QSplitter, QGroupBox, QTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal, QSize, QMimeData, QTimer
from PySide6.QtGui import (
    QAction, QPalette, QPainter, QFont, QPixmap, 
    QDrag, QColor, QBrush, QPen
)


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
        self.setFixedSize(80, 100)
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
        self.resize(400, 500)
        
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
            self.grid_rows = 2
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
                
                # Update our data list
                for i, vial in enumerate(self.vial_data_list):
                    if vial.get('vial_index') == updated_data.get('vial_index'):
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
        """Get the next available vial index."""
        existing_indices = set()
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
                        min-height: 60px;
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
                large_vial_layout.addWidget(placeholder, row, col)
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
                
                grid_layout.addWidget(vial_widget, row, col)
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
                
                # Update our data list
                for i, vial in enumerate(self.vial_data_list):
                    if vial.get('vial_index') == updated_data.get('vial_index'):
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
            
            self.large_vial_grid.addWidget(placeholder, row, col)
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
        """Get the next available vial index."""
        existing_indices = set()
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


class VialManagerMainWindow(QMainWindow):
    """Main window for visual vial management."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Vial Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        self.status_file_path = None
        self.original_vials_data = []
        self.rack_widgets = {}  # location_name -> VialRackWidget
        
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
        
        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self._save_file)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.reload_button = QPushButton("Reload from File")
        self.reload_button.clicked.connect(self._reload_file)
        self.reload_button.setEnabled(False)
        button_layout.addWidget(self.reload_button)
        
        button_layout.addStretch()
        
        # Stats label
        self.stats_label = QLabel("No vials loaded")
        button_layout.addWidget(self.stats_label)
        
        layout.addLayout(button_layout)
    
    def _setup_menu_bar(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_action = QAction("&Load Status File...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_file_dialog)
        file_menu.addAction(load_action)
        
        save_action = QAction("&Save Changes", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_file)
        file_menu.addAction(save_action)
        
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
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                # Auto-detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                for row in reader:
                    vials_data.append(row)
            
            if not vials_data:
                QMessageBox.warning(self, "Empty File", "No vial data found in file.")
                return False
            
            self.original_vials_data = vials_data.copy()
            self.status_file_path = file_path
            
            self._populate_racks(vials_data)
            self._update_ui_state()
            
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", 
                               f"Failed to load {file_path}:\n{str(e)}")
            return False
    
    def _populate_racks(self, vials_data: List[Dict]):
        """Populate rack widgets with vial data."""
        # Clear existing tabs
        self.tab_widget.clear()
        self.rack_widgets.clear()
        
        # Group vials by location
        locations = set(vial.get('location', 'unknown') for vial in vials_data)
        
        # Create special combined view for auxiliary locations
        aux_locations = {'large_vial_rack', 'photoreactor_array', 'clamp'}
        
        # First add main_8mL_rack tab (ensure it's first)
        main_locations = {loc for loc in locations if '8mL' in loc or 'main' in loc}
        for location in sorted(main_locations):
            if location and location != 'unknown':
                rack_widget = VialRackWidget(location)
                rack_widget.vial_edited.connect(self._on_vial_edited)
                rack_widget.vial_added.connect(self._on_vial_edited)  # Handle new vials
                
                self.tab_widget.addTab(rack_widget, location)
                self.rack_widgets[location] = rack_widget
                rack_widget.add_vials(vials_data)
        
        # Then add auxiliary racks tab
        has_aux = any(loc in aux_locations for loc in locations)
        
        if has_aux:
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
    
    def _on_vial_edited(self, updated_vial_data: Dict):
        """Handle vial edit from any rack."""
        # Handle removal or addition/edit
        vial_index = updated_vial_data.get('vial_index')
        
        if updated_vial_data.get('_remove'):
            # Handle vial removal
            if vial_index is not None:
                # Remove from master data list
                self.original_vials_data = [v for v in self.original_vials_data 
                                          if str(v.get('vial_index', '')) != str(vial_index)]
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
            
            self.status_bar.showMessage("Vial data modified - remember to save changes")
        
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI state based on loaded data."""
        has_file = self.status_file_path is not None
        
        self.save_button.setEnabled(has_file)
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
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.status_file_path}.backup_{timestamp}"
            shutil.copy2(self.status_file_path, backup_path)
            
            # Collect all current vial data from rack widgets
            all_vials_data = []
            for rack_widget in self.rack_widgets.values():
                all_vials_data.extend(rack_widget.get_vials_data())
            
            # Also include vials not in any visible rack
            visible_indices = set()
            for vial in all_vials_data:
                if 'vial_index' in vial:
                    visible_indices.add(str(vial.get('vial_index')))
            
            for vial in self.original_vials_data:
                if str(vial.get('vial_index', '')) not in visible_indices:
                    all_vials_data.append(vial)
            
            # Sort by vial_index if present
            try:
                all_vials_data.sort(key=lambda x: int(x.get('vial_index', 999)))
            except (ValueError, TypeError):
                pass  # Keep original order if sorting fails
            
            # Write to CSV
            if all_vials_data:
                # Collect all unique fieldnames from all vials
                all_fieldnames = set()
                for vial in all_vials_data:
                    all_fieldnames.update(vial.keys())
                fieldnames = sorted(all_fieldnames)  # Sort for consistent column order
                
                with open(self.status_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_vials_data)
                
                self.status_bar.showMessage(f"File saved successfully (backup: {Path(backup_path).name})")
                
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