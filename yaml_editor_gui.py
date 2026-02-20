#!/usr/bin/env python3
"""
YAML Robot Status Editor - PySide6 GUI

A robust desktop application for editing robot status YAML files.
Supports robot_status.yaml, track_status.yaml, and vial_positions.yaml.

Usage:
    python yaml_editor_gui.py

Features:
- Tab-based interface for multiple YAML files
- Tree view for nested data structures
- Type validation (int, float, bool, str, null)
- Real-time editing with save/revert functionality
- Backup creation before saving
"""

import sys
import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QPushButton, QMessageBox, QFileDialog, QLabel,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSplitter, QGroupBox, QStatusBar,
    QMenuBar, QToolBar, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QFont, QIcon


class YAMLTreeWidget(QTreeWidget):
    """Enhanced QTreeWidget for YAML data editing with validation."""
    
    data_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Key", "Value", "Type"])
        self.setColumnCount(3)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.itemChanged.connect(self._on_item_changed)
        
        # Configure column widths
        header = self.header()
        header.setStretchLastSection(False)
        header.resizeSection(0, 200)  # Key column
        header.resizeSection(1, 150)  # Value column
        header.resizeSection(2, 80)   # Type column
        
        self._original_data = None
        self._file_path = None
        
    def load_yaml_file(self, file_path: str) -> bool:
        """Load YAML file into the tree widget."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
            
            self._original_data = data.copy() if isinstance(data, dict) else data
            self._file_path = file_path
            
            self.clear()
            self._populate_tree(data)
            self.expandAll()
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", 
                               f"Failed to load {file_path}:\n{str(e)}")
            return False
    
    def _populate_tree(self, data: Any, parent: Optional[QTreeWidgetItem] = None):
        """Recursively populate tree from YAML data."""
        if isinstance(data, dict):
            for key, value in data.items():
                item = QTreeWidgetItem(parent or self)
                item.setText(0, str(key))
                
                if isinstance(value, (dict, list)):
                    item.setText(1, f"({len(value)} items)")
                    item.setText(2, type(value).__name__)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._populate_tree(value, item)
                else:
                    self._set_item_value(item, value)
                    
        elif isinstance(data, list):
            for i, value in enumerate(data):
                item = QTreeWidgetItem(parent)
                item.setText(0, f"[{i}]")
                
                if isinstance(value, (dict, list)):
                    item.setText(1, f"({len(value)} items)")
                    item.setText(2, type(value).__name__)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._populate_tree(value, item)
                else:
                    self._set_item_value(item, value)
    
    def _set_item_value(self, item: QTreeWidgetItem, value: Any):
        """Set item value and type with proper formatting."""
        if value is None:
            item.setText(1, "null")
            item.setText(2, "null")
        elif isinstance(value, bool):
            item.setText(1, str(value).lower())
            item.setText(2, "bool")
        elif isinstance(value, int):
            item.setText(1, str(value))
            item.setText(2, "int")
        elif isinstance(value, float):
            item.setText(1, f"{value:.6g}")  # Compact float representation
            item.setText(2, "float")
        else:
            item.setText(1, str(value))
            item.setText(2, "str")
        
        # Make value column editable
        item.setFlags(item.flags() | Qt.ItemIsEditable)
    
    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle item value changes with validation."""
        if column != 1:  # Only process value column changes
            return
            
        value_text = item.text(1).strip()
        value_type = item.text(2)
        
        try:
            # Convert text to appropriate type
            if value_type == "null":
                if value_text.lower() in ("null", "none", ""):
                    validated_value = None
                else:
                    raise ValueError("null values must be 'null', 'none', or empty")
            elif value_type == "bool":
                if value_text.lower() in ("true", "1", "yes"):
                    validated_value = True
                elif value_text.lower() in ("false", "0", "no"):
                    validated_value = False
                else:
                    raise ValueError("bool values must be true/false/1/0/yes/no")
                item.setText(1, str(validated_value).lower())
            elif value_type == "int":
                validated_value = int(value_text)
                item.setText(1, str(validated_value))
            elif value_type == "float":
                validated_value = float(value_text)
                item.setText(1, f"{validated_value:.6g}")
            else:  # str
                validated_value = value_text
            
            self.data_changed.emit()
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Value", 
                              f"Invalid {value_type} value: {value_text}\n{str(e)}")
            # Revert to previous value (this triggers recursion, but that's OK)
            self.blockSignals(True)
            item.setText(1, item.text(1))  # This should restore from model
            self.blockSignals(False)
    
    def get_yaml_data(self) -> Dict:
        """Extract current data from tree into dictionary format."""
        return self._extract_item_data(self.invisibleRootItem())
    
    def _extract_item_data(self, parent_item: QTreeWidgetItem) -> Any:
        """Recursively extract data from tree items."""
        if parent_item.childCount() == 0:
            return None
            
        # Determine if this is a list or dict based on child keys
        child_keys = [parent_item.child(i).text(0) for i in range(parent_item.childCount())]
        is_list = all(key.startswith('[') and key.endswith(']') for key in child_keys)
        
        if is_list:
            result = []
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.childCount() > 0:
                    result.append(self._extract_item_data(child))
                else:
                    result.append(self._parse_value(child.text(1), child.text(2)))
        else:
            result = {}
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                key = child.text(0)
                if child.childCount() > 0:
                    result[key] = self._extract_item_data(child)
                else:
                    result[key] = self._parse_value(child.text(1), child.text(2))
        
        return result
    
    def _parse_value(self, value_str: str, value_type: str) -> Any:
        """Parse string value according to type."""
        if value_type == "null" or value_str.lower() in ("null", "none", ""):
            return None
        elif value_type == "bool":
            return value_str.lower() in ("true", "1", "yes")
        elif value_type == "int":
            return int(value_str)
        elif value_type == "float":
            return float(value_str)
        else:
            return value_str
    
    def has_unsaved_changes(self) -> bool:
        """Check if current data differs from original."""
        try:
            current_data = self.get_yaml_data()
            return current_data != self._original_data
        except:
            return True  # Assume changes if we can't compare
    
    def save_yaml_file(self, backup: bool = True) -> bool:
        """Save current data to YAML file with optional backup."""
        if not self._file_path:
            return False
            
        try:
            # Create backup if requested
            if backup:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self._file_path}.backup_{timestamp}"
                shutil.copy2(self._file_path, backup_path)
            
            # Save current data
            current_data = self.get_yaml_data()
            with open(self._file_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(current_data, file, default_flow_style=False, 
                              allow_unicode=True, indent=2)
            
            # Update original data reference
            self._original_data = current_data.copy() if isinstance(current_data, dict) else current_data
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                               f"Failed to save {self._file_path}:\n{str(e)}")
            return False
    
    def revert_changes(self):
        """Revert to original file data."""
        if self._original_data is not None:
            self.clear()
            self._populate_tree(self._original_data)
            self.expandAll()


class YAMLEditorMainWindow(QMainWindow):
    """Main window for YAML robot status editor."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Status YAML Editor")
        self.setGeometry(100, 100, 1000, 700)
        
        # Default robot_state directory
        self.robot_state_dir = Path("robot_state")
        if not self.robot_state_dir.exists():
            # Try relative paths
            for possible_path in [Path("."), Path(".."), Path("../.."), Path("../../..")]:
                test_path = possible_path / "robot_state"
                if test_path.exists():
                    self.robot_state_dir = test_path
                    break
        
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._load_default_files()
    
    def _setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different YAML files
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Button bar
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Current")
        self.save_button.clicked.connect(self._save_current_tab)
        button_layout.addWidget(self.save_button)
        
        self.save_all_button = QPushButton("Save All")
        self.save_all_button.clicked.connect(self._save_all_tabs)
        button_layout.addWidget(self.save_all_button)
        
        self.revert_button = QPushButton("Revert Current")
        self.revert_button.clicked.connect(self._revert_current_tab)
        button_layout.addWidget(self.revert_button)
        
        self.reload_button = QPushButton("Reload from Files")
        self.reload_button.clicked.connect(self._reload_all_files)
        button_layout.addWidget(self.reload_button)
        
        button_layout.addStretch()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Connect tab change signal to update buttons
        self.tab_widget.currentChanged.connect(self._update_button_states)
    
    def _setup_menu_bar(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open YAML File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_yaml_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Current Tab", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_current_tab)
        file_menu.addAction(save_action)
        
        save_all_action = QAction("Save &All Tabs", self)
        save_all_action.setShortcut("Ctrl+Shift+S")
        save_all_action.triggered.connect(self._save_all_tabs)
        file_menu.addAction(save_all_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        revert_action = QAction("&Revert Current Tab", self)
        revert_action.triggered.connect(self._revert_current_tab)
        edit_menu.addAction(revert_action)
        
        reload_action = QAction("Re&load All Files", self)
        reload_action.setShortcut("F5")
        reload_action.triggered.connect(self._reload_all_files)
        edit_menu.addAction(reload_action)
    
    def _setup_status_bar(self):
        """Set up status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _load_default_files(self):
        """Load default robot status YAML files."""
        default_files = [
            ("robot_status.yaml", "Robot Status"),
            ("track_status.yaml", "Track Status"), 
            ("vial_positions.yaml", "Vial Positions")
        ]
        
        for filename, tab_name in default_files:
            file_path = self.robot_state_dir / filename
            if file_path.exists():
                self._add_yaml_tab(str(file_path), tab_name)
            else:
                self.status_bar.showMessage(f"Warning: {filename} not found in {self.robot_state_dir}")
    
    def _add_yaml_tab(self, file_path: str, tab_name: Optional[str] = None) -> bool:
        """Add a new tab for editing a YAML file."""
        if tab_name is None:
            tab_name = Path(file_path).name
        
        # Create tree widget for this file
        tree_widget = YAMLTreeWidget()
        tree_widget.data_changed.connect(self._update_button_states)
        
        if tree_widget.load_yaml_file(file_path):
            self.tab_widget.addTab(tree_widget, tab_name)
            self._update_button_states()
            return True
        return False
    
    def _get_current_tree_widget(self) -> Optional[YAMLTreeWidget]:
        """Get the currently active tree widget."""
        current_widget = self.tab_widget.currentWidget()
        return current_widget if isinstance(current_widget, YAMLTreeWidget) else None
    
    def _update_button_states(self):
        """Update button enabled states based on current tab."""
        tree_widget = self._get_current_tree_widget()
        has_current_tab = tree_widget is not None
        has_changes = tree_widget.has_unsaved_changes() if tree_widget else False
        
        self.save_button.setEnabled(has_current_tab and has_changes)
        self.revert_button.setEnabled(has_current_tab and has_changes)
        
        # Check if any tab has changes
        any_changes = False
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, YAMLTreeWidget) and widget.has_unsaved_changes():
                any_changes = True
                break
        
        self.save_all_button.setEnabled(any_changes)
        
        # Update status bar
        if has_changes:
            self.status_bar.showMessage("Unsaved changes")
        else:
            self.status_bar.showMessage("Ready")
    
    def _save_current_tab(self):
        """Save the currently active tab."""
        tree_widget = self._get_current_tree_widget()
        if tree_widget and tree_widget.save_yaml_file():
            self.status_bar.showMessage("File saved successfully")
            self._update_button_states()
        
    def _save_all_tabs(self):
        """Save all tabs with unsaved changes."""
        saved_count = 0
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, YAMLTreeWidget) and widget.has_unsaved_changes():
                if widget.save_yaml_file():
                    saved_count += 1
        
        self.status_bar.showMessage(f"Saved {saved_count} file(s)")
        self._update_button_states()
    
    def _revert_current_tab(self):
        """Revert the currently active tab."""
        tree_widget = self._get_current_tree_widget()
        if tree_widget:
            reply = QMessageBox.question(self, "Revert Changes", 
                                       "Are you sure you want to revert all changes?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                tree_widget.revert_changes()
                self.status_bar.showMessage("Changes reverted")
                self._update_button_states()
    
    def _reload_all_files(self):
        """Reload all files from disk."""
        reply = QMessageBox.question(self, "Reload Files", 
                                   "Reload all files from disk? Any unsaved changes will be lost.",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Store current file paths and tab names
            tabs_to_reload = []
            for i in range(self.tab_widget.count()):
                widget = self.tab_widget.widget(i)
                if isinstance(widget, YAMLTreeWidget):
                    tab_name = self.tab_widget.tabText(i)
                    file_path = widget._file_path
                    tabs_to_reload.append((file_path, tab_name))
            
            # Clear all tabs
            self.tab_widget.clear()
            
            # Reload tabs
            for file_path, tab_name in tabs_to_reload:
                if file_path and Path(file_path).exists():
                    self._add_yaml_tab(file_path, tab_name)
            
            self.status_bar.showMessage("Files reloaded")
    
    def _open_yaml_file(self):
        """Open a new YAML file in a tab."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML File", str(self.robot_state_dir), 
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self._add_yaml_tab(file_path)
    
    def closeEvent(self, event):
        """Handle window close event, checking for unsaved changes."""
        unsaved_tabs = []
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            if isinstance(widget, YAMLTreeWidget) and widget.has_unsaved_changes():
                tab_name = self.tab_widget.tabText(i)
                unsaved_tabs.append(tab_name)
        
        if unsaved_tabs:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"The following tabs have unsaved changes:\n{', '.join(unsaved_tabs)}\n\nSave before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self._save_all_tabs()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:  # Cancel
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("YAML Robot Status Editor")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')  # Modern look across platforms
    
    window = YAMLEditorMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()