import os
import sys
import csv
import html
import json
import math
import time
from statistics import median
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import yaml
from scheduler.console_output import ConsoleLogMirror
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parent


def read_runtime_estimates():
    """Return workflow medians from completed real runs, never execution limits."""
    path = REPO_ROOT / "experiment_tracking" / "experiment_runs.csv"
    durations = {}
    skipped = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"workflow_name", "simulate", "status", "duration_sec"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("Run log requires workflow_name, simulate, status and duration_sec columns.")
            for record in reader:
                if record["status"] != "completed" or str(record["simulate"]).strip().casefold() != "false":
                    continue
                try:
                    duration = float(record["duration_sec"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("Invalid duration")
                    name = record["workflow_name"].strip().replace("\\", "/").rsplit("/", 1)[-1]
                    if name.lower().endswith(".py"):
                        name = name[:-3]
                    if not name or name.casefold() == "unknown":
                        raise ValueError("Unknown workflow")
                    durations.setdefault(name.casefold(), []).append(duration)
                except (ValueError, TypeError, AttributeError):
                    skipped += 1
    except (OSError, ValueError, csv.Error) as error:
        return {}, f"Cannot read runtime history: {error}"
    estimates = {
        name: (median(values), len(values), min(values), max(values), values)
        for name, values in durations.items()
    }
    warning = f"Skipped {skipped} completed real-run records with invalid duration or workflow identity." if skipped else ""
    return estimates, warning


def format_runtime(seconds):
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def read_workflow_vial_path(name, config_file=None):
    config_path = (REPO_ROOT / config_file).resolve() if config_file is not None else REPO_ROOT / "workflow_configs" / f"{name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Workflow config must be a YAML mapping.")
    vial_value = config["INPUT_VIAL_STATUS_FILE"]
    if vial_value is None:
        return config_path, None
    if not isinstance(vial_value, str) or not vial_value.strip():
        raise ValueError("INPUT_VIAL_STATUS_FILE must be a nonempty path or explicit null for no vial tracking.")
    vial_path = (REPO_ROOT / vial_value).resolve()
    if not vial_path.is_file():
        raise FileNotFoundError(f"Vial file does not exist: {vial_path}")
    return config_path, vial_path


def read_vial_occupancy(workflow_names, areas):
    """Read distinct vial files once; current location alone determines occupancy."""
    sources = {}
    errors = []
    occupancy = {}
    for job in workflow_names:
        name, config_file = (job, None) if isinstance(job, str) else job
        label = name if config_file is None else f"{name} [{config_file}]"
        try:
            _, path = read_workflow_vial_path(name, config_file)
            if path is not None:
                sources.setdefault(path, set()).add(label)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
            errors.append(f"{label}: {error}")
    for path, names in sources.items():
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                required = {"vial_name", "location", "location_index", "vial_volume"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError("Vial CSV requires vial_name, location, location_index and vial_volume columns.")
                for line, record in enumerate(reader, start=2):
                    try:
                        location = record["location"].strip()
                        if location not in areas or not areas[location]["rack_present"]:
                            raise ValueError(f"Unknown or disabled current location: {location!r}")
                        number = Decimal(record["location_index"])
                        if not number.is_finite() or number != number.to_integral_value():
                            raise ValueError("Current location_index must be an integer.")
                        index = int(number)
                        if not 0 <= index < areas[location]["rack_size"]:
                            raise ValueError(f"Current slot {index} is outside {location}.")
                        record.update(workflows=sorted(names), source_file=str(path), source_line=line)
                        occupancy.setdefault((location, index), []).append(record)
                    except (ValueError, InvalidOperation, TypeError, AttributeError) as error:
                        errors.append(f"{path.name}, row {line}: {error}")
        except (OSError, ValueError, csv.Error) as error:
            errors.append(f"{path.name}: {error}")
    return occupancy, errors


class VialLayout(QWidget):
    def __init__(self):
        super().__init__()
        self.slots = {}
        self.areas = {}
        self.occupancy = {}
        self.load_error = None
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        for text, color in (("Empty", "#e4e7eb"), ("Occupied", "#23854a"), ("Conflict", "#c63636")):
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background: {color}; border: 1px solid #aaa;")
            toolbar.addWidget(swatch)
            toolbar.addWidget(QLabel(text))
        toolbar.addStretch()
        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_button.setToolTip("Refresh vial files")
        self.refresh_button.setAccessibleName("Refresh vial files")
        toolbar.addWidget(self.refresh_button)
        layout.addLayout(toolbar)
        self.summary = QLabel("No workflows selected")
        layout.addWidget(self.summary)
        self.errors = QLabel()
        self.errors.setWordWrap(True)
        self.errors.setTextFormat(Qt.PlainText)
        self.errors.setStyleSheet("color: #a52222;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self.errors)
        sections = QGridLayout()
        sections.setSpacing(20)
        sections.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        auxiliary = QWidget()
        auxiliary_layout = QGridLayout(auxiliary)
        auxiliary_layout.setContentsMargins(0, 0, 0, 0)
        auxiliary_layout.setSpacing(14)
        auxiliary_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        sections.addWidget(auxiliary, 0, 1, Qt.AlignLeft | Qt.AlignTop)
        content_layout.addLayout(sections)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        try:
            self.areas = yaml.safe_load((REPO_ROOT / "robot_state" / "vial_positions.yaml").read_text(encoding="utf-8"))
            if not isinstance(self.areas, dict) or not self.areas:
                raise ValueError("Vial positions must define at least one area.")
            for name, area in self.areas.items():
                if type(area["rack_present"]) is not bool:
                    raise ValueError(f"rack_present must be true or false: {name}")
            self.areas = {name: area for name, area in self.areas.items() if area["rack_present"]}
            for section_index, (name, area) in enumerate(self.areas.items()):
                size = area["rack_size"]
                rows = area["grid_params"]["num_rows"]
                columns = area["grid_params"]["num_cols"]
                if any(type(value) is not int or value < 1 for value in (size, rows, columns)) or size > rows * columns:
                    raise ValueError(f"Invalid rack dimensions: {name}")
                section = QWidget()
                section_layout = QVBoxLayout(section)
                heading = QLabel(name.replace("_", " "))
                heading.setWordWrap(True)
                heading.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                heading.setStyleSheet("font-weight: 600;")
                section_layout.addWidget(heading)
                grid = QGridLayout()
                grid.setSpacing(5)
                section_layout.addLayout(grid)
                section_layout.addStretch()
                for index in range(size):
                    slot = QLabel(str(index))
                    slot.setAlignment(Qt.AlignCenter)
                    slot.setFixedSize(58, 52)
                    slot.setTextFormat(Qt.PlainText)
                    slot.setAccessibleName(f"{name} slot {index}")
                    grid.addWidget(slot, index % rows, columns - 1 - index // rows)
                    self.slots[name, index] = slot
                if name == "main_8mL_rack":
                    sections.addWidget(section, 0, 0, Qt.AlignLeft | Qt.AlignTop)
                else:
                    positions = {
                        "heater": (0, 0, 1, 2),
                        "large_vial_rack": (1, 0, 2, 1),
                        "photoreactor_array": (1, 1, 1, 1),
                        "clamp": (2, 1, 1, 1),
                    }
                    position = positions[name] if name in positions else (3 + section_index, 0, 1, 2)
                    auxiliary_layout.addWidget(section, *position, Qt.AlignLeft | Qt.AlignTop)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            self.load_error = f"Cannot load vial area configuration: {error}"
        self.refresh([])

    def refresh(self, names):
        if self.load_error is not None:
            self.errors.setText(self.load_error)
            self.summary.setText("Layout unavailable")
            return
        self.occupancy, errors = read_vial_occupancy(names, self.areas)
        conflicts = 0
        for key, slot in self.slots.items():
            claims = self.occupancy.get(key, [])
            conflict = len(claims) > 1
            conflicts += int(conflict)
            state = "conflict" if conflict else "occupied" if claims else "empty"
            slot.setProperty("occupancy", state)
            color = "#c63636" if conflict else "#23854a" if claims else "#e4e7eb"
            foreground = "white" if claims else "#50555b"
            slot.setStyleSheet(f"background: {color}; color: {foreground}; border: 1px solid #b7bbc0; border-radius: 4px;")
            slot.setText(f"{key[1]}\n{len(claims)} vials" if conflict else str(key[1]))
            details = [f"{key[0]} [{key[1]}]"]
            for claim in claims:
                details.append(
                    f"{claim['vial_name']} | {claim['vial_volume']} mL\n"
                    f"Workflow: {', '.join(claim['workflows'])}\n"
                    f"File: {claim['source_file']} (row {claim['source_line']})"
                )
            slot.setToolTip("<qt><pre>" + html.escape("\n\n".join(details)) + "</pre></qt>")
        self.summary.setText(f"{len(self.occupancy)} occupied positions | {conflicts} conflicting positions | {len(errors)} input issues")
        self.errors.setText("\n".join(errors))
        self.errors.setVisible(bool(errors))


class SchedulerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Workflow Scheduler")
        self.resize(1000, 725)
        self.setup_window = None
        self.setup_root = REPO_ROOT / "temp" / "scheduler_setup" / uuid4().hex
        self._moving_rows = False
        self.execution_mode = "simulation"
        self.approved_simulation_inputs = None
        self.pending_simulation_inputs = None
        self.simulation_process = None
        self.simulation_session = None
        self.simulation_rows = []
        self.simulation_index = 0
        self.simulation_job_start_time = None
        self.stop_simulation_requested = False
        self._execution_busy = False
        self._loading_shared_state = False
        self.shared_state_baselines = {}
        self.runtime_estimates, self.runtime_history_warning = read_runtime_estimates()
        self.workflow_names = sorted(
            path.stem
            for path in (REPO_ROOT / "workflows").glob("*.py")
            if not path.name.startswith("_")
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        heading = QLabel("Experiment Queue")
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(heading)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        queue = QWidget()
        queue_layout = QVBoxLayout(queue)
        queue_layout.setContentsMargins(0, 10, 0, 0)
        self.tabs.addTab(queue, "Queue")
        self.vial_layout = VialLayout()
        self.tabs.addTab(self.vial_layout, "Vial Layout")
        self.vial_layout.refresh_button.clicked.connect(self.refresh_vial_layout)
        toolbar = QHBoxLayout()
        self.queue_toolbar = toolbar
        add = QPushButton("Add Workflow")
        add.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        add.clicked.connect(lambda: self.add_row())
        toolbar.addWidget(add)
        for label, icon, action in (
            ("Remove selected workflow", QStyle.SP_TrashIcon, self.remove_row),
            ("Move selected workflow up", QStyle.SP_ArrowUp, lambda: self.move_row(-1)),
            ("Move selected workflow down", QStyle.SP_ArrowDown, lambda: self.move_row(1)),
        ):
            button = QPushButton()
            button.setIcon(self.style().standardIcon(icon))
            button.setToolTip(label)
            button.setAccessibleName(label)
            button.setFixedSize(34, 34)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        toolbar.addStretch()
        refresh_history = QPushButton()
        refresh_history.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_history.setToolTip("Refresh runtime estimates from the master run log")
        refresh_history.setAccessibleName("Refresh runtime estimates")
        refresh_history.clicked.connect(self.refresh_runtime_estimates)
        toolbar.addWidget(refresh_history)
        self.count_label = QLabel()
        toolbar.addWidget(self.count_label)
        queue_layout.addLayout(toolbar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Workflow", "Setup", "Status", "Notes", "Estimated Time", "Actual Time"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(48)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(4, 135)
        self.table.setColumnWidth(5, 135)
        self.table.cellDoubleClicked.connect(self.show_simulation_notes)
        queue_layout.addWidget(self.table)

        footer = QHBoxLayout()
        footer.addStretch()
        self.stop_button = QPushButton("Stop After Current")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_after_current)
        footer.addWidget(self.stop_button)
        for label, icon in (
            ("Simulate", QStyle.SP_MediaPlay),
            ("Run", QStyle.SP_DialogApplyButton),
        ):
            button = QPushButton(label)
            button.setIcon(self.style().standardIcon(icon))
            button.setEnabled(False)
            button.setMinimumSize(110, 36)
            if label == "Simulate":
                self.simulate_button = button
                button.clicked.connect(self.start_simulation)
            else:
                self.run_button = button
                button.clicked.connect(self.start_live_run)
            footer.addWidget(button)
        layout.addLayout(footer)
        self.statusBar().showMessage("Queue not running")
        self.setup_shared_state_tabs()
        self.add_row()

    def setup_shared_state_tabs(self):
        from vial_manager_gui import RobotStatusWidget, TrackStatusWidget

        self.robot_status_widget = RobotStatusWidget(
            REPO_ROOT / "robot_state" / "robot_status.yaml", show_load_errors=False)
        self.track_status_widget = TrackStatusWidget(
            REPO_ROOT / "robot_state" / "track_status.yaml", show_load_errors=False)
        self.shared_state_widgets = (self.robot_status_widget, self.track_status_widget)
        for widget, title, save, reload in (
            (self.robot_status_widget, "Robot Status", self.robot_status_widget._save_robot_status,
             self.robot_status_widget._load_robot_status),
            (self.track_status_widget, "Track Status", self.track_status_widget._save_track_status,
             self.track_status_widget._load_track_status),
        ):
            self.tabs.addTab(widget, title)
            buttons = QHBoxLayout()
            save_button = QPushButton("Save")
            save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
            save_button.setAccessibleName(f"Save {title}")
            save_button.clicked.connect(lambda checked=False, widget=widget, save=save:
                                        self.save_shared_state(widget, save))
            buttons.addWidget(save_button)
            reload_button = QPushButton("Reload")
            reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
            reload_button.setAccessibleName(f"Reload {title}")
            reload_button.setToolTip("Reload saved state and discard pending edits")
            reload_button.clicked.connect(lambda checked=False, widget=widget, reload=reload:
                                         self.reload_shared_state(widget, reload))
            buttons.addWidget(reload_button)
            buttons.addStretch()
            widget.layout().addLayout(buttons)
            self.shared_state_baselines[widget] = self.shared_state_snapshot(widget)
            widget.state_saved.connect(lambda widget=widget, reload=reload:
                                       self.shared_state_saved(widget, reload))
            for control in widget.findChildren(QWidget):
                if isinstance(control, QLineEdit):
                    control.textChanged.connect(self.shared_state_changed)
                elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                    control.valueChanged.connect(self.shared_state_changed)
                elif isinstance(control, QComboBox):
                    control.currentIndexChanged.connect(self.shared_state_changed)

    def shared_state_snapshot(self, widget):
        values = []
        for control in widget.findChildren(QWidget):
            if isinstance(control, QLineEdit):
                values.append(control.text())
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                values.append(control.value())
            elif isinstance(control, QComboBox):
                values.append(control.currentText())
        return tuple(values)

    def shared_state_dirty(self):
        return any(self.shared_state_snapshot(widget) != self.shared_state_baselines[widget]
                   for widget in self.shared_state_widgets)

    def shared_state_ready(self):
        return all(widget.state_loaded for widget in self.shared_state_widgets) and not self.shared_state_dirty()

    def shared_state_changed(self, *args):
        if self._loading_shared_state:
            return
        self.approved_simulation_inputs = None
        self.pending_simulation_inputs = None
        self.refresh_vial_layout()
        if self.shared_state_dirty():
            self.statusBar().showMessage("Unsaved robot/track edits: Save or Reload before Simulate or Run")

    def save_shared_state(self, widget, save):
        if self._execution_busy or self.setup_window is not None:
            return
        if not widget.state_loaded:
            QMessageBox.warning(self, "Cannot Save Shared State", "Load a valid shared state file with Reload before saving.")
            return
        save(silent=True)

    def shared_state_saved(self, widget, reload):
        self._loading_shared_state = True
        try:
            if reload():
                self.shared_state_baselines[widget] = self.shared_state_snapshot(widget)
        finally:
            self._loading_shared_state = False
        self.shared_state_changed()
        if self.shared_state_ready():
            self.statusBar().showMessage("Shared state saved; simulate again before Run")

    def reload_shared_state(self, widget, reload, confirm=True):
        if self._execution_busy or self.setup_window is not None:
            return
        dirty = self.shared_state_snapshot(widget) != self.shared_state_baselines[widget]
        if confirm and dirty and QMessageBox.question(
            self, "Discard Shared Edits?", "Reload saved state and discard pending edits in this tab?",
            QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Discard:
            return
        self.shared_state_saved(widget, reload)
        if not widget.state_loaded:
            self.statusBar().showMessage("Cannot load shared robot/track state; check the YAML files and Reload")
        elif self.shared_state_ready():
            self.statusBar().showMessage("Shared state reloaded; simulate again before Run")

    def add_row(self, selected_workflow=None):
        if self.setup_window is not None or self.simulation_process is not None:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        selector = QComboBox()
        selector.setProperty("job_id", uuid4().hex)
        selector.setEditable(True)
        selector.setInsertPolicy(QComboBox.NoInsert)
        selector.addItem("Select workflow...", None)
        for name in self.workflow_names:
            selector.addItem(name, name)
        selector.completer().setCaseSensitivity(Qt.CaseInsensitive)
        if selected_workflow is not None:
            selector.setCurrentIndex(selector.findData(selected_workflow))
        self.table.setCellWidget(row, 0, selector)

        setup = QPushButton("Setup")
        setup.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        setup.setEnabled(selected_workflow in self.workflow_names)
        setup.setToolTip("Edit this row's configuration and vial file")
        self.table.setCellWidget(row, 1, setup)
        status = QTableWidgetItem("Not configured" if selected_workflow else "Not selected")
        self.table.setItem(row, 2, status)
        notes = QTableWidgetItem("")
        self.table.setItem(row, 3, notes)
        estimate = QTableWidgetItem("")
        self.table.setItem(row, 4, estimate)
        self.update_runtime_estimate(selector, estimate)
        actual = QTableWidgetItem("")
        self.table.setItem(row, 5, actual)
        if selected_workflow in self.workflow_names:
            self.initialize_row_config(selector, status, notes)

        def selection_changed():
            if self._moving_rows:
                return
            self.approved_simulation_inputs = None
            name = selector.currentData()
            valid = name is not None and selector.currentText() == name
            setup.setEnabled(valid)
            if selector.property("snapshot_workflow") != (name if valid else None):
                selector.setProperty("config_file", None)
                selector.setProperty("snapshot_workflow", name if valid else None)
                selector.setProperty("config_source", None)
                status.setText("Not configured" if valid else "Not selected")
                notes.setText("")
                if valid:
                    self.initialize_row_config(selector, status, notes)
            notes.setToolTip("")
            notes.setData(Qt.UserRole, None)
            self.set_simulation_row_colour(self.table.indexFromItem(status).row(), None)
            self.update_runtime_estimate(selector, estimate)
            self.refresh_vial_layout()

        selector.currentIndexChanged.connect(selection_changed)
        selector.editTextChanged.connect(selection_changed)
        setup.clicked.connect(lambda: self.open_setup(selector, status, notes))
        self.table.selectRow(row)
        self.count_label.setText(f"{self.table.rowCount()} queued")
        self.refresh_vial_layout()

    def row_config_path(self, selector):
        return selector.property("config_file")

    def initialize_row_config(self, selector, status, notes):
        name = selector.currentData()
        selector.setProperty("snapshot_workflow", name)
        target = self.setup_root / f"{selector.property('job_id')}.yaml"
        selector.setProperty("config_file", str(target))
        try:
            target.unlink(missing_ok=True)
            self.set_row_config(selector, None)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
            status.setText("Review setup")
            notes.setText(str(error))

    def set_row_config(self, selector, path):
        if self.setup_window is not None or self.simulation_process is not None:
            raise ValueError("Cannot change configuration while setup or execution is open.")
        name = selector.currentData()
        source = (REPO_ROOT / path).resolve() if path is not None else REPO_ROOT / "workflow_configs" / f"{name}.yaml"
        content = source.read_bytes()
        config = yaml.safe_load(content)
        if not isinstance(config, dict):
            raise ValueError("Workflow config must be a YAML mapping.")
        target = self.setup_root / f"{selector.property('job_id')}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        selector.setProperty("config_file", str(target))
        selector.setProperty("config_source", str(source))
        selector.setProperty("snapshot_workflow", name)
        self.approved_simulation_inputs = None
        self.pending_simulation_inputs = None
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 0) is selector:
                self.table.item(row, 2).setText("Not configured")
                notes = self.table.item(row, 3)
                notes.setText("")
                notes.setToolTip("")
                notes.setData(Qt.UserRole, None)
                self.set_simulation_row_colour(row, None)
                break
        self.refresh_vial_layout()

    def update_runtime_estimate(self, selector, item):
        name = selector.currentData()
        if name not in self.workflow_names or selector.currentText() != name:
            item.setText("")
            item.setToolTip("")
            return
        estimate = self.runtime_estimates.get(name.casefold())
        if estimate is None:
            item.setText("History unavailable" if self.runtime_history_warning.startswith("Cannot read") else "No history")
            item.setToolTip(self.runtime_history_warning or "No completed non-simulated runs for this workflow.")
            return
        seconds, count, shortest, longest, values = estimate
        item.setText(format_runtime(seconds))
        details = (
            f"Median of {count} completed non-simulated runs.\n"
            f"Historical range: {format_runtime(shortest)} to {format_runtime(longest)}.\n"
            "\nIndividual completed run durations:\n"
            + "\n".join(f"{index}. {format_runtime(duration)}" for index, duration in enumerate(values, start=1))
        )
        if self.runtime_history_warning:
            details += "\n" + self.runtime_history_warning
        item.setToolTip(details)

    def refresh_runtime_estimates(self):
        self.runtime_estimates, self.runtime_history_warning = read_runtime_estimates()
        for row in range(self.table.rowCount()):
            self.update_runtime_estimate(self.table.cellWidget(row, 0), self.table.item(row, 4))

    def refresh_vial_layout(self):
        if self._moving_rows:
            return
        names = []
        for row in range(self.table.rowCount()):
            selector = self.table.cellWidget(row, 0)
            name = selector.currentData()
            if name in self.workflow_names and selector.currentText() == name:
                names.append((name, self.row_config_path(selector)))
        self.vial_layout.refresh(names)
        if not self._execution_busy:
            self.simulate_button.setEnabled(self.setup_window is None and self.shared_state_ready()
                                            and bool(names) and len(names) == self.table.rowCount())
            self.update_live_run_enabled()
        if not all(widget.state_loaded for widget in self.shared_state_widgets):
            self.statusBar().showMessage("Cannot load shared robot/track state; check the YAML files and Reload")

    def set_simulation_busy(self, busy):
        self._execution_busy = busy and self.setup_window is None
        self.table.setEnabled(not busy)
        for widget in self.shared_state_widgets:
            widget.setEnabled(not busy)
        for index in range(self.queue_toolbar.count()):
            widget = self.queue_toolbar.itemAt(index).widget()
            if isinstance(widget, QPushButton):
                widget.setEnabled(not busy)
        self.simulate_button.setEnabled(not busy and self.shared_state_ready() and bool(self.simulation_rows))
        self.stop_button.setEnabled(self._execution_busy)
        self.run_button.setEnabled(False)

    def queue_input_fingerprint(self):
        from scheduler.live_runner import fingerprint
        names = []
        jobs = []
        paths = [REPO_ROOT / "robot_state" / "robot_status.yaml", REPO_ROOT / "robot_state" / "track_status.yaml"]
        paths.extend((REPO_ROOT / "robot_state").glob("*.yaml"))
        for row in range(self.table.rowCount()):
            selector = self.table.cellWidget(row, 0)
            name = selector.currentData()
            if name not in self.workflow_names or selector.currentText() != name:
                raise ValueError("Every queue row must select a workflow.")
            config_path, vial_path = read_workflow_vial_path(name, self.row_config_path(selector))
            names.append(name)
            jobs.append((selector.property("job_id"), name, str(config_path.resolve())))
            paths.extend([config_path, REPO_ROOT / "workflows" / f"{name}.py"])
            if vial_path is not None:
                paths.append(vial_path)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            for value in config.values():
                if isinstance(value, str):
                    candidate = REPO_ROOT / value
                    if candidate.is_file():
                        paths.append(candidate)
        paths.extend((REPO_ROOT / "workflows").glob("*.py"))
        return {"workflows": names, "jobs": jobs, "files": fingerprint(paths)}

    def update_live_run_enabled(self):
        allowed = (not self._execution_busy and self.setup_window is None and self.shared_state_ready()
               and self.approved_simulation_inputs is not None)
        if allowed:
            try:
                allowed = self.queue_input_fingerprint() == self.approved_simulation_inputs
            except (OSError, ValueError, KeyError, yaml.YAMLError):
                allowed = False
        allowed = allowed and not self.vial_layout.errors.text() and not any(
            len(claims) > 1 for claims in self.vial_layout.occupancy.values()
        )
        self.run_button.setEnabled(bool(allowed))

    def start_live_run(self):
        if self._execution_busy or self.setup_window is not None:
            return
        self.refresh_vial_layout()
        if not self.run_button.isEnabled():
            QMessageBox.warning(self, "Run Blocked", "Simulate this unchanged queue successfully with zero errors and no vial conflicts first.")
            return
        response = QMessageBox.question(
            self, "Start Real Hardware?", "Run the approved queue on REAL hardware?\n"
            "Confirm the physical inventory matches the saved state and no other hardware workflow is running.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return
        from uuid import uuid4
        try:
            if self.queue_input_fingerprint() != self.approved_simulation_inputs:
                raise ValueError("Inputs changed during confirmation; simulate again.")
            root = REPO_ROOT / "scheduler" / "state" / ("live_" + uuid4().hex)
            root.mkdir(parents=True)
            vial_files = []
            jobs = []
            for row in range(self.table.rowCount()):
                selector = self.table.cellWidget(row, 0)
                name = selector.currentData()
                config_path, vial_path = read_workflow_vial_path(name, self.row_config_path(selector))
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                jobs.append((row, name, config_path, config, vial_path))
                if vial_path is not None:
                    vial_files.append(str(vial_path))
            for row, name, config_path, config, vial_path in jobs:
                config["SIMULATE"] = False
                config["INPUT_VIAL_STATUS_FILE"] = str(vial_path) if vial_path is not None else None
                folder = root / "jobs" / f"{row:03d}"
                folder.mkdir(parents=True)
                (folder / "input.json").write_text(json.dumps({
                    "workflow": name, "config_file": str(config_path.resolve()), "config": config,
                    "input_hashes": {path: digest for path, digest in self.approved_simulation_inputs["files"].items()
                                     if path not in set(vial_files) | {
                                         str((REPO_ROOT / "robot_state" / "robot_status.yaml").resolve()),
                                         str((REPO_ROOT / "robot_state" / "track_status.yaml").resolve())
                                     } or (vial_path is not None and path == str(vial_path))},
                    "state_root": str(REPO_ROOT / "robot_state"), "vial_files": vial_files,
                }), encoding="utf-8")
            if self.queue_input_fingerprint() != self.approved_simulation_inputs:
                raise ValueError("Inputs changed during preparation; simulate again.")
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            QMessageBox.warning(self, "Run Blocked", str(error))
            return
        self.execution_mode = "live"
        self.approved_simulation_inputs = None
        self.simulation_session = root
        self.simulation_rows = list(range(self.table.rowCount()))
        self.simulation_index = 0
        self.stop_simulation_requested = False
        self.table.clearSelection()
        for row in self.simulation_rows:
            self.table.item(row, 2).setText("Pending run")
            self.table.item(row, 3).setText("")
            self.table.item(row, 3).setData(Qt.UserRole, None)
            self.table.item(row, 5).setText("")
            self.table.item(row, 5).setToolTip("")
            self.set_simulation_row_colour(row, None)
        self.set_simulation_busy(True)
        self.launch_next_simulation()

    def set_simulation_row_colour(self, row, state):
        colours = {"running": "#fff0a6", "success": "#ccebd5", "error": "#f5cccc"}
        colour = colours.get(state)
        for column in range(self.table.columnCount()):
            item = self.table.item(row, column)
            if item is not None:
                item.setBackground(QBrush(QColor(colour)) if colour else QBrush())
                item.setForeground(QBrush(QColor("#202020")) if colour else QBrush())
            widget = self.table.cellWidget(row, column)
            if widget is not None:
                widget.setStyleSheet(
                    f"background-color: {colour}; color: #202020;" if colour else ""
                )

    def start_simulation(self):
        if self._execution_busy or self.setup_window is not None:
            return
        if not self.shared_state_ready():
            QMessageBox.warning(self, "Cannot Simulate Queue", "Save or Reload robot/track state before launching.")
            return
        from scheduler.simulation_state import create_session
        from scheduler.simulation_runner import handoff_issues
        jobs = []
        self.execution_mode = "simulation"
        self.approved_simulation_inputs = None
        allow_conflicts = False
        try:
            self.refresh_vial_layout()
            if self.vial_layout.errors.text():
                raise ValueError(self.vial_layout.errors.text())
            if any(len(claims) > 1 for claims in self.vial_layout.occupancy.values()):
                response = QMessageBox.question(
                    self, "Simulate With Vial Conflicts?",
                    "Current vial positions conflict. Continue this simulation for testing?\n"
                    "Conflicts will remain reported; this does not permit live execution.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if response != QMessageBox.Yes:
                    return
                allow_conflicts = True
            self.pending_simulation_inputs = self.queue_input_fingerprint()
            for row in range(self.table.rowCount()):
                selector = self.table.cellWidget(row, 0)
                name = selector.currentData()
                if name not in self.workflow_names or selector.currentText() != name:
                    raise ValueError(f"Select a workflow for row {row + 1}.")
                config_path, vial_path = read_workflow_vial_path(name, self.row_config_path(selector))
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                jobs.append((row, name, config, vial_path, config_path))
            if not jobs:
                return
            root = create_session([job[3] for job in jobs])
            issues = handoff_issues(root)
            if allow_conflicts:
                issues = [issue for issue in issues if not issue.startswith("Current vial-position conflict")]
            if issues:
                raise ValueError("Initial state requires review:\n" + "\n".join(issues))
            for index, (row, name, config, vial_path, config_path) in enumerate(jobs):
                folder = root / "jobs" / f"{index:03d}"
                folder.mkdir(parents=True)
                config["INPUT_VIAL_STATUS_FILE"] = str(vial_path) if vial_path is not None else None
                (folder / "input.json").write_text(json.dumps({
                    "workflow": name, "config_file": str(config_path.resolve()),
                    "config": config, "allow_vial_conflicts": allow_conflicts
                }), encoding="utf-8")
            if self.queue_input_fingerprint() != self.pending_simulation_inputs:
                raise ValueError("Inputs changed during preparation; simulate again.")
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            QMessageBox.warning(self, "Cannot Simulate Queue", str(error))
            return
        self.simulation_session = root
        self.simulation_rows = [job[0] for job in jobs]
        self.simulation_index = 0
        self.stop_simulation_requested = False
        self.table.clearSelection()
        for row in self.simulation_rows:
            self.table.item(row, 2).setText("Pending simulation")
            self.table.item(row, 3).setText("")
            self.table.item(row, 3).setData(Qt.UserRole, None)
            self.table.item(row, 5).setText("")
            self.table.item(row, 5).setToolTip("")
            self.set_simulation_row_colour(row, None)
        self.set_simulation_busy(True)
        self.launch_next_simulation()

    def launch_next_simulation(self):
        if self.stop_simulation_requested or self.simulation_index >= len(self.simulation_rows):
            self.finish_simulation_queue()
            return
        row = self.simulation_rows[self.simulation_index]
        folder = self.simulation_session / "jobs" / f"{self.simulation_index:03d}"
        process = QProcess(self)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setStandardOutputFile(str(folder / "console.log"))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(environment)
        self.console_mirror = ConsoleLogMirror(folder / "console.log", sys.stdout, self)
        process.finished.connect(self.simulation_finished)
        process.errorOccurred.connect(self.simulation_process_error)
        self.simulation_process = process
        live = self.execution_mode == "live"
        self.table.item(row, 2).setText("Running" if live else "Simulating")
        self.set_simulation_row_colour(row, "running")
        self.statusBar().showMessage(f"{'Running' if live else 'Simulating'} {self.simulation_index + 1} of {len(self.simulation_rows)}")
        self.simulation_job_start_time = time.monotonic()
        arguments = (["-u", "-m", "scheduler.live_runner", "--job-folder", str(folder)] if live else
                 ["-u", "-m", "scheduler.simulation_runner", "--session", str(self.simulation_session), "--job", folder.name])
        process.start(sys.executable, arguments)

    def simulation_process_error(self, error):
        if error == QProcess.FailedToStart:
            self.simulation_finished(-1, QProcess.CrashExit)

    def simulation_finished(self, exit_code, exit_status):
        process = self.simulation_process
        if process is None:
            return
        self.console_mirror.finish()
        self.console_mirror = None
        self.simulation_process = None
        row = self.simulation_rows[self.simulation_index]
        folder = self.simulation_session / "jobs" / f"{self.simulation_index:03d}"
        elapsed = time.monotonic() - self.simulation_job_start_time
        self.table.item(row, 5).setText(format_runtime(elapsed))
        self.table.item(row, 5).setToolTip(
            f"Actual wall-clock duration of this {'run' if self.execution_mode == 'live' else 'simulation'}: {round(elapsed)} s."
        )
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            records = result["records"]
            errors = sum(record["level"] in {"ERROR", "CRITICAL"} for record in records)
            warnings = sum(record["level"] == "WARNING" for record in records)
            okay = exit_code == 0 and exit_status == QProcess.NormalExit and result["completed"] and result["handoff_ok"]
            live = self.execution_mode == "live"
            if live and errors:
                okay = False
            self.set_simulation_row_colour(row, "success" if okay and not errors else "error")
            self.table.item(row, 2).setText(("Completed" if okay else "Run failed") if live else
                                          "Needs attention" if okay and errors else "Simulated" if okay else "Simulation incomplete")
            self.table.item(row, 3).setText(f"{errors} errors, {warnings} warnings")
            result["console_log"] = str(folder / "console.log")
            if not live:
                result["end_state"] = str(self.simulation_session / "end_states" / folder.name)
            self.table.item(row, 3).setData(Qt.UserRole, result)
            ordered_records = sorted(records, key=lambda record: record["level"] not in {"ERROR", "CRITICAL"})
            messages = "\n".join(f"{record['level']}: {record['message']}" for record in ordered_records)
            self.table.item(row, 3).setToolTip(messages[:6000] or "No logged errors or warnings")
        except (OSError, ValueError, KeyError, TypeError) as error:
            okay = False
            self.set_simulation_row_colour(row, "error")
            message = f"Child did not return a valid simulation report: {error}. {process.errorString()}"
            self.table.item(row, 2).setText("Run failed" if self.execution_mode == "live" else "Simulation incomplete")
            self.table.item(row, 3).setText("1 error")
            self.table.item(row, 3).setToolTip(message)
            self.table.item(row, 3).setData(Qt.UserRole, {
                "records": [{"level": "ERROR", "message": message}], "console_log": str(folder / "console.log")
            })
        process.deleteLater()
        self.simulation_index += 1
        if not okay:
            self.stop_simulation_requested = True
        self.launch_next_simulation()

    def finish_simulation_queue(self):
        live = self.execution_mode == "live"
        for row in self.simulation_rows[self.simulation_index:]:
            self.table.item(row, 2).setText("Not run" if live else "Not simulated")
        if not live and not self.stop_simulation_requested and all(
            self.table.item(row, 2).text() == "Simulated" for row in self.simulation_rows
        ):
            self.approved_simulation_inputs = self.pending_simulation_inputs
        self.set_simulation_busy(False)
        if live and not self.shared_state_dirty():
            self.reload_shared_state(self.robot_status_widget, self.robot_status_widget._load_robot_status, confirm=False)
            self.reload_shared_state(self.track_status_widget, self.track_status_widget._load_track_status, confirm=False)
        self.stop_button.setEnabled(False)
        self.refresh_vial_layout()
        label = "Run" if live else "Simulation"
        self.statusBar().showMessage(f"{label} {'stopped' if self.stop_simulation_requested else 'finished'} | {self.simulation_session}")

    def stop_after_current(self):
        self.stop_simulation_requested = True
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopping after the current workflow; child will not be terminated")

    def show_simulation_notes(self, row, column):
        if column in (4, 5):
            self.show_time_details(row, column)
            return
        if column != 3:
            return
        result = self.table.item(row, 3).data(Qt.UserRole)
        if not result:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Workflow Notes")
        dialog.resize(760, 500)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        ordered_records = sorted(result["records"], key=lambda record: record["level"] not in {"ERROR", "CRITICAL"})
        messages = [f"{record['level']}: {record['message']}" for record in ordered_records]
        for key in ("tips_used", "plates_used", "log_file", "console_log", "end_state"):
            if key in result:
                messages.append(f"{key}: {result[key]}")
        text.setPlainText("\n\n".join(messages) or "No logged errors or warnings")
        layout.addWidget(text)
        dialog.exec()

    def show_time_details(self, row, column):
        item = self.table.item(row, column)
        if item is None:
            return
        details = item.toolTip()
        if not details:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Estimated Time Details" if column == 4 else "Actual Time Details")
        dialog.resize(600, 300)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(details)
        layout.addWidget(text)
        dialog.exec()

    def closeEvent(self, event):
        if self.simulation_process is not None:
            QMessageBox.warning(self, "Workflow Running", "Wait for the current workflow to finish before closing.")
            event.ignore()
            return
        super().closeEvent(event)

    def open_setup(self, selector, status, notes):
        if self._execution_busy:
            return
        if selector.currentData() not in self.workflow_names or selector.currentText() != selector.currentData():
            return
        if self.setup_window is not None:
            self.setup_window.raise_()
            self.setup_window.activateWindow()
            return
        import yaml
        from vial_manager_gui import VialManagerMainWindow

        name = selector.currentData()
        config_file = self.row_config_path(selector)
        editor = None
        draft = None
        try:
            if config_file is None or not Path(config_file).is_file():
                self.set_row_config(selector, None)
                config_file = self.row_config_path(selector)
            config_path, vial_path = read_workflow_vial_path(name, config_file)
            draft = config_path.with_name(config_path.stem + "_draft_" + uuid4().hex + ".yaml")
            draft.write_bytes(config_path.read_bytes())
            editor = VialManagerMainWindow(preparation_mode=True, include_shared_state=False)
            editor.setup_preparation(vial_path, name, draft,
                                     config_source=selector.property("config_source"), row_owned=True)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
            if editor is not None:
                editor.deleteLater()
            if draft is not None:
                draft.unlink(missing_ok=True)
            notes.setText(str(error))
            QMessageBox.warning(self, "Cannot Open Setup", str(error))
            return

        def finished(saved):
            self.setup_window = None
            self.set_simulation_busy(False)
            if not saved:
                draft.unlink(missing_ok=True)
                self.refresh_vial_layout()
                return
            try:
                _, updated_path = read_workflow_vial_path(name, draft)
                config_path.write_bytes(draft.read_bytes())
                selector.setProperty("config_source", editor.config_editor.config_source)
                self.approved_simulation_inputs = None
                self.pending_simulation_inputs = None
                if updated_path != vial_path:
                    status.setText("Review vial file")
                    notes.setText("Vial path changed. Open Setup again to review the new file.")
                    return
                status.setText("Setup saved")
                notes.setText("")
                notes.setData(Qt.UserRole, None)
                self.set_simulation_row_colour(self.table.indexFromItem(status).row(), None)
            except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
                status.setText("Review setup")
                notes.setText(str(error))
            finally:
                draft.unlink(missing_ok=True)
                self.refresh_vial_layout()

        editor.setParent(self, Qt.Window)
        editor.setWindowModality(Qt.WindowModal)
        editor.setAttribute(Qt.WA_DeleteOnClose)
        editor.preparation_finished.connect(finished)
        self.setup_window = editor
        self.set_simulation_busy(True)
        editor.show()

    def remove_row(self):
        if self.setup_window is not None or self.simulation_process is not None:
            return
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.count_label.setText(f"{self.table.rowCount()} queued")
            self.refresh_vial_layout()

    def move_row(self, direction):
        if self.setup_window is not None or self.simulation_process is not None:
            return
        row = self.table.currentRow()
        destination = row + direction
        if row < 0 or not 0 <= destination < self.table.rowCount():
            return
        current = self.table.cellWidget(row, 0)
        neighbor = self.table.cellWidget(destination, 0)
        properties = ("job_id", "config_file", "config_source", "snapshot_workflow")
        current_config = [current.property(key) for key in properties]
        neighbor_config = [neighbor.property(key) for key in properties]
        current_report = self.table.item(row, 3).data(Qt.UserRole)
        neighbor_report = self.table.item(destination, 3).data(Qt.UserRole)
        current_details = [self.table.item(row, column).text() for column in (2, 3, 5)]
        neighbor_details = [self.table.item(destination, column).text() for column in (2, 3, 5)]
        current_tooltip = self.table.item(row, 3).toolTip()
        neighbor_tooltip = self.table.item(destination, 3).toolTip()
        current_actual_tooltip = self.table.item(row, 5).toolTip()
        neighbor_actual_tooltip = self.table.item(destination, 5).toolTip()
        current_index = current.currentIndex()
        self._moving_rows = True
        try:
            current.setCurrentIndex(neighbor.currentIndex())
            neighbor.setCurrentIndex(current_index)
            for key, current_value, neighbor_value in zip(properties, current_config, neighbor_config):
                current.setProperty(key, neighbor_value)
                neighbor.setProperty(key, current_value)
            self.table.cellWidget(row, 1).setEnabled(current.currentData() in self.workflow_names)
            self.table.cellWidget(destination, 1).setEnabled(neighbor.currentData() in self.workflow_names)
            self.update_runtime_estimate(current, self.table.item(row, 4))
            self.update_runtime_estimate(neighbor, self.table.item(destination, 4))
            self.approved_simulation_inputs = None
            self.pending_simulation_inputs = None
        finally:
            self._moving_rows = False
        for column, current_detail, neighbor_detail in zip((2, 3, 5), current_details, neighbor_details):
            self.table.item(row, column).setText(neighbor_detail)
            self.table.item(destination, column).setText(current_detail)
        self.table.item(row, 3).setData(Qt.UserRole, neighbor_report)
        self.table.item(destination, 3).setData(Qt.UserRole, current_report)
        self.table.item(row, 3).setToolTip(neighbor_tooltip)
        self.table.item(destination, 3).setToolTip(current_tooltip)
        self.table.item(row, 5).setToolTip(neighbor_actual_tooltip)
        self.table.item(destination, 5).setToolTip(current_actual_tooltip)
        for position in (row, destination):
            text = self.table.item(position, 2).text()
            state = "success" if text in {"Simulated", "Completed"} else "error" if text in {"Needs attention", "Simulation incomplete", "Run failed"} else None
            self.set_simulation_row_colour(position, state)
        self.table.selectRow(destination)
        self.refresh_vial_layout()


def main():
    os.chdir(REPO_ROOT)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SchedulerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
