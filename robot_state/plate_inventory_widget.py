"""Position-based editor of recorded track state. Never commands hardware."""

from pathlib import Path

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QPushButton, QMessageBox, QInputDialog, QDialog, QFormLayout, QComboBox,
    QSpinBox, QCheckBox, QDialogButtonBox, QHeaderView, QScrollArea, QSplitter,
    QMenu, QToolButton, QFrame, QGridLayout,
)

from robot_state.plate_inventory import PlateInventory
from robot_state.plate_layout_view import PlateLayoutView, position_label

_ROOT = Path(__file__).resolve().parents[1]


class PlateInventoryWidget(QWidget):
    state_saved = Signal()
    draft_changed = Signal()

    def __init__(self, file_path=None, show_load_errors=True):
        super().__init__()
        self.track_file_path = str(file_path or _ROOT / "robot_state/track_status.yaml")
        self.show_load_errors = show_load_errors
        self.state_loaded = False
        self.inventory = None
        self._setup_ui()
        self._load_track_status()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel("Plate positions")
        heading.setStyleSheet("font-size: 17px; font-weight: 600; color: #243247;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Select or drag plates to update the recorded setup. Editing does not move hardware."))
        splitter = QSplitter(Qt.Horizontal)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(0, 0, 0, 0)
        self.diagram = PlateLayoutView()
        self.diagram.position_activated.connect(self.select_position)
        self.diagram.item_dropped.connect(self._drop_item)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.diagram)
        overview_layout.addWidget(scroll)
        note = QLabel("Drag plates between holders or stack layers. A plate carries its lid; drag the lid’s top edge to move it separately.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #64748b; padding: 4px;")
        overview_layout.addWidget(note)
        splitter.addWidget(overview)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(10, 0, 0, 0)
        self.position_selector = QComboBox()
        self.position_selector.setAccessibleName("Position to edit")
        self.position_selector.currentIndexChanged.connect(lambda *_: self._show_contents())
        details_layout.addWidget(self.position_selector)
        self.contents_summary = QLabel()
        self.contents_summary.setWordWrap(True)
        details_layout.addWidget(self.contents_summary)
        self.contents_tree = QTreeWidget()
        self.contents_tree.setHeaderLabels(["Plate / lid", "Type", "Lid"])
        self.contents_tree.setColumnWidth(0, 170)
        self.contents_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.contents_tree.header().setStretchLastSection(True)
        self.contents_tree.currentItemChanged.connect(lambda *_: self._sync_selection())
        self.contents_tree.itemDoubleClicked.connect(lambda *_: self._guard(self._edit_dialog))
        details_layout.addWidget(self.contents_tree)
        actions = QGridLayout()
        self.add_button = QPushButton("Add…")
        self.add_button.clicked.connect(lambda: self._guard(self._add_dialog))
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(lambda: self._guard(self._edit_dialog))
        self.remove_button = QPushButton("Remove")
        self.move_button = QPushButton("Move to…")
        self.higher_button = QPushButton("Move higher")
        self.lower_button = QPushButton("Move lower")
        for button, action in ((self.remove_button, self._remove_selected),
                               (self.move_button, self._relocate_dialog),
                               (self.higher_button, lambda: self._reorder(1)),
                               (self.lower_button, lambda: self._reorder(-1))):
            button.clicked.connect(lambda checked=False, fn=action: self._guard(fn))
        menu = QMenu(self)
        for label, action in (("Move to another position…", self._relocate_dialog),
                              ("Move higher in stack", lambda: self._reorder(1)),
                              ("Move lower in stack", lambda: self._reorder(-1)),
                              ("Remove from setup", self._remove_selected)):
            menu.addAction(label, lambda checked=False, fn=action: self._guard(fn))
        for index, button in enumerate((self.add_button, self.edit_button, self.remove_button,
                                        self.move_button, self.higher_button, self.lower_button)):
            actions.addWidget(button, index // 3, index % 3)
        details_layout.addLayout(actions)
        self.contents_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.contents_tree.customContextMenuRequested.connect(lambda point: menu.exec(self.contents_tree.viewport().mapToGlobal(point)))
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 450])
        layout.addWidget(splitter, 1)
        self.recovery_panel = QFrame()
        self.recovery_panel.setStyleSheet("QFrame { background: #fff7e6; border-radius: 6px; }")
        recovery_layout = QHBoxLayout(self.recovery_panel)
        self.recovery_label = QLabel()
        self.recovery_label.setWordWrap(True)
        recovery_layout.addWidget(self.recovery_label, 1)
        self.confirm_positions = QPushButton("Positions are correct")
        self.confirm_positions.clicked.connect(lambda: self._guard(self._confirm_positions))
        recovery_layout.addWidget(self.confirm_positions)
        self.recovery_panel.hide()
        layout.addWidget(self.recovery_panel)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        self.diagram.drop_feedback.connect(self.feedback.setText)
        layout.addWidget(self.feedback)
        coordinates = QToolButton()
        coordinates.setText("Track coordinates")
        coordinates.setCheckable(True)
        layout.addWidget(coordinates)
        self.pose_label = QLabel()
        self.pose_label.hide()
        coordinates.toggled.connect(self.pose_label.setVisible)
        layout.addWidget(self.pose_label)

    def _confirm_positions(self):
        self.inventory.finish()

    def _drop_item(self, object_id, destination, index=None):
        before = self.state_snapshot()
        try:
            self.inventory.relocate(object_id, destination, index)
            pending = self.inventory.data.get("pending_transfer")
            if pending and pending["object_id"] == object_id:
                self.inventory.finish()
        except (ValueError, KeyError, TypeError) as error:
            self.feedback.setText(str(error))
            return
        if self.state_snapshot() != before:
            self._populate_form()
            self.select_position(destination, object_id)
            self.draft_changed.emit()
            self.feedback.setText("Position updated in the draft. Save to keep it; no hardware moved.")

    @property
    def track_data(self):
        return self.inventory.data if self.inventory else {}

    def _guard(self, action):
        try:
            action()
            self.feedback.setText("Draft changed; use the main window's Save to persist.")
            self._populate_form()
            self.draft_changed.emit()
        except (ValueError, KeyError, TypeError) as error:
            self.feedback.setText(str(error))

    def _load_track_status(self):
        self.state_loaded = False
        try:
            definitions = yaml.safe_load((_ROOT / "robot_state/wellplates.yaml").read_text())
            positions = yaml.safe_load((_ROOT / "robot_state/track_positions.yaml").read_text())
            self.inventory = PlateInventory.load(self.track_file_path, definitions, positions)
            self.state_loaded = True
            self._populate_form()
            if self.inventory.legacy:
                self.feedback.setText("Review the plate types, positions and lid states before saving this setup.")
            return True
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            self.feedback.setText(str(error))
            if self.show_load_errors:
                QMessageBox.warning(self, "Track state could not be loaded", str(error))
            return False

    def selected_position(self):
        return self.position_selector.currentData() or "source_stack"

    def select_position(self, location, object_id=None):
        index = self.position_selector.findData(location)
        if index < 0:
            return
        self.position_selector.setCurrentIndex(index)
        if object_id:
            self._select_object(object_id)

    def _select_object(self, object_id):
        for index in range(self.contents_tree.topLevelItemCount()):
            row = self.contents_tree.topLevelItem(index)
            candidates = [row] + [row.child(i) for i in range(row.childCount())]
            for item in candidates:
                if item.data(0, Qt.UserRole) == object_id:
                    self.contents_tree.setCurrentItem(item)
                    return

    def state_snapshot(self):
        return yaml.safe_dump(self.track_data, sort_keys=True)

    def selected_object(self):
        item = self.contents_tree.currentItem()
        if not item:
            raise ValueError("Select a plate or lid first")
        return item.data(0, Qt.UserRole)

    def _populate_form(self):
        if not self.inventory:
            return
        selected = self.selected_position()
        old_item = self.contents_tree.currentItem()
        old_id = old_item.data(0, Qt.UserRole) if old_item else None
        self.position_selector.blockSignals(True)
        self.position_selector.clear()
        for location in [*self.inventory.holders(), "gripper"]:
            self.position_selector.addItem(position_label(location), location)
        self.position_selector.setCurrentIndex(max(0, self.position_selector.findData(selected)))
        self.position_selector.blockSignals(False)
        pending = self.inventory.data.get("pending_transfer")
        self.recovery_panel.setVisible(bool(pending))
        if pending:
            obj = pending["object_id"]
            self.recovery_label.setText(f"A transfer involving {obj} stopped. Check the physical setup and update the positions above. Then choose ‘Positions are correct’. Save to keep the correction.")
        pose = self.inventory.data.get("track_pose", {})
        self.pose_label.setText(f"{pose.get('location')} · X {pose.get('x')}, Z {pose.get('z')}")
        self._show_contents()
        if old_id:
            self._select_object(old_id)

    def _description(self, object_id):
        obj = self.inventory.objects[object_id]
        text = f"{obj['kind']} {object_id} ({self.inventory.plate_type(object_id)})"
        return text + (" with lid" if object_id in self.inventory.attachments else "")

    def _show_contents(self):
        self.contents_tree.clear()
        if not self.inventory:
            return
        contents = self.inventory.contents(self.selected_position())
        for index, object_id in enumerate(contents):
            obj = self.inventory.objects[object_id]
            lid = self.inventory.attachments.get(object_id)
            label = (f"Plate {index + 1}" if obj["kind"] == "plate" else "Lid")
            if len(contents) > 1 and index == len(contents)-1:
                label += " (next pickup)"
            row = QTreeWidgetItem([label, self.inventory.plate_type(object_id), "On" if lid else ("Off" if obj['kind'] == 'plate' else "Lid only")])
            row.setData(0, Qt.UserRole, object_id)
            row.setToolTip(0, object_id)
            self.contents_tree.addTopLevelItem(row)
            if lid:
                child = QTreeWidgetItem(["Lid", self.inventory.plate_type(lid), "Attached"])
                child.setData(0, Qt.UserRole, lid)
                child.setToolTip(0, lid)
                row.addChild(child)
                row.setExpanded(True)
        if contents:
            self.contents_tree.setCurrentItem(self.contents_tree.topLevelItem(len(contents)-1))
        location = self.selected_position()
        self.contents_summary.setText("Bottom to top; the last row is picked first." if len(contents) > 1 else ("Select a plate or lid to edit it." if contents else "This position is empty. Use Add to record an item here."))
        self.diagram.set_inventory(self.inventory, location)
        self._sync_selection()

    def _sync_selection(self):
        item = self.contents_tree.currentItem()
        self.edit_button.setEnabled(item is not None)
        self.remove_button.setEnabled(item is not None)
        self.move_button.setEnabled(item is not None)
        contents = self.inventory.contents(self.selected_position()) if self.inventory else []
        object_id = item.data(0, Qt.UserRole) if item else None
        index = contents.index(object_id) if object_id in contents else -1
        stack = self.inventory is not None and self.inventory.holders().get(self.selected_position()) == "plate"
        self.higher_button.setEnabled(stack and 0 <= index < len(contents) - 1)
        self.lower_button.setEnabled(stack and index > 0)
        if self.inventory:
            self.diagram.set_inventory(self.inventory, self.selected_position(), item.data(0, Qt.UserRole) if item else None)

    def _select_gripper(self):
        self.select_position("gripper")

    def add_item(self, location, plate_type, kind="plate", with_lid=False, quantity=1):
        with self.inventory.edit():
            created = [self.inventory.add(location, plate_type, kind, with_lid) for _ in range(quantity)]
        self.draft_changed.emit()
        return created

    def _object_dialog(self, title, object_id=None):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        form = QFormLayout(dialog)
        kind = QComboBox()
        kind.addItems(["plate", "lid"])
        types = QComboBox()
        types.addItems([key for key, value in self.inventory.definitions.items() if "num_wells" in value])
        count = QSpinBox()
        count.setRange(1, 1000)
        lid = QCheckBox("Lid attached")
        replace = QCheckBox("This is a replacement physical object (new ID)")
        if object_id:
            obj = self.inventory.objects[object_id]
            kind.setCurrentText(obj["kind"])
            kind.setEnabled(False)
            types.setCurrentText(self.inventory.plate_type(object_id))
            lid.setChecked(object_id in self.inventory.attachments)
        kind.currentTextChanged.connect(lambda value: lid.setEnabled(value == "plate"))
        lid.setEnabled(kind.currentText() == "plate")
        form.addRow("Item", kind)
        form.addRow("Plate definition", types)
        if not object_id:
            form.addRow("Quantity", count)
        form.addRow(lid)
        if object_id:
            form.addRow(replace)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return kind.currentText(), types.currentText(), count.value(), lid.isChecked(), replace.isChecked()

    def _add_dialog(self):
        result = self._object_dialog("Record an item at " + self.selected_position())
        if result:
            kind, plate_type, count, lid, _ = result
            self.add_item(self.selected_position(), plate_type, kind, lid and kind == "plate", count)

    def _remove_selected(self):
        self.inventory.remove(self.selected_object())

    def _edit_dialog(self):
        object_id = self.selected_object()
        result = self._object_dialog("Correct item description", object_id)
        if not result:
            return
        kind, plate_type, _, lid_on, replace = result
        with self.inventory.edit():
            if replace:
                location = self.inventory.locate(object_id)
                if location.startswith("attached:"):
                    parent = location.split(":", 1)[1]
                    self.inventory.remove(object_id)
                    self.inventory.attachments[parent] = self.inventory.create(kind, plate_type)
                else:
                    contents = self.inventory.contents(location)
                    layer = contents.index(object_id)
                    self.inventory.remove(object_id)
                    new_id = self.inventory.add(location, plate_type, kind, lid_on and kind == "plate")
                    if isinstance(self.inventory.positions[location], list):
                        self.inventory.positions[location].remove(new_id)
                        self.inventory.positions[location].insert(layer, new_id)
            else:
                self.inventory.objects[object_id]["type" if kind == "plate" else "plate_type"] = plate_type
                if kind == "plate":
                    old_lid = self.inventory.attachments.get(object_id)
                    if lid_on and not old_lid:
                        self.inventory.attachments[object_id] = self.inventory.create("lid", plate_type)
                    elif not lid_on and old_lid:
                        # This explicitly records a lid removed from the setup.
                        # To park/reuse that physical lid, select its child row and relocate it.
                        self.inventory.remove(old_lid)

    def _reorder(self, direction):
        object_id = self.selected_object()
        contents = self.inventory.positions[self.selected_position()]
        if not isinstance(contents, list) or object_id not in contents:
            raise ValueError("Select a plate in an ordered position")
        index = contents.index(object_id)
        other = index + direction
        if 0 <= other < len(contents):
            with self.inventory.edit():
                contents[index], contents[other] = contents[other], contents[index]

    def _relocate_dialog(self):
        object_id = self.selected_object()
        options = list(self.inventory.holders()) + ["gripper"]
        location, ok = QInputDialog.getItem(self, "Record physical relocation", "Destination (no hardware movement)", options, 0, False)
        if ok:
            self.inventory.relocate(object_id, location)
            pending = self.inventory.data.get("pending_transfer")
            if pending and pending["object_id"] == object_id:
                self.inventory.finish()

    def _save_track_status(self, silent=False):
        if not self.state_loaded:
            return False
        try:
            self.inventory.save(self.track_file_path)
            self.state_saved.emit()
            self.feedback.setText("Track positions saved. No hardware was moved.")
            return True
        except (OSError, ValueError, KeyError) as error:
            self.feedback.setText(str(error))
            if not silent:
                QMessageBox.warning(self, "Track state not saved", str(error))
            return False
