"""Dialogs for sample tracking: add a stock, and load or replace a reservoir or cartridge.

The logic lives in experiment_tracking.sample_tracking; these dialogs only collect input.
"""

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from PySide6.QtCore import Qt

from experiment_tracking import sample_tracking

UNITS = ["mM", "M", "µM", "mg/mL", "% w/v", "% v/v"]
NEW_STOCK = "__new__"


class AddStockDialog(QDialog):
    """One stock, with one or more components. The ID and date are filled in automatically."""

    def __init__(self, parent=None, stocks_file=None):
        super().__init__(parent)
        self.stocks_file = stocks_file
        self.stock_id = None
        self.setWindowTitle("Add Stock")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Components (one row each; most stocks have one):"))
        self.resize(520, 380)
        self.table = QTableWidget(1, 3)
        self.table.setHorizontalHeaderLabels(["Material", "Concentration", "Units"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 120)
        self._unit_box(0)
        layout.addWidget(self.table)
        add_row = QPushButton("Add component")
        add_row.clicked.connect(self._add_row)
        layout.addWidget(add_row)
        form = QFormLayout()
        self.solvent = QLineEdit()
        self.lot = QLineEdit()
        self.lot.setPlaceholderText("Supplier and lot number")
        self.notes = QLineEdit()
        form.addRow("Solvent:", self.solvent)
        form.addRow("Lot:", self.lot)
        form.addRow("Notes:", self.notes)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _unit_box(self, row):
        box = QComboBox()
        box.setEditable(True)
        box.addItems(UNITS)
        self.table.setCellWidget(row, 2, box)

    def _add_row(self):
        self.table.insertRow(self.table.rowCount())
        self._unit_box(self.table.rowCount() - 1)

    def components(self):
        rows = []
        for row in range(self.table.rowCount()):
            material = (self.table.item(row, 0).text() if self.table.item(row, 0) else "").strip()
            concentration = (self.table.item(row, 1).text() if self.table.item(row, 1) else "").strip()
            if not material:
                continue
            if concentration:
                float(concentration)  # raises ValueError for a non-number
            rows.append({"material": material, "concentration": concentration,
                         "units": self.table.cellWidget(row, 2).currentText() if concentration else "",
                         "solvent": self.solvent.text().strip(), "lot": self.lot.text().strip(),
                         "notes": self.notes.text().strip()})
        return rows

    def _save(self):
        try:
            components = self.components()
        except ValueError:
            QMessageBox.warning(self, "Add Stock", "Concentration must be a number.")
            return
        if not components and self.solvent.text().strip():
            # A pure solvent (e.g. water, ethanol): the solvent itself, with nothing dissolved
            solvent = self.solvent.text().strip()
            components = [{"material": solvent, "concentration": "", "units": "", "solvent": solvent,
                           "lot": self.lot.text().strip(), "notes": self.notes.text().strip()}]
        if not components:
            QMessageBox.warning(self, "Add Stock", "Enter at least one material, or only a solvent for a pure solvent.")
            return
        try:
            self.stock_id = sample_tracking.add_stock(components, self.stocks_file)
        except Exception as error:
            QMessageBox.critical(self, "Add Stock", f"Could not save the stock: {error}")
            return
        self.accept()


def fill_stock_combo(combo, current=None, stocks_file=None):
    """(none), every stock, then 'Add new...'. Selects ``current`` if given."""
    combo.clear()
    combo.addItem("(none)", "")
    for stock_id, label in sample_tracking.stock_choices(sample_tracking.load_stocks(stocks_file)):
        combo.addItem(label, stock_id)
    combo.addItem("Add new...", NEW_STOCK)
    index = combo.findData(current or "")
    combo.setCurrentIndex(index if index >= 0 else 0)


def handle_add_new(combo, parent, previous, stocks_file=None):
    """If 'Add new...' was picked, open the Add Stock form and select the new stock."""
    if combo.currentData() != NEW_STOCK:
        return
    dialog = AddStockDialog(parent, stocks_file)
    new_id = dialog.stock_id if dialog.exec() == QDialog.Accepted else None
    fill_stock_combo(combo, new_id or previous, stocks_file)


class StocksWidget(QWidget):
    """Scheduler tab: every stock, and what holds it now. Stocks are history: only notes change."""

    COLUMNS = ["Stock ID", "Contents", "Solvent", "Lot", "Made on", "Notes", "Used in now"]

    def __init__(self, parent=None, stocks_file=None, sources_file=None, status_dir=None):
        super().__init__(parent)
        self.stocks_file, self.sources_file, self.status_dir = stocks_file, sources_file, status_dir
        layout = QVBoxLayout(self)
        header = QLabel("Stock solutions and materials made or bought by hand")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add stock…")
        self.add_button.clicked.connect(self.add_stock)
        self.notes_button = QPushButton("Edit notes…")
        self.notes_button.clicked.connect(self.edit_notes)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        for button in (self.add_button, self.notes_button, self.reload_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.reload()

    def reload(self):
        stocks = sample_tracking.load_stocks(self.stocks_file)
        try:
            usage = sample_tracking.stock_usage(self.status_dir, self.sources_file)
        except Exception:
            usage = {}
        grouped = {}
        for row in stocks:
            grouped.setdefault(row["stock_id"], []).append(row)
        self.table.setRowCount(len(grouped))
        for index, (stock_id, rows) in enumerate(grouped.items()):
            contents = " + ".join(" ".join(str(v) for v in (r["material"], r.get("concentration", ""), r.get("units", ""))
                                           if str(v)) for r in rows)
            first = rows[0]
            places = usage.get(stock_id, [])
            values = [stock_id, contents, first.get("solvent", ""), first.get("lot", ""), first.get("made_on", ""),
                      first.get("notes", ""), sample_tracking.usage_summary(places)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value or "")
                if column == len(values) - 1 and places:
                    item.setToolTip("\n".join(places))
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()

    def selected_stock(self):
        row = self.table.currentRow()
        return self.table.item(row, 0).text() if row >= 0 and self.table.item(row, 0) else None

    def add_stock(self):
        dialog = AddStockDialog(self, self.stocks_file)
        if dialog.exec() == QDialog.Accepted:
            self.reload()

    def edit_notes(self):
        stock_id = self.selected_stock()
        if not stock_id:
            QMessageBox.information(self, "Edit notes", "Select a stock first.")
            return
        current = self.table.item(self.table.currentRow(), 5).text()
        notes, ok = QInputDialog.getText(self, f"Notes for {stock_id}", "Notes:", text=current)
        if ok:
            try:
                sample_tracking.update_stock_notes(stock_id, notes, self.stocks_file)
            except Exception as error:
                QMessageBox.critical(self, "Edit notes", f"Could not save: {error}")
            self.reload()


class ReservoirsWidget(QWidget):
    """Scheduler tab: each reservoir pump, what's loaded, and roughly how much is left."""

    COLUMNS = ["Reservoir", "Liquid type", "Stock loaded", "Loaded on", "Loaded", "Left (est.)"]

    def __init__(self, parent=None, pumps_file=None, sources_file=None, stocks_file=None, transfers_file=None):
        super().__init__(parent)
        self.pumps_file, self.sources_file = pumps_file, sources_file
        self.stocks_file, self.transfers_file = stocks_file, transfers_file
        layout = QVBoxLayout(self)
        header = QLabel("Reservoirs: what's loaded and roughly how much is left")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #8a4b00; background: #fff3cd; padding: 6px;")
        layout.addWidget(self.warning)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        note = QLabel("Left = amount loaded minus what the robot has dispensed since. It's an estimate; "
                      "nobody measures it.")
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.load_button = QPushButton("Load/replace…")
        self.load_button.clicked.connect(self.load)
        self.history_button = QPushButton("History…")
        self.history_button.clicked.connect(self.history)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        for button in (self.load_button, self.history_button, self.reload_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.rows = []
        self.reload()

    def reload(self):
        try:
            self.rows = sample_tracking.reservoir_status(self.pumps_file, self.sources_file, self.transfers_file)
        except Exception as error:
            self.rows = []
            self.warning.setText(f"Could not read the reservoirs: {error}")
            self.warning.show()
            return
        labels = dict(sample_tracking.stock_choices(sample_tracking.load_stocks(self.stocks_file)))
        self.table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            load = row["load"]
            stock = (labels.get(load["stock_id"], load["stock_id"]) or "no stock") if load else "Nothing recorded"
            left = "" if row["left_mL"] is None else f"about {max(row['left_mL'], 0):.0f} mL"
            values = [f"Pump {row['index']}", row["liquid"], stock, load["loaded_on"] if load else "",
                      f"{float(load['amount_loaded']):.0f} mL" if load else "", left]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row["low"]:
                    item.setBackground(Qt.yellow)
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        warnings = sample_tracking.reservoir_warnings(self.pumps_file, self.sources_file, self.transfers_file)
        self.warning.setText("\n".join(warnings))
        self.warning.setVisible(bool(warnings))
        if self.rows and self.table.currentRow() < 0:
            self.table.selectRow(0)

    def selected(self):
        row = self.table.currentRow()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def load(self):
        row = self.selected()
        if row is None:
            QMessageBox.information(self, "Load/replace", "Select a reservoir first.")
            return
        dialog = LoadSourceDialog("reservoir", self, self.sources_file, self.stocks_file,
                                  slot=row["index"], default_amount=row["capacity_mL"])
        if dialog.exec() == QDialog.Accepted:
            self.reload()

    def history(self):
        row = self.selected()
        if row is None:
            return
        loads = [r for r in sample_tracking.load_sources(self.sources_file) if r["slot"] == row["slot"]]
        lines = [f"{r['loaded_on']}: {r['stock_id'] or 'no stock'}, {float(r['amount_loaded']):.0f} {r['units']}"
                 for r in reversed(loads)]
        QMessageBox.information(self, f"Pump {row['index']} history", "\n".join(lines) or "Nothing recorded yet.")


class PowdersWidget(QWidget):
    """Scheduler tab: each powder cartridge slot, what's loaded, and roughly how much is left."""

    COLUMNS = ["Cartridge", "Stock loaded", "Loaded on", "Loaded", "Left (est.)"]

    def __init__(self, parent=None, channels_file=None, sources_file=None, stocks_file=None, transfers_file=None):
        super().__init__(parent)
        self.channels_file, self.sources_file = channels_file, sources_file
        self.stocks_file, self.transfers_file = stocks_file, transfers_file
        layout = QVBoxLayout(self)
        header = QLabel("Powder cartridges: what's loaded and roughly how much is left")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        note = QLabel("Slots come from robot_state/powder_channels.yaml. Left = mg loaded minus what the robot "
                      "has dispensed since; an estimate.")
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.load_button = QPushButton("Load/replace…")
        self.load_button.clicked.connect(self.load)
        self.empty_button = QPushButton("Empty…")
        self.empty_button.setToolTip("Record that the cartridge was taken out or used up")
        self.empty_button.clicked.connect(self.empty)
        self.history_button = QPushButton("History…")
        self.history_button.clicked.connect(self.history)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        for button in (self.load_button, self.empty_button, self.history_button, self.reload_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.rows = []
        self.reload()

    def reload(self):
        try:
            self.rows = sample_tracking.cartridge_status(self.channels_file, self.sources_file, self.transfers_file)
        except Exception as error:
            self.rows = []
            QMessageBox.warning(self, "Powders", f"Could not read the cartridge slots: {error}")
        labels = dict(sample_tracking.stock_choices(sample_tracking.load_stocks(self.stocks_file)))
        self.table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            load = row["load"]
            if load:
                values = [f"Channel {row['channel']}", labels.get(load["stock_id"], load["stock_id"]), load["loaded_on"],
                          f"{float(load['amount_loaded']):.4g} mg", f"about {max(row['left_mg'], 0):.4g} mg"]
            else:
                emptied = f"Empty (since {row['emptied_on']})" if row["emptied_on"] else "Nothing loaded"
                values = [f"Channel {row['channel']}", emptied, "", "", ""]
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        if self.rows and self.table.currentRow() < 0:
            self.table.selectRow(0)

    def selected(self):
        row = self.table.currentRow()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def load(self):
        row = self.selected()
        if row is None:
            QMessageBox.information(self, "Load/replace", "Select a cartridge first.")
            return
        dialog = LoadSourceDialog("cartridge", self, self.sources_file, self.stocks_file, slot=row["channel"])
        if dialog.exec() == QDialog.Accepted:
            self.reload()

    def empty(self):
        row = self.selected()
        if row is None or row["load"] is None:
            return
        answer = QMessageBox.question(self, "Empty cartridge",
                                      f"Record that channel {row['channel']} was taken out or used up?")
        if answer == QMessageBox.Yes:
            sample_tracking.record_empty(row["slot"], self.sources_file)
            self.reload()

    def history(self):
        row = self.selected()
        if row is None:
            return
        loads = [r for r in sample_tracking.load_sources(self.sources_file) if r["slot"] == row["slot"]]
        lines = [f"{r['loaded_on']}: " + (f"{r['stock_id']}, {float(r['amount_loaded']):.4g} {r['units']}"
                                          if r["stock_id"] else "emptied") for r in reversed(loads)]
        QMessageBox.information(self, f"Channel {row['channel']} history", "\n".join(lines) or "Nothing recorded yet.")


class LiquidDialog(QDialog):
    """Add a liquid, or edit one's density and tip settings (robot_state/liquids.yaml)."""

    def __init__(self, parent=None, row=None, registry=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox
        self.registry, self.new = registry, row is None
        self.setWindowTitle("Add liquid" if self.new else f"Edit {row['liquid']}")
        form = QFormLayout(self)
        self.name = QLineEdit(row["liquid"] if row else "")
        self.name.setEnabled(self.new)
        self.name.setPlaceholderText("Exactly as workflows name it, e.g. 6M_HCl")
        form.addRow("Liquid:", self.name)
        self.density = QDoubleSpinBox()
        self.density.setRange(0.001, 30.0)
        self.density.setDecimals(3)
        self.density.setSuffix(" g/mL")
        self.density.setValue(row["density"] if row else 1.0)
        form.addRow("Density:", self.density)
        self.reuse = QCheckBox("Reuse the tip between calibration measurements")
        self.reuse.setChecked(row["reuse_tip"] if row else True)
        self.reuse.setToolTip("Unticked = a fresh tip after every calibration measurement (refill_pipets: true)")
        form.addRow("Reuse tip:", self.reuse)
        self.condition = QCheckBox("Condition a fresh tip before pipetting from it")
        self.condition.setChecked(row["condition_tip"] if row else False)
        form.addRow("Condition tip:", self.condition)
        if self.new:
            form.addRow(QLabel("A new liquid stays 'not calibrated' until a calibration is run and merged."))
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _save(self):
        from pipetting_data.liquid_overview import save_liquid
        try:
            kwargs = {"registry": self.registry} if self.registry else {}
            save_liquid(self.name.text(), self.density.value(), self.reuse.isChecked(), self.condition.isChecked(),
                        new=self.new, **kwargs)
        except Exception as error:
            QMessageBox.warning(self, self.windowTitle(), str(error))
            return
        self.accept()


class LiquidsWidget(QWidget):
    """Scheduler tab: each liquid's settings, calibrated volumes and last calibration/validation."""

    COLUMNS = ["Liquid", "Used (90 days)", "Density (g/mL)", "Reuse tip", "Condition tip", "Calibrated: small tip",
               "Calibrated: large tip", "Last calibrated", "Last validated"]

    def __init__(self, parent=None, registry=None, directory=None, sessions_file=None):
        super().__init__(parent)
        self.registry, self.directory, self.sessions_file = registry, directory, sessions_file
        layout = QVBoxLayout(self)
        header = QLabel("Liquids: settings, calibrations and validations")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(header)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda _: self.edit())
        layout.addWidget(self.table)
        note = QLabel("Most used first: dispenses in the transfer log plus validation sessions, last 90 days "
                      "(approximate). Settings are in robot_state/liquids.yaml; changing them means queued runs need "
                      "simulating again. Calibrated volumes come from optimal_conditions_<liquid>.csv. 'Unknown' "
                      "calibration date: calibrated before dates were recorded.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add liquid…")
        self.add_button.clicked.connect(self.add)
        self.edit_button = QPushButton("Edit…")
        self.edit_button.clicked.connect(self.edit)
        self.delete_button = QPushButton("Delete…")
        self.delete_button.setToolTip("Remove the liquid from liquids.yaml; its calibration and history are kept")
        self.delete_button.clicked.connect(self.delete)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        for button in (self.add_button, self.edit_button, self.delete_button, self.reload_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.rows = []
        self.reload()

    def delete_message(self, name, usage):
        """The text of the Delete window."""
        shown = usage["files"][:10]
        lines = [f"Delete {name}?", ""]
        if usage["files"]:
            lines.append("Still used by:")
            lines += [f"  • {path}" for path in shown]
            if len(usage["files"]) > len(shown):
                lines.append(f"  • and {len(usage['files']) - len(shown)} more")
        else:
            lines.append("No workflow or saved config names it.")
        lines.append("Calibration file: " + (", ".join(usage["calibration_files"]) or "none"))
        lines.append(f"Validation history: {usage['validation_sessions'] or 'none'}"
                     + (" sessions" if usage["validation_sessions"] else ""))
        lines.append("")
        if usage["files"]:
            lines.append('These would stop with "Unknown liquid" when you next simulate them.')
        lines.append("Its calibration file and history are kept.")
        return "\n".join(lines)

    def delete(self):
        from pipetting_data.liquid_overview import delete_liquid, liquid_usage
        row = self.table.currentRow()
        if not 0 <= row < len(self.rows):
            QMessageBox.information(self, "Delete", "Select a liquid first.")
            return
        name = self.rows[row]["liquid"]
        message = self.delete_message(name, liquid_usage(name))
        if QMessageBox.question(self, f"Delete {name}", message, QMessageBox.Yes | QMessageBox.Cancel,
                                QMessageBox.Cancel) != QMessageBox.Yes:
            return
        try:
            delete_liquid(name, **({"registry": self.registry} if self.registry else {}))
        except Exception as error:
            QMessageBox.warning(self, "Delete", f"Could not delete {name}: {error}")
        self.reload()

    def _kwargs(self):
        return {k: v for k, v in (("registry", self.registry), ("directory", self.directory),
                                  ("sessions_file", self.sessions_file)) if v is not None}

    def reload(self):
        from pipetting_data.liquid_overview import liquids_table
        try:
            self.rows = liquids_table(**self._kwargs())
        except Exception as error:
            self.rows = []
            QMessageBox.warning(self, "Liquids", f"Could not read the liquids: {error}")
        yes_no = lambda value: "yes" if value else "no"
        self.table.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            values = [row["liquid"], str(row["recent_use"]), f"{row['density']:g}", yes_no(row["reuse_tip"]),
                      yes_no(row["condition_tip"]),
                      row["small_tip"], row["large_tip"], row["last_calibrated"] or "unknown",
                      row["last_validated"] or "never"]
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def add(self):
        if LiquidDialog(self, registry=self.registry).exec() == QDialog.Accepted:
            self.reload()

    def edit(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.rows):
            QMessageBox.information(self, "Edit", "Select a liquid first.")
            return
        if LiquidDialog(self, self.rows[row], registry=self.registry).exec() == QDialog.Accepted:
            self.reload()


class LoadSourceDialog(QDialog):
    """Load or replace a reservoir ('reservoir') or powder cartridge ('cartridge')."""

    def __init__(self, kind, parent=None, sources_file=None, stocks_file=None, slot=None, default_amount=None):
        super().__init__(parent)
        self.kind, self.sources_file, self.stocks_file = kind, sources_file, stocks_file
        self.default_amount = default_amount
        self.container_id = None
        noun = "Reservoir" if kind == "reservoir" else "Cartridge"
        self.setWindowTitle(f"Load/Replace {noun}")
        form = QFormLayout(self)
        self.slot = QSpinBox()
        self.slot.setRange(0, 20)
        if slot is not None:
            self.slot.setValue(int(slot))
            self.slot.setEnabled(False)  # opened for one reservoir or cartridge
        elif kind == "reservoir":
            self.slot.setValue(1)  # pump 0 is the pipetting pump
        form.addRow("Pump index:" if kind == "reservoir" else "Powder channel:", self.slot)
        self.stock = QComboBox()
        self.stock.activated.connect(lambda _: handle_add_new(self.stock, self, self._previous_stock(),
                                                              self.stocks_file))
        form.addRow("Stock loaded:", self.stock)
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.0, 100000.0)
        self.amount.setDecimals(3)
        self.amount.setSuffix(" mL" if kind == "reservoir" else " mg")
        form.addRow("Amount loaded:", self.amount)
        self.status = QLabel()
        form.addRow("Now:", self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.slot.valueChanged.connect(self._show_current)
        self._show_current()

    def slot_name(self):
        return (sample_tracking.reservoir_slot if self.kind == "reservoir"
                else sample_tracking.cartridge_slot)(self.slot.value())

    def _previous_stock(self):
        load = sample_tracking.current_load(self.slot_name(), self.sources_file)
        return load["stock_id"] if load else ""

    def _show_current(self):
        load = sample_tracking.current_load(self.slot_name(), self.sources_file)
        fill_stock_combo(self.stock, load["stock_id"] if load else "", self.stocks_file)
        if self.default_amount:
            self.amount.setValue(float(self.default_amount))  # a refill fills it up
        elif load:
            self.amount.setValue(float(load["amount_loaded"]))
        if load:
            left = sample_tracking.amount_left(self.slot_name(), self.sources_file)
            self.status.setText(f"{load['stock_id'] or 'no stock'}, loaded {load['loaded_on']}; "
                                f"about {left[0]:.4g} {left[1]} left (estimated)")
        else:
            self.status.setText("Nothing recorded yet")

    def _save(self):
        if self.amount.value() <= 0:
            QMessageBox.warning(self, self.windowTitle(), "Enter the amount loaded.")
            return
        try:
            self.container_id = sample_tracking.record_load(
                self.slot_name(), self.stock.currentData() or "", self.amount.value(),
                "mL" if self.kind == "reservoir" else "mg", self.sources_file)
        except Exception as error:
            QMessageBox.critical(self, self.windowTitle(), f"Could not save: {error}")
            return
        self.accept()
