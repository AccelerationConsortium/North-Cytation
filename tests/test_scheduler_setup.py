import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDrag, QDragEnterEvent, QDragMoveEvent, QDropEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

import scheduler_gui
from vial_manager_gui import VIAL_MOVE_MIME, VialManagerMainWindow


class SchedulerSetupTests(unittest.TestCase):
    def test_dialog_position_edit_relocates_grid(self):
        editor = self.editor()
        rack = editor.rack_widgets["main_8mL_rack"]
        source = next(iter(rack.vials))
        destination = next(index for index in range(48) if index not in rack.vials)
        original = rack.vials[source].get_vial_data()

        def accept_move(dialog):
            dialog.location_index_spin.setValue(destination)
            return QDialog.Accepted

        with patch("vial_manager_gui.VialEditDialog.exec", accept_move):
            rack._on_vial_clicked(original)
        self.assertIn(destination, rack.vials)
        self.assertNotIn(source, rack.vials)
        self.assertEqual(rack.vials[destination].vial_data["vial_index"], original["vial_index"])

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "status").mkdir()
        (self.root / "workflow_configs").mkdir()
        self.vials = self.root / "status" / "surfactant_grid_ailsa_vials.csv"
        self.config = self.root / "workflow_configs" / "surfactant_grid_ailsa.yaml"
        shutil.copyfile(REPO_ROOT / "status" / self.vials.name, self.vials)
        shutil.copyfile(REPO_ROOT / "workflow_configs" / self.config.name, self.config)
        self.shared_status = {
            path: path.read_bytes()
            for path in (REPO_ROOT / "robot_state").glob("*.yaml")
        }
        self.track_file = self.root / "track.yaml"
        self.robot_file = self.root / "robot.yaml"
        shutil.copyfile(REPO_ROOT / "robot_state" / "track_status.yaml", self.track_file)
        shutil.copyfile(REPO_ROOT / "robot_state" / "robot_status.yaml", self.robot_file)

    def redirect_status(self, editor):
        if editor.track_status_widget is not None:
            editor.track_status_widget.track_file_path = str(self.track_file)
        if editor.robot_status_widget is not None:
            editor.robot_status_widget.robot_file_path = str(self.robot_file)

    def row_window(self):
        (self.root / "workflows").mkdir()
        (self.root / "workflows" / "surfactant_grid_ailsa.py").touch()
        shutil.copytree(REPO_ROOT / "robot_state", self.root / "robot_state")
        root_patch = patch.object(scheduler_gui, "REPO_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        window = scheduler_gui.SchedulerWindow()
        self.addCleanup(window.deleteLater)
        selector = window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData("surfactant_grid_ailsa"))
        return window, selector

    def open_row(self, window, selector):
        row = next(row for row in range(window.table.rowCount()) if window.table.cellWidget(row, 0) is selector)
        window.table.cellWidget(row, 1).click()
        editor = window.setup_window
        self.assertIsNotNone(editor)
        self.redirect_status(editor)
        return editor

    def editor(self):
        editor = VialManagerMainWindow(preparation_mode=True)
        editor.setup_preparation(self.vials, "surfactant_grid_ailsa", self.config)
        self.redirect_status(editor)
        self.addCleanup(editor.deleteLater)
        return editor

    def test_normal_workflow_run_and_abort_are_preserved(self):
        normal = VialManagerMainWindow()
        self.addCleanup(normal.deleteLater)
        coordinator = SimpleNamespace(logger=logging.getLogger("scheduler_setup_test"))
        with patch.object(normal, "load_status_file", return_value=True), patch.object(
            normal, "_populate_racks"
        ):
            normal._setup_workflow_mode(str(self.vials), coordinator)
        self.assertFalse(normal._preparation_mode)
        self.assertFalse(normal.run_workflow_button.isHidden())
        self.assertFalse(normal.abort_workflow_button.isHidden())
        with patch.object(normal, "_has_unsaved_changes", return_value=False):
            normal._run_workflow()
        self.assertTrue(normal._workflow_continue)
        with self.assertRaises(SystemExit) as caught:
            normal._abort_workflow()
        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(normal._workflow_continue)

    def test_preparation_buttons_menus_and_cli_isolation(self):
        with patch.object(sys, "argv", ["scheduler_gui.py", "not-a-vial-file"]), patch.object(
            QMessageBox, "critical"
        ) as error_dialog:
            editor = VialManagerMainWindow(preparation_mode=True)
            self.addCleanup(editor.deleteLater)
            error_dialog.assert_not_called()
            self.assertIsNone(editor.status_file_path)
        editor = self.editor()
        self.assertTrue(editor.run_workflow_button.isHidden())
        self.assertTrue(editor.abort_workflow_button.isHidden())
        self.assertEqual(editor.save_return_button.text(), "Save and Return")
        self.assertIsNone(editor._lash_e_instance)
        menu_action = editor.menuBar().actions()[0]
        menu = menu_action.menu()
        self.assertFalse(any("Workflow" in action.text() for action in menu.actions()))
        self.assertTrue(editor.track_status_widget.isEnabled())
        self.assertTrue(editor.robot_status_widget.isEnabled())

    def test_save_and_return_writes_only_temporary_vials_and_config(self):
        editor = self.editor()
        results = []
        editor.preparation_finished.connect(results.append)
        editor.original_vials_data[0]["vial_volume"] = "7.25"
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        editor.track_status_widget.num_source_spin.setValue(3)
        editor.track_status_widget.track_data["current_gripper_location"] = "home"
        editor._save_and_return()
        self.assertEqual(results, [True])
        with self.vials.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(next(csv.DictReader(stream))["vial_volume"], "7.25")
        self.assertEqual(yaml.safe_load(self.config.read_text())["MAX_WELLS"], 48)
        self.assertEqual(yaml.safe_load(self.track_file.read_text())["num_in_source"], 3)
        self.assertEqual(yaml.safe_load(self.track_file.read_text())["current_gripper_location"], "home")
        for path, content in self.shared_status.items():
            self.assertEqual(path.read_bytes(), content)

    def test_save_and_return_persists_moved_current_and_home_positions(self):
        editor = self.editor()
        vial = editor.original_vials_data[0]
        vial_index = str(vial["vial_index"])
        before = self.vials.read_bytes()
        results = []
        editor.preparation_finished.connect(results.append)
        self.assertTrue(editor._move_vial(vial_index, "heater", 11))
        self.assertEqual(self.vials.read_bytes(), before)
        editor._save_and_return()
        self.assertEqual(results, [True])
        with self.vials.open(newline="", encoding="utf-8") as stream:
            saved = next(row for row in csv.DictReader(stream) if row["vial_index"] == vial_index)
        self.assertEqual(saved["location"], "heater")
        self.assertEqual(saved["location_index"], "11")
        self.assertEqual(saved["home_location"], "heater")
        self.assertEqual(saved["home_location_index"], "11")
        for path, content in self.shared_status.items():
            self.assertEqual(path.read_bytes(), content)

    def test_close_cancel_and_discard_do_not_save(self):
        editor = self.editor()
        original = self.vials.read_bytes()
        editor.original_vials_data[0]["vial_volume"] = "7.25"
        results = []
        editor.preparation_finished.connect(results.append)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel):
            self.assertFalse(editor.close())
        self.assertEqual(results, [])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            self.assertTrue(editor.close())
        self.assertEqual(results, [False])
        self.assertEqual(self.vials.read_bytes(), original)

    def test_scheduler_saved_move_refreshes_layout_and_preserves_shared_state(self):
        window, selector = self.row_window()
        editor = self.open_row(window, selector)
        vial = editor.original_vials_data[0]
        vial_index = str(vial["vial_index"])
        vial_name = vial["vial_name"]
        before = self.vials.read_bytes()
        self.assertTrue(editor._move_vial(vial_index, "heater", 11))
        self.assertEqual(self.vials.read_bytes(), before)
        editor._save_and_return()
        self.assertIsNone(window.setup_window)
        slot = window.vial_layout.slots["heater", 11]
        self.assertEqual(slot.property("occupancy"), "occupied")
        self.assertIn(vial_name, slot.toolTip())
        with self.vials.open(newline="", encoding="utf-8") as stream:
            saved = next(row for row in csv.DictReader(stream) if row["vial_index"] == vial_index)
        self.assertEqual((saved["location"], saved["home_location"]), ("heater", "heater"))
        self.assertEqual((saved["location_index"], saved["home_location_index"]), ("11", "11"))
        for path, content in self.shared_status.items():
            self.assertEqual(path.read_bytes(), content)

    def test_failed_save_does_not_return_success(self):
        editor = self.editor()
        results = []
        editor.preparation_finished.connect(results.append)
        editor.original_vials_data.append(editor.original_vials_data[0].copy())
        with patch.object(QMessageBox, "critical"):
            editor._save_and_return()
        self.assertFalse(editor._preparation_saved)
        self.assertEqual(results, [])
        editor.original_vials_data.pop()
        with patch.object(editor.config_editor, "_save_config", return_value=False):
            editor._save_and_return()
        self.assertFalse(editor._preparation_saved)
        self.assertEqual(results, [])

    def test_scheduler_setup_and_saved_status(self):
        workflow_modules_before = {name for name in sys.modules if name.startswith("workflows.")}
        window = scheduler_gui.SchedulerWindow()
        self.addCleanup(window.deleteLater)
        selector = window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData("surfactant_grid_ailsa"))
        with patch.object(scheduler_gui, "REPO_ROOT", self.root):
            window.set_row_config(selector, self.config)
            window.table.cellWidget(0, 1).click()
            self.assertIsNotNone(window.setup_window)
            self.redirect_status(window.setup_window)
            window.setup_window.original_vials_data[0]["location"] = "clamp"
            window.setup_window.original_vials_data[0]["location_index"] = "0"
            window.setup_window._save_and_return()
            self.assertEqual(window.vial_layout.slots["clamp", 0].property("occupancy"), "occupied")
        self.assertIsNone(window.setup_window)

        self.assertEqual(window.table.item(0, 2).text(), "Setup saved")
        window.add_row()
        window.table.selectRow(0)
        window.move_row(1)
        self.assertEqual(window.table.item(1, 2).text(), "Setup saved")
        buttons = window.findChildren(QPushButton)
        self.assertTrue(all(not button.isEnabled() for button in buttons if button.text() == "Run"))
        self.assertNotIn("master_usdl_coordinator", sys.modules)
        self.assertEqual({name for name in sys.modules if name.startswith("workflows.")}, workflow_modules_before)

    def test_shared_only_edit_prompts_and_failed_save_does_not_close(self):
        editor = self.editor()
        editor.track_status_widget.num_source_spin.setValue(
            (editor.track_status_widget.num_source_spin.value() + 1) % 100
        )
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel) as prompt:
            self.assertFalse(editor.close())
        prompt.assert_called_once()
        with patch.object(editor.track_status_widget, "_save_track_status", return_value=False):
            editor._save_and_return()
        self.assertFalse(editor._preparation_saved)

    def test_no_vial_setup_saves_config_and_shared_state_without_csv(self):
        config = yaml.safe_load(self.config.read_text())
        config["INPUT_VIAL_STATUS_FILE"] = None
        self.config.write_text(yaml.safe_dump(config))
        editor = VialManagerMainWindow(preparation_mode=True)
        self.addCleanup(editor.deleteLater)
        editor.setup_preparation(None, "surfactant_grid_ailsa", self.config)
        self.redirect_status(editor)
        self.assertIsNone(editor.status_file_path)
        self.assertFalse(editor.return_home_button.isEnabled())
        self.assertTrue(editor.robot_status_widget.isEnabled())
        self.assertTrue(editor.track_status_widget.isEnabled())
        self.assertTrue(all(not widget.isEnabled() for widget in editor.rack_widgets.values()))
        before = self.vials.read_bytes()
        editor.track_status_widget.num_source_spin.setValue(3)
        self.assertTrue(editor._save_preparation())
        self.assertIsNone(yaml.safe_load(self.config.read_text())["INPUT_VIAL_STATUS_FILE"])
        self.assertEqual(yaml.safe_load(self.track_file.read_text())["num_in_source"], 3)
        self.assertEqual(self.vials.read_bytes(), before)
        editor.close()

    def test_second_workflow_setup_uses_its_own_config_and_vial_file(self):
        name = "fluorescence_calibration_workflow"
        second_vials = self.root / "status" / "fluorescence_calibration_vials.csv"
        second_config = self.root / "workflow_configs" / f"{name}.yaml"
        shutil.copyfile(REPO_ROOT / "status" / second_vials.name, second_vials)
        config = yaml.safe_load((REPO_ROOT / "workflow_configs" / second_config.name).read_text())
        config["INPUT_VIAL_STATUS_FILE"] = str(second_vials)
        second_config.write_text(yaml.safe_dump(config))
        window = scheduler_gui.SchedulerWindow()
        self.addCleanup(window.deleteLater)
        selector = window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData(name))
        self.assertTrue(window.table.cellWidget(0, 1).isEnabled())
        with patch.object(scheduler_gui, "REPO_ROOT", self.root):
            window.set_row_config(selector, second_config)
            window.table.cellWidget(0, 1).click()
        editor = window.setup_window
        self.assertEqual(Path(editor.status_file_path), second_vials)
        self.assertNotEqual(Path(editor.config_editor.config_file), Path(window.row_config_path(selector)))
        self.assertNotEqual(Path(editor.config_editor.config_file), second_config)
        editor.close()
        self.assertIsNone(window.setup_window)

    def test_row_apply_cancel_and_reopen_keep_original_yaml_unchanged(self):
        window, selector = self.row_window()
        window.add_row("surfactant_grid_ailsa")
        other = Path(window.row_config_path(window.table.cellWidget(1, 0)))
        other_before = other.read_bytes()
        original = self.config.read_bytes()
        snapshot = Path(window.row_config_path(selector))
        before = snapshot.read_bytes()
        editor = self.open_row(window, selector)
        draft = Path(editor.config_editor.config_file)
        self.assertNotEqual(draft, snapshot)
        self.assertTrue(editor.save_all_button.isHidden())
        self.assertEqual(editor.close_setup_button.text(), "Cancel")
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(24)
        self.assertTrue(editor.config_editor._save_config(silent=True))
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            editor.close()
        self.assertEqual(snapshot.read_bytes(), before)
        self.assertFalse(draft.exists())
        editor = self.open_row(window, selector)
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        editor._save_and_return()
        self.assertIsNone(window.setup_window)
        self.assertEqual(yaml.safe_load(snapshot.read_text())["MAX_WELLS"], 48)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual(other.read_bytes(), other_before)
        editor = self.open_row(window, selector)
        self.assertEqual(editor.config_editor.config_widgets["MAX_WELLS"].value(), 48)
        editor.close()

    def test_explicit_save_to_disk_parses_values_and_only_writes_configuration(self):
        config = yaml.safe_load(self.config.read_text())
        config.update(VOLUME_TARGETS_ML=[0.02], EXTRA={"count": 1}, FLOAT_VALUE=0.01)
        self.config.write_text(yaml.safe_dump(config))
        window, selector = self.row_window()
        window.add_row("surfactant_grid_ailsa")
        snapshot = Path(window.row_config_path(selector))
        other = Path(window.row_config_path(window.table.cellWidget(1, 0)))
        before = {path: path.read_bytes() for path in (snapshot, other, self.vials, self.track_file, self.robot_file)}
        editor = self.open_row(window, selector)
        controls = editor.config_editor.config_widgets
        controls["MAX_WELLS"].setValue(32)
        controls["VOLUME_TARGETS_ML"].setText("[0.01, 0.005]")
        controls["EXTRA"].setPlainText("count: 3\nvalues: [1, 2]")
        controls["FLOAT_VALUE"].setValue(0.123456)
        with patch.object(scheduler_gui.QFileDialog, "getSaveFileName", return_value=(str(self.config), "")), patch.object(
            QMessageBox, "question", return_value=QMessageBox.Yes
        ) as confirmation:
            editor.config_editor.save_disk_button.click()
        confirmation.assert_called_once()
        saved = yaml.safe_load(self.config.read_text())
        self.assertEqual(saved["MAX_WELLS"], 32)
        self.assertEqual(saved["VOLUME_TARGETS_ML"], [0.01, 0.005])
        self.assertEqual(saved["EXTRA"], {"count": 3, "values": [1, 2]})
        self.assertEqual(saved["FLOAT_VALUE"], 0.123456)
        self.assertIs(window.setup_window, editor)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            editor.close()
        self.assertEqual(snapshot.read_bytes(), before[snapshot])
        self.assertEqual(yaml.safe_load(self.config.read_text())["MAX_WELLS"], 32)

    def test_explicit_load_preset_can_be_declined_then_applied_without_source_write(self):
        window, selector = self.row_window()
        preset = self.root / "workflow_configs" / "alternate.yaml"
        config = yaml.safe_load(self.config.read_text())
        config["MAX_WELLS"] = 16
        preset.write_text(yaml.safe_dump(config))
        original = self.config.read_bytes()
        preset_before = preset.read_bytes()
        editor = self.open_row(window, selector)
        draft = editor.config_editor.config_file
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        with patch.object(scheduler_gui.QFileDialog, "getOpenFileName", return_value=(str(preset), "")):
            with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
                editor.config_editor.load_preset_button.click()
            self.assertEqual(editor.config_editor.config_widgets["MAX_WELLS"].value(), 48)
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                editor.config_editor.load_preset_button.click()
        self.assertEqual(editor.config_editor.config_file, draft)
        self.assertEqual(editor.config_editor.config_widgets["MAX_WELLS"].value(), 16)
        editor._save_and_return()
        self.assertEqual(yaml.safe_load(Path(window.row_config_path(selector)).read_text())["MAX_WELLS"], 16)
        self.assertEqual(selector.property("config_source"), str(preset.resolve()))
        self.assertEqual(self.config.read_bytes(), original)
        self.assertEqual(preset.read_bytes(), preset_before)

    def test_save_all_then_later_edits_and_discard_do_not_commit_row(self):
        window, selector = self.row_window()
        snapshot = Path(window.row_config_path(selector))
        before = snapshot.read_bytes()
        editor = self.open_row(window, selector)
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        self.assertTrue(editor._save_all())
        self.assertFalse(editor._preparation_saved)
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(24)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel):
            self.assertFalse(editor.close())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            editor.close()
        self.assertEqual(snapshot.read_bytes(), before)

    def test_row_owned_apply_does_not_create_or_save_shared_state(self):
        window, selector = self.row_window()
        snapshot = Path(window.row_config_path(selector))
        before = {path: path.read_bytes() for path in (self.root / "robot_state").glob("*.yaml")}
        editor = self.open_row(window, selector)
        self.assertIsNone(editor.track_status_widget)
        self.assertIsNone(editor.robot_status_widget)
        tabs = [editor.tab_widget.tabText(index) for index in range(editor.tab_widget.count())]
        self.assertNotIn("Robot Status", tabs)
        self.assertNotIn("Track Status", tabs)
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        editor._save_and_return()
        self.assertIsNone(window.setup_window)
        self.assertEqual(yaml.safe_load(snapshot.read_text())["MAX_WELLS"], 48)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_missing_default_is_row_error_and_setup_retries_after_file_created(self):
        content = self.config.read_bytes()
        self.config.unlink()
        window, selector = self.row_window()
        self.assertEqual(window.table.item(0, 2).text(), "Review setup")
        self.assertTrue(window.vial_layout.errors.text())
        self.config.write_bytes(content)
        editor = self.open_row(window, selector)
        self.assertTrue(Path(window.row_config_path(selector)).is_file())
        editor.close()

    def test_standalone_config_editor_keeps_original_save_behavior(self):
        editor = self.editor()
        self.assertTrue(editor.config_editor.load_preset_button.isHidden())
        self.assertTrue(editor.config_editor.save_disk_button.isHidden())
        editor.config_editor.config_widgets["MAX_WELLS"].setValue(48)
        self.assertTrue(editor._save_all())
        self.assertEqual(yaml.safe_load(self.config.read_text())["MAX_WELLS"], 48)
        editor.close()

    def test_default_standalone_keeps_shared_tabs_and_save_all(self):
        with patch.object(sys, "argv", ["vial_manager_gui.py"]):
            editor = VialManagerMainWindow()
        self.addCleanup(editor.deleteLater)
        editor.load_status_file(str(self.vials))
        self.redirect_status(editor)
        tabs = [editor.tab_widget.tabText(index) for index in range(editor.tab_widget.count())]
        self.assertIn("Robot Status", tabs)
        self.assertIn("Track Status", tabs)
        editor.robot_status_widget.small_tip_rack_1_spin.setValue(9)
        editor.track_status_widget.num_source_spin.setValue(4)
        editor._save_all()
        self.assertEqual(yaml.safe_load(self.robot_file.read_text())["pipets_used"]["small_tip_rack_1"], 9)
        self.assertEqual(yaml.safe_load(self.track_file.read_text())["num_in_source"], 4)
        for path, content in self.shared_status.items():
            self.assertEqual(path.read_bytes(), content)

    def test_row_owned_default_constructor_removes_existing_shared_tabs(self):
        editor = VialManagerMainWindow(preparation_mode=True)
        self.addCleanup(editor.deleteLater)
        editor._populate_racks([])
        robot = editor.robot_status_widget
        track = editor.track_status_widget
        with patch.object(robot, "_save_robot_status") as robot_save, patch.object(
            track, "_save_track_status"
        ) as track_save:
            editor.setup_preparation(None, "surfactant_grid_ailsa", self.config, row_owned=True)
            self.assertIsNone(editor.robot_status_widget)
            self.assertIsNone(editor.track_status_widget)
            self.assertEqual(editor.tab_widget.indexOf(robot), -1)
            self.assertEqual(editor.tab_widget.indexOf(track), -1)
            self.assertTrue(editor._save_preparation())
            robot_save.assert_not_called()
            track_save.assert_not_called()
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                editor._reload_all()
            self.assertIsNone(editor.robot_status_widget)
            self.assertIsNone(editor.track_status_widget)
        editor.close()

    def test_scheduler_row_constructor_never_creates_hidden_shared_widgets(self):
        window, selector = self.row_window()
        with patch("vial_manager_gui.RobotStatusWidget", side_effect=AssertionError("Hidden robot widget")), patch(
            "vial_manager_gui.TrackStatusWidget", side_effect=AssertionError("Hidden track widget")
        ):
            editor = self.open_row(window, selector)
            editor._save_and_return()
        self.assertIsNone(window.setup_window)

    def test_scheduler_shared_tabs_save_temp_files_and_invalidate_approval(self):
        window, selector = self.row_window()
        self.assertEqual([window.tabs.tabText(index) for index in range(window.tabs.count())],
                         ["Queue", "Vial Layout", "Robot Status", "Track Status"])
        robot = window.robot_status_widget
        track = window.track_status_widget
        self.assertEqual(Path(robot.robot_file_path), self.root / "robot_state" / "robot_status.yaml")
        self.assertEqual(Path(track.track_file_path), self.root / "robot_state" / "track_status.yaml")
        window.approved_simulation_inputs = window.queue_input_fingerprint()
        window.update_live_run_enabled()
        self.assertTrue(window.run_button.isEnabled())
        before = Path(robot.robot_file_path).read_bytes()
        robot.small_tip_rack_1_spin.setValue(9)
        track.num_source_spin.setValue(4)
        self.assertTrue(window.shared_state_dirty())
        self.assertFalse(window.simulate_button.isEnabled())
        self.assertFalse(window.run_button.isEnabled())
        self.assertIsNone(window.approved_simulation_inputs)
        self.assertEqual(Path(robot.robot_file_path).read_bytes(), before)
        for widget, label in ((robot, "Robot Status"), (track, "Track Status")):
            save = next(button for button in widget.findChildren(QPushButton)
                        if button.accessibleName() == f"Save {label}")
            self.assertTrue(save.isEnabled())
            window.tabs.setCurrentWidget(widget)
            save.click()
        self.assertFalse(window.shared_state_dirty())
        self.assertTrue(window.simulate_button.isEnabled())
        self.assertFalse(window.run_button.isEnabled())
        self.assertEqual(yaml.safe_load(Path(robot.robot_file_path).read_text())["pipets_used"]["small_tip_rack_1"], 9)
        self.assertEqual(yaml.safe_load(Path(track.track_file_path).read_text())["num_in_source"], 4)
        window.approved_simulation_inputs = window.queue_input_fingerprint()
        self.assertTrue(robot._save_robot_status(silent=True))
        self.assertIsNone(window.approved_simulation_inputs)
        for path, content in self.shared_status.items():
            self.assertEqual(path.read_bytes(), content)

    def test_shared_reload_cancel_then_discard_does_not_write(self):
        window, selector = self.row_window()
        track = window.track_status_widget
        initial = track.num_source_spin.value()
        path = Path(track.track_file_path)
        before = path.read_bytes()
        reload = next(button for button in track.findChildren(QPushButton)
                      if button.accessibleName() == "Reload Track Status")
        track.num_source_spin.setValue((initial + 1) % 100)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel):
            reload.click()
        self.assertTrue(window.shared_state_dirty())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            reload.click()
        self.assertEqual(track.num_source_spin.value(), initial)
        self.assertFalse(window.shared_state_dirty())
        self.assertTrue(window.simulate_button.isEnabled())
        self.assertEqual(path.read_bytes(), before)

    def test_shared_controls_are_disabled_during_setup_and_execution(self):
        window, selector = self.row_window()
        editor = self.open_row(window, selector)
        self.assertFalse(window.table.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())
        self.assertTrue(all(not widget.isEnabled() for widget in window.shared_state_widgets))
        self.assertFalse(window.simulate_button.isEnabled())
        editor.close()
        self.assertTrue(window.table.isEnabled())
        self.assertTrue(all(widget.isEnabled() for widget in window.shared_state_widgets))
        window.set_simulation_busy(True)
        self.assertTrue(all(not widget.isEnabled() for widget in window.shared_state_widgets))
        with patch.object(window.robot_status_widget, "_save_robot_status") as save:
            window.save_shared_state(window.robot_status_widget, save)
            save.assert_not_called()
        window.set_simulation_busy(False)

    def test_shared_failed_load_and_save_keep_launches_blocked(self):
        window, selector = self.row_window()
        track = window.track_status_widget
        path = Path(track.track_file_path)
        before = path.read_bytes()
        track.num_source_spin.setValue((track.num_source_spin.value() + 1) % 100)
        track.track_file_path = str(self.root / "missing" / "track.yaml")
        with patch.object(QMessageBox, "critical"):
            self.assertFalse(track._save_track_status(silent=True))
        self.assertTrue(window.shared_state_dirty())
        self.assertFalse(window.simulate_button.isEnabled())
        window.reload_shared_state(track, track._load_track_status, confirm=False)
        self.assertFalse(track.state_loaded)
        self.assertFalse(window.simulate_button.isEnabled())
        self.assertIn("Cannot load", window.statusBar().currentMessage())
        with patch.object(QMessageBox, "warning"), patch.object(track, "_save_track_status") as save:
            window.save_shared_state(track, save)
            save.assert_not_called()
        track.track_file_path = str(path)
        window.reload_shared_state(track, track._load_track_status, confirm=False)
        self.assertTrue(window.shared_state_ready())
        self.assertEqual(path.read_bytes(), before)


class VialMovementSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        print_patch = patch("builtins.print")
        print_patch.start()
        self.addCleanup(print_patch.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.vials = Path(temp.name) / "vials.csv"
        self.records = [
            {
                "vial_index": str(index),
                "vial_name": f"sample_{index}",
                "location": "main_8mL_rack",
                "location_index": str(index),
                "home_location": "main_8mL_rack",
                "home_location_index": str(index),
                "vial_volume": "2.500",
                "vial_type": "8_mL",
                "capped": "True",
                "cap_type": "closed",
                "notes": f"original note {index}",
                "batch_metadata": f"batch-{index}",
                "external_sample_id": f"external-{index}",
            }
            for index in range(2)
        ]
        with self.vials.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
        self.disk_before = self.vials.read_bytes()
        self.editor = self.new_editor()

    def new_editor(self):
        editor = VialManagerMainWindow(preparation_mode=True, include_shared_state=False)
        self.addCleanup(editor.deleteLater)
        self.assertTrue(editor.load_status_file(str(self.vials)))
        return editor

    def snapshot(self):
        return [record.copy() for record in self.editor.original_vials_data]

    def record(self, vial_index):
        return next(record for record in self.editor.original_vials_data
                    if str(record["vial_index"]) == str(vial_index))

    def assert_position(self, vial_index, location, index, home_location=None, home_index=None):
        record = self.record(vial_index)
        self.assertEqual(record["location"], location)
        self.assertEqual(int(record["location_index"]), index)
        self.assertEqual(record["home_location"], location if home_location is None else home_location)
        self.assertEqual(int(record["home_location_index"]), index if home_index is None else home_index)

    def assert_metadata_preserved(self):
        position_keys = {"location", "location_index", "home_location", "home_location_index"}
        for original in self.records:
            moved = self.record(original["vial_index"])
            self.assertEqual({key: value for key, value in moved.items() if key not in position_keys},
                             {key: value for key, value in original.items() if key not in position_keys})

    def assert_main_grid(self):
        rack = self.editor.rack_widgets["main_8mL_rack"]
        expected = {int(record["location_index"]): str(record["vial_index"])
                    for record in self.editor.original_vials_data
                    if record["location"] == "main_8mL_rack"}
        self.assertEqual(rack.grid_layout.count(), 48)
        self.assertEqual({index: str(widget.get_vial_data()["vial_index"])
                          for index, widget in rack.vials.items()}, expected)
        self.assertEqual(set(rack.empty_slots), set(range(48)) - set(expected))
        for index in range(48):
            row = index % rack.grid_rows
            column = rack.grid_cols - 1 - index // rack.grid_rows
            widget = rack.grid_layout.itemAtPosition(row, column).widget()
            self.assertIs(widget, rack.vials[index] if index in expected else rack.empty_slots[index])

    def accept_dialog(self, vial_index, edit_fields):
        rack = self.editor.rack_widgets["main_8mL_rack"]
        index = int(self.record(vial_index)["location_index"])
        original = rack.vials[index].get_vial_data()

        def accept(dialog):
            edit_fields(dialog)
            return QDialog.Accepted

        with patch("vial_manager_gui.VialEditDialog.exec", accept):
            rack._on_vial_clicked(original)

    def test_empty_slot_move_updates_current_and_home_without_changing_metadata(self):
        untouched = self.record("1").copy()
        self.assertTrue(self.editor._move_vial("0", "main_8mL_rack", 2))
        self.assert_position("0", "main_8mL_rack", 2)
        self.assertEqual(self.record("1"), untouched)
        self.assert_metadata_preserved()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assert_main_grid()

    def drag_mime(self, **overrides):
        payload = {
            "owner": self.editor._drag_owner,
            "vial_index": "0",
            "source": ["main_8mL_rack", 0],
        }
        payload.update(overrides)
        mime = QMimeData()
        mime.setData(VIAL_MOVE_MIME, json.dumps(payload).encode("utf-8"))
        return mime

    def show_interaction_editor(self):
        self.editor.show()
        QApplication.processEvents()

    def drop_on(self, target, mime):
        self.assertIs(target.window(), self.editor)
        enter = QDragEnterEvent(QPoint(5, 5), Qt.MoveAction, mime,
                                Qt.LeftButton, Qt.NoModifier)
        target.dragEnterEvent(enter)
        self.assertTrue(enter.isAccepted())
        self.assertEqual(enter.dropAction(), Qt.MoveAction)
        drop = QDropEvent(QPointF(5, 5), Qt.MoveAction, mime,
                          Qt.LeftButton, Qt.NoModifier)
        target.dropEvent(drop)
        self.assertTrue(drop.isAccepted())
        self.assertEqual(drop.dropAction(), Qt.MoveAction)
        self.assertTrue(self.editor._layout_refresh_pending)

    def test_qt_drop_on_empty_main_slot_refreshes_grid_after_events(self):
        self.show_interaction_editor()
        rack = self.editor.rack_widgets["main_8mL_rack"]
        source = rack.vials[0]
        untouched = self.record("1").copy()
        self.drop_on(rack.empty_slots[2], self.drag_mime())
        self.assert_position("0", "main_8mL_rack", 2)
        self.assertEqual(self.record("1"), untouched)
        self.assertIs(rack.vials[0], source)
        QApplication.processEvents()
        self.assertFalse(self.editor._layout_refresh_pending)
        self.assert_main_grid()
        self.assert_metadata_preserved()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_drop_on_occupied_main_vial_swaps_grid_after_events(self):
        self.show_interaction_editor()
        rack = self.editor.rack_widgets["main_8mL_rack"]
        self.drop_on(rack.vials[1], self.drag_mime())
        self.assert_position("0", "main_8mL_rack", 1)
        self.assert_position("1", "main_8mL_rack", 0)
        QApplication.processEvents()
        self.assert_main_grid()
        self.assert_metadata_preserved()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_drop_on_empty_auxiliary_clamp_moves_between_racks(self):
        self.show_interaction_editor()
        combined = self.editor.rack_widgets["_combined_aux"]
        target = combined.clamp_placeholders[("clamp", 0)][0]
        untouched = self.record("1").copy()
        self.drop_on(target, self.drag_mime())
        self.assert_position("0", "clamp", 0)
        self.assertEqual(self.record("1"), untouched)
        QApplication.processEvents()
        self.assert_main_grid()
        occupied = combined.vials[("clamp", 0)]
        self.assertEqual(str(occupied.get_vial_data()["vial_index"]), "0")
        self.assertIs(combined.clamp_grid.itemAtPosition(0, 0).widget(), occupied)
        self.assertEqual(combined.clamp_grid.count(), 1)
        self.assert_metadata_preserved()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_drop_on_occupied_auxiliary_clamp_swaps_between_racks(self):
        self.show_interaction_editor()
        self.assertTrue(self.editor._move_vial("1", "clamp", 0))
        combined = self.editor.rack_widgets["_combined_aux"]
        self.drop_on(combined.vials[("clamp", 0)], self.drag_mime())
        self.assert_position("0", "clamp", 0)
        self.assert_position("1", "main_8mL_rack", 0)
        QApplication.processEvents()
        self.assert_main_grid()
        occupied = combined.vials[("clamp", 0)]
        self.assertEqual(str(occupied.get_vial_data()["vial_index"]), "0")
        self.assertIs(combined.clamp_grid.itemAtPosition(0, 0).widget(), occupied)
        self.assertEqual(combined.clamp_grid.count(), 1)
        self.assert_metadata_preserved()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_drop_rejects_invalid_payloads_and_readonly_without_mutation(self):
        self.show_interaction_editor()
        malformed = QMimeData()
        malformed.setData(VIAL_MOVE_MIME, b"{invalid-json")
        cases = [
            ("malformed JSON", malformed, False),
            ("foreign owner", self.drag_mime(owner="foreign-editor"), False),
            ("stale source", self.drag_mime(source=["main_8mL_rack", 1]), False),
            ("no MIME", QMimeData(), False),
            ("unknown vial", self.drag_mime(vial_index="missing-vial"), False),
            ("read-only", self.drag_mime(), True),
        ]
        for name, mime, readonly in cases:
            with self.subTest(case=name):
                before = self.snapshot()
                self.editor._set_interface_readonly(readonly)
                target = self.editor.rack_widgets["main_8mL_rack"].empty_slots[2]
                self.assertIs(target.window(), self.editor)
                enter = QDragEnterEvent(QPoint(5, 5), Qt.MoveAction, mime,
                                        Qt.LeftButton, Qt.NoModifier)
                drop = QDropEvent(QPointF(5, 5), Qt.MoveAction, mime,
                                  Qt.LeftButton, Qt.NoModifier)
                with patch.object(QMessageBox, "warning") as warning:
                    target.dragEnterEvent(enter)
                    target.dropEvent(drop)
                warning.assert_not_called()
                self.assertFalse(enter.isAccepted())
                self.assertFalse(drop.isAccepted())
                QApplication.processEvents()
                self.assertEqual(self.snapshot(), before)
                self.assertFalse(self.editor._layout_refresh_pending)
                self.assert_main_grid()
                self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.editor._set_interface_readonly(False)

    def test_qt_click_and_subthreshold_move_emit_one_click_without_drag(self):
        self.show_interaction_editor()
        rack = self.editor.rack_widgets["main_8mL_rack"]
        source = rack.vials[0]
        source.vial_clicked.disconnect(rack._on_vial_clicked)
        clicks = []
        source.vial_clicked.connect(clicks.append)
        before = self.snapshot()
        position = QPoint(5, 5)
        with patch.object(QDrag, "exec", autospec=True, return_value=Qt.IgnoreAction) as drag:
            QTest.mouseClick(source, Qt.LeftButton, pos=position)
            self.assertEqual(clicks, [source.get_vial_data()])
            clicks.clear()
            QTest.mousePress(source, Qt.LeftButton, pos=position)
            moved = position + QPoint(1, 0)
            self.assertLess((moved - position).manhattanLength(), QApplication.startDragDistance())
            event = QMouseEvent(QEvent.MouseMove, QPointF(moved),
                                QPointF(source.mapToGlobal(moved)), Qt.NoButton,
                                Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(source, event)
            QTest.mouseRelease(source, Qt.LeftButton, pos=moved)
            drag.assert_not_called()
        self.assertEqual(clicks, [source.get_vial_data()])
        self.assertEqual(self.snapshot(), before)
        self.assert_main_grid()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_threshold_move_starts_drag_without_click_or_mutation(self):
        self.show_interaction_editor()
        rack = self.editor.rack_widgets["main_8mL_rack"]
        source = rack.vials[0]
        source.vial_clicked.disconnect(rack._on_vial_clicked)
        clicks = []
        source.vial_clicked.connect(clicks.append)
        before = self.snapshot()
        position = QPoint(5, 5)
        moved = position + QPoint(QApplication.startDragDistance(), 0)
        QTest.mousePress(source, Qt.LeftButton, pos=position)
        event = QMouseEvent(QEvent.MouseMove, QPointF(moved),
                            QPointF(source.mapToGlobal(moved)), Qt.NoButton,
                            Qt.LeftButton, Qt.NoModifier)
        with patch.object(QDrag, "exec", autospec=True, return_value=Qt.IgnoreAction) as drag:
            QApplication.sendEvent(source, event)
            QTest.mouseRelease(source, Qt.LeftButton, pos=moved)
            drag.assert_called_once_with(Qt.MoveAction)
        self.assertEqual(clicks, [])
        self.assertFalse(self.editor._active_drag)
        self.assertFalse(self.editor._drag_tab_timer.isActive())
        self.assertEqual(self.snapshot(), before)
        self.assert_main_grid()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_drag_hover_switches_rack_tab_and_rejects_configuration_tab(self):
        configuration = QPushButton("Unsaved configuration")
        config_index = self.editor.tab_widget.addTab(configuration, "Configuration")
        self.show_interaction_editor()
        tabs = self.editor.tab_widget
        tabs.setCurrentWidget(self.editor.rack_widgets["main_8mL_rack"])
        bar = tabs.tabBar()
        heater_index = tabs.indexOf(self.editor.rack_widgets["heater"])
        heater_position = bar.tabRect(heater_index).center()
        mime = self.drag_mime()
        self.editor._active_drag = True
        self.addCleanup(setattr, self.editor, "_active_drag", False)
        self.addCleanup(self.editor._drag_tab_timer.stop)
        before = self.snapshot()
        original_index = tabs.currentIndex()
        for event_type in (QDragEnterEvent, QDragMoveEvent):
            event = event_type(heater_position, Qt.MoveAction, mime,
                               Qt.LeftButton, Qt.NoModifier)
            self.assertTrue(self.editor.eventFilter(bar, event))
            self.assertTrue(event.isAccepted())
            self.assertEqual(event.dropAction(), Qt.MoveAction)
            self.assertEqual(self.editor._drag_tab_index, heater_index)
            self.assertTrue(self.editor._drag_tab_timer.isActive())
            self.assertEqual(tabs.currentIndex(), original_index)
        self.editor._switch_drag_tab()
        self.assertEqual(tabs.currentIndex(), heater_index)
        self.assertTrue(self.editor._active_drag)
        for event_type in (QDragEnterEvent, QDragMoveEvent):
            event = event_type(bar.tabRect(config_index).center(), Qt.MoveAction, mime,
                               Qt.LeftButton, Qt.NoModifier)
            self.assertTrue(self.editor.eventFilter(bar, event))
            self.assertFalse(event.isAccepted())
            self.assertFalse(self.editor._drag_tab_timer.isActive())
            self.assertEqual(self.editor._drag_tab_index, -1)
        self.editor._switch_drag_tab()
        self.assertEqual(tabs.currentIndex(), heater_index)
        self.assertEqual(self.snapshot(), before)
        self.assert_main_grid()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_qt_malformed_source_position_rejects_drag_without_click(self):
        self.show_interaction_editor()
        rack = self.editor.rack_widgets["main_8mL_rack"]
        source = rack.vials[0]
        source.vial_clicked.disconnect(rack._on_vial_clicked)
        clicks = []
        source.vial_clicked.connect(clicks.append)
        self.record("0")["location_index"] = "NaN"
        source.vial_data["location_index"] = "NaN"
        before = self.snapshot()
        position = QPoint(5, 5)
        moved = position + QPoint(QApplication.startDragDistance(), 0)
        QTest.mousePress(source, Qt.LeftButton, pos=position)
        event = QMouseEvent(QEvent.MouseMove, QPointF(moved),
                            QPointF(source.mapToGlobal(moved)), Qt.NoButton,
                            Qt.LeftButton, Qt.NoModifier)
        with patch.object(QDrag, "exec", autospec=True) as drag, patch.object(
                QMessageBox, "warning") as warning:
            QApplication.sendEvent(source, event)
            QTest.mouseRelease(source, Qt.LeftButton, pos=moved)
        warning.assert_called_once()
        drag.assert_not_called()
        self.assertEqual(clicks, [])
        self.assertFalse(self.editor._active_drag)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_same_rack_occupied_move_swaps_both_current_and_home_positions(self):
        self.assertTrue(self.editor._move_vial("0", "main_8mL_rack", 1))
        self.assert_position("0", "main_8mL_rack", 1)
        self.assert_position("1", "main_8mL_rack", 0)
        self.assert_metadata_preserved()
        self.assert_main_grid()

    def test_cross_rack_occupied_move_swaps_both_current_and_home_positions(self):
        self.assertTrue(self.editor._move_vial("1", "heater", 3))
        self.assertTrue(self.editor._move_vial("0", "heater", 3))
        self.assert_position("0", "heater", 3)
        self.assert_position("1", "main_8mL_rack", 0)
        self.assertEqual(str(self.editor.rack_widgets["heater"].vials[3].get_vial_data()["vial_index"]), "0")
        self.assert_metadata_preserved()
        self.assert_main_grid()

    def test_same_current_slot_is_noop_even_when_home_differs(self):
        self.record("0")["home_location"] = "heater"
        self.record("0")["home_location_index"] = "3"
        before = self.snapshot()
        with patch.object(QMessageBox, "warning") as warning:
            self.editor._move_vial("0", "main_8mL_rack", 0)
        warning.assert_not_called()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assert_main_grid()

    def test_move_is_only_persisted_on_save_and_reloads_with_metadata(self):
        self.assertTrue(self.editor._move_vial("0", "heater", 11))
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assertTrue(self.editor._save_file())
        with self.vials.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        expected = [record.copy() for record in self.records]
        expected[0].update(location="heater", location_index="11",
                           home_location="heater", home_location_index="11")
        self.assertEqual(saved, expected)
        reloaded = self.new_editor()
        self.assertEqual(reloaded.original_vials_data, expected)
        self.assertNotIn(0, reloaded.rack_widgets["main_8mL_rack"].vials)
        self.assertEqual(str(reloaded.rack_widgets["heater"].vials[11].get_vial_data()["vial_index"]), "0")

    def test_invalid_move_targets_warn_without_mutating_any_record(self):
        targets = [
            ("0", "main_8mL_rack", "not-an-index"),
            ("0", "main_8mL_rack", ""),
            ("0", "main_8mL_rack", None),
            ("0", "main_8mL_rack", "2.5"),
            ("0", "main_8mL_rack", 2.5),
            ("0", "main_8mL_rack", "NaN"),
            ("0", "main_8mL_rack", "Infinity"),
            ("0", "main_8mL_rack", float("nan")),
            ("0", "main_8mL_rack", float("inf")),
            ("0", "main_8mL_rack", -1),
            ("0", "main_8mL_rack", 48),
            ("0", "heater", 12),
            ("0", "small_vial_rack", 4),
            ("0", "12_well_ilya", 12),
            ("0", "unknown_rack", 0),
            ("missing-vial", "main_8mL_rack", 2),
        ]
        for vial_index, location, index in targets:
            with self.subTest(vial_index=vial_index, location=location, index=index):
                before = self.snapshot()
                with patch.object(QMessageBox, "warning") as warning:
                    self.assertFalse(self.editor._move_vial(vial_index, location, index))
                warning.assert_called_once()
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.vials.read_bytes(), self.disk_before)
                self.assert_main_grid()

    def test_readonly_rejects_moves_and_can_be_cleared(self):
        before = self.snapshot()
        self.editor._set_interface_readonly(True)
        self.assertTrue(self.editor._layout_readonly)
        with patch.object(QMessageBox, "warning"):
            self.assertFalse(self.editor._move_vial("0", "main_8mL_rack", 2))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.editor._set_interface_readonly(False)
        self.assertFalse(self.editor._layout_readonly)
        self.assertTrue(self.editor._move_vial("0", "main_8mL_rack", 2))
        self.assert_main_grid()

    def test_duplicate_vial_identity_blocks_save_without_touching_csv(self):
        self.record("1")["vial_index"] = "0"
        with patch.object(QMessageBox, "critical") as error:
            self.assertFalse(self.editor._save_file())
        error.assert_called_once()
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_repeated_moves_leave_exact_grid_mapping_without_ghost_placeholders(self):
        for location, index in (("main_8mL_rack", 2), ("main_8mL_rack", 47),
                                ("heater", 11), ("main_8mL_rack", 0),
                                ("main_8mL_rack", 1), ("main_8mL_rack", 2)):
            with self.subTest(location=location, index=index):
                self.assertTrue(self.editor._move_vial("0", location, index))
                self.assert_main_grid()
                self.editor._reload_all_widgets()
                self.assert_main_grid()
        self.assert_metadata_preserved()

    def test_refresh_false_updates_model_and_explicit_reload_updates_all_racks(self):
        original_widget = self.editor.rack_widgets["main_8mL_rack"].vials[0]
        self.assertTrue(self.editor._move_vial("0", "heater", 2, refresh=False))
        self.assert_position("0", "heater", 2)
        self.assertIs(self.editor.rack_widgets["main_8mL_rack"].vials[0], original_widget)
        self.editor._reload_all_widgets()
        self.assert_main_grid()
        self.assertEqual(str(self.editor.rack_widgets["heater"].vials[2].get_vial_data()["vial_index"]), "0")

    def test_configured_empty_destinations_and_yaml_grid_dimensions(self):
        for location, rows, columns in (("heater", 3, 4),):
            with self.subTest(location=location):
                config = self.editor.vial_areas[location]
                self.assertEqual(config["grid_params"]["num_rows"], rows)
                self.assertEqual(config["grid_params"]["num_cols"], columns)
                rack = self.editor.rack_widgets[location]
                self.assertEqual((rack.grid_rows, rack.grid_cols), (rows, columns))
                self.assertEqual(rack.grid_layout.count(), config["rack_size"])
                self.assertEqual(rack.vials, {})
                self.assertEqual(set(rack.empty_slots), set(range(config["rack_size"])))
                self.assertGreaterEqual(self.editor.tab_widget.indexOf(rack), 0)
        auxiliary = {"large_vial_rack", "photoreactor_array", "clamp"}
        for location in set(self.editor.vial_areas) - auxiliary - {"main_8mL_rack"}:
            with self.subTest(empty_destination=location):
                if not self.editor.vial_areas[location]['rack_present']:
                    self.assertNotIn(location, self.editor.rack_widgets)
                    continue
                rack = self.editor.rack_widgets[location]
                self.assertEqual(rack.get_vials_data(), [])
                self.assertGreaterEqual(self.editor.tab_widget.indexOf(rack), 0)
        combined = self.editor.rack_widgets["_combined_aux"]
        self.assertGreaterEqual(self.editor.tab_widget.indexOf(combined), 0)
        self.assertEqual(combined.get_vials_data(), [])

    def test_absent_racks_are_hidden_and_cannot_receive_vials(self):
        for location in ("12_well_ilya", "small_vial_rack", "50mL_vial_rack"):
            with self.subTest(location=location):
                self.assertFalse(self.editor.vial_areas[location]['rack_present'])
                self.assertNotIn(location, self.editor.rack_widgets)
                before = self.snapshot()
                with patch.object(QMessageBox, "warning") as warning:
                    self.assertFalse(self.editor._move_vial("0", location, 0))
                warning.assert_called_once()
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_presence_flag_enables_empty_rack_and_keeps_absent_auxiliary_hidden(self):
        self.editor.vial_areas['small_vial_rack']['rack_present'] = True
        self.editor.vial_areas['clamp']['rack_present'] = False
        self.editor._populate_racks(self.editor.original_vials_data)
        rack = self.editor.rack_widgets['small_vial_rack']
        self.assertEqual((rack.grid_rows, rack.grid_cols), (2, 2))
        self.assertTrue(self.editor._move_vial('0', 'small_vial_rack', 0))
        combined = self.editor.rack_widgets['_combined_aux']
        clamp = combined.clamp_grid.parentWidget()
        self.assertTrue(clamp.isHidden())
        self.editor._reload_all_widgets()
        self.assertTrue(combined.clamp_grid.parentWidget().isHidden())

    def test_absent_home_assignment_blocks_save_without_deleting_vial(self):
        self.record('0')['home_location'] = 'small_vial_rack'
        before = self.snapshot()
        with patch.object(QMessageBox, 'critical') as error:
            self.assertFalse(self.editor._save_file())
        error.assert_called_once()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.vials.read_bytes(), self.disk_before)

    def test_controller_rejects_absent_racks_without_hardware_initialization(self):
        from North_Safe import North_Robot

        robot = North_Robot.__new__(North_Robot)
        robot.VIAL_POSITIONS = {name: area.copy() for name, area in self.editor.vial_areas.items()}
        self.assertEqual(robot.get_config_parameter('vial_positions', 'heater', 'rack_size'), 12)
        for location in ('small_vial_rack', '50mL_vial_rack', '12_well_ilya'):
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, 'not present'):
                robot.get_location(True, location, 0)
        robot.VIAL_POSITIONS['heater']['rack_present'] = 'false'
        with self.assertRaisesRegex(ValueError, 'rack_present'):
            robot.get_config_parameter('vial_positions', 'heater', 'rack_size')
        del robot.VIAL_POSITIONS['heater']['rack_present']
        with self.assertRaises(KeyError):
            robot.get_config_parameter('vial_positions', 'heater', 'rack_size')

    def test_reload_preserves_other_tabs_their_identity_and_unsaved_values(self):
        other_tab = QPushButton("Unsaved configuration")
        self.editor.tab_widget.addTab(other_tab, "Other settings")
        self.editor.tab_widget.setCurrentWidget(other_tab)
        tabs = [(self.editor.tab_widget.widget(index), self.editor.tab_widget.tabText(index))
                for index in range(self.editor.tab_widget.count())]
        self.assertTrue(self.editor._move_vial("0", "heater", 4))
        self.editor._reload_all_widgets()
        self.assertEqual([(self.editor.tab_widget.widget(index), self.editor.tab_widget.tabText(index))
                          for index in range(self.editor.tab_widget.count())], tabs)
        self.assertIs(self.editor.tab_widget.currentWidget(), other_tab)
        self.assertEqual(other_tab.text(), "Unsaved configuration")
        self.assert_main_grid()

    def test_manual_accepted_dialog_moves_cross_rack_without_overwriting_home(self):
        untouched = self.record("1").copy()

        def edit(dialog):
            dialog.location_edit.setText("heater")
            dialog.location_index_spin.setValue(5)

        with patch.object(QMessageBox, "warning") as warning:
            self.accept_dialog("0", edit)
        warning.assert_not_called()
        self.assert_position("0", "heater", 5, "main_8mL_rack", 0)
        self.assertEqual(self.record("1"), untouched)
        self.assertEqual(self.record("0")["batch_metadata"], "batch-0")
        self.assertEqual(str(self.editor.rack_widgets["heater"].vials[5].get_vial_data()["vial_index"]), "0")
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assert_main_grid()

    def test_manual_accepted_home_only_dialog_does_not_relocate_current_vial(self):
        def edit(dialog):
            dialog.home_location_edit.setText("heater")
            dialog.home_index_spin.setValue(7)

        with patch.object(QMessageBox, "warning") as warning:
            self.accept_dialog("0", edit)
        warning.assert_not_called()
        self.assert_position("0", "main_8mL_rack", 0, "heater", 7)
        self.assertEqual(self.record("1"), self.records[1])
        self.assertEqual(self.record("0")["external_sample_id"], "external-0")
        self.assertEqual(self.editor.rack_widgets["heater"].vials, {})
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assert_main_grid()

    def test_manual_accepted_dialog_rejects_occupied_slot_without_swapping(self):
        before = self.snapshot()

        def edit(dialog):
            dialog.location_index_spin.setValue(1)
            dialog.name_edit.setText("must-not-commit")

        with patch.object(QMessageBox, "warning") as warning:
            self.accept_dialog("0", edit)
        warning.assert_called_once()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.vials.read_bytes(), self.disk_before)
        self.assert_main_grid()


if __name__ == "__main__":
    unittest.main()