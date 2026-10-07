import csv
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
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

import scheduler_gui
from vial_manager_gui import VialManagerMainWindow


class SchedulerSetupTests(unittest.TestCase):
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
        editor.track_status_widget.track_file_path = str(self.track_file)
        editor.robot_status_widget.robot_file_path = str(self.robot_file)

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
            window.table.cellWidget(0, 1).click()
        editor = window.setup_window
        self.assertEqual(Path(editor.status_file_path), second_vials)
        self.assertEqual(Path(editor.config_editor.config_file), second_config)
        editor.close()
        self.assertIsNone(window.setup_window)


if __name__ == "__main__":
    unittest.main()