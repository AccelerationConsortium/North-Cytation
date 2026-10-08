import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from PySide6.QtCore import QEventLoop, QProcess, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
import scheduler_gui
from scheduler import simulation_state as state


class HarmlessProcess(QProcess):
    launches = []

    def start(self, program, arguments):
        self.launches.append((program, arguments))
        if "--job-folder" in arguments:
            folder = Path(arguments[arguments.index("--job-folder") + 1])
            root = str(folder.parent.parent)
            job = folder.name
        else:
            root = arguments[arguments.index("--session") + 1]
            job = arguments[arguments.index("--job") + 1]
        fixture_args = [str(REPO_ROOT / "tests" / "fixtures" / "scheduler_simulation_child.py"), root, job]
        if "--job-folder" in arguments:
            fixture_args.append("--live-fixture")
        super().start(program, fixture_args)


class SimulationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("workflows", "workflow_configs", "robot_state", "status"):
            (self.root / name).mkdir()
        (self.root / "robot_state" / "vial_positions.yaml").write_text(
            yaml.safe_dump(yaml.safe_load((REPO_ROOT / "robot_state" / "vial_positions.yaml").read_text()))
        )
        (self.root / "robot_state" / "robot_status.yaml").write_text(yaml.safe_dump({
            "gripper_status": None, "gripper_vial_index": None, "held_pipet_type": None,
            "pipet_fluid_vial_index": None, "pipet_fluid_volume": 0, "pipets_used": {"rack": 0}
        }))
        (self.root / "robot_state" / "track_status.yaml").write_text(yaml.safe_dump({"active_wellplate_position": None}))
        for index, name in enumerate(("first", "second")):
            (self.root / "workflows" / f"{name}.py").touch()
            (self.root / "workflow_configs" / f"{name}.yaml").write_text(yaml.safe_dump({
                "SIMULATE": False, "INPUT_VIAL_STATUS_FILE": f"status/{name}.csv"
            }))
            (self.root / "status" / f"{name}.csv").write_text(
                f"vial_name,location,location_index,vial_volume\n{name},main_8mL_rack,{index},2\n"
            )
        HarmlessProcess.launches = []
        for item in (patch.object(scheduler_gui, "REPO_ROOT", self.root),
                     patch.object(state, "REPO_ROOT", self.root),
                     patch.object(state, "STATE_ROOT", self.root / "private"),
                     patch.object(scheduler_gui, "QProcess", HarmlessProcess)):
            item.start()
            self.addCleanup(item.stop)
        self.window = scheduler_gui.SchedulerWindow()
        self.addCleanup(self.window.deleteLater)
        selector = self.window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData("first"))
        self.window.add_row("second")

    def wait_for_queue(self):
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(10)

        def finished():
            if self.window.simulation_process is None:
                loop.quit()

        timer.timeout.connect(finished)
        timer.start()
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        watchdog.start(10000)
        loop.exec()
        timer.stop()
        watchdog.stop()
        self.assertIsNone(self.window.simulation_process, "Test child did not finish; watchdog is only a test limit")

    def test_children_run_in_order_with_state_and_notes(self):
        original = (self.root / "robot_state" / "robot_status.yaml").read_bytes()
        self.window.start_simulation()
        self.assertFalse(self.window.table.isEnabled())
        self.assertEqual(self.window.table.item(0, 2).background().color().name(), "#fff0a6")
        self.wait_for_queue()
        session = self.window.simulation_session
        self.assertEqual((session / "order.txt").read_text().splitlines(), ["000", "001"])
        second = json.loads((session / "jobs" / "001" / "result.json").read_text())
        self.assertEqual(second["tips_before"], 1)
        self.assertEqual(self.window.table.item(0, 3).text(), "0 errors, 1 warnings")
        self.assertEqual(self.window.table.item(1, 2).text(), "Simulated")
        self.assertEqual(self.window.table.item(1, 2).background().color().name(), "#ccebd5")
        self.assertEqual(HarmlessProcess.launches[0][0], sys.executable)
        self.assertEqual((self.root / "robot_state" / "robot_status.yaml").read_bytes(), original)
        self.assertTrue(self.window.table.isEnabled())

    def test_fatal_first_child_does_not_launch_second(self):
        config_path = self.root / "workflow_configs" / "first.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["TEST_FAILURE"] = True
        config_path.write_text(yaml.safe_dump(config))
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertEqual(len(HarmlessProcess.launches), 1)
        self.assertEqual(self.window.table.item(0, 2).text(), "Simulation incomplete")
        self.assertEqual(self.window.table.item(0, 2).background().color().name(), "#f5cccc")
        self.assertEqual(self.window.table.item(1, 2).text(), "Not simulated")

    def test_stop_after_current_and_close_do_not_terminate_child(self):
        self.window.start_simulation()
        with patch.object(QMessageBox, "warning"):
            self.assertFalse(self.window.close())
        self.window.stop_after_current()
        self.wait_for_queue()
        self.assertEqual(len(HarmlessProcess.launches), 1)
        self.assertEqual(self.window.table.item(0, 2).text(), "Simulated")
        self.assertEqual(self.window.table.item(1, 2).text(), "Not simulated")

    def test_conflict_requires_confirmation_and_can_be_simulated(self):
        path = self.root / "status" / "second.csv"
        path.write_text("vial_name,location,location_index,vial_volume\nsecond,main_8mL_rack,0,2\n")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.window.start_simulation()
        self.assertEqual(HarmlessProcess.launches, [])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.start_simulation()
        self.wait_for_queue()
        self.assertEqual(len(HarmlessProcess.launches), 2)
        specification = json.loads((self.window.simulation_session / "jobs" / "000" / "input.json").read_text())
        self.assertTrue(specification["allow_vial_conflicts"])
        self.assertFalse(self.window.run_button.isEnabled())

    def test_live_run_requires_clean_simulation_and_forces_live_config(self):
        self.assertFalse(self.window.run_button.isEnabled())
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        HarmlessProcess.launches = []
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.start_live_run()
        self.assertEqual(self.window.table.item(0, 2).text(), "Running")
        self.assertEqual(self.window.table.item(0, 2).background().color().name(), "#fff0a6")
        self.wait_for_queue()
        self.assertEqual(len(HarmlessProcess.launches), 2)
        self.assertIn("scheduler.live_runner", HarmlessProcess.launches[0][1])
        specification = json.loads((self.window.simulation_session / "jobs" / "000" / "input.json").read_text())
        self.assertIs(specification["config"]["SIMULATE"], False)
        self.assertEqual(specification["config"]["INPUT_VIAL_STATUS_FILE"], str((self.root / "status" / "first.csv").resolve()))
        self.assertEqual(self.window.table.item(1, 2).text(), "Completed")
        self.assertEqual(self.window.table.item(1, 2).background().color().name(), "#ccebd5")
        self.assertFalse(self.window.run_button.isEnabled())

    def test_live_error_even_with_successful_child_exit_stops_queue(self):
        path = self.root / "workflow_configs" / "first.yaml"
        config = yaml.safe_load(path.read_text())
        config["TEST_LIVE_ERROR"] = True
        path.write_text(yaml.safe_dump(config))
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        HarmlessProcess.launches = []
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.start_live_run()
        self.wait_for_queue()
        self.assertEqual(len(HarmlessProcess.launches), 1)
        self.assertEqual(self.window.table.item(0, 2).text(), "Run failed")
        self.assertEqual(self.window.table.item(0, 2).background().color().name(), "#f5cccc")
        self.assertEqual(self.window.table.item(1, 2).text(), "Not run")

    def test_changed_inputs_and_order_invalidate_live_approval(self):
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        path = self.root / "status" / "first.csv"
        path.write_text(path.read_text().replace(",2\n", ",3\n"))
        self.window.refresh_vial_layout()
        self.assertFalse(self.window.run_button.isEnabled())
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.table.selectRow(0)
        self.window.move_row(1)
        self.assertFalse(self.window.run_button.isEnabled())

    def test_simulation_error_blocks_live_run_even_when_completed(self):
        path = self.root / "workflow_configs" / "first.yaml"
        config = yaml.safe_load(path.read_text())
        config["TEST_SIMULATION_ERROR"] = True
        path.write_text(yaml.safe_dump(config))
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertEqual(self.window.table.item(0, 2).text(), "Needs attention")
        self.assertFalse(self.window.run_button.isEnabled())
        with patch.object(QMessageBox, "warning") as blocked:
            self.window.start_live_run()
        blocked.assert_called_once()
        self.assertEqual(self.window.execution_mode, "simulation")

    def test_vial_less_queue_simulates_and_live_spec_keeps_null(self):
        for name in ("first", "second"):
            path = self.root / "workflow_configs" / f"{name}.yaml"
            config = yaml.safe_load(path.read_text())
            config["INPUT_VIAL_STATUS_FILE"] = None
            path.write_text(yaml.safe_dump(config))
        self.window.refresh_vial_layout()
        self.assertFalse(self.window.vial_layout.occupancy)
        self.assertFalse(self.window.vial_layout.errors.text())
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertEqual(list((self.window.simulation_session / "vials").iterdir()), [])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.start_live_run()
        self.wait_for_queue()
        specification = json.loads((self.window.simulation_session / "jobs" / "000" / "input.json").read_text())
        self.assertIsNone(specification["config"]["INPUT_VIAL_STATUS_FILE"])
        self.assertEqual(specification["vial_files"], [])
        self.assertEqual(self.window.table.item(1, 2).text(), "Completed")


if __name__ == "__main__":
    unittest.main()