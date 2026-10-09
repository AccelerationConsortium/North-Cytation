import json
import io
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
        self.assertTrue(all(not widget.isEnabled() for widget in self.window.shared_state_widgets))
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
        self.assertTrue(all(widget.isEnabled() for widget in self.window.shared_state_widgets))

    def test_child_output_reaches_terminal_before_exit_and_remains_in_log(self):
        for row in range(2):
            path = Path(self.window.row_config_path(self.window.table.cellWidget(row, 0)))
            config = yaml.safe_load(path.read_text())
            config["TEST_OUTPUT_DELAY"] = 0.6
            path.write_text(yaml.safe_dump(config))
        for live in (False, True):
            with self.subTest(live=live):
                terminal = io.StringIO()
                observed_while_running = []
                timer = QTimer()
                timer.setInterval(20)
                timer.timeout.connect(lambda: observed_while_running.append(
                    self.window.simulation_process is not None
                    and "Child stderr is visible" in terminal.getvalue()
                ))
                with patch.object(sys, "stdout", terminal):
                    timer.start()
                    if live:
                        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                            self.window.start_live_run()
                    else:
                        self.window.start_simulation()
                    self.wait_for_queue()
                    timer.stop()
                saved = "".join((self.window.simulation_session / "jobs" / f"{row:03d}" / "console.log").read_text()
                                for row in range(2))
                self.assertTrue(any(observed_while_running), "No output reached terminal while child was running")
                self.assertEqual(terminal.getvalue(), saved)

    def test_unsaved_shared_edits_block_simulation_and_live_without_writes(self):
        before = {path: path.read_bytes() for path in (self.root / "robot_state").glob("*.yaml")}
        self.window.approved_simulation_inputs = self.window.queue_input_fingerprint()
        self.window.update_live_run_enabled()
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.robot_status_widget.gripper_status_edit.setText("held")
        self.assertFalse(self.window.simulate_button.isEnabled())
        self.assertFalse(self.window.run_button.isEnabled())
        with patch.object(QMessageBox, "warning") as warning, patch.object(QMessageBox, "question") as confirm:
            self.window.start_simulation()
            self.window.start_live_run()
        self.assertEqual(warning.call_count, 2)
        confirm.assert_not_called()
        self.assertEqual(HarmlessProcess.launches, [])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_simulation_does_not_reload_shared_editors_or_expand_minimal_yaml(self):
        robot = self.window.robot_status_widget
        track = self.window.track_status_widget
        before = {path: path.read_bytes() for path in (self.root / "robot_state").glob("*.yaml")}
        with patch.object(robot, "_load_robot_status", wraps=robot._load_robot_status) as robot_reload, patch.object(
            track, "_load_track_status", wraps=track._load_track_status
        ) as track_reload:
            self.window.start_simulation()
            self.wait_for_queue()
        robot_reload.assert_not_called()
        track_reload.assert_not_called()
        self.assertFalse(self.window.shared_state_dirty())
        self.assertTrue(self.window.run_button.isEnabled())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_live_finish_reloads_clean_shared_editors_without_saving(self):
        robot = self.window.robot_status_widget
        track = self.window.track_status_widget
        robot_path = Path(robot.robot_file_path)
        updated = yaml.safe_load(robot_path.read_text())
        updated["pipet_fluid_volume"] = 17
        robot_path.write_text(yaml.safe_dump(updated))
        before = robot_path.read_bytes()
        self.window.execution_mode = "live"
        with patch.object(robot, "_save_robot_status") as robot_save, patch.object(
            track, "_save_track_status"
        ) as track_save:
            self.window.finish_simulation_queue()
        self.assertEqual(robot.pipet_fluid_volume_spin.value(), 17)
        self.assertFalse(self.window.shared_state_dirty())
        self.assertEqual(robot_path.read_bytes(), before)
        robot_save.assert_not_called()
        track_save.assert_not_called()

    def test_live_finish_preserves_pending_shared_edits(self):
        robot = self.window.robot_status_widget
        robot.pipet_fluid_volume_spin.setValue(17)
        self.window.execution_mode = "live"
        with patch.object(robot, "_load_robot_status") as reload:
            self.window.finish_simulation_queue()
        reload.assert_not_called()
        self.assertEqual(robot.pipet_fluid_volume_spin.value(), 17)
        self.assertTrue(self.window.shared_state_dirty())
        self.assertFalse(self.window.simulate_button.isEnabled())

    def test_fatal_first_child_does_not_launch_second(self):
        config_path = Path(self.window.row_config_path(self.window.table.cellWidget(0, 0)))
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
        path = Path(self.window.row_config_path(self.window.table.cellWidget(0, 0)))
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
        path = Path(self.window.row_config_path(self.window.table.cellWidget(0, 0)))
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
        for row in range(self.window.table.rowCount()):
            path = Path(self.window.row_config_path(self.window.table.cellWidget(row, 0)))
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

    def test_repeated_workflow_configs_survive_simulation_and_live_preparation(self):
        alternate = self.root / "workflow_configs" / "first_other.yaml"
        alternate.write_text(yaml.safe_dump({
            "SIMULATE": True, "INPUT_VIAL_STATUS_FILE": "status/second.csv",
            "LIQUID": "glycerol", "VOLUME_TARGETS_ML": [0.2]
        }))
        second = self.window.table.cellWidget(1, 0)
        second.setCurrentIndex(second.findData("first"))
        self.window.set_row_config(second, alternate)
        before = alternate.read_bytes()
        self.assertEqual(len(self.window.vial_layout.occupancy), 2)
        self.window.start_simulation()
        self.wait_for_queue()
        simulation = self.window.simulation_session
        first_job = json.loads((simulation / "jobs/000/input.json").read_text())
        second_job = json.loads((simulation / "jobs/001/input.json").read_text())
        self.assertEqual(first_job["workflow"], second_job["workflow"])
        self.assertNotEqual(first_job["config_file"], second_job["config_file"])
        self.assertEqual(second_job["config"]["LIQUID"], "glycerol")
        self.assertTrue(self.window.run_button.isEnabled())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.window.start_live_run()
        self.wait_for_queue()
        live = json.loads((self.window.simulation_session / "jobs/001/input.json").read_text())
        self.assertEqual(live["config_file"], self.window.row_config_path(second))
        self.assertNotEqual(live["config_file"], str(alternate.resolve()))
        self.assertEqual(live["config"]["VOLUME_TARGETS_ML"], [0.2])
        self.assertIs(live["config"]["SIMULATE"], False)
        self.assertEqual(alternate.read_bytes(), before)

    def test_repeated_workflow_move_preserves_config_and_invalidates_approval(self):
        alternate = self.root / "workflow_configs" / "first_range.yaml"
        alternate.write_text(yaml.safe_dump({
            "SIMULATE": True, "INPUT_VIAL_STATUS_FILE": "status/first.csv",
            "VOLUME_TARGETS_ML": [0.01, 0.005]
        }))
        second = self.window.table.cellWidget(1, 0)
        second.setCurrentIndex(second.findData("first"))
        self.window.set_row_config(second, alternate)
        first_path = self.window.row_config_path(self.window.table.cellWidget(0, 0))
        second_path = self.window.row_config_path(second)
        second_id = second.property("job_id")
        self.assertEqual(len(self.window.vial_layout.occupancy), 1)
        self.window.start_simulation()
        self.wait_for_queue()
        before = self.window.queue_input_fingerprint()
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.table.selectRow(1)
        self.window.move_row(-1)
        self.assertEqual(self.window.row_config_path(self.window.table.cellWidget(0, 0)),
                         second_path)
        self.assertEqual(self.window.table.cellWidget(0, 0).property("job_id"), second_id)
        self.assertEqual(self.window.table.cellWidget(0, 0).property("config_source"), str(alternate.resolve()))
        self.assertEqual(self.window.row_config_path(self.window.table.cellWidget(1, 0)), first_path)
        self.assertNotEqual(before, self.window.queue_input_fingerprint())
        self.assertFalse(self.window.run_button.isEnabled())

    def test_new_rows_snapshot_default_once_and_explicit_loading_is_independent(self):
        source = self.root / "workflow_configs" / "first.yaml"
        before = source.read_bytes()
        selector = self.window.table.cellWidget(0, 0)
        first_path = Path(self.window.row_config_path(selector))
        config = yaml.safe_load(source.read_text())
        config["VOLUME_TARGETS_ML"] = [0.01]
        source.write_text(yaml.safe_dump(config))
        self.window.add_row("first")
        third = self.window.table.cellWidget(2, 0)
        third_path = Path(self.window.row_config_path(third))
        self.assertEqual(first_path.read_bytes(), before)
        self.assertEqual(yaml.safe_load(third_path.read_text())["VOLUME_TARGETS_ML"], [0.01])
        self.assertNotEqual(first_path, third_path)
        config["VOLUME_TARGETS_ML"] = [0.2]
        source.write_text(yaml.safe_dump(config))
        self.window.set_row_config(third, source)
        self.assertEqual(first_path.read_bytes(), before)
        self.assertEqual(yaml.safe_load(third_path.read_text())["VOLUME_TARGETS_ML"], [0.2])
        self.assertEqual(self.window.table.columnCount(), 5)

    def test_snapshot_edit_invalidates_approval_but_source_edit_does_not(self):
        alternate = self.root / "workflow_configs" / "first_extra.yaml"
        alternate.write_bytes((self.root / "workflow_configs" / "first.yaml").read_bytes())
        self.window.set_row_config(self.window.table.cellWidget(0, 0), alternate)
        self.window.start_simulation()
        self.wait_for_queue()
        self.assertTrue(self.window.run_button.isEnabled())
        config = yaml.safe_load(alternate.read_text())
        config["VOLUME_TARGETS_ML"] = [0.2]
        alternate.write_text(yaml.safe_dump(config))
        self.window.refresh_vial_layout()
        self.assertTrue(self.window.run_button.isEnabled())
        snapshot = Path(self.window.row_config_path(self.window.table.cellWidget(0, 0)))
        snapshot.write_text(yaml.safe_dump(config))
        self.window.refresh_vial_layout()
        self.assertFalse(self.window.run_button.isEnabled())

    def test_switch_to_missing_workflow_config_never_uses_previous_snapshot(self):
        selector = self.window.table.cellWidget(0, 0)
        previous = Path(self.window.row_config_path(selector))
        (self.root / "workflow_configs" / "second.yaml").unlink()
        selector.setCurrentIndex(selector.findData("second"))
        self.assertFalse(previous.exists())
        self.assertEqual(self.window.table.item(0, 2).text(), "Review setup")
        self.assertTrue(self.window.vial_layout.errors.text())
        with patch.object(QMessageBox, "warning") as blocked:
            self.window.start_simulation()
        blocked.assert_called_once()
        self.assertFalse(HarmlessProcess.launches)


if __name__ == "__main__":
    unittest.main()
