"""Scheduler Check Deck gating. Camera/server are mocked; nothing touches hardware."""
import csv
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
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from scheduler import deck_check, scheduler_gui
from scheduler import simulation_state as state


def issue(kind, index, message="fix it"):
    return deck_check.Issue(kind, "main_8mL_rack", index, "first", "closed-cap", "empty", message)


class DeckCheckGuiTests(unittest.TestCase):
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
            yaml.safe_dump(yaml.safe_load((REPO_ROOT / "robot_state" / "vial_positions.yaml").read_text())))
        (self.root / "robot_state" / "robot_status.yaml").write_text(yaml.safe_dump({
            "gripper_status": None, "gripper_vial_index": None, "held_pipet_type": None,
            "pipet_fluid_vial_index": None, "pipet_fluid_volume": 0, "pipets_used": {"rack": 0}}))
        (self.root / "robot_state" / "track_status.yaml").write_text(yaml.safe_dump({"active_wellplate_position": None}))
        (self.root / "workflows" / "first.py").touch()
        (self.root / "workflow_configs" / "first.yaml").write_text(yaml.safe_dump({
            "SIMULATE": False, "INPUT_VIAL_STATUS_FILE": "status/first.csv"}))
        self.vial_file = self.root / "status" / "first.csv"
        self.vial_file.write_text("vial_name,location,location_index,vial_volume,capped,cap_type\n"
                                  "first,main_8mL_rack,0,2,True,closed\n")
        self.outcomes = []      # DeckChecks the mocked run_check returns, in order
        self.decisions = []     # (decision, note) the mocked dialog returns, in order
        self.seen_occupancy = []
        for item in (patch.object(scheduler_gui, "REPO_ROOT", self.root),
                     patch.object(state, "REPO_ROOT", self.root),
                     patch.object(deck_check, "run_check", side_effect=self.fake_run_check),
                     patch.object(scheduler_gui.SchedulerWindow, "ask_deck_check_decision",
                                  side_effect=lambda check: self.decisions.pop(0))):
            item.start()
            self.addCleanup(item.stop)
        self.window = scheduler_gui.SchedulerWindow()
        self.addCleanup(self.window.deleteLater)
        selector = self.window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData("first"))
        # Emulate a successful simulation of the unchanged queue.
        self.window.approved_simulation_inputs = self.window.queue_input_fingerprint()
        self.window.update_live_run_enabled()

    def fake_run_check(self, occupancy, records_root, settings):
        self.seen_occupancy.append(occupancy)
        check = self.outcomes.pop(0)
        folder = Path(records_root) / f"check_{len(self.seen_occupancy)}"
        folder.mkdir(parents=True)
        check.folder, check.timestamp = str(folder), folder.name
        return check

    def check(self, outcome="match", issues=(), error=None):
        return deck_check.DeckCheck(outcome=outcome, timestamp="", issues=list(issues), error=error, checked=1)

    def run_deck_check(self, *checks_and_decisions):
        for check, decision, *flags in checks_and_decisions:
            self.outcomes.append(check)
            self.decisions.append((decision, "note", flags[0] if flags else []))
        self.window.start_deck_check()
        loop, timer, watchdog = QEventLoop(), QTimer(), QTimer()
        timer.timeout.connect(lambda: loop.quit() if self.window.deck_check_worker is None and not self.outcomes else None)
        timer.start(10)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(loop.quit)
        watchdog.start(10000)
        loop.exec()
        timer.stop()
        self.assertIsNone(self.window.deck_check_worker, "deck check did not finish")

    def index_rows(self):
        return list(csv.DictReader((self.root / "scheduler" / "state" / "deck_checks" / "index.csv").open()))

    def test_run_requires_deck_check_and_match_enables_it(self):
        self.assertTrue(self.window.deck_check_button.isEnabled())
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertIn("Check Deck", self.window.run_button.toolTip())
        self.run_deck_check((self.check("match"), "continue"))
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIn(("main_8mL_rack", 0), self.seen_occupancy[0])
        self.assertEqual(self.index_rows()[0]["decision"], "continue")
        self.assertEqual(self.window.deck_check_run_label(), "Deck check: passed")

    def test_mismatch_cancel_blocks_run_marks_layout_and_is_saved(self):
        self.run_deck_check((self.check("mismatch", [issue("missing", 0, "Place the vial")]), "cancel"))
        self.assertFalse(self.window.run_button.isEnabled())
        slot = self.window.vial_layout.slots["main_8mL_rack", 0]
        self.assertTrue(slot.property("deck_mismatch"))
        self.assertIn("Place the vial", slot.toolTip())
        row = self.index_rows()[0]
        self.assertEqual((row["outcome"], row["decision"], row["issues"]), ("mismatch", "cancel", "1"))
        report = json.loads((Path(row["folder"]) / "report.json").read_text())
        self.assertEqual(report["issues"][0]["kind"], "missing")

    def test_mismatch_continue_enables_run_with_warning_label(self):
        self.run_deck_check((self.check("mismatch", [issue("wrong_cap", 0)]), "continue"))
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIn("1 MISMATCH(ES) ACKNOWLEDGED", self.window.deck_check_run_label())

    def test_all_camera_errors_continue_and_are_saved_for_training(self):
        self.run_deck_check((self.check("mismatch", [issue("unexpected", 47), issue("unexpected", 41)]),
                             "continue", [0, 1]))
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertEqual(self.window.deck_check_run_label(), "Deck check: passed (2 camera error(s) flagged)")
        row = self.index_rows()[0]
        self.assertEqual(row["camera_wrong"], "2")
        errors = self.root / "scheduler" / "state" / "deck_checks" / "camera_errors"
        text = next(errors.glob("*.txt")).read_text()
        self.assertIn("main_8mL_rack [47]", text)
        self.assertIn("Note: note", text)

    def test_comms_failure_can_be_continued(self):
        self.run_deck_check((self.check("comms_failure", error="camera capture failed: x"), "continue"))
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertIn("COULD NOT RUN", self.window.deck_check_run_label())
        self.assertEqual(self.index_rows()[0]["outcome"], "comms_failure")

    def test_recheck_runs_again(self):
        self.run_deck_check((self.check("mismatch", [issue("missing", 0)]), "recheck"),
                            (self.check("match"), "continue"))
        self.assertEqual(len(self.seen_occupancy), 2)
        self.assertEqual([r["decision"] for r in self.index_rows()], ["recheck", "continue"])
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertFalse(self.window.vial_layout.slots["main_8mL_rack", 0].property("deck_mismatch"))

    def test_vial_file_edit_invalidates_check(self):
        self.run_deck_check((self.check("mismatch", [issue("missing", 0)]), "continue"))
        self.assertTrue(self.window.deck_check_resolved())
        self.vial_file.write_text("vial_name,location,location_index,vial_volume,capped,cap_type\n"
                                  "first,main_8mL_rack,1,2,True,closed\n")
        self.window.approved_simulation_inputs = self.window.queue_input_fingerprint()
        self.window.refresh_vial_layout()
        self.assertFalse(self.window.deck_check_resolved())
        self.assertFalse(self.window.run_button.isEnabled())
        self.assertFalse(self.window.vial_layout.slots["main_8mL_rack", 0].property("deck_mismatch"))

    def test_live_run_consumes_check_and_records_it(self):
        self.run_deck_check((self.check("match"), "continue"))
        folder = Path(self.index_rows()[0]["folder"])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as question, \
                patch.object(self.window, "launch_next_simulation"):
            self.window.start_live_run()
        self.assertIn("Deck check: passed", question.call_args[0][2])
        self.assertIsNone(self.window.resolved_deck_check)
        self.assertIn("live_", json.loads((folder / "report.json").read_text())["live_run"])


class DeckCheckDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mismatch_continue_needs_acknowledgement(self):
        check = deck_check.DeckCheck("mismatch", "t", issues=[issue("missing", 0)])
        dialog = scheduler_gui.DeckCheckDialog(check)
        self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.continue_button.isEnabled())
        dialog.acknowledge.setChecked(True)
        self.assertTrue(dialog.continue_button.isEnabled())
        dialog.continue_button.click()
        self.assertEqual(dialog.decision, "continue")

    def test_marking_every_issue_camera_wrong_enables_continue(self):
        from PySide6.QtCore import Qt
        check = deck_check.DeckCheck("mismatch", "t", issues=[issue("unexpected", 47), issue("wrong_cap", 6)])
        dialog = scheduler_gui.DeckCheckDialog(check)
        self.addCleanup(dialog.deleteLater)
        table = dialog.issue_table
        table.item(0, 3).setCheckState(Qt.Checked)
        self.assertEqual(dialog.camera_wrong(), [0])
        self.assertFalse(dialog.continue_button.isEnabled())   # one real issue left
        table.item(1, 3).setCheckState(Qt.Checked)
        self.assertTrue(dialog.continue_button.isEnabled())
        self.assertEqual(dialog.continue_button.text(), "Continue (camera errors)")
        table.item(1, 3).setCheckState(Qt.Unchecked)
        dialog.acknowledge.setChecked(True)                    # partial flags + deck checked also works
        self.assertTrue(dialog.continue_button.isEnabled())
        self.assertEqual(dialog.continue_button.text(), "Continue anyway")

    def test_comms_failure_continue_needs_no_tick(self):
        dialog = scheduler_gui.DeckCheckDialog(deck_check.DeckCheck("comms_failure", "t", error="down"))
        self.addCleanup(dialog.deleteLater)
        self.assertIsNone(dialog.acknowledge)
        self.assertTrue(dialog.continue_button.isEnabled())
        self.assertIn("down", dialog.findChildren(scheduler_gui.QLabel)[0].text())


if __name__ == "__main__":
    unittest.main()
