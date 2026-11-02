import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
import scheduler_gui


class SchedulerRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_only_completed_real_runs_contribute_to_median(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment_tracking").mkdir()
            path = root / "experiment_tracking" / "experiment_runs.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["workflow_name", "simulate", "status", "duration_sec"])
                writer.writerows([
                    ["demo", "False", "completed", 3600],
                    ["demo.py", "False", "completed", 7200],
                    [r"workflows\demo.py", "false", "completed", 999999],
                    ["demo", "True", "completed", 1],
                    ["demo", "False", "failed", 2],
                    ["demo", "False", "unknown_backfilled", 3],
                    ["demo", "", "completed", 4],
                    ["demo", "False", "completed", "nan"],
                    ["demo", "False", "completed", -1],
                    ["demo", "False", "completed", ""],
                ])
            with patch.object(scheduler_gui, "REPO_ROOT", root):
                estimates, warning = scheduler_gui.read_runtime_estimates()
            self.assertEqual(estimates["demo"], (7200.0, 3, 3600.0, 999999.0))
            self.assertIn("Skipped 3", warning)

    def test_missing_or_malformed_log_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(scheduler_gui, "REPO_ROOT", root):
                estimates, error = scheduler_gui.read_runtime_estimates()
                self.assertFalse(estimates)
                self.assertIn("Cannot read", error)
                (root / "experiment_tracking").mkdir()
                (root / "experiment_tracking" / "experiment_runs.csv").write_text("wrong_header\n")
                estimates, error = scheduler_gui.read_runtime_estimates()
                self.assertFalse(estimates)
                self.assertIn("requires", error)

    def test_selection_reorder_refresh_and_no_history(self):
        first = "surfactant_grid_ailsa"
        second = "fluorescence_calibration_workflow"
        history = {first: (7200, 3, 3600, 9000)}
        with patch.object(scheduler_gui, "read_runtime_estimates", return_value=(history, "")):
            window = scheduler_gui.SchedulerWindow()
        self.addCleanup(window.deleteLater)
        selector = window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData(first))
        self.assertEqual(window.table.item(0, 4).text(), "2 h 00 min")
        self.assertIn("3 completed", window.table.item(0, 4).toolTip())
        window.add_row(second)
        self.assertEqual(window.table.item(1, 4).text(), "No history")
        window.table.selectRow(0)
        window.move_row(1)
        self.assertEqual(window.table.item(1, 4).text(), "2 h 00 min")
        self.assertEqual(window.table.item(0, 4).text(), "No history")
        with patch.object(scheduler_gui, "read_runtime_estimates", return_value=({second: (1800, 1, 1800, 1800)}, "")):
            window.refresh_runtime_estimates()
        self.assertEqual(window.table.item(0, 4).text(), "30 min")
        self.assertEqual(window.table.item(1, 4).text(), "No history")


if __name__ == "__main__":
    unittest.main()