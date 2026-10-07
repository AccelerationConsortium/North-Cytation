import csv
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
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
import scheduler_gui


class SchedulerVialLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("workflows", "workflow_configs", "status", "robot_state"):
            (self.root / folder).mkdir()
        self.areas = yaml.safe_load((REPO_ROOT / "robot_state" / "vial_positions.yaml").read_text())
        (self.root / "robot_state" / "vial_positions.yaml").write_text(yaml.safe_dump(self.areas))
        self.root_patch = patch.object(scheduler_gui, "REPO_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.fields = ["vial_name", "location", "location_index", "vial_volume", "home_location", "home_location_index"]

    def workflow(self, name, records):
        (self.root / "workflows" / f"{name}.py").write_text("raise RuntimeError('Must not import workflow')\n")
        config = self.root / "workflow_configs" / f"{name}.yaml"
        config.write_text(yaml.safe_dump({"INPUT_VIAL_STATUS_FILE": f"status/{name}.csv"}))
        self.write_records(name, records)

    def write_records(self, name, records):
        with (self.root / "status" / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(records)

    def record(self, name, location, index):
        return dict(vial_name=name, location=location, location_index=index, vial_volume="2.5",
                    home_location="clamp", home_location_index=0)

    def test_current_clamp_conflict_and_shared_home_ignored(self):
        self.workflow("first", [self.record("first vial", "main_8mL_rack", 0)])
        self.workflow("second", [self.record("second vial", "main_8mL_rack", 1)])
        view = scheduler_gui.VialLayout()
        self.addCleanup(view.deleteLater)
        view.refresh(["first", "second"])
        self.assertEqual(view.slots["clamp", 0].property("occupancy"), "empty")
        self.assertEqual(view.slots["main_8mL_rack", 0].property("occupancy"), "occupied")
        self.write_records("first", [self.record("first vial", "clamp", 0)])
        self.write_records("second", [self.record("second vial", "clamp", "0.0")])
        view.refresh(["first", "second"])
        clamp = view.slots["clamp", 0]
        self.assertEqual(clamp.property("occupancy"), "conflict")
        self.assertIn("first vial", clamp.toolTip())
        self.assertIn("second vial", clamp.toolTip())
        self.assertIn("second.csv", clamp.toolTip())
        self.assertEqual(view.slots["main_8mL_rack", 0].property("occupancy"), "empty")

    def test_selection_removal_refresh_and_no_editable_slots(self):
        self.workflow("first", [self.record("vial", "heater", 2)])
        self.workflow("second", [self.record("other", "heater", 2)])
        window = scheduler_gui.SchedulerWindow()
        self.addCleanup(window.deleteLater)
        selector = window.table.cellWidget(0, 0)
        selector.setCurrentIndex(selector.findData("first"))
        self.assertEqual(window.vial_layout.slots["heater", 2].property("occupancy"), "occupied")
        window.add_row("second")
        self.assertEqual(window.vial_layout.slots["heater", 2].property("occupancy"), "conflict")
        window.remove_row()
        self.assertEqual(window.vial_layout.slots["heater", 2].property("occupancy"), "occupied")
        self.write_records("first", [self.record("vial", "large_vial_rack", 3)])
        window.vial_layout.refresh_button.click()
        self.assertEqual(window.vial_layout.slots["heater", 2].property("occupancy"), "empty")
        self.assertEqual(window.vial_layout.slots["large_vial_rack", 3].property("occupancy"), "occupied")
        self.assertTrue(all(type(slot) is QLabel for slot in window.vial_layout.slots.values()))
        self.assertTrue(all(not button.isEnabled() for button in window.findChildren(QPushButton)
                            if button.text() == "Run"))

    def test_invalid_records_and_missing_config_are_reported(self):
        self.workflow("first", [self.record("bad", "unknown", 0), self.record("bad slot", "heater", 12),
                                self.record("fraction", "heater", "0.5"), self.record("valid", "clamp", 0)])
        claims, errors = scheduler_gui.read_vial_occupancy(["first", "missing"], self.areas)
        self.assertEqual(len(errors), 4)
        self.assertEqual(set(claims), {("clamp", 0)})
        (self.root / "status" / "first.csv").write_text("vial_name\nvial\n")
        claims, errors = scheduler_gui.read_vial_occupancy(["first"], self.areas)
        self.assertFalse(claims)
        self.assertIn("requires", errors[0])

    def test_same_csv_counted_once_and_duplicate_rows_detected(self):
        self.workflow("first", [self.record("vial", "clamp", 0)])
        self.workflow("second", [])
        (self.root / "workflow_configs" / "second.yaml").write_text(
            yaml.safe_dump({"INPUT_VIAL_STATUS_FILE": "status/first.csv"})
        )
        claims, errors = scheduler_gui.read_vial_occupancy(["first", "first", "second"], self.areas)
        self.assertFalse(errors)
        self.assertEqual(len(claims["clamp", 0]), 1)
        self.assertEqual(claims["clamp", 0][0]["workflows"], ["first", "second"])
        self.write_records("first", [self.record("vial", "clamp", 0), self.record("other", "clamp", 0)])
        claims, errors = scheduler_gui.read_vial_occupancy(["first"], self.areas)
        self.assertEqual(len(claims["clamp", 0]), 2)

    def test_bad_area_configuration_is_visible(self):
        (self.root / "robot_state" / "vial_positions.yaml").write_text("{}")
        view = scheduler_gui.VialLayout()
        self.addCleanup(view.deleteLater)
        self.assertEqual(view.summary.text(), "Layout unavailable")
        self.assertIn("Cannot load", view.errors.text())


if __name__ == "__main__":
    unittest.main()