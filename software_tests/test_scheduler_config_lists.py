import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scheduler.config_lists import (config_fingerprint, live_experiment_id, read_config_entries,
                                    read_config_list, read_planned_consumables, read_planned_queues,
                                    write_config_list)


class ConfigListTests(unittest.TestCase):
    def test_save_round_trip_keeps_order_and_independent_configuration_copies(self):
        second = self.root / "second.yaml"
        second.write_text("PREPARE_ONLY: false\n", encoding="utf-8")
        write_config_list(self.path, [("demo", second), ("demo", self.config)])
        jobs = read_config_list(self.path, ["demo"])
        self.assertEqual([name for name, _ in jobs], ["demo", "demo"])
        self.assertEqual([yaml.safe_load(p.read_bytes()) for _, p in jobs],
                         [{"PREPARE_ONLY": False}, {"PREPARE_ONLY": True}])
        second.unlink()
        self.config.write_text("PREPARE_ONLY: changed\n", encoding="utf-8")
        write_config_list(self.path, [("demo", self.config)])
        self.assertEqual(yaml.safe_load(jobs[0][1].read_bytes()), {"PREPARE_ONLY": False})
        self.assertEqual(yaml.safe_load(jobs[1][1].read_bytes()), {"PREPARE_ONLY": True})

    def test_invalid_save_does_not_replace_existing_queue(self):
        write_config_list(self.path, [("demo", self.config)])
        before = self.path.read_bytes()
        for jobs in ([], [("demo", None)], [("demo", self.root / "missing.yaml")]):
            with self.subTest(jobs=jobs), self.assertRaises((ValueError, OSError)):
                write_config_list(self.path, jobs)
            self.assertEqual(self.path.read_bytes(), before)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "queue.yaml"
        self.config = self.root / "prepare.yaml"
        self.config.write_text("PREPARE_ONLY: true\n", encoding="utf-8")

    def test_order_duplicates_and_relative_and_absolute_paths(self):
        execution = self.root / "execute.yaml"
        execution.write_text("PREPARE_ONLY: false\n", encoding="utf-8")
        self.path.write_text(yaml.safe_dump([
            {"workflow": "demo", "config_file": "prepare.yaml"},
            {"workflow": "demo", "config_file": str(execution)},
            {"workflow": "demo", "config_file": "prepare.yaml"},
        ]), encoding="utf-8")
        self.assertEqual(read_config_list(self.path, ["demo"]), [
            ("demo", self.config), ("demo", execution), ("demo", self.config),
        ])

    def test_invalid_lists_are_rejected_with_entry_details(self):
        for content, message in (
            ([], "nonempty YAML list"),
            ({"workflow": "demo"}, "nonempty YAML list"),
            ([{"workflow": "demo"}], "Entry 1"),
            ([{"workflow": "missing", "config_file": "prepare.yaml"}], "unknown workflow"),
            ([{"workflow": "demo", "config_file": None}], "nonempty path"),
            ([{"workflow": "demo", "config_file": "absent.yaml"}], "Entry 1"),
            ([{"workflow": "demo", "config_file": "prepare.yaml"},
              {"workflow": "demo", "config_file": "absent.yaml"}], "Entry 2"),
        ):
            with self.subTest(content=content):
                self.path.write_text(yaml.safe_dump(content), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_config_list(self.path, ["demo"])

    def test_invalid_config_mapping_is_rejected(self):
        self.path.write_text("- workflow: demo\n  config_file: prepare.yaml\n", encoding="utf-8")
        self.config.write_text("- not-a-config\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "YAML mapping"):
            read_config_list(self.path, ["demo"])


class PlannedQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "queue.yaml"
        (self.root / "configs").mkdir()
        self.config = self.root / "configs" / "00.yaml"
        self.config.write_text("PREPARE_ONLY: true\n", encoding="utf-8")

    def test_planner_keys_load_and_unknown_keys_are_rejected(self):
        self.path.write_text(yaml.safe_dump([
            {"workflow": "demo", "config_file": "configs/00.yaml", "experiment_id": "EXP-1",
             "config_sha256": "abc", "label": "First run"},
            {"workflow": "demo", "config_file": "configs/00.yaml", "experiment_id": None},
        ]), encoding="utf-8")
        entries = read_config_entries(self.path, ["demo"])
        self.assertEqual(entries[0], {"workflow": "demo", "config_path": self.config, "experiment_id": "EXP-1",
                                      "config_sha256": "abc", "label": "First run"})
        self.assertEqual((entries[1]["experiment_id"], entries[1]["config_sha256"], entries[1]["label"]),
                         (None, None, None))
        self.assertEqual(read_config_list(self.path, ["demo"]), [("demo", self.config)] * 2)
        for extra, message in (({"priority": 1}, "Entry 1"), ({"experiment_id": 5}, "text or null")):
            with self.subTest(extra=extra):
                self.path.write_text(yaml.safe_dump([
                    {"workflow": "demo", "config_file": "configs/00.yaml", **extra}]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_config_entries(self.path, ["demo"])

    def test_fingerprint_matches_lab_agent_and_ignores_unhashed_keys(self):
        self.assertEqual(config_fingerprint({"A": 1, "SIMULATE": True}), hashlib.sha256(b'{"A": 1}').hexdigest())
        base = {"A": 1, "B": [1, 2]}
        reference = config_fingerprint(base)
        self.assertEqual(config_fingerprint({**base, "INPUT_VIAL_STATUS_FILE": "x.csv", "SIMULATE": False}), reference)
        self.assertNotEqual(config_fingerprint({**base, "A": 2}), reference)
        self.assertNotEqual(config_fingerprint({**base, "C": None}), reference)

    def test_live_experiment_id(self):
        config = {"A": 1, "SIMULATE": False}
        planned = config_fingerprint(config)
        random_id = live_experiment_id(None, None, config)
        self.assertRegex(random_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(random_id, live_experiment_id(None, None, config))
        self.assertEqual(live_experiment_id("EXP-1", planned, {**config, "SIMULATE": True}), "EXP-1")
        changed = {"A": 2}
        self.assertEqual(live_experiment_id("EXP-1", planned, changed),
                         f"EXP-1-changed-{config_fingerprint(changed)[:8]}")

    def test_read_planned_queues(self):
        index = self.root / "index.json"
        self.assertEqual(read_planned_queues(index), [])
        index.write_text(json.dumps({"updated_at": "now", "queues": [{"setup_day": "2026-11-09"}]}), encoding="utf-8")
        self.assertEqual(read_planned_queues(index), [{"setup_day": "2026-11-09"}])
        with patch.dict(os.environ, {"LAB_AGENT_QUEUE_INDEX": str(index)}):
            self.assertEqual(read_planned_queues(), [{"setup_day": "2026-11-09"}])
        index.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "queue index"):
            read_planned_queues(index)

    def test_read_planned_consumables(self):
        self.assertIsNone(read_planned_consumables(self.path))
        (self.root / "manifest.json").write_text(json.dumps({"plate_types": ["96 WELL PLATE"], "jobs": [
            {"plates": 1, "tips": {"small": 7, "large": 17}}, {"plates": 0, "tips": {"small": 2}}, {}]}),
            encoding="utf-8")
        self.assertEqual(read_planned_consumables(self.path), {
            "plates_per_job": [1, 0, 0], "plate_types": ["96 WELL PLATE"], "tips": {"small": 9, "large": 17}})
        (self.root / "manifest.json").write_text("{broken", encoding="utf-8")
        self.assertIsNone(read_planned_consumables(self.path))
