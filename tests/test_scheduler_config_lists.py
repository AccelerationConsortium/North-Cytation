import tempfile
import unittest
from pathlib import Path

import yaml

from scheduler.config_lists import read_config_list, write_config_list


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
