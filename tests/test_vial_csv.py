"""CSV regressions; exercise GUI methods without requiring Qt or robot drivers."""
import ast
import csv
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vial_csv import parse_vial_csv, read_vial_csv, save_vial_csv


def gui_method(name, **namespace):
    tree = ast.parse((ROOT / "vial_manager_gui.py").read_text(encoding="utf-8"))
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "VialManagerMainWindow")
    method = next(node for node in window.body if isinstance(node, ast.FunctionDef)
                  and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "vial_manager_gui.py"), "exec"), namespace)
    return namespace[name]


class VialCsvGuiTests(unittest.TestCase):
    def test_load_actual_fluorescence_file(self):
        messages = SimpleNamespace(critical=Mock(), warning=Mock())
        window = SimpleNamespace(_populate_racks=Mock(), _update_ui_state=Mock())
        load = gui_method("load_status_file", read_vial_csv=read_vial_csv, QMessageBox=messages)
        result = load(window, str(ROOT / "status/fluorescence_calibration_vials.csv"))
        self.assertTrue(result, messages.critical.call_args)
        with (ROOT / "status/fluorescence_calibration_vials.csv").open(newline="") as stream:
            self.assertEqual(window.original_vials_data, list(csv.DictReader(stream)))
        messages.critical.assert_not_called()

    def test_invalid_gui_save_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vials.csv"
            path.write_text("original contents")
            records = [dict(vial_index="0", vial_name="vial", location="main_8mL_rack",
                            location_index="0", vial_volume="nan")]
            window = SimpleNamespace(status_file_path=str(path), original_vials_data=records,
                                     original_fieldnames=list(records[0]),
                                     _validate_vial_positions=Mock(), status_bar=Mock())
            messages = SimpleNamespace(critical=Mock())
            save = gui_method("_save_file", save_vial_csv=save_vial_csv, QMessageBox=messages)
            self.assertFalse(save(window))
            self.assertEqual(path.read_text(), "original contents")
            self.assertIn("vial_volume", str(messages.critical.call_args))


class VialCsvTests(unittest.TestCase):
    fields = ["vial_index", "vial_name", "location", "location_index", "vial_volume", "notes"]

    def record(self):
        return dict(vial_index="0", vial_name="sample", location="main_8mL_rack",
                    location_index="0", vial_volume="1.2345678901234567", notes="")

    def serialize(self, records, delimiter=",", fields=None, quoting=csv.QUOTE_MINIMAL):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields or self.fields, delimiter=delimiter, quoting=quoting)
        writer.writeheader()
        writer.writerows(records)
        return stream.getvalue()

    def test_truncated_sample_does_not_break_valid_csv(self):
        record = self.record()
        record["vial_name"] = "a" * 1500
        text = self.serialize([record])
        with self.assertRaisesRegex(csv.Error, "Could not determine delimiter"):
            csv.Sniffer().sniff(text[:1024])
        self.assertEqual(parse_vial_csv(text), (self.fields, [record]))

    def test_supported_separators_bom_quotes_and_multiline_notes(self):
        record = self.record()
        record["notes"] = 'comma, semicolon; pipe| tab\t newline\nquoted "text"'
        for delimiter in (",", ";", "\t", "|"):
            for quoting in (csv.QUOTE_MINIMAL, csv.QUOTE_ALL):
                with self.subTest(delimiter=delimiter, quoting=quoting):
                    text = "\ufeff" + self.serialize([record], delimiter, quoting=quoting)
                    self.assertEqual(parse_vial_csv(text), (self.fields, [record]))

    def test_malformed_rows_headers_and_quotes_are_rejected(self):
        valid = self.serialize([self.record()])
        for text in (valid.rstrip("\r\n") + ",extra\n", valid.replace(",1.2345678901234567,", ","),
                     valid.replace("notes\r\n", "vial_name\r\n"), valid + '1,"unterminated',
                     "wrong,header\nvalue,value\n", self.serialize([])):
            with self.subTest(text=text):
                with self.assertRaises((ValueError, csv.Error)):
                    parse_vial_csv(text)

    def test_invalid_names_identities_and_volumes_are_rejected(self):
        for field, value in (("vial_name", ""), ("vial_index", ""), ("vial_volume", "nan"),
                             ("vial_volume", "inf"), ("vial_volume", "-1"), ("vial_volume", "abc")):
            record = self.record()
            record[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse_vial_csv(self.serialize([record]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_vial_csv(self.serialize([self.record(), self.record()]))

    def test_atomic_save_round_trip_preserves_extra_fields_and_order(self):
        record = self.record()
        record["notes"] = 'x,y\n"quoted"'
        record["custom"] = "custom value"
        fields = list(reversed(self.fields + ["custom"]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vials.csv"
            path.write_text("previous contents")
            save_vial_csv(path, fields, [record])
            self.assertEqual(read_vial_csv(path), (fields, [record]))
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_replace_preserves_original_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vials.csv"
            path.write_bytes(b"previous contents")
            with patch("vial_csv.os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    save_vial_csv(path, self.fields, [self.record()])
            self.assertEqual(path.read_bytes(), b"previous contents")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_verification_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vials.csv"
            path.write_bytes(b"previous contents")
            with patch("vial_csv.read_vial_csv", return_value=([], [])):
                with self.assertRaisesRegex(ValueError, "verification"):
                    save_vial_csv(path, self.fields, [self.record()])
            self.assertEqual(path.read_bytes(), b"previous contents")
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
