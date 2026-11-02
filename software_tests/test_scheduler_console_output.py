import io
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from scheduler.console_output import ConsoleLogMirror


class ConsoleOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "console.log"

    def test_split_unicode_and_windows_newline_and_final_tail(self):
        stream = io.StringIO()
        mirror = ConsoleLogMirror(self.path, stream)
        self.assertFalse(mirror.poll())  # Child has not created its log yet.
        self.path.write_bytes(b"volume \xce")
        mirror.poll()
        with self.path.open("ab") as log:
            log.write(b"\xbcL\r")
        mirror.poll()
        with self.path.open("ab") as log:
            log.write(b"\nfinal tail")
        mirror.finish()
        self.assertEqual(stream.getvalue(), "volume \u03bcL\nfinal tail")
        self.assertEqual(self.path.read_bytes(), "volume \u03bcL\r\nfinal tail".encode())

    def test_closed_terminal_does_not_affect_saved_log(self):
        stream = io.StringIO()
        stream.close()
        mirror = ConsoleLogMirror(self.path, stream)
        self.path.write_bytes(b"still saved\n")
        mirror.poll()
        mirror.finish()
        self.assertEqual(self.path.read_bytes(), b"still saved\n")

    def test_finish_drains_more_than_one_chunk(self):
        stream = io.StringIO()
        mirror = ConsoleLogMirror(self.path, stream)
        output = "x" * 150000
        self.path.write_text(output, encoding="utf-8")
        mirror.finish()
        self.assertEqual(stream.getvalue(), output)
