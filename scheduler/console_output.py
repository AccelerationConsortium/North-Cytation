"""Mirror a child's saved console log without putting its output in a pipe."""

import codecs
from pathlib import Path

from PySide6.QtCore import QObject, QTimer


class ConsoleLogMirror(QObject):
    def __init__(self, path, stream, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.stream = stream
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending_cr = ""
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def _write(self, text):
        if not text or self.stream is None:
            return
        try:
            try:
                self.stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self.stream.write(text.encode(encoding, errors="backslashreplace").decode(encoding))
            self.stream.flush()
        except (OSError, ValueError):
            # A closed terminal must not interrupt the workflow or its saved log.
            self.stream = None

    def poll(self):
        if self.stream is None:
            return False
        try:
            with self.path.open("rb") as log:
                log.seek(self.offset)
                chunk = log.read(65536)
        except OSError:
            return False
        self.offset += len(chunk)
        text = self.pending_cr + self.decoder.decode(chunk)
        self.pending_cr = "\r" if text.endswith("\r") else ""
        if self.pending_cr:
            text = text[:-1]
        self._write(text.replace("\r\n", "\n"))
        return bool(chunk)

    def finish(self):
        self.timer.stop()
        while self.poll():
            pass
        self._write((self.pending_cr + self.decoder.decode(b"", final=True)).replace("\r\n", "\n"))
        self.deleteLater()
