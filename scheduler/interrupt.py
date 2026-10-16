"""Deliver a scheduler stop request as KeyboardInterrupt in a child process."""

import _thread
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread


STOP_FILE = "interrupt.requested"


@contextmanager
def watch_for_interrupt(job_folder):
    request = Path(job_folder) / STOP_FILE
    finished = Event()

    def watch():
        while not finished.wait(0.1):
            if request.exists():
                _thread.interrupt_main()
                return

    thread = Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        finished.set()
        thread.join(timeout=0.2)
