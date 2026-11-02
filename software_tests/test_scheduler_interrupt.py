import subprocess
import sys

from scheduler.interrupt import STOP_FILE


def test_stop_request_raises_keyboard_interrupt_in_child(tmp_path):
    child = subprocess.Popen([
        sys.executable, "-u", "-c",
        "import time, sys\nfrom scheduler.interrupt import watch_for_interrupt\n"
        "print('ready', flush=True)\n"
        "with watch_for_interrupt(sys.argv[1]):\n"
        " while True: time.sleep(0.02)",
        str(tmp_path),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        (tmp_path / STOP_FILE).touch()
        _, stderr = child.communicate(timeout=5)
        assert child.returncode != 0
        assert "KeyboardInterrupt" in stderr
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
