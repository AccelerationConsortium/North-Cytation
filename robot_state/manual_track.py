"""Manual track I/O, independent of workflow initialization and inventory writes."""
import math
import queue
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class BoundedSDK:
    """Bound even SDK calls whose internal serial wait has no timeout.

    A timed-out request may still be outstanding. Quarantine that connection
    instead of issuing subsequent commands concurrently with it.
    """
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.failed = False

    def call(self, label, function, *args, **kwargs):
        if self.failed:
            raise RuntimeError("Connection has an unresolved request. Restart Track Control after checking the controller.")
        result = queue.Queue(maxsize=1)

        def invoke():
            try:
                result.put((True, function(*args, **kwargs)))
            except BaseException as error:
                result.put((False, error))

        threading.Thread(target=invoke, daemon=True).start()
        try:
            success, value = result.get(timeout=self.timeout)
        except queue.Empty:
            self.failed = True
            raise TimeoutError(f"{label}: no SDK reply within {self.timeout:g}s. "
                               "The physical result is unknown; no further commands will be sent on this connection.") from None
        if not success:
            raise RuntimeError(f"{label}: {value}") from value
        return value


class OperationRunner:
    """One background operation; the UI drains events on its own thread."""
    def __init__(self):
        self.events = queue.Queue()
        self.busy = False

    def start(self, label, function):
        if self.busy:
            return False
        self.busy = True

        def run():
            try:
                value = function(lambda text: self.events.put(("progress", text)))
                self.events.put(("done", (label, value)))
            except BaseException as error:
                self.events.put(("error", (label, str(error))))

        threading.Thread(target=run, daemon=True).start()
        return True


class ManualTrack:
    def __init__(self, controller, positions, hardware, sdk=None, motion_timeout=30):
        self.controller = controller
        self.positions = positions
        self.hardware = hardware
        self.sdk = sdk or BoundedSDK()
        self.motion_timeout = motion_timeout
        self.cancel = threading.Event()

    @classmethod
    def connect(cls, progress, factory=None, root=ROOT, sdk=None):
        positions = yaml.safe_load((root / "robot_state/track_positions.yaml").read_text())
        hardware = yaml.safe_load((root / "robot_state/robot_hardware.yaml").read_text())
        sdk = sdk or BoundedSDK()
        if factory is None:
            progress("Loading North SDK")
            from north import NorthC9
            factory = NorthC9
        progress("Opening controller A / USB AU06CNCF (no startup motion)")
        controller = sdk.call("Open controller", factory, "A", network_serial="AU06CNCF")
        track = cls(controller, positions, hardware, sdk)
        # Connection success means the actual track axes answered.
        track.read_position(progress)
        return track

    def axis(self, name):
        return self.hardware["track_axes"][name + "_axis"]

    def read_position(self, progress):
        values = []
        for name in ("x", "z"):
            progress(f"Reading {name.upper()} encoder (axis {self.axis(name)})")
            value = self.sdk.call(f"Read {name.upper()} encoder", self.controller.get_axis_position, self.axis(name))
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {name.upper()} encoder reading: {value!r}")
            values.append(value)
        return tuple(values)

    def _move_axis(self, name, target, progress):
        if self.cancel.is_set():
            raise RuntimeError("Operation stopped before the next movement")
        axis = self.axis(name)
        progress(f"Moving {name.upper()} axis {axis} to {target:g} counts")
        speed = self.positions["speeds"]["default_" + name]
        token = self.sdk.call(f"Send {name.upper()} move", self.controller.move_axis,
                              axis, target, vel=speed, wait=False)
        deadline = time.monotonic() + self.motion_timeout
        while True:
            if self.cancel.is_set():
                raise RuntimeError("Stop requested; movement completion is unconfirmed")
            if self.sdk.call(f"Wait for {name.upper()} move", token.is_done):
                return
            # The SDK token otherwise polls forever even when the axis faults.
            state = self.sdk.call(f"Read {name.upper()} status", self.controller.get_axis_status, axis)
            if state == 9:  # North SDK AxisState.ERROR
                raise RuntimeError(f"{name.upper()} axis {axis} reports an axis error")
            if time.monotonic() >= deadline:
                try:
                    self.stop(progress)
                finally:
                    self.sdk.failed = True
                raise TimeoutError(f"{name.upper()} axis {axis} did not complete within {self.motion_timeout:g}s. "
                                   "A quick stop was requested. Check the controller and restart Track Control before continuing.")
            self.cancel.wait(0.05)

    def jog(self, name, delta, progress):
        x, z = self.read_position(progress)
        target = (x if name == "x" else z) + delta
        if name == "z" and target < 0:
            raise ValueError("Cannot move above Z=0")
        self._move_axis(name, target, progress)
        return self.read_position(progress)

    def move_to(self, x, z, progress):
        if z < 0:
            raise ValueError("Cannot move above Z=0")
        self._move_axis("z", 0, progress)
        self._move_axis("x", x, progress)
        if z != 0:
            self._move_axis("z", z, progress)
        return self.read_position(progress)

    def set_gripper(self, opened, progress):
        progress("Opening gripper" if opened else "Closing gripper")
        outputs = self.hardware["track_pneumatics"]
        for name, value in (("gripper_open", opened), ("gripper_close", not opened)):
            self.sdk.call(f"Set {name}", self.controller.set_output, outputs[name], value)
        self.cancel.wait(self.positions.get("timing", {}).get("gripper_delay", 2))
        if self.cancel.is_set():
            raise RuntimeError("Stop requested during gripper operation; inspect its state")
        return self.read_position(progress)

    def stop(self, progress):
        self.cancel.set()
        progress("Sending SDK quick-stop request (not the physical E-stop)")
        self.sdk.call("Quick stop", self.controller.quick_stop)
