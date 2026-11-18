"""Fault injection for simulation: make the simulated C9 controller fail like the real one.

Off unless asked for. Then the stand-in controller raises a C9 error on the Nth move
command, shaped like the North SDK's (RuntimeError carrying .source and .error, see
north_c9.py send_packet), so error records, safe_stop and the codes list run exactly as
they would on the robot. After the fault every command fails, as on the real controller,
until a "power cycle": the next Lash_E start gets a fresh controller.

Ask for a fault with the environment variable (works across scheduler child processes)
or, in tests, the inject_fault() context manager:

    NORTH_SIM_FAULT=C9_MOTOR_FAULT:3     motor fault on the 3rd move command
"""
import os
from contextlib import contextmanager
from unittest.mock import DEFAULT, MagicMock

FAULT_ENV = "NORTH_SIM_FAULT"

# Commands that move an axis (counted to decide where the fault happens)
MOVE_COMMANDS = ("move_axis", "move_z", "goto_xy_safe", "goto_safe", "goto", "move_carousel", "move_xyz")

# Every controller command our code calls; all of them fail once the controller has faulted
ALL_COMMANDS = MOVE_COMMANDS + (
    "set_output", "delay", "get_axis_position", "set_pump_valve", "set_pump_speed", "open_gripper",
    "close_gripper", "move_pump", "dispense_ml", "aspirate_ml", "zero_scale", "uncap", "cap",
    "reduce_axis_position", "read_steady_scale", "read_scale", "open_clamp", "close_clamp",
    "home_robot", "home_carousel", "home_pump", "home_axis",
)

# (source name, C9 error number, message), copied from what the real controller sent
FAULTS = {
    "C9_MOTOR_FAULT": ("IO", 21, "IO: MOTOR FAULT: A critical error has occured on a motor driver, restart the controller"),
    "C9_HARD_ESTOP": ("IO", 8, "IO: SYSTEM ERROR: Hard E-Stop triggered, restart the system when it is safe to do so"),
}


class SimulatedC9Error:
    """Same shape as north_c9.C9Errors: .source, .error, and the controller's message."""

    def __init__(self, source, error, message):
        self.source, self.error, self.message = source, error, message

    def __str__(self):
        return self.message


def requested_fault():
    """(code, move number) from NORTH_SIM_FAULT, or None when no fault is asked for."""
    value = os.environ.get(FAULT_ENV, "").strip()
    if not value:
        return None
    code, _, after = value.partition(":")
    if code not in FAULTS or not after.isdigit() or int(after) < 1:
        raise ValueError(f"{FAULT_ENV} must look like C9_MOTOR_FAULT:3 (known: {', '.join(FAULTS)})")
    return code, int(after)


def simulated_c9():
    """The simulated controller: a plain MagicMock, or a faulty one if a fault is asked for."""
    controller = MagicMock()
    fault = requested_fault()
    if fault is None:
        return controller
    code, after = fault
    source, number, message = FAULTS[code]
    controller.fault_state = {"moves": 0, "faulted": False, "code": code, "after": after}

    def fail(*args, **kwargs):
        raise RuntimeError(SimulatedC9Error(source, number, message))

    def counted(name):
        def move(*args, **kwargs):
            if controller.fault_state["faulted"]:
                fail()
            controller.fault_state["moves"] += 1
            if controller.fault_state["moves"] >= after:
                controller.fault_state["faulted"] = True
                for command in ALL_COMMANDS:  # the controller is now down
                    getattr(controller, command).side_effect = fail
                fail()
            return DEFAULT  # behave like the plain stand-in
        return move

    for name in MOVE_COMMANDS:
        getattr(controller, name).side_effect = counted(name)
    return controller


@contextmanager
def inject_fault(code="C9_MOTOR_FAULT", after_moves=1):
    """For tests: every Lash_E started inside this block gets a controller that faults."""
    previous = os.environ.get(FAULT_ENV)
    os.environ[FAULT_ENV] = f"{code}:{after_moves}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FAULT_ENV, None)
        else:
            os.environ[FAULT_ENV] = previous
