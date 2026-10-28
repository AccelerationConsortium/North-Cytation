"""Manual track integration tests: delayed SDK replies, no real hardware or Tk."""
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import track_control_gui as gui
from robot_state.manual_track import BoundedSDK, ManualTrack, OperationRunner, ROOT


@pytest.fixture
def track():
    positions = yaml.safe_load((ROOT / "robot_state/track_positions.yaml").read_text())
    positions["timing"]["gripper_delay"] = 0
    hardware = yaml.safe_load((ROOT / "robot_state/robot_hardware.yaml").read_text())
    device = Mock()
    coordinates = {7: 120000, 6: 75000}
    device.get_axis_position.side_effect = coordinates.__getitem__
    device.get_axis_status.return_value = 7
    token = Mock()
    token.is_done.return_value = True

    def move(axis, target, **kwargs):
        coordinates[axis] = target
        return token

    device.move_axis.side_effect = move
    return ManualTrack(device, positions, hardware, BoundedSDK(timeout=0.5)), device, coordinates, token


@pytest.fixture
def controller(monkeypatch, track):
    monkeypatch.setattr(gui.messagebox, "askyesno", Mock(return_value=True))
    monkeypatch.setattr(gui.messagebox, "showerror", Mock())
    control = gui.TrackControlGUI.__new__(gui.TrackControlGUI)
    control.simulation_mode = False
    control.hardware = track[0]
    control.runner = OperationRunner()
    control.stop_runner = OperationRunner()
    control.root = Mock()
    for label in ("mode_label", "connection_label", "operation_label"):
        setattr(control, label, Mock())
    control.current_x, control.current_z = 0, 0
    control.UNITS_PER_MM = 100
    control.x_increment_var = Mock(get=Mock(return_value=1))
    control.z_increment_var = Mock(get=Mock(return_value=1))
    control.gripper_open = None
    control.position_history = []
    control._closed = False
    control._stage = "Ready"
    control._position_known = True
    control._operation_done = None
    control.update_position_display = Mock()
    return control


def drain(control, timeout=2):
    deadline = time.monotonic() + timeout
    while control.runner.busy or control.stop_runner.busy:
        control._poll_operations()
        assert time.monotonic() < deadline, "Worker failed to deliver a completion/error event"
        time.sleep(0.005)


def test_zero_preset_retracts_then_moves_x(controller, track):
    control, device = controller, track[1]
    control.go_to_preset("Source", 200, 0)
    drain(control)
    assert device.move_axis.call_args_list == [call(6, 0, vel=50, wait=False), call(7, 200, vel=50, wait=False)]
    assert (control.current_x, control.current_z) == (200, 0)
    assert control.position_history[-1][-1] == "Source"


def test_lower_preset_moves_in_order(controller, track):
    controller.go_to_preset("Active", 131650, 75000)
    drain(controller)
    assert track[1].move_axis.call_args_list == [
        call(6, 0, vel=50, wait=False), call(7, 131650, vel=50, wait=False), call(6, 75000, vel=50, wait=False)]
    assert (controller.current_x, controller.current_z) == (131650, 75000)


def test_preset_callback_returns_while_hardware_is_waiting(controller, track):
    entered, release = threading.Event(), threading.Event()
    original = track[1].move_axis.side_effect

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(1)
        return original(*args, **kwargs)

    track[1].move_axis.side_effect = delayed
    try:
        started = time.monotonic()
        controller.go_to_preset("Source", 200, 0)
        assert time.monotonic() - started < 0.1, "Hardware waits block the Tk event loop"
        assert entered.wait(0.5)
        # The event pump still runs while hardware is held; another command is not queued.
        controller._poll_operations()
        controller.move_x(1)
        assert track[1].move_axis.call_count == 1
        assert controller.runner.busy
    finally:
        release.set()
        drain(controller)


def test_refresh_reads_encoders_without_inventory_reload(controller, track):
    controller.refresh_track_status()
    drain(controller)
    assert (controller.current_x, controller.current_z) == (120000, 75000)
    track[1].get_track_status.assert_not_called()


def test_jog_uses_encoder_not_simulation_guess(controller, track):
    controller.current_x = 68325
    controller.move_x(1)
    drain(controller)
    track[1].move_axis.assert_called_once_with(7, 120100, vel=50, wait=False)
    assert controller.current_x == 120100


def test_z_jog_sign_and_top_limit(controller, track):
    controller.move_z(1)
    drain(controller)
    assert controller.current_z == 74900
    track[2][6] = 0
    track[1].move_axis.reset_mock()
    controller.move_z(1)
    drain(controller)
    track[1].move_axis.assert_not_called()
    assert not controller._position_known
    gui.messagebox.showerror.assert_called_once()


def test_typed_increment_used_and_invalid_input_rejected(controller, track):
    controller.x_increment_var.get.return_value = 2.5
    controller.move_x(1)
    drain(controller)
    assert controller.current_x == 120250
    track[1].move_axis.reset_mock()
    controller.x_increment_var.get.return_value = float("nan")
    controller.move_x(1)
    track[1].move_axis.assert_not_called()


def test_failed_preset_does_not_claim_destination(controller, track):
    track[1].move_axis.side_effect = RuntimeError("axis fault")
    controller.go_to_preset("Source", 200, 0)
    drain(controller)
    assert not controller._position_known
    assert not controller.position_history
    gui.messagebox.showerror.assert_called_once()


def test_gripper_outputs_and_unknown_initial_state(controller, track):
    assert controller.gripper_open is None
    controller.set_gripper(True)
    drain(controller)
    assert controller.gripper_open is True
    assert track[1].set_output.call_args_list == [call(4, True), call(5, False)]
    controller.set_gripper(False)
    drain(controller)
    assert track[1].set_output.call_args_list[-2:] == [call(4, False), call(5, True)]


def test_simulation_never_sends_commands_to_parked_connection(controller, track):
    controller.toggle_simulation_mode()
    assert controller.simulation_mode
    controller.move_x(1)
    controller.move_z(-1)
    controller.go_to_preset("Source", 200, 0)
    controller.set_gripper(True)
    controller.go_to_origin()
    assert not controller.runner.busy
    assert not track[1].mock_calls


def test_connect_uses_only_controller_a_and_does_not_move_or_write(track, tmp_path):
    # A private config root with no inventory or workflow config is sufficient.
    state = tmp_path / "robot_state"
    state.mkdir()
    for name in ("track_positions.yaml", "robot_hardware.yaml"):
        (state / name).write_bytes((ROOT / "robot_state" / name).read_bytes())
    before = {p.name: p.read_bytes() for p in state.iterdir()}
    factory = Mock(return_value=track[1])
    progress = Mock()
    connected = ManualTrack.connect(progress, factory=factory, root=tmp_path)
    factory.assert_called_once_with("A", network_serial="AU06CNCF")
    track[1].move_axis.assert_not_called()
    track[1].set_output.assert_not_called()
    assert connected.read_position(progress) == (120000, 75000)
    assert {p.name: p.read_bytes() for p in state.iterdir()} == before


def test_hardware_toggle_only_completes_after_encoder_verification(controller, monkeypatch, track):
    controller.hardware = None
    controller.simulation_mode = True
    connect = Mock(return_value=track[0])
    monkeypatch.setattr(gui.ManualTrack, "connect", connect)
    controller.toggle_simulation_mode()
    assert controller.simulation_mode
    drain(controller)
    assert not controller.simulation_mode
    assert controller.gripper_open is None
    controller.toggle_simulation_mode()
    controller.toggle_simulation_mode()
    drain(controller)
    assert connect.call_count == 1


def test_connection_failure_keeps_simulation(controller, monkeypatch):
    controller.hardware = None
    controller.simulation_mode = True
    monkeypatch.setattr(gui.ManualTrack, "connect", Mock(side_effect=RuntimeError("USB unavailable")))
    controller.toggle_simulation_mode()
    drain(controller)
    assert controller.simulation_mode
    assert controller.hardware is None
    gui.messagebox.showerror.assert_called_once()


def test_default_startup_does_not_connect_or_import_workflow(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(gui.ManualTrack, "connect", connect)
    monkeypatch.setattr(gui.tk, "Tk", Mock())
    for method in ("setup_gui", "setup_keybindings", "update_position_display", "refresh_track_status"):
        monkeypatch.setattr(gui.TrackControlGUI, method, Mock())
    control = gui.TrackControlGUI()
    assert control.simulation_mode
    connect.assert_not_called()
    assert not control.runner.busy
    assert "Lash_E" not in vars(gui)


def test_presets_follow_yaml_even_when_started_outside_repository(controller, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    controller.load_track_positions()
    presets = dict((name, (x, z)) for name, x, z in controller.preset_positions())
    source = controller.track_positions["positions"]["source_stack"]
    assert presets["Source Stack"] == (source["x"], source["z_transfer"])
    assert "96-well Lid Storage" in presets


def test_sdk_reply_timeout_reports_error_and_quarantines_connection():
    release = threading.Event()
    sdk = BoundedSDK(timeout=0.02)
    try:
        with pytest.raises(TimeoutError, match="physical result is unknown"):
            sdk.call("Read encoder", lambda: release.wait(1))
        next_call = Mock()
        with pytest.raises(RuntimeError, match="unresolved request"):
            sdk.call("Move", next_call)
        next_call.assert_not_called()
    finally:
        release.set()


def test_motion_timeout_does_not_poll_forever_or_continue_path(track):
    hardware, device, coordinates, token = track
    hardware.motion_timeout = 0.01
    token.is_done.return_value = False
    with pytest.raises(TimeoutError, match="did not complete"):
        hardware.move_to(200, 0, Mock())
    assert device.move_axis.call_count == 1
    device.quick_stop.assert_called_once()
    assert hardware.sdk.failed


def test_axis_error_does_not_poll_forever(track):
    hardware, device, coordinates, token = track
    token.is_done.return_value = False
    device.get_axis_status.return_value = 9
    with pytest.raises(RuntimeError, match="axis error"):
        hardware.move_to(200, 0, Mock())
    assert device.move_axis.call_count == 1


def test_stop_sends_real_quick_stop_while_motion_is_pending(controller, track):
    entered, release = threading.Event(), threading.Event()
    original = track[1].move_axis.side_effect

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(1)
        return original(*args, **kwargs)

    track[1].move_axis.side_effect = delayed
    try:
        controller.go_to_preset("Source", 200, 0)
        assert entered.wait(0.5)
        controller.emergency_stop()
        assert controller.stop_runner.busy
        release.set()
        drain(controller)
        track[1].quick_stop.assert_called_once()
        assert track[1].move_axis.call_count == 1
        assert not controller.position_history
    finally:
        release.set()


def test_runner_error_is_delivered_on_ui_thread(controller, track):
    main_thread = threading.get_ident()
    observed = []
    track[1].get_axis_position.side_effect = RuntimeError("communication fault")
    gui.messagebox.showerror.side_effect = lambda *a: observed.append(threading.get_ident())
    controller.refresh_track_status()
    drain(controller)
    assert observed == [main_thread]


def test_close_does_not_wait_on_outstanding_sdk_request(controller):
    controller._close()
    controller.root.destroy.assert_called_once()
    controller.root.reset_mock()
    controller._poll_operations()
    controller.root.after.assert_not_called()
