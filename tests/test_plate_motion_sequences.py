"""Differential motor-command traces against the synced pre-refactor version.

No motors are connected. The reference is fixed, rather than a moving HEAD.
Inventory persistence is excluded; actual axis/pneumatic commands are compared.
"""
import ast
from copy import deepcopy
import logging
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from North_Safe import North_Track
from master_usdl_coordinator import Lash_E
from robot_state.plate_inventory import PlateInventory

REFERENCE = "9ba125e0"


def reference_file(name):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{REFERENCE}:{name}"],
        cwd=ROOT, text=True, encoding="utf-8")


@pytest.fixture(scope="module")
def reference_class():
    tree = ast.parse(reference_file("North_Safe.py"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("North_Base", "North_Track")]
    namespace = {}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "synced_North_Safe.py", "exec"), namespace)
    return namespace["North_Track"]


def make_track(cls, active=False, source_count=2):
    track = object.__new__(cls)
    track.c9 = MagicMock()
    track.logger = logging.getLogger("motion-trace")
    track.simulate = True
    track.WELLPLATES = yaml.safe_load(reference_file("robot_state/wellplates.yaml"))
    track.TRACK_POSITIONS = yaml.safe_load(reference_file("robot_state/track_positions.yaml"))
    track.ROBOT_HARDWARE = yaml.safe_load((ROOT / "robot_state/robot_hardware.yaml").read_text())
    track.CURRENT_GRIPPER_LOCATION = "origin"
    track.CURRENT_GRIPPER_POSITION = {"x": 0, "z": 0}
    track.save_track_status = lambda: None
    legacy = {"wellplate_type": "96 WELL PLATE", "num_in_source": source_count,
              "num_in_waste": 1, "active_wellplate_position": "pipetting_area" if active else None}
    if cls is North_Track:
        track.WELLPLATES["quartz"]["lid_gripping_z_offset"] = track.WELLPLATES["quartz_lid"]["gripping_z_offset"]
        track.WELLPLATES["96 WELL PLATE"]["lid_gripping_z_offset"] = track.WELLPLATES["96_wellplate_lid"]["gripping_z_offset"]
        track._inventory = PlateInventory(legacy, track.WELLPLATES, track.TRACK_POSITIONS)
    else:
        track.CURRENT_WP_TYPE = "96 WELL PLATE"
        track.NUM_SOURCE = source_count
        track.NUM_WASTE = 1
        track.ACTIVE_WELLPLATE_POSITION = legacy["active_wellplate_position"]
    return track


def motor_trace(track):
    return [call for call in track.c9.mock_calls
            if call[0] in ("move_axis", "set_output", "delay")]


@pytest.mark.parametrize("method,active", [("get_new_wellplate", False), ("discard_wellplate", True)])
@pytest.mark.parametrize("home_after", [False, True])
def test_stack_helpers_preserve_synced_motor_order(reference_class, method, active, home_after):
    baseline = make_track(reference_class, active)
    current = make_track(North_Track, active)
    getattr(baseline, method)(move_home_afterwards=home_after)
    getattr(current, method)(move_home_afterwards=home_after)
    assert motor_trace(current) == motor_trace(baseline)


def test_source_pickup_raises_before_horizontal_travel(reference_class):
    track = make_track(North_Track)
    # Simulate starting below the safe transfer height, not just at home.
    track.CURRENT_GRIPPER_POSITION = {"x": 131650, "z": 75000}
    track.get_new_wellplate(move_home_afterwards=False)
    axis_calls = [call for call in motor_trace(track) if call[0] == "move_axis"]
    assert axis_calls[0].args == (track.get_axis("z_axis"), 0)
    assert axis_calls[1].args == (track.get_axis("z_axis"), 0)
    assert axis_calls[2].args == (track.get_axis("x_axis"), 200)


@pytest.fixture(scope="module")
def reference_coordinator():
    tree = ast.parse(reference_file("master_usdl_coordinator.py"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Lash_E")
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "synced_coordinator.py", "exec"), namespace)
    return namespace["Lash_E"]


def make_coordinator(cls, track):
    coordinator = object.__new__(cls)
    coordinator.nr_track = track
    coordinator.nr_robot = MagicMock()
    coordinator.nr_robot.get_config_parameter.side_effect = lambda category, key, parameter, **kwargs: track.WELLPLATES[key].get(parameter)
    coordinator.cytation = MagicMock()
    coordinator.logger = track.logger
    coordinator.simulate = False  # Reader is mocked; include carrier calls in the trace.
    recorder = MagicMock()
    recorder.attach_mock(track.c9, "track")
    recorder.attach_mock(coordinator.cytation, "reader")
    return coordinator, recorder


@pytest.mark.parametrize("safe_movement", [False, True])
def test_cytation_load_and_return_preserve_all_motor_and_carrier_order(reference_class, reference_coordinator, safe_movement):
    traces = []
    for coordinator_class, track_class in ((reference_coordinator, reference_class), (Lash_E, North_Track)):
        coordinator, recorder = make_coordinator(coordinator_class, make_track(track_class, active=True))
        coordinator.move_wellplate_to_cytation(safe_movement=safe_movement)
        coordinator.move_wellplate_back_from_cytation()
        traces.append(recorder.mock_calls)
    assert traces[1] == traces[0]


@pytest.mark.parametrize("cycles", [1, 2])
def test_complete_cycles_preserve_synced_motor_and_carrier_sequence(reference_class, reference_coordinator, cycles):
    traces = []
    for coordinator_class, track_class in ((reference_coordinator, reference_class), (Lash_E, North_Track)):
        coordinator, recorder = make_coordinator(coordinator_class, make_track(track_class, source_count=cycles))
        for _ in range(cycles):
            coordinator.grab_new_wellplate()
            assert coordinator.measure_wellplate(None) is None
            coordinator.discard_used_wellplate()
        traces.append(recorder.mock_calls)
        assert coordinator.nr_track.NUM_SOURCE == 0
        assert coordinator.nr_track.NUM_WASTE == 1 + cycles
    assert traces[1] == traces[0]


def test_discard_from_home_visits_intermediate_x_before_lowering(reference_class):
    track = make_track(North_Track, active=True)
    track.discard_wellplate(move_home_afterwards=False)
    movements = [call for call in motor_trace(track) if call[0] == "move_axis"]
    assert movements[0].args == (track.get_axis("x_axis"), 28275)
    assert movements[1].args == (track.get_axis("z_axis"), 75000)
    assert movements[2].args == (track.get_axis("x_axis"), 131650)


@pytest.mark.parametrize("lid_alias,plate_type,storage", [
    ("quartz_lid", "quartz", "lid_storage"),
    ("96_wellplate_lid", "96 WELL PLATE", "lid_storage_96"),
])
def test_lid_park_and_replace_preserve_synced_motor_order(reference_class, lid_alias, plate_type, storage):
    traces = []
    for cls in (reference_class, North_Track):
        track = make_track(cls, active=True)
        if cls is North_Track:
            track.inventory.add("pipetting_area", plate_type, kind="lid")
        track.grab_wellplate_from_location("pipetting_area", lid_alias, waypoint_locations=["cytation_safe_area"])
        track.release_wellplate_in_location(storage, lid_alias)
        track.grab_wellplate_from_location(storage, lid_alias, waypoint_locations=["cytation_safe_area"])
        track.release_wellplate_in_location("pipetting_area", lid_alias)
        traces.append(motor_trace(track))
    assert traces[1] == traces[0]


def test_requested_mixed_round_trip_changes_only_stack_depths(reference_class, reference_coordinator):
    """Use original routes with the requested sum substituted only for stack math."""
    old_track = make_track(reference_class, source_count=2)
    old_track.NUM_WASTE = 0
    source_types = ["24 WELL PLATE", "96 WELL PLATE"]
    waste_types = []
    def sum_height(num, plate_type, location, operation):
        supports = source_types[:-1] if operation == "grab" else waste_types
        base = old_track.get_position(location)["z_grab" if operation == "grab" else "z_release"]
        return base - sum(old_track.WELLPLATES[t]["thickness"] for t in supports)
    old_track.calculate_wp_stack_height = sum_height
    old, old_recorder = make_coordinator(reference_coordinator, old_track)
    current_track = make_track(North_Track, source_count=2)
    current_track._inventory = PlateInventory.load(
        ROOT / "tests/fixtures/mixed_wellplate_track_status.yaml",
        current_track.WELLPLATES, current_track.TRACK_POSITIONS)
    current, current_recorder = make_coordinator(Lash_E, current_track)
    for plate_type in ("96 WELL PLATE", "24 WELL PLATE"):
        old_track.CURRENT_WP_TYPE = plate_type
        old.grab_new_wellplate()
        source_types.pop()
        old.measure_wellplate(None, plate_type=plate_type)
        old.discard_used_wellplate()
        waste_types.append(plate_type)
        current.grab_new_wellplate()
        current.measure_wellplate(None)
        current.discard_used_wellplate()
    assert current_recorder.mock_calls == old_recorder.mock_calls
