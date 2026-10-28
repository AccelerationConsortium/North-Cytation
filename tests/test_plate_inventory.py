import logging
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_state.plate_inventory import PlateInventory
from North_Safe import North_Track, North_Robot
from master_usdl_coordinator import Lash_E


@pytest.fixture
def inventory():
    return PlateInventory.load(ROOT / "tests/fixtures/mixed_wellplate_track_status.yaml",
        yaml.safe_load((ROOT / "robot_state/wellplates.yaml").read_text()),
        yaml.safe_load((ROOT / "robot_state/track_positions.yaml").read_text()))


@pytest.fixture
def workstation(inventory, monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    track = object.__new__(North_Track)
    track._inventory = inventory
    track.WELLPLATES = inventory.definitions
    track.TRACK_POSITIONS = inventory.position_config
    track.ROBOT_HARDWARE = yaml.safe_load((ROOT / "robot_state/robot_hardware.yaml").read_text())
    track.CURRENT_GRIPPER_LOCATION = "origin"
    track.CURRENT_GRIPPER_POSITION = {"x": 0, "z": 0}
    track.TRACK_STATUS_FILE = str(tmp_path / "track_status.yaml")
    track.simulate = True
    track.logger = logging.getLogger("plate-test")
    track.c9 = MagicMock()
    lash = object.__new__(Lash_E)
    lash.nr_track = track
    lash.nr_robot = MagicMock()
    lash.cytation = MagicMock()
    lash.simulate = True
    lash.logger = track.logger
    lash._workflow_should_continue = True
    return lash


def test_discrete_sum_matches_old_formula_and_limit(inventory):
    for plate_type in ("96 WELL PLATE", "24 WELL PLATE", "48 WELL PLATE"):
        for count in range(1, 16):
            inv = PlateInventory({"wellplate_type": plate_type, "num_in_source": count,
                                  "num_in_waste": count-1}, inventory.definitions, inventory.position_config)
            for location, operation, base in (("source_stack", "grab", 85500), ("waste_stack", "release", 83000)):
                expected = base - (count-1)*inv.definitions[plate_type]["thickness"]
                if expected >= 20000:
                    assert inv.stack_z(location, operation) == expected
                else:
                    with pytest.raises(ValueError, match="height limit"):
                        inv.stack_z(location, operation)
    assert inventory.stack_z("source_stack", "grab") == 76000
    assert inventory.stack_z("waste_stack", "release") == 83000


@pytest.mark.parametrize("simulate_reader", [True, False])
def test_requested_two_plate_workflow_and_actual_z(workstation, monkeypatch, simulate_reader):
    lash = workstation
    lash.simulate = simulate_reader  # c9 and reader always mocks; track is always simulated.
    namespace = runpy.run_path(str(ROOT / "tests/mixed_wellplate_round_trip.py"))
    execute = namespace["execute"]
    monkeypatch.setitem(execute.__globals__, "Lash_E", Mock(return_value=lash))
    assert execute(config={"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": None}, show_gui=False) == ["plate-96", "plate-24"]
    inv = lash.nr_track.inventory
    assert inv.contents("waste_stack") == ["plate-96", "plate-24"]
    assert inv.contents("source_stack") == []
    assert inv.contents("pipetting_area") == inv.contents("cytation_tray") == []
    assert inv.positions["gripper"] is None
    lash.cytation.run_protocol.assert_not_called()
    # 96 pickup above a 9500-thick 24; 24 release above a 6400-thick 96.
    z_values = [call.args[1] for call in lash.nr_track.c9.move_axis.call_args_list
                if call.args[0] == lash.nr_track.get_axis("z_axis")]
    assert 76000 in z_values
    assert 76600 in z_values
    if not simulate_reader:
        assert [call.kwargs["plate_type"] for call in lash.cytation.CarrierIn.call_args_list] == [
            "96 WELL PLATE", "96 WELL PLATE", "24 WELL PLATE", "24 WELL PLATE"]


@pytest.mark.parametrize("source_types", [
    [], ["96 WELL PLATE"], ["24 WELL PLATE"], ["48 WELL PLATE"],
    ["96 WELL PLATE", "24 WELL PLATE"],
    ["24 WELL PLATE", "48 WELL PLATE", "96 WELL PLATE"],
    ["96 WELL PLATE", "96 WELL PLATE", "24 WELL PLATE"],
    ["48 WELL PLATE", "24 WELL PLATE", "96 WELL PLATE", "48 WELL PLATE"],
])
def test_source_workflow_accepts_configured_combinations_and_existing_waste(workstation, monkeypatch, source_types):
    lash = workstation
    inv = lash.nr_track.inventory
    for object_id in inv.contents("source_stack"):
        inv.remove(object_id)
    before = [inv.add("waste_stack", "96 WELL PLATE")]
    source = [inv.add("source_stack", plate_type) for plate_type in source_types]
    namespace = runpy.run_path(str(ROOT / "tests/mixed_wellplate_round_trip.py"))
    execute = namespace["execute"]
    monkeypatch.setitem(execute.__globals__, "Lash_E", Mock(return_value=lash))
    assert execute(config={"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": None}, show_gui=False) == list(reversed(source))
    assert inv.contents("waste_stack") == before + list(reversed(source))
    assert inv.contents("source_stack") == []
    lash.cytation.run_protocol.assert_not_called()
    if not source:
        lash.nr_track.c9.move_axis.assert_not_called()


def test_async_allows_other_plate_and_retries_return(workstation):
    lash = workstation
    lash.grab_new_wellplate()
    handle = lash.measure_wellplate_async(None)
    handle._done_event.wait(2)
    lash.grab_new_wellplate()  # No return reservation blocks the 24-well plate.
    assert lash.nr_track.inventory.top("pipetting_area") == "plate-24"
    before = len(lash.nr_track.c9.mock_calls)
    with pytest.raises(ValueError, match="capacity"):
        handle.complete()
    assert lash.nr_track.inventory.top("cytation_tray") == "plate-96"
    assert len(lash.nr_track.c9.mock_calls) == before
    lash.discard_used_wellplate()
    assert handle.complete() is None
    assert lash.nr_track.inventory.top("pipetting_area") == "plate-96"
    after = len(lash.nr_track.c9.mock_calls)
    assert handle.complete() is None
    assert len(lash.nr_track.c9.mock_calls) == after


def test_lid_detach_park_reattach_and_normal_plate_offset(workstation):
    lash = workstation
    inv = lash.nr_track.inventory
    lash.grab_new_wellplate()
    lid = inv.add("pipetting_area", "96 WELL PLATE", kind="lid")
    assert inv.attachments["plate-96"] == lid
    assert lash.nr_track.grab_lid_from_location("pipetting_area") == lid
    assert inv.contents("pipetting_area") == ["plate-96"]
    assert not inv.attachments
    lash.nr_track.release_wellplate_in_location("lid_storage_96")
    lash.nr_track.grab_lid_from_location("lid_storage_96")
    lash.nr_track.release_wellplate_in_location("pipetting_area")
    assert inv.attachments["plate-96"] == lid
    lash.nr_track.c9.reset_mock()
    lash.measure_wellplate(None)
    assert inv.attachments["plate-96"] == lid
    # Plate's normal 89000 pickup, not the detached lid's 86000 pickup.
    assert any(call.args[1] == 89000 for call in lash.nr_track.c9.move_axis.call_args_list)


def test_type_and_capacity_checks_before_track_motion(workstation):
    lash = workstation
    with pytest.raises(ValueError, match="expected"):
        lash.nr_track.move_wellplate("source_stack", "pipetting_area", "24 WELL PLATE")
    lash.nr_track.c9.move_axis.assert_not_called()
    lash.grab_new_wellplate()
    lash.nr_track.c9.reset_mock()
    with pytest.raises(ValueError, match="capacity"):
        lash.grab_new_wellplate()
    lash.nr_track.c9.move_axis.assert_not_called()


def test_failed_pickup_records_pending_without_guessing_placement(workstation):
    lash = workstation
    lash.nr_track.close_gripper = Mock(side_effect=RuntimeError("grip failed"))
    with pytest.raises(RuntimeError, match="grip failed"):
        lash.grab_new_wellplate()
    inv = lash.nr_track.inventory
    assert inv.contents("source_stack") == ["plate-24", "plate-96"]
    assert inv.data["pending_transfer"]["object_id"] == "plate-96"
    assert inv.positions["gripper"] is None


def test_failed_release_keeps_held_plate_and_pending_intent(workstation):
    track = workstation.nr_track
    track.grab_wellplate_from_location("source_stack", waypoint_locations=["max_height"])
    track.open_gripper = Mock(side_effect=RuntimeError("release failed"))
    with pytest.raises(RuntimeError, match="release failed"):
        track.release_wellplate_in_location("pipetting_area", waypoint_locations=["transfer_stack"])
    assert track.inventory.positions["gripper"] == "plate-96"
    assert track.inventory.contents("pipetting_area") == []
    assert track.inventory.data["pending_transfer"]["destination"] == "pipetting_area"


def test_reader_active_conflict_is_local_to_reader_motion(workstation):
    import threading
    lash = workstation
    lash._active_measurement = SimpleNamespace(_done_event=threading.Event())
    lash.grab_new_wellplate()  # Track source/active transfers remain available.
    before = len(lash.nr_track.c9.mock_calls)
    with pytest.raises(ValueError, match="read is active"):
        lash.move_wellplate_to_cytation()
    assert len(lash.nr_track.c9.mock_calls) == before


def test_save_revision_backup_and_roundtrip(inventory, tmp_path):
    path = tmp_path / "track_status.yaml"
    old = {"wellplate_type": "96 WELL PLATE", "num_in_source": 2, "num_in_waste": 0}
    path.write_text(yaml.safe_dump(old))
    first = PlateInventory.load(path, inventory.definitions, inventory.position_config)
    second = PlateInventory.load(path, inventory.definitions, inventory.position_config)
    first.save(path)
    assert yaml.safe_load(path.with_suffix(".yaml.legacy.bak").read_text()) == old
    with pytest.raises(ValueError, match="changed on disk"):
        second.save(path)
    loaded = PlateInventory.load(path, inventory.definitions, inventory.position_config)
    assert loaded.data == first.data
    assert not loaded.legacy


def test_missing_thickness_only_matters_when_plate_is_beneath(inventory):
    inv = PlateInventory({"wellplate_type": "quartz", "num_in_source": 1}, inventory.definitions, inventory.position_config)
    assert inv.stack_z("source_stack", "grab") == 85500
    inv.add("source_stack", "96 WELL PLATE")
    with pytest.raises(ValueError, match="supporting plate"):
        inv.stack_z("source_stack", "grab")


def test_plate_ids_prevent_reuse_of_replaced_plate(inventory):
    from workflows.two_run_plate import ready_for_second_run
    inventory.relocate("plate-96", "pipetting_area")
    marker = {"status": "second_half_ready", "plate_id": "plate-96", "plate_type": "96 WELL PLATE"}
    assert ready_for_second_run(inventory.data, marker)
    inventory.remove("plate-96")
    inventory.add("pipetting_area", "96 WELL PLATE")
    assert not ready_for_second_run(inventory.data, marker)


def test_pipetting_resolves_actual_plate_before_liquid_handling(inventory):
    inv = inventory
    inv.relocate("plate-96", "pipetting_area")
    robot = object.__new__(North_Robot)
    robot.plate_inventory = inv
    robot.WELLPLATES = inv.definitions
    assert robot._resolve_pipetting_plate(None, [95]) == "96 WELL PLATE"
    inv.add("pipetting_area", "96 WELL PLATE", kind="lid")
    with pytest.raises(ValueError, match="Remove the lid"):
        robot._resolve_pipetting_plate(None, [0])
    inv.remove(inv.attachments["plate-96"])
    inv.relocate("plate-96", "waste_stack")
    inv.relocate("plate-24", "pipetting_area")
    robot.get_tip_if_needed = Mock()
    with pytest.raises(ValueError, match="Locator grid"):
        robot.pipet_from_wellplate(0, 0.1)
    robot.get_tip_if_needed.assert_not_called()


def test_scheduled_handoff_keeps_independent_plates(inventory):
    from scheduler.plate_state import operation_issues, source_ids
    inventory.relocate("plate-96", "cytation_tray")
    inventory.relocate("plate-24", "pipetting_area")
    assert operation_issues(inventory.data) == []
    assert source_ids(inventory.data) == set()


def test_complete_workflow_initializes_real_simulators_and_saves_private_state(monkeypatch, tmp_path):
    from scheduler import simulation_state as state
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path / "private")
    root = state.create_session([])
    (root / "track_status.yaml").write_bytes((ROOT / "tests/fixtures/mixed_wellplate_track_status.yaml").read_bytes())
    live_paths = [ROOT / "robot_state" / name for name in ("robot_status.yaml", "track_status.yaml")]
    before = [path.read_bytes() for path in live_paths]
    namespace = runpy.run_path(str(ROOT / "tests/mixed_wellplate_round_trip.py"))
    execute = namespace["execute"]
    created = []
    # Keep the atexit run-register write inside this test's temporary directory.
    from experiment_tracking import experiment_run_logger
    trackers = []
    real_start = experiment_run_logger.start_run
    def private_start(*args, **kwargs):
        tracker = real_start(*args, **kwargs)
        tracker.csv_path = str(tmp_path / "experiment_runs.csv")
        trackers.append(tracker)
        return tracker
    monkeypatch.setattr(experiment_run_logger, "start_run", private_start)
    def initialize(**kwargs):
        lash = Lash_E(logging_folder=str(tmp_path / "logs"), **kwargs)
        created.append(lash)
        return lash
    monkeypatch.setitem(execute.__globals__, "Lash_E", initialize)
    with state.workflow_state(root, "mixed_round_trip"):
        assert execute(config={"SIMULATE": True, "INPUT_VIAL_STATUS_FILE": None}, show_gui=False) == ["plate-96", "plate-24"]
    assert created[0].nr_robot.plate_inventory is created[0].nr_track.inventory
    final = yaml.safe_load((root / "track_status.yaml").read_text())
    assert final["positions"]["waste_stack"] == ["plate-96", "plate-24"]
    assert final["pending_transfer"] is None
    assert final["positions"]["gripper"] is None
    assert [path.read_bytes() for path in live_paths] == before
    for tracker in trackers:
        experiment_run_logger.cancel_run(tracker)
