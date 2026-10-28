"""Exercise real coordinator/track transfers with hardware interfaces mocked."""

import logging
from pathlib import Path
import runpy
import sys
from unittest.mock import MagicMock, Mock

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from master_usdl_coordinator import Lash_E
from North_Safe import North_Track


@pytest.fixture
def round_trip(monkeypatch, tmp_path):
    monkeypatch.chdir(_ROOT)
    # Isolate input and all track status writes from the live state files.
    status_file = tmp_path / "track_status.yaml"
    status_file.write_text(yaml.safe_dump({
        "wellplate_type": "24 WELL PLATE", "num_in_source": 1,
        "num_in_waste": 0, "active_wellplate_position": None,
        "current_gripper_location": "origin",
        "current_gripper_position": {"x": 0, "z": 0},
    }))
    original_load = North_Track.get_track_status

    def load_private(track):
        track.TRACK_STATUS_FILE = str(status_file)
        original_load(track)

    monkeypatch.setattr(North_Track, "get_track_status", load_private)
    lash = object.__new__(Lash_E)
    lash.simulate = True
    lash.logger = logging.getLogger(__name__)
    lash._workflow_should_continue = True
    lash.nr_track = North_Track(MagicMock(), simulate=True, logger=lash.logger)
    lash.nr_track.set_wellplate_position = Mock(wraps=lash.nr_track.set_wellplate_position)
    lash.nr_robot = MagicMock()
    lash.nr_robot.get_config_parameter.return_value = "24 WELL PLATE"
    lash.nr_robot.pause_after_error.side_effect = AssertionError("Unexpected robot error")
    lash.nr_track.pause_after_error = Mock(side_effect=AssertionError("Unexpected track error"))
    lash.cytation = MagicMock()
    namespace = runpy.run_path(str(_ROOT / "tests" / "wellplate_24_round_trip.py"))
    execute = namespace["execute"]
    constructor = Mock(return_value=lash)
    monkeypatch.setitem(execute.__globals__, "Lash_E", constructor)
    return execute, lash, constructor


@pytest.mark.parametrize("reader_simulation", [True, False])
def test_round_trip_uses_real_transfers_and_null_read(round_trip, reader_simulation):
    execute, lash, constructor = round_trip
    # Also exercise the live coordinator's carrier calls, with a mocked reader.
    # The track always remains simulated and c9 is always a mock.
    lash.simulate = reader_simulation
    assert execute(config={"SIMULATE": reader_simulation, "INPUT_VIAL_STATUS_FILE": None},
                   show_gui=False) is None
    assert not constructor.call_args.kwargs["show_gui"]
    assert [call.args[0] for call in lash.nr_track.set_wellplate_position.call_args_list] == [
        "gripper", "pipetting_area", "gripper", "cytation_tray",
        "gripper", "pipetting_area", "gripper", "waste_stack", None,
    ]
    assert lash.nr_track.NUM_SOURCE == 0
    assert lash.nr_track.NUM_WASTE == 1
    assert lash.nr_track.CURRENT_GRIPPER_LOCATION == "origin"
    lash.cytation.run_protocol.assert_not_called()
    assert lash.cytation.CarrierOut.call_count == (0 if reader_simulation else 2)
    assert lash.cytation.CarrierIn.call_count == (0 if reader_simulation else 2)
    for call in lash.cytation.CarrierIn.call_args_list:
        assert call.kwargs == {"plate_type": "24 WELL PLATE", "use_lid": False}


@pytest.mark.parametrize("field,value", [
    ("CURRENT_WP_TYPE", "96 WELL PLATE"), ("NUM_SOURCE", 3),
    ("NUM_WASTE", 1), ("ACTIVE_WELLPLATE_POSITION", "cytation_tray"),
])
def test_bad_initial_state_stops_before_plate_transfers(round_trip, field, value):
    execute, lash, _ = round_trip
    setattr(lash.nr_track, field, value)
    with pytest.raises(ValueError):
        execute()
    lash.nr_track.set_wellplate_position.assert_not_called()


def test_cancel_does_not_transfer_a_plate(round_trip):
    execute, lash, _ = round_trip
    lash._workflow_should_continue = False
    assert execute() is None
    lash.nr_track.set_wellplate_position.assert_not_called()


def test_failed_measurement_does_not_discard_or_attempt_recovery(round_trip):
    execute, lash, _ = round_trip
    lash.measure_wellplate = Mock(side_effect=RuntimeError("Reader unavailable"))
    lash.discard_used_wellplate = Mock()
    with pytest.raises(RuntimeError, match="Reader unavailable"):
        execute()
    lash.discard_used_wellplate.assert_not_called()
    assert lash.nr_track.ACTIVE_WELLPLATE_POSITION == "pipetting_area"
