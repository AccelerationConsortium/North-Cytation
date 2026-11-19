"""Powder dispenser: channel list, simulation, the capped-vial check, safe stop and tracking."""
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

import master_usdl_coordinator
from North_Safe import North_Powder
from experiment_tracking import sample_tracking as st
from robot_state import powder_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def powder(monkeypatch):
    monkeypatch.chdir(ROOT)
    return North_Powder(MagicMock(), simulate=True, logger=logging.getLogger(__name__))


def test_simulated_dispense_runs_the_real_moves_and_returns_the_target(powder):
    assert powder.dispense_powder_mg(12.5, channel=0) == 12.5
    assert powder.c9.move_carousel.call_args_list == [call(66.5, 70), call(0, 0)]
    powder.p2.activate_powder_channel.assert_called_once_with(0)


def test_a_channel_not_in_the_list_is_refused_before_anything_moves(powder):
    with pytest.raises(ValueError, match="channel 3 isn't listed"):
        powder.dispense_powder_mg(5.0, channel=3)
    powder.c9.move_carousel.assert_not_called()
    powder.p2.activate_powder_channel.assert_not_called()


def test_only_the_fitted_slot_is_listed(powder):
    assert list(powder.powder_channels()) == [0]


def test_safe_stop_closes_every_listed_valve(powder):
    powder.close_all_valves()
    assert powder.p2.move_axis.call_args_list == [call(0, 0.0)]


def test_safe_stop_no_longer_lists_the_powder_dispenser_as_not_handled():
    assert "powder dispenser" not in master_usdl_coordinator.Lash_E._SAFE_STOP_NOT_HANDLED


def coordinator(pipetable, location="main_8mL_rack", dispenser=True):
    robot = MagicMock()
    robot.GRIPPER_STATUS = None
    robot.get_vial_info.side_effect = lambda vial, key: location if key == "location" else None
    robot.is_vial_pipetable.return_value = pipetable
    robot._vial_container_id.return_value = None
    robot.uncap_clamp_vial.side_effect = lambda: setattr(robot, "GRIPPER_STATUS", "Cap")
    robot.recap_clamp_vial.side_effect = lambda: setattr(robot, "GRIPPER_STATUS", None)
    lash = SimpleNamespace(nr_robot=robot, logger=logging.getLogger(__name__),
                           powder_dispenser=MagicMock() if dispenser else None)
    if dispenser:
        lash.powder_dispenser.dispense_powder_mg.return_value = 9.8
    return lash


def test_a_closed_cap_is_taken_off_and_put_back_like_a_reservoir_dispense():
    lash = coordinator(pipetable=False)
    assert master_usdl_coordinator.Lash_E.mass_dispense_into_vial(lash, "v1", 10.0) == 9.8
    robot = lash.nr_robot
    robot.move_vial_to_location.assert_called_once_with("v1", "clamp", 0)
    robot.uncap_clamp_vial.assert_called_once_with()
    robot.recap_clamp_vial.assert_called_once_with()
    robot.return_vial_home.assert_called_once_with("v1")
    names = [c[0] for c in robot.method_calls]
    assert names.index("uncap_clamp_vial") < names.index("move_home") < names.index("recap_clamp_vial")


def test_powder_goes_through_an_open_cap_without_uncapping():
    lash = coordinator(pipetable=True)
    assert master_usdl_coordinator.Lash_E.mass_dispense_into_vial(lash, "v1", 10.0) == 9.8
    lash.nr_robot.uncap_clamp_vial.assert_not_called()
    lash.nr_robot.recap_clamp_vial.assert_not_called()


def test_a_vial_already_in_the_clamp_is_not_moved():
    lash = coordinator(pipetable=True, location="clamp")
    master_usdl_coordinator.Lash_E.mass_dispense_into_vial(lash, "v1", 10.0, return_home=False)
    lash.nr_robot.move_vial_to_location.assert_not_called()
    lash.nr_robot.return_vial_home.assert_not_called()


def test_a_dispenser_that_was_not_started_gives_a_clear_error():
    lash = coordinator(pipetable=True, dispenser=False)
    with pytest.raises(RuntimeError, match="initialize_p2=True"):
        master_usdl_coordinator.Lash_E.mass_dispense_into_vial(lash, "v1", 10.0)
    lash.nr_robot.move_vial_to_location.assert_not_called()


def test_settings_load_from_yaml_with_the_old_values():
    ps = powder_settings.default_ps
    assert (ps.tol, ps.scale_delay) == (0.2, 1)
    assert (ps.fast_settings.opening_deg, ps.fast_settings.percent_target, ps.fast_settings.max_growth) == (50, 0.75, 3)
    assert (ps.med_settings.thresh, ps.med_settings.amplitude) == (5, 75)
    assert (ps.slow_settings.thresh, ps.slow_settings.shut_valve) == (3, False)
    assert set(powder_settings.load_protocols()) == {"default", "sticky", "friendly", "flowing"}


def test_cartridge_status_load_and_empty(tmp_path):
    channels, sources, log = tmp_path / "channels.yaml", tmp_path / "sources.csv", tmp_path / "transfers.csv"
    channels.write_text("0:\n  description: Cartridge slot 0\n")
    assert st.cartridge_status(channels, sources, log)[0]["load"] is None
    load = st.record_load("powder-0", "TTAB-004", 2000, "mg", sources)
    st.record_transfer(SimpleNamespace(simulate=False, run_id="r"), load, "vial-a", 12.5, "mg", "powder", path=log)
    row = st.cartridge_status(channels, sources, log)[0]
    assert row["load"]["stock_id"] == "TTAB-004" and row["left_mg"] == pytest.approx(1987.5)
    st.record_empty("powder-0", sources)
    row = st.cartridge_status(channels, sources, log)[0]
    assert row["load"] is None and row["emptied_on"]


def test_powder_limit_is_in_robot_hardware_yaml():
    import yaml
    assert yaml.safe_load((ROOT / "robot_state/robot_hardware.yaml").read_text())["powder_limits"] == {"max_mg": 1000}


@pytest.mark.parametrize("mass, message", [(0, "must be > 0"), (-5, "must be > 0"), (float("nan"), "must be > 0"),
                                           ("a", "must be > 0"), (1000.1, "above 1000 mg")])
def test_a_bad_mass_is_refused_before_anything_moves(powder, mass, message):
    with pytest.raises(ValueError, match=f"refused before moving: mass_mg.*{message}|refused before moving: mass_mg must be > 0"):
        powder.dispense_powder_mg(mass, channel=0)
    powder.c9.move_carousel.assert_not_called()
    powder.p2.activate_powder_channel.assert_not_called()


def test_the_largest_allowed_mass_is_dispensed(powder):
    assert powder.dispense_powder_mg(1000, channel=0) == 1000


def test_the_vial_flow_refuses_a_bad_mass_before_the_vial_moves(powder):
    lash = coordinator(pipetable=False)
    lash.powder_dispenser = powder
    with pytest.raises(ValueError, match="above 1000 mg"):
        master_usdl_coordinator.Lash_E.mass_dispense_into_vial(lash, "v1", 5000)
    lash.nr_robot.move_vial_to_location.assert_not_called()
    lash.nr_robot.uncap_clamp_vial.assert_not_called()
