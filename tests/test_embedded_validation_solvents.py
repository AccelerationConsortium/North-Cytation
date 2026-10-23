"""Exercise the solvent runner against the controller's real vial movement guard."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from North_Safe import North_Robot


ROOT = Path(__file__).resolve().parents[1]


class MovementRobot(North_Robot):
    def __init__(self):
        # Use real movement/guard methods, with no hardware connection.
        self.simulate = True
        self.logger = Mock()
        self.VIAL_FILE = str(ROOT / "status/calibration_vials_short.csv")
        self.HELD_PIPET_TYPE = None
        self.locations = {"acetone": ("main_8mL_rack", 36),
                          "ethyl_acetate": ("main_8mL_rack", 38)}
        self.tips = []
        self.discards = 0

    def normalize_vial_index(self, vial):
        return vial

    def get_vial_info(self, vial, key):
        return {"location": self.locations[vial][0],
                "location_index": self.locations[vial][1], "vial_name": vial}[key]

    def pause_after_error(self, message):
        raise AssertionError(message)

    def grab_vial(self, vial, move_speed=None):
        pass

    def drop_off_vial(self, vial, location, index, move_speed=None):
        self.locations[vial] = (location, index)

    def get_pipet(self, tip):
        assert self.HELD_PIPET_TYPE is None
        self.HELD_PIPET_TYPE = tip
        self.tips.append(tip)

    def remove_pipet(self):
        self.HELD_PIPET_TYPE = None
        self.discards += 1

    def return_vial_home(self, vial):
        self.move_vial_to_location(vial, "main_8mL_rack", 36 if vial == "acetone" else 38)

    def move_home(self):
        pass


def test_four_sessions_move_vials_without_held_tips(monkeypatch, tmp_path):
    robot = MovementRobot()
    lash = SimpleNamespace(nr_robot=robot, simulate=True, logger=Mock(),
                           _workflow_should_continue=True)
    calls = []

    def validate(**kwargs):
        # The embedded validator moves the source to clamp for conditioning,
        # then the destination to clamp before the stage measurements.
        robot.move_vial_to_location(kwargs["source_vial"], "clamp", 0)
        robot.move_vial_to_location(kwargs["destination_vial"], "clamp", 0)
        calls.append(kwargs)
        return {"success": True}

    def initialize(**kwargs):
        g = kwargs["workflow_globals"]
        lash.workflow_config = {k: v for k, v in g.items()
                                if k.isupper() and not k.startswith("_")}
        return lash

    monkeypatch.setitem(sys.modules, "master_usdl_coordinator", SimpleNamespace(Lash_E=initialize))
    monkeypatch.setitem(sys.modules, "pipetting_data.embedded_calibration_validation",
                        SimpleNamespace(validate_pipetting_accuracy=validate))
    module = runpy.run_path(str(ROOT / "tests/embedded_validation_solvents.py"))
    # Keep runner output in this test's temporary directory.
    module["execute"].__globals__["__file__"] = str(tmp_path / "tests" / "runner.py")
    result = module["execute"](show_gui=False)
    assert list(result) == ["acetone_small", "acetone_large",
                            "ethyl_acetate_small", "ethyl_acetate_large"]
    assert robot.tips == ["small_tip", "large_tip"] * 2
    assert robot.discards == 4
    assert robot.locations == {"acetone": ("main_8mL_rack", 36),
                               "ethyl_acetate": ("main_8mL_rack", 38)}
    assert len(calls) == 4
    assert [call["tip_type"] for call in calls] == ["small_tip", "large_tip"] * 2
    assert all(call["compensate_overvolume"] is False for call in calls)
