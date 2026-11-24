"""blowout_speed is applied only when a workflow opts in (APPLY_BLOWOUT_SPEED); off, nothing changes."""
import logging
import unittest
from types import SimpleNamespace

from North_Safe import North_Robot
from pipetting_data.pipetting_parameters import PipettingParameters


class FakeC9:
    PUMP_VALVE_LEFT, PUMP_VALVE_RIGHT = "left", "right"

    def __init__(self, calls):
        self.calls = calls

    def dispense_ml(self, pump, amount):
        self.calls.append(("dispense", amount))

    def aspirate_ml(self, pump, amount):
        self.calls.append(("aspirate", amount))

    def set_pump_valve(self, pump, valve):
        self.calls.append(("valve", valve))


def fake_robot(apply=None):
    calls = []
    robot = SimpleNamespace(simulate=True, c9=FakeC9(calls), CURRENT_PUMP_SPEEDS={0: 14},
                            logger=logging.getLogger("test"), calls=calls)
    robot.adjust_pump_speed = lambda pump, speed: calls.append(("speed", speed))
    if apply is not None:
        robot.APPLY_BLOWOUT_SPEED = apply
    return robot


class BlowoutSpeedTests(unittest.TestCase):
    def test_off_by_default_so_calibrated_values_are_ignored(self):
        params = PipettingParameters(blowout_speed=30)
        self.assertIsNone(North_Robot._blowout_speed(fake_robot(), params))
        self.assertIsNone(North_Robot._blowout_speed(fake_robot(False), params))
        self.assertEqual(North_Robot._blowout_speed(fake_robot(True), params), 30)

    def test_without_blowout_speed_the_air_is_pushed_at_the_dispense_speed_as_before(self):
        robot = fake_robot()
        North_Robot.pipet_dispense(robot, 0.1, blowout_vol=0.2)
        self.assertEqual(robot.calls, [("dispense", 0.1), ("speed", 5), ("valve", "left"), ("aspirate", 0.2),
                                       ("speed", 14), ("valve", "right"), ("dispense", 0.2)])

    def test_with_blowout_speed_the_air_is_pushed_at_it_then_the_speed_is_restored(self):
        robot = fake_robot()
        North_Robot.pipet_dispense(robot, 0.1, blowout_vol=0.2, blowout_speed=30)
        self.assertEqual(robot.calls, [("dispense", 0.1), ("speed", 5), ("valve", "left"), ("aspirate", 0.2),
                                       ("speed", 30), ("valve", "right"), ("dispense", 0.2), ("speed", 14)])


if __name__ == "__main__":
    unittest.main()
