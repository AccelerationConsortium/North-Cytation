import logging
import unittest
from pathlib import Path
from unittest.mock import Mock

import yaml
from North_Safe import North_Robot


class TipSelectionTests(unittest.TestCase):
    def setUp(self):
        self.robot = object.__new__(North_Robot)
        self.robot.logger = logging.getLogger(__name__)
        path = Path(__file__).resolve().parents[1] / 'robot_state' / 'pipet_tips.yaml'
        self.robot.PIPET_TIPS = yaml.safe_load(path.read_text())
        self.robot.pause_after_error = Mock()

    def test_configured_capacity_boundaries(self):
        for volume, expected in [(.199, 'small_tip'), (.2, 'large_tip'),
                                 (.201, 'large_tip'), (1.0, 'large_tip')]:
            with self.subTest(volume=volume):
                self.assertEqual(self.robot.select_pipet_tip(volume), expected)
        self.robot.pause_after_error.assert_not_called()

    def test_selection_uses_configured_capacities(self):
        self.robot.PIPET_TIPS = {
            'smaller': {'volume': .3, 'min_suggested_volume': .001},
            'larger': {'volume': 2.5, 'min_suggested_volume': .15},
        }
        self.assertEqual(self.robot.select_pipet_tip(.299), 'smaller')
        self.assertEqual(self.robot.select_pipet_tip(.3), 'larger')
        self.assertEqual(self.robot.select_pipet_tip(2.5), 'larger')
        self.assertIsNone(self.robot.select_pipet_tip(2.501))
        self.robot.pause_after_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
