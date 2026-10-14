from unittest.mock import Mock

import pytest

from workflows.interrupt_cleanup import cleanup_after_keyboard_interrupt


def test_interrupt_cleanup_runs_track_before_robot():
    events = []
    lash = Mock()
    lash.nr_track.reset_after_initialization.side_effect = lambda: events.append("track")
    lash.nr_robot._perform_physical_cleanup.side_effect = lambda: events.append("robot")

    cleanup_after_keyboard_interrupt(lash)

    assert events == ["track", "robot"]


@pytest.mark.parametrize("error", [RuntimeError("track failed"), KeyboardInterrupt()])
def test_robot_cleanup_still_runs_when_track_cleanup_fails(error):
    lash = Mock()
    lash.nr_track.reset_after_initialization.side_effect = error

    cleanup_after_keyboard_interrupt(lash)

    lash.nr_robot._perform_physical_cleanup.assert_called_once_with()
    lash.logger.exception.assert_called_once_with(
        "Keyboard interrupt %s cleanup failed", "track"
    )


def test_track_cleanup_still_runs_when_robot_is_absent():
    lash = Mock(nr_robot=None)

    cleanup_after_keyboard_interrupt(lash)

    lash.nr_track.reset_after_initialization.assert_called_once_with()
