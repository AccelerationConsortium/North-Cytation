"""Best-effort cleanup after a workflow's keyboard interrupt."""


def cleanup_after_keyboard_interrupt(lash_e):
    """Release a held plate, return the track, then clear the robot's held items."""
    for label, controller, method_name in (
        ("track", getattr(lash_e, "nr_track", None), "reset_after_initialization"),
        ("robot", getattr(lash_e, "nr_robot", None), "_perform_physical_cleanup"),
    ):
        if controller is None:
            continue
        try:
            getattr(controller, method_name)()
        except (Exception, KeyboardInterrupt):
            lash_e.logger.exception("Keyboard interrupt %s cleanup failed", label)
