"""Best-effort cleanup after a workflow fails or is interrupted."""


def cleanup_after_keyboard_interrupt(lash_e, reason="workflow failed or was interrupted"):
    """Run Lash_E.safe_stop, the one cleanup used after any error or interrupt.

    Kept under its original name so existing callers keep working; it now covers errors as well
    as Ctrl-C. safe_stop turns off instruments, parks the track (unless it holds a plate),
    clears the robot, returns vials home and reports what is left for an operator.
    """
    try:
        return lash_e.safe_stop(reason)
    except (Exception, KeyboardInterrupt):
        lash_e.logger.exception("Safe stop failed")
