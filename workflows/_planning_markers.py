"""Log markers that let planning tools reconstruct scheduled waits.

Simulation skips real waits, so a simulated log alone cannot show that a run
idles until a measurement timepoint. Call these at the same places in live and
simulated runs; they only log, and never change control flow or timing.

    clock = f"plate{plate}_block{block}"
    mark_clock_start(logger, clock)            # when the schedule clock starts
    for minutes in schedule:
        mark_wait_until(logger, clock, minutes * 60)   # before waiting/reading
        ...

The lab Slack agent's duration model advances a virtual clock to
clock_start + offset at each wait_until, matching the workflow's
`wait = target - time.monotonic(); if wait > 0: sleep(wait)`.
"""

import re

_CLOCK_ID = re.compile(r"[A-Za-z0-9_.:-]{1,80}")


def _clock(clock_id):
    clock_id = str(clock_id)
    if not _CLOCK_ID.fullmatch(clock_id):
        raise ValueError(f"Planning clock id must be 1-80 of [A-Za-z0-9_.:-]: {clock_id!r}")
    return clock_id


def mark_clock_start(logger, clock_id):
    """A schedule clock starts now (e.g. right after dispensing a plate/block)."""
    logger.info("PLANNING clock_start id=%s", _clock(clock_id))


def mark_wait_until(logger, clock_id, offset_seconds):
    """The run will not proceed before clock start + offset_seconds."""
    offset = float(offset_seconds)
    if not offset >= 0:
        raise ValueError("Planning wait offset must be a nonnegative number of seconds")
    logger.info("PLANNING wait_until id=%s offset_s=%.3f", _clock(clock_id), offset)


def mark_wait(logger, clock_id, seconds):
    """A fixed wait of `seconds` starting now (e.g. a reaction or settling time)."""
    mark_clock_start(logger, clock_id)
    mark_wait_until(logger, clock_id, seconds)
