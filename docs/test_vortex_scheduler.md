# Vortex scheduler test (vial mixing)

Scheduler-compatible copy of `tests/test_vortex.py`; the original is left
untouched. Run normally for the existing startup GUI:

```powershell
python -m workflows.test_vortex_scheduler
```

The scheduler instead calls `execute(config=complete_config, show_gui=False)`.
`Lash_E` owns YAML/GUI review in the normal path; the experiment uses the
reviewed/confirmed values afterward. Vortexes `TARGET_VIAL` for
`VORTEX_TIME` seconds, returns the vial home, then homes the robot.

Requires a vial CSV (`INPUT_VIAL_STATUS_FILE`) because `vortex_vial` and
`return_vial_home` need vial identity and location — vial-less tasks may set
`INPUT_VIAL_STATUS_FILE=None`, but this task must not be run that way.
