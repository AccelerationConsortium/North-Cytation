# Vortex scheduler test (vial mixing)

The reviewed vortex diagnostic supports both normal and scheduler execution.
The redundant standalone hardware script has been retired. Run normally for
the existing startup GUI:

```powershell
python -m hardware_tests.test_vortex_scheduler
```

The scheduler instead calls `execute(config=complete_config, show_gui=False)`.
`Lash_E` owns YAML/GUI review in the normal path; the experiment uses the
reviewed/confirmed values afterward. Vortexes `TARGET_VIAL` for
`VORTEX_TIME` seconds, returns the vial home, then homes the robot.

Requires a vial CSV (`INPUT_VIAL_STATUS_FILE`) because `vortex_vial` and
`return_vial_home` need vial identity and location — vial-less tasks may set
`INPUT_VIAL_STATUS_FILE=None`, but this task must not be run that way.
