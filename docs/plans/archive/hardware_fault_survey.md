# Hardware fault survey

Surveyed 2026-11-16 to support designing structured error records. Read-only: no code, settings or packages were changed, and the robot was not run.

**Sources**

- `experiment_tracking/experiment_runs.csv` (4,393 rows)
- `logs/experiment_run_errors/*.txt` (126 files)
- `logs/*.log`
- North SDK `north` 0.3.0.6 and `ftdi_serial` 0.1.12 in `anaconda3/envs/north_async_test`
- `biotek_driver` 0.1, an editable install from `Desktop/cytation_driver`
- The repo files named in each section

**Line numbers.** In Part 1, file:line refers to the code as it was when the failure happened, so it is taken from the traceback. The code has moved since then, so the same function often shows different line numbers on different dates. In Parts 2 and 3, line numbers refer to the current files.

`logs/` is git-ignored, so none of this evidence is under version control.

---

## Part 1 — Real failures from the logs

### Selection

There are 108 live runs (`simulate == False`) with status `failed` or `interrupted_unknown_crash`, dated 2026-09-04 to 2026-11-10:

- 84 have status `failed`, and 24 have `interrupted_unknown_crash`.
- 82 have a traceback, and 26 do not (2 of these also have no log file).

### Failures by group

The counts add up to 108. "Origin" is the deepest frame in our own code; for hardware errors, the SDK frame where the error was raised is given as well.

| Group | Count | Exception type(s) | Example message (exact) | Origin (file:line at the time) | Dates |
|---|---|---|---|---|---|
| **Motor or axis fault** | 41 | `RuntimeError` (39 raw, plus 2 wrapped by our code: see notes) | `IO: MOTOR FAULT: A critical error has occured on a motor driver, restart the controller` | Raised at `north/north_c9.py:591 send_packet`. Surfaced in `North_Safe.py`: `drop_off_vial` ×19, `goto_location_if_not_there` ×11, `reset_after_initialization` (via `c9.open_clamp()`) ×5, `move_home` ×2, `home_robot_components` ×1, `dispense_into_wellplate` ×1, `_move_to_pipet_tip` ×1. Also `surfactant_grid_ailsa.py:1462` ×1 (wrapped) | 2026-09-04 → 2026-11-10 |
| **Hard E-stop** (its own group, because we can't tell whether an operator pressed it) | 2 | `RuntimeError` | `IO: SYSTEM ERROR: Hard E-Stop triggered, restart the system when it is safe to do so` | `north_c9.py:591`, reached from `North_Safe.py:1291 remove_pipet` and `North_Safe.py:3915 goto_location_if_not_there` | 2026-11-04 → 2026-11-05 |
| **Serial or communication failure** (all on the C9 controller link; none came from the scale, pumps or photoreactor) | 6 | `ftdi_serial.SerialReadTimeoutException` | `Read timeout` | Raised at `ftdi_serial.py:596 read`, reached through `north_c9.py` `serial_worker`. In all 6, the command was `SETO` from `c9.open_clamp()` at startup in `North_Safe.py reset_after_initialization` (lines 1150–1382 over time); one went through a temporary `traced_send` wrapper at `North_Safe.py:1043` | 2026-10-09 → 2026-11-05 |
| **Cytation or plate reader** | 2 | `RuntimeError`, `xml.etree.ElementTree.ParseError` | `Plate read error encountered` / `no element found: line 1, column 0` | `biotek_new.py:140 monitor_plate_read`; `workflows/cytation_labware_test.py:44 _labware` (that file no longer exists) | 2026-10-29 → 2026-10-30 |
| **Track or plate handling** (checks in our inventory code; no physical fault) | 9 | `ValueError` | `Reconcile pending transfer of plate-16c9f046243f before this move` | `plate_inventory.py:264 begin` ×2, `plate_inventory.py:137 plate_at` ×1, `mixed_wellplate_round_trip.py:35` ×4, `fluorescence_calibration_workflow.py:272/403` ×2 (`Load enough plates for the complete experiment`) | 2026-09-30 → 2026-10-30 |
| **Vial or deck state check** (our code refused the action) | 6 | `ValueError`, and `EOFError` at an operator prompt | `Clamp must be empty before starting the test` | `rack_cap_cycle.py:40/43` ×2, `ailsa_clamp_test.py:35`, `North_Safe.py:1722 resolve_pipetting_setup`, `North_Safe.py:32 pause_after_error` ×2 (the underlying errors were `Cannot move vial to destination, destination full` and `Can't move vial, vial is uncapped.`; see notes) | 2026-10-05 → 2026-11-05 |
| **Liquid handling or measurement check** | 2 | `ValueError` | `Validation measured volume must be a finite number greater than zero` | `embedded_calibration_validation.py:889`, `pipette_calibration_workflow.py:109 _number` (the measured mass was -0.20 mg) | 2026-09-04 → 2026-10-16 |
| **Software bug** (Python error in our own code) | 10 | `KeyError` ×5, `TypeError` ×2, `AttributeError` ×2, `ValueError` ×1 | `KeyError: None` | `North_Safe.py:129/131 get_config_parameter` ×4 (a held vial with no location during reset cleanup), `North_Safe.py:3833 get_vial_in_location` (`KeyError: 'location'`), `North_Safe.py:3897 uncap_clamp_vial`, `North_Safe.py:1192 reset_after_initialization`, `North_Safe.py:3941 vortex_vial`, `tests/test_move_vial.py:16`, `tests/test_gripper.py:16` | 2026-09-04 → 2026-11-04 |
| **Environment** (wrong interpreter) | 2 | `ModuleNotFoundError` | `No module named 'north'` | `master_usdl_coordinator.py:307 __init__`, run from the `north_env` env rather than `north_async_test` | 2026-09-30 |
| **Operator stop or prompt** | 2 | `KeyboardInterrupt`, `EOFError` | `EOF when reading a line` | `North_Safe.py:1274 check_input_file` (`input()` with no console); `North_Safe.py:1100 home_robot_components` (Ctrl-C during homing after `Reset stopped for operator review`) | 2026-09-30 → 2026-11-10 |
| **Unknown** (no traceback) | 26 | — | — | See the next table | 2026-10-01 → 2026-11-10 |

#### Notes on the groups

- **Hidden motor faults.** Two motor faults were wrapped by our own code, which hid them:
  - `ad8634a6` failed with `Failed to initialize measurement protocol: IO: MOTOR FAULT…`, wrapped at `sdl_pipette_calibration/experiment.py:720`.
  - `3435e2b5` failed with `Substock preparation failed: SDS_dilution_2`. Its log shows that five substock steps each failed with `IO: MOTOR FAULT…` within 10 seconds before the workflow gave up.
- **A motor fault does not name a motor.** The C9 reports source `IO` and error code 21 (`ERROR_MOTOR_FAULT`), but no axis. The fault is reported by whatever command comes next, which in this data was:
  - `AXST` (axis status poll during `goto_z` or `goto_xy_safe`) in most cases
  - `SETO` (`open_clamp`) in 5 cases
  - `MOAX` in 2 cases
  - `HOAX` in 1 case

  So the traceback location shows where the fault was *noticed*, not which motor faulted.
- **Read timeouts happen at the first command after connecting.** All 6 serial timeouts happened on the first real command after connecting (`open_clamp` in `reset_after_initialization`). That call sits outside the startup retry loop (see Part 3).
- **Two state-check failures ended as `EOFError`.** In `8f10de29` and `9ecfe627`, `check_for_errors` detected a bad vial state and called `pause_after_error`. Its `input()` then raised `EOFError` because no console was attached. The calibration protocol wrapped this as `Hardware initialization failed: ` with an empty message.
- **Three "live" runs started in simulate mode.** Their logs end in `_simulate.log` and show `SIMULATE flag updated: True -> False` partway through: `53bb9cce`, `7e669e9a` and `21f14304`.

### Failures with no traceback (26)

None of these 26 has a file in `logs/experiment_run_errors/`.

| Evidence in the log | Count | Run IDs |
|---|---|---|
| No log file either | 2 | `98a12c64`, `68bdfd47` (both `failed`, 2026-10-01) |
| Last line is `Cytation5 not connected... May need to restart` (`biotek_new.py` then waits at `input()`) | 3 | `2329daac`, `3560a2d0`, `16fde2af` |
| Last line is an ERROR from an operator pause: pipetting validation (`Stage 1 dispensed 0.36uL vs 400.0uL target…`, `VALIDATION FAILURE: All measurements below 10% of target…`), `Cannot dispense 0.025 mL into vial ethanol: would exceed capacity…`, or `Cannot move vial 12 while holding pipet…` | 5 | `d4277e0e`, `79b7fea1`, `47204490`, `c379caab`, `181fe5fc` (this one also logged a swallowed `Warning: Could not condition tip with DMSO_rf_1: IO: MOTOR FAULT…` one second earlier) |
| Log stops mid-action with no error (for example `Homing carousel...`, `Removing pipet`, `Dispensing 0.744 mL…`) | 16 | `c9d0cc6e`, `3280531b`, `21f14304`, `989f0184`, `17d20c8c`, `6b824f83`, `3ede0905`, `78e1055a`, `d13c972a`, `f4f1cb92`, `c6e00261`, `ff3e3878`, `c2e4d210`, `f98e371d`, `d84b891d`, `0f0be509` |

### Tracebacks with no row in the CSV (outside the selection)

Twenty traceback files in `logs/experiment_run_errors/` have no matching `run_id` in the CSV. Their only date is the file's modified time (2026-09-15 to 2026-10-30). They add to the hardware picture:

| Type | Count | Message | Dates (file mtime) |
|---|---|---|---|
| `ftdi_serial.SerialReadTimeoutException` | 9 | `Read timeout` (8 at `reset_after_initialization` and `open_clamp`, 1 at `goto_location_if_not_there` and `AXST`) | 2026-09-16 → 2026-09-23 |
| `RuntimeError` motor fault | 4 | `IO: MOTOR FAULT: …` | 2026-09-15 → 2026-09-23 |
| `RuntimeError` hard E-stop | 3 | `IO: SYSTEM ERROR: Hard E-Stop triggered, …` | 2026-09-15 |
| Our-code errors | 4 | `NameError`, `ValueError` ×2, `FileNotFoundError` | 2026-09-16 → 2026-10-30 |

### Hardware errors that were caught and did not end a run (all of `logs/*.log`)

| Message | Log files | Lines | Notes |
|---|---|---|---|
| `Main: SCALE ERROR: Overload condition` | 9 | 17 | All logged as `Scale reading failed: …` (the reading was skipped). Dates 2026-05-29 → 2026-08-07 |
| `Sequence: CAPPING FAULT: Torque threshold not reached` | 2 | 4 | 2025-09-25 and 2025-10-09, inside the startup retry loop. The second one gave up after 2 attempts |
| `Physical cleanup failed (attempt n/2): …` | 467 | 498 | Of the 2026 logs, 306 are `Main: HOME REQUIRED: Home the axis before moving`. This is the normal first pass that triggers homing. Others include `Read timeout` ×6, `IO: MOTOR FAULT…` ×6 and Python errors (`None` ×49, `'location'` ×9, …) |
| `Cytation5 not connected... May need to restart` | 109 | 109 | `biotek_new` startup status was not 0 |
| `Failed to start read.` | 3 | 3 | `start_read()` returned `None`. Dates 2025-09-26 and 2026-01-07 |
| `Plate read was aborted`, `Error running mpremote` | 0 | 0 | Never seen. The mpremote message is only printed, so it would not reach a log anyway |
| `PUMP ERROR`, `AXIS ERROR`, `Soft E-Stop` | 0 | 0 | Never seen |

---

## Part 2 — What the hardware packages raise

### North SDK (`north` 0.3.0.6) and `ftdi_serial` 0.1.12

| Item | Where | What it is |
|---|---|---|
| `C9Errors` | `north/north_c9.py:3113` | **Not an exception class.** It is a plain object with `.source` (cog: 0 Main, 1 IO, 2 Axis Network, 3 C9 Network, 4 Motion, 5 Aux COM, 6 Sequence) and `.error` (codes 0–23). Its `__str__` builds messages such as `IO: MOTOR FAULT: …` |
| How controller errors are raised | `north_c9.py:586`, `591` (`send_packet`) | `raise RuntimeError(C9Errors(src, code))`. The exception type is always plain `RuntimeError`; the structured codes are in `exc.args[0].source` and `exc.args[0].error` |
| Codes relevant here | `north_c9.py:3122–3145` | `ERROR_MOTOR_FAULT = 21`, `ERROR_HARD_STOP = 8`, `ERROR_SOFT_STOP = 9`, `ERROR_AXIS_NO_RESPONSE = 1`, `ERROR_PUMP_NO_RESPONSE = 10`, `ERROR_SCALE_NO_RESPONSE = 13`, `ERROR_SCALE_OVERLOAD = 17`, `ERROR_CAPPING_UNDER_TORQUE = 22`, `ERROR_HOME_REQUIRED = 23`, plus CRC, overflow and parse errors |
| Timeouts and communication | `ftdi_serial.py:26–61` | `SerialException` → `SerialTimeoutException` → `SerialReadTimeoutException` / `SerialWriteTimeoutException`; also `SerialInvalidDeviceException`, `SerialDeviceOpenException`, `SerialDeviceNotImplementedException`, `SerialFtd2xxNotAvailableException`. These pass through `FTDISerialControllerNetwork.serial_worker` (`north_c9.py:2889`) and are re-raised in `send()` (`north_c9.py:2853`) |
| Bad CRC | `north_c9.py:2865` | Bare `raise IOError`, after `print("CRC Error...")` |
| Retries inside the SDK | `north_c9.py:525`, `207`, `206` | `send_packet(..., retries=3)`: up to 4 attempts with `ERROR_DELAY = 0.5` s between them. The retry applies **both** to transport exceptions **and** to `ERR!` replies, so a motor fault is re-sent 3 times before it is raised. Pump commands use `N_PUMP_RETRIES = 3`. Serial read timeout is `TIMEOUT = 0.6` s (`north_c9.py:2753`) |
| No timeout on motion waits | `north_c9.py:96–103` (`CmdToken.wait`) | `while not self.is_done(): sleep(...)` with no limit. A hung move only ends if a status poll raises |
| Other `RuntimeError`s | `north_c9.py:1985`, `2014` | `Pump volume too full to aspirate …`, `Cannot move pump to …` |
| Other exception classes | `north/north_project.py:155` | `LegacyProjectError(ValueError)`, for project files only. There are no motor- or communication-specific exception classes |

### `biotek_driver`

It handles errors in three different ways depending on the call:

| Behaviour | Calls | Where |
|---|---|---|
| **Every COM error becomes `RuntimeError`** | `pythoncom.com_error` is turned into `RuntimeError("Failed to invoke method …")` | `ole_handler.py:17–92` |
| **Raises** | Gen5 startup (`RuntimeError("COM Error: …")`), `configure_*_reader`, `test_reader_communication`, `carrier_in`/`carrier_out`/`carrier_in_ex`, and the `PlateReadMonitor.errors_count`/`get_error_code`/`get_error_message` methods | `biotek.py:57–64`; `application.py:213–280`, `391–455`; `monitors/plate_read_monitor.py:59–114` |
| **Returns an error value** | `get_reader_status()` returns -1 on a COM error (the same as "Busy"; -2 means not communicating, -3 not configured). `Plate.read_status` returns -1 on a COM error (otherwise 0–5, where 4 is Error). `start_read()` returns `None`. `new_experiment()` returns `None`. `get_procedure()` and `get_plate_layout()` return `""`. Several `Experiment` property getters return `False` | `application.py:282–302`, `322–349`; `plates.py:116–148`, `245–279`, `638–664`; `experiment.py:63–216` |
| **Prints only** | A failed connection test at startup: `print(f"Failed to connect to {self.app_name}. Error: …")`. It does not raise | `biotek.py:76–81` |

In our wrapper (`biotek_new.py`):

- `monitor_plate_read` raises `Plate read error encountered` on status 4, but never asks the monitor *why*: `errors_count` and `get_error_message` are not called.
- `ParseError: no element found` in `0eac013c` is consistent with `get_procedure()` returning `""` after a failed COM call. This can't be confirmed, because the calling test file has since been deleted.

### `photoreactor_controller.py`

`run_command` (lines 4–21) never raises and never returns a value:

- It calls `subprocess.run(['mpremote', 'connect', 'COM6', 'exec', command], ...)` without `check=True` and without a `timeout`.
- A non-zero exit code is ignored. If stderr is not empty, it is printed after `Errors:`.
- If `mpremote` cannot start (for example, it is not installed), the exception is caught and printed as `Error running mpremote: …`.
- Nothing goes to the logger, so the caller cannot tell that the call failed.
- A hung `mpremote` blocks forever.

The controller is created in `master_usdl_coordinator.py:354–355`. No photoreactor errors appear in any log.

---

## Part 3 — Retries and recovery already in our code

### `North_Safe.py`

| Function | Lines | What is caught | What happens |
|---|---|---|---|
| `pause_after_error` | 23–50 | (Called by our code, not a catch) | Logs the error, sends a Slack message, and **waits at `input()`** for the operator; this is the main "wait" mechanism (35 call sites). `KeyboardInterrupt` is re-raised. With no console attached, `input()` raises `EOFError` (seen 3 times in Part 1) |
| `reset_after_initialization` | 1156–1202 | `Exception` from `load_pumps()` + `_perform_physical_cleanup()` | **Retry loop** (`max_retries=2`): logs `Physical cleanup failed (attempt n/2)`, calls `home_robot_components()`, then tries again. Re-raises after 2 failures, or straight away if homing fails. The calls just before the loop (`open_clamp`, `zero_scale`, `set_output` at 1168–1171) are **not covered**; that is where all 6 logged startup `Read timeout`s and 5 motor faults happened |
| `home_robot_components` | ~1100–1154 | `Exception` | Logs `Failed to home robot components` and re-raises. No recovery |
| `pipet_aspirate` | 1527–1534 | Bare `except:` around `c9.aspirate_ml` | Assumes the volume limit was exceeded and moves the pump to the maximum position (`move_pump`). Because the `except` is bare, it would also catch a communication error or motor fault |
| `aspirate_from_vial` → nested `read_scale_continuous` | 2115–2146 | `Exception` from `c9.read_scale()` | Logs `Scale reading failed: …`, skips the sample and continues (5 Hz loop) |
| `dispense_into_vial` → nested `read_scale_continuous` | 2684–2714 | Same | Same |
| `dispense_into_vial_from_reservoir` → nested `read_scale_continuous` | 3484–3516 | Same | Same |
| `monitor_weight_over_time` | 2988–3056 | `Exception` from `c9.read_scale()` | Logs, sleeps 0.1 s, continues |

Catches in `North_Safe.py` that are not about hardware:

- File and state reading: `_load_yaml_file` 81–90, `_load_state_files` 966/1016/1018, `get_robot_status` 1286.
- Calibration lookup: 1671, 1740, 1818.
- Transfer splitting: `plan_vial_transfer` 2363–2368.
- Index parsing: `dispense_into_wellplate` 3110, `get_location` 4208/4213.
- `cl_pow_dispense` (701–740) has a `max_tries=20` loop, but it repeats powder dosing; it is not an error retry.

### `master_usdl_coordinator.py`

| Function | Lines | What is caught | What happens |
|---|---|---|---|
| `MeasurementHandle._run_cytation` (background thread) | 116–150 | `RuntimeError` from `cytation.run_protocol` (139); any `Exception` (147) | Saves the error and stops the remaining reads. No retry |
| `MeasurementHandle.complete` | 153–177 | (Passes the saved error on) | `_done_event.wait()` with **no timeout** (173). If an error was saved, calls `pause_after_error` (operator wait) and then re-raises |
| `measure_wellplate` | 584–586 | `RuntimeError` from a plate read | Logs `Plate read failed` and re-raises. No recovery; the plate is left in the reader |
| `_run_plate_protocol` | 649–654 | `try/finally` only | Clears the "reading" flag |
| `move_wellplate_back_from_cytation` | 515–530 | — | Its comment says it is ordered "so completion is retryable", but nothing retries it |

There are **no retry loops** in this file. The catches at 20 (`ImportError`) and 452/470 (status GUI) are not about hardware.

### `biotek_new.py`

The file has no `try`/`except` at all.

| Function | Lines | Waiting or recovery behaviour |
|---|---|---|
| `__init__` | 81–90 | If the reader status is not 0, it logs `Cytation5 not connected... May need to restart` and **waits at `input()`**. Afterwards it carries on without checking the status again |
| `monitor_plate_read` | 106–142 | Polls `read_status` every 2 s with **no timeout**. It raises on 2 (aborted) and 4 (error) and keeps waiting on 3 (paused). It treats -1 (the driver's COM-error value) as "keep polling", so a lost connection could loop forever |
| `run_protocol` | 298–371 | Raises `Failed to create experiment from protocol file` if `new_experiment` returns `None`. No retry |

---

## Could not determine

- **Cause of the 26 failures with no traceback.**
  - For 16 of them, the log just stops with no error, consistent with the process being killed (window closed, power, or a crash in native code); the logs alone can't tell these apart.
  - For 8 of them, the last line is an operator-pause or Cytation prompt, which suggests the process was killed while waiting at `input()`. This is unconfirmed.
- **Which motor or axis faulted.** The C9 motor-fault reply has no axis number, and the logs don't record one.
- **Whether a person pressed the hard E-stop** in `ed623d36` and `d58d4430`.
- **Root cause of the Cytation `Plate read error encountered`** (`aa753b36`). Gen5's error details were never read.
- **Root cause of `0eac013c`.** The test file `workflows/cytation_labware_test.py` has since been deleted.
- **Exact dates of the 20 orphan tracebacks.** They come from file modified times only.
- **Photoreactor and scale/pump faults during the CSV period.** None were found, but `photoreactor_controller.py` only prints (it doesn't log), so failures there would not appear in the logs.
- **Behaviour on the real devices.** No hardware was exercised; everything in Part 2 comes from reading the code only.
