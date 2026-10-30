# Cytation labware diagnostic

## XML-only test: no physical plate required

From the repository root, with the same Python environment used for normal reads:

```powershell
python -m scripts.cytation_xml_only
```

This first tests the 96-well procedure with the known rejected `WellSelection`
and `Options` blocks removed, matching the existing absorbance builder's
top-level structure. Original read steps are retained. The original and returned
XML preserve the evidence of any resulting option/default changes. If that
succeeds, test changing the labware:

```powershell
python -m scripts.cytation_xml_only --plate-type "48 WELL PLATE"
```

The script uses the existing reader initialization and `run_protocol()` hook,
with `run_plate()` overridden to return without starting a read. It does not
initialize `Lash_E`, the robot or track, or call carrier movement, abort or close.
Gen5/the reader must be available for normal initialization and validation.
Do not run alongside another reader job. It never modifies the on-disk `.prt`.

Each run prints its folder under `output/cytation_xml_only/`. Start review with
`report.md`; then compare `requested.diff`, `original.xml`, `requested.xml` and
`readback.xml`. `summary.json` records accepted labware, Gen5 row/column counts,
whether procedure steps were retained, A1 selection acceptance and validation.
`diagnostic.log` captures driver errors, including those swallowed by setters.
The script makes one attempt and stops on a rejection; it does not strip
additional tokens automatically. A1 acceptance means the API returned without
a logged error; only an eventual physical read can confirm measured wells.

These probes test XML acceptance only, not a plate read or fluorescence support.

## Physical read test

Run from the repository root:

```powershell
python -m workflows.cytation_labware_test
```

The normal startup configuration review defaults to simulation, a lidless
`96 WELL PLATE`, and `C:\Protocols\CMC_Absorbance_96.prt`. Simulation exercises
the transfer sequence only; it cannot verify Gen5 XML support or a real read.
Select `SIMULATE: false` in the review when ready for a live test.

Start with exactly one active plate in the pipetting area, with a sample already
in A1. The Cytation tray and track gripper must be empty. Reconcile the saved
active plate type and lid state with the physical plate. Source-stack inventory
is not used. No pipetting, separate shake protocol, or discard step runs. A
successful test returns the same plate to the pipetting area, ready for another
run. Startup initialization itself may home hardware.

Use the same turbidity `.prt` for each mode; it is never rewritten on disk:

| TEST_MODE | WELLPLATE_TYPE | Action |
| --- | --- | --- |
| `baseline` | `96 WELL PLATE` | Read the original procedure without modifying it. Start here. |
| `roundtrip` | `96 WELL PLATE` | Reapply the exact original procedure XML through `set_procedure()`. |
| `replace_labware` | `48 WELL PLATE` | Change only `Labware/Name` in the original procedure XML, then call `set_procedure()`. |
| `modifiable_labware` | `48 WELL PLATE` | Try the corresponding change through `set_modifiable_procedure()`. Driver documentation predicts rejection across plate formats. |

The workflow calls the existing `measure_wellplate()` method, which transfers
the active plate, calls the existing reader `run_protocol()`, and returns the
plate. An optional `plate_setup` callback runs immediately after `plates.add()`
and before the normal read. All other callers retain their existing behavior.
The callback saves the original and modifiable XML, logs original validation,
applies the selected change, saves readback, and validates again. It refuses
to read unless the returned labware matches the physical plate's Cytation
name, the procedure remains lidless, the StepList is unchanged, and Gen5
validation returns success. This catches a setter that merely logs a failure.

The existing reader code selects A1 and starts and monitors the read. Connection,
carrier movement, read monitoring, data extraction, return and track origin use
the usual coordinator/wrapper path. The diagnostic adds no separate polling,
abort, timeout, session close, or driver initialization. Logs include the
protocol SHA-256, XML changes, validation results and normal reader messages.

Artifacts under `output/cytation_labware_test/<UTC timestamp>/` include:

- `diagnostic.log` and `summary.json` (phase, success, exception, final inventory).
- `original.xml`, `modifiable.xml`, `requested.xml`, `readback.xml` when available.
- `requested.diff` for modification modes.
- `results.csv` containing the normal coordinator result when a read succeeds.

Failures follow the existing read path and trigger no additional recovery motion.
Inspect the logged location and reader state before recovery. Earlier versions'
`READ_TIMEOUT_SECONDS` and `ABORT_TIMEOUT_SECONDS` settings are no longer used.
The diagnostic does not call `experiment.close()` or restart Gen5 or the reader.

Success establishes this procedure's XML round trip and A1 read on the selected
plate. It does not establish all-well addressing, fluorescence compatibility,
or compatibility with `shake_5_wait_5.prt`.
