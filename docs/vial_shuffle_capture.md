# Vial shuffle photo collection

From the repository root, run:

```powershell
python workflows/vial_shuffle_capture_workflow.py
```

The normal workflow review selects the vial CSV, simulation mode and settings.
Settings are saved in `workflow_configs/vial_shuffle_capture_workflow.yaml`.
The default is simulation; change `SIMULATE` to `false` for hardware collection.
Use a CSV supported by the robot, with accurate current positions, cap flags and
vial types. Other file formats are not converted by this workflow.

| Setting | Default | Meaning |
| --- | --- | --- |
| `INPUT_VIAL_STATUS_FILE` | `status/sample_input_vials.csv` | Robot's tracked vial input |
| `ITERATIONS` | 10 | Complete shuffled photo/CSV pairs to collect |
| `TARGET_EXPERIMENT_TIME_HOURS` | 0 | Target collection duration, including cycle work; 0 disables pacing |
| `MIN_VIALS_MOVED`, `MAX_VIALS_MOVED` | 3, 8 | Random number of distinct participating vials per image |
| `MIN_OFF_RACK`, `MAX_OFF_RACK` | 0, 14 | Total participating vials outside the main rack at exposure |
| `CAMERA_FOLDER` | Desktop `camera-control` | Existing C110 camera program directory |
| `CAPTURE_TIMEOUT_S` | 20 | Timeout per capture attempt |
| `CAPTURE_ATTEMPTS` | 3 | Maximum attempts on the same arrangement |
| `SETTLE_SECONDS` | 1 | Delay after moving the robot home, before capture |
| `RANDOM_SEED` | null | Optional repeatable random seed |

Only capped `8_mL` vials participate. Other vials remain fixed and block their
occupied positions. Random destinations include all available positions in
`main_8mL_rack`, `heater`, `photoreactor_array`, and `clamp`. They must be present
and movable in the robot's configuration. Equipment is used only for placement;
the workflow does not enable heating, illumination or stirring. Off-rack counts
include participating vials only. If fewer vials are available than the move
limits, both move limits are capped at the available number. Off-rack targets
are selected within the configured limits and the selected subset's reach.
Impossible settings stop the run rather than violating the limits.

Swaps use an empty position as a temporary buffer; this may require an extra
transfer of a selected vial. At least one destination must be empty. Participating
vials must start within the enabled destinations. The robot must have no held
pipette or object when collection begins.

Each iteration finishes vial transfers, calls the robot's `move_home()`, waits
for settling, and captures one fresh frame. Snapshot CSVs come from the current
`VIAL_DF`; normal robot state persistence continues unchanged. Capture never
calls detection. After the requested number of pairs, participating vials return
to their recorded starting positions (which may differ from CSV home fields),
and the robot moves home again.

For example, `TARGET_EXPERIMENT_TIME_HOURS: 24` and `ITERATIONS: 500` gives
each cycle a 172.8-second interval. Planning, movement, homing, capture retries,
file saves and optional verification consume that interval: a 120-second cycle
leaves 52.8 seconds to wait. Shuffle starts follow deadlines anchored to the
first shuffle using a monotonic clock, so small delays do not accumulate.
If a cycle runs late, the workflow skips the wait, logs the delay and continues.
It still collects every requested sample, even if collection takes longer than
the target. Faster later cycles can catch up to the original schedule.

The robot remains home while waiting. There is no initial or final wait. This
is a pacing target, not a minimum runtime or a shutdown deadline. Startup and
returning vials afterward are additional time. Photo spacing can vary because
individual shuffles take different amounts of time. Simulation logs each cycle's
estimated remaining wait without sleeping; it does not predict hardware speed.

`summary.json` records `sample_interval_seconds`, each pair's `cycle_seconds`,
`wait_seconds` (planned delay) and `schedule_lag_seconds`, and total
`elapsed_seconds` including restoration. The parameter replaces
`ESTIMATED_LENGTH_HOURS`: update old presets to use the new name. Existing
scheduler rows use private config copies; reload the updated preset or recreate
the row to pick up the new parameter.

Outputs go in a unique directory under
`output/vial_shuffle_capture_workflow/<timestamp>_LIVE/`:

```text
000001.jpg
000001.csv
000002.jpg
000002.csv
initial_state.csv
final_state.csv
config.json
summary.json
```

`summary.json` lists complete pairs, actual transfers, distinct vials moved,
off-rack occupancy, completion status, restoration status and verification
counts. Initial and final states are metadata and do not count toward the
requested images. Files are staged before publishing; consumers should use the
manifest's complete pairs. Simulation writes clearly labeled synthetic images
in a separate `_SIMULATION` directory and never connects to the camera or Slack.

Capture failures retry without additional vial motion. Exhausted retries stop
collection. Motion failures and operator interruptions preserve the tracked
state and available pairs for recovery; they do not trigger automatic return
movements when physical state may be uncertain. Successful completion restores
starting positions. A live run attempts one best-effort end-of-run Slack message
with image count, outcome and restoration status. Notification failure is logged
and does not fail the collection.

## Separate capture and detection

`tools/c110_camera.py` provides independent `capture_photo(...)` and
`detect_photo(...)` functions. The standalone `tests/Get_C110_Frame.py` now
captures only by default. Explicit `--detect` restores its optional detection
step; importing the script performs no capture, detection or file writes.

## Future vision verification

The model-to-CSV comparison is deferred. A callable can later be supplied through
`execute(..., verifier=callback)`. For each complete pair, it receives the photo
and CSV paths while the arrangement is still unchanged, and must return a
JSON-serializable mapping containing a boolean `matched` plus any comparison
evidence or raw model response. Each result is saved as
`000001.verification.json`. Matching rules and detector label/position mapping
belong in that callback, separate from capture and movement.

When a verifier is supplied, the summary and Slack notification include matches,
successfully verified images, mismatches, and verification errors. A verification
error does not discard the pair or count as a match. Without a callback, the
workflow performs no verification and explicitly reports it disabled.

Hardware-free checks:

```powershell
python -m unittest discover -s tests -p test_vial_shuffle_capture.py
```
