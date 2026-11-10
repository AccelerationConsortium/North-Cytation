# Scheduler deck check — plan

Date: 2026-11-10 · Status: implemented 2026-11-10 (see scheduler/README.md "Deck Check");
records live in `scheduler/state/deck_checks/` rather than `deck_check_<timestamp>/` folders.

## Goal

Before a live run, photograph the north deck, read it with the Labaware service
([API](../platform/labaware_api.md), client `tools/labaware_client.py`) and check
that every vial the queue's vial files expect is physically there with the right
cap, and nothing unexpected is. Flag differences, suggest fixes, and keep a record
of every check, especially mismatches.

Required like simulation: Run stays disabled until a deck check has been resolved
for the current inputs. It is never a hard block on the lab: comms failures and
mismatches can both be acknowledged and the run continued.

## What exists today (checked)

- `VialLayout` / `read_vial_occupancy()` already merge every queue row's
  `INPUT_VIAL_STATUS_FILE` into `{(location, index): [records]}`. Rows may use
  different vial files today; this union is the expected deck, so the check works
  now and when the queue moves to a single file.
- Run gating: `update_live_run_enabled()` requires a matching approved simulation
  fingerprint, no layout errors, no conflicts. Approval is cleared by any input or
  shared-state edit, Setup, Reset Robot, and consumed when Run starts.
- Camera: `tools/c110_camera.capture_photo(path, camera_folder, timeout)` grabs one
  RTSP frame via `Desktop/camera-control`. A frame captured this way was read
  correctly by the server today (history_id 224, 14 vials, no warnings).
- The server stores every frame it receives (by `history_id`).
- Cap columns: `capped` (bool) and `cap_type` (`open` = open-top/pierceable,
  `closed` = solid). `North_Safe` treats a vial as pipettable if uncapped or open.

## What the camera can and can't check

| Location | Checked? |
|---|---|
| `main_8mL_rack` (48 slots) | yes — slot by slot |
| `clamp`, `clamp_2` | yes (assumes server `clamp_1` = robot `clamp`) |
| `large_vial_rack`, `photoreactor_array`, heater, others | **no** — listed as "not checked" in the report |

## Comparison rules (per observable slot)

| Vial file expects | Camera sees | Result |
|---|---|---|
| nothing | empty | ok |
| nothing | vial | **Unexpected vial** |
| vial | empty | **Missing vial** |
| capped + `cap_type: closed` | closed-cap | ok |
| capped + `cap_type: open` | open-cap | ok |
| closed | open-cap (or vice versa) | **Wrong cap** |
| uncapped (`capped: False`) | ? | see decision 1 |
| anything | labware `null` | **Not seen** (rack/clamp not detected) |

Also flagged: any server `warnings`; and (if we use `detail=full`) slots with
confidence below a threshold, marked "low confidence" rather than a mismatch.

Suggested fixes come straight from the rows, e.g.
`B4 (index 9): expected "SDS_stock" (closed cap) — slot is empty. Place the vial or update the vial file.`
For a missing vial plus an unexpected vial of the same cap type elsewhere, suggest
"possibly moved from X to Y".

## User flow

1. New **Check Deck** button in the footer, before Simulate/Run. It's enabled
   whenever the queue has valid vial files, no layout errors and no pending shared edits.
   It's independent of simulation order (either can come first).
2. Click → runs in a background thread (capture can take up to ~20 s; first
   inference after server restart ~4 s). GUI stays responsive and shows a status line.
3. Result dialog, one of:
   - **Match** → "Deck matches the vial files (N vials checked; large rack/photoreactor not checked)". Check passes.
   - **Mismatch** → table of issues with suggested fixes, the photo, and buttons:
     *Recheck* (after fixing) · *Continue anyway* · *Cancel*. Continue requires
     ticking "I have checked these positions on the deck" (and optionally a note, decision 3).
   - **Comms failure** (camera or server) → "Deck check could not run: <reason>".
     Buttons *Retry* · *Continue without deck check*.
4. The Vial Layout tab colours mismatched slots (e.g. orange) with the issue in the
   tooltip until the next check or refresh.
5. Run becomes enabled only when simulation approval is valid **and** the deck check
   is resolved (pass, acknowledged mismatch, or acknowledged comms failure) for the
   same inputs. The Run confirmation dialog states which of those it was.

## Invalidation

The deck check result is tied to a fingerprint of the queue's vial files (contents),
`vial_positions.yaml`, and the queue rows. It is cleared by: any vial-file edit, Setup
save, queue change, shared-state save/reload, Reset Robot, and is consumed when a
live run starts (same as simulation approval). Optional staleness timeout — decision 4.

## Records

Every check is saved to `scheduler/state/deck_check_<timestamp>/` (Git-ignored like
other scheduler state):

- `frame.jpg` — the photo sent
- `response.json` — raw server reply (`detail=full`, so per-slot confidence is kept)
- `expected.csv` — the merged expected occupancy that was compared
- `report.json` — outcome (`match` / `mismatch` / `comms_failure`), issues, server
  `history_id`, user decision (recheck / continued / cancelled), note, timestamp,
  queue fingerprint, and whether a live run followed
- Appended one line to `scheduler/state/deck_checks.csv` (index of all checks),
  so mismatches can be filtered later.

Mismatches keep the `history_id` prominent, so the frame can be found and
corrected in the Labaware dashboard (useful for retraining when the model was wrong
rather than the deck).

## Code layout

- `scheduler/deck_check.py` (no Qt): `expected_from_occupancy()`, `compare(expected, reading) -> issues`,
  `suggest_fixes(issues)`, `save_check_record(...)`, `run_check(capture, client, ...)`.
  Pure functions, so they're fully unit-tested.
- `tools/labaware_client.py`: add per-slot confidence parsing for `detail=full`.
- `scheduler/scheduler_gui.py`: button, worker thread, result dialog, gating in
  `update_live_run_enabled()`, invalidation hooks, layout colouring.
- Settings (camera folder, server URL, confidence threshold) in one place, e.g.
  `robot_state/deck_check.yaml` or constants in `deck_check.py` (decision 5).

## Tests (software only, all mocked — never runs the robot)

- `test_deck_check.py`: every row of the comparison table; null labware; warnings;
  "moved" suggestion; record files written; comms failure path.
- `test_scheduler_deck_check_gui.py` (offscreen Qt, mocked capture + client): Run
  disabled until check resolved; mismatch → continue requires acknowledgement;
  comms failure → continue allowed; vial-file edit invalidates; Run consumes it.
- One manual live check against the real deck/server before relying on it.

## Risks / notes

- Camera view must be clear — the arm or a hand over the deck gives a false
  mismatch. Robot should be home before checking (it is before a run).
- Daytime colour frames are less reliable (per API doc); a mismatch may be the model.
- Only the starting deck is checked; vials added mid-queue by workflows aren't.
- Every check is stored on the server.

## Decisions (2026-11-10)

1. **Uncapped vials** are rare and are *not* `open-cap`. The camera has no uncapped
   state, so for them only presence is checked (any non-empty reading is ok) and the
   report notes "cap not checked".
2. **Clamps**: server `clamp_1` = robot `clamp`, `clamp_2` = `clamp_2` (naming only).
3. **Continue past a mismatch**: tick-box only; optional free-text note.
4. **Staleness**: no time expiry; invalidated only by input changes.
5. **Settings**: `robot_state/deck_monitoring.yaml` (general computer-vision settings).
6. **Confidence**: ignored. The check is an aid, not a guarantee; `detail=compact` is enough.
