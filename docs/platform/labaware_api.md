# Labaware API — integration guide

Labaware reads the state of every vial position on the north deck from one camera frame.
You POST an image; you get back, for each position, whether it is `empty`, `open-cap` or `closed-cap`.

| | |
|---|---|
| Base URL | `http://sdl4-not-for-gaming:8010` (Tailscale; IP `100.79.67.125`) |
| App id | `north_deck` |
| Auth | none — reachable only from inside the Tailscale network |
| Interactive docs | `http://sdl4-not-for-gaming:8010/docs` (OpenAPI / Swagger) |
| Dashboard | `http://sdl4-not-for-gaming:8010/apps/north_deck` |

---

## `POST /api/apps/north_deck/infer`

Send one frame, get the state of every slot.

**Request** — `multipart/form-data` with one field:

| field | type | notes |
|---|---|---|
| `image` | file | JPEG/PNG/BMP/TIFF/WebP from the deck camera. Must have an `image/*` content type. |

Optional query parameter: `detail=compact` (default) or `detail=full` (see [Full response](#full-response-detailfull)).

**Response** — `200 OK`, JSON:

```json
{
  "app_id": "north_deck",
  "states": {
    "vial_rack": [
      ["open-cap","open-cap","open-cap","empty","open-cap","empty"],
      ["empty","open-cap","open-cap","empty","empty","open-cap"],
      ["open-cap","empty","open-cap","open-cap","open-cap","open-cap"],
      ["closed-cap","closed-cap","closed-cap","closed-cap","closed-cap","closed-cap"],
      ["empty","empty","empty","empty","empty","empty"],
      ["empty","empty","empty","empty","empty","empty"],
      ["empty","empty","empty","empty","empty","empty"],
      ["empty","empty","empty","empty","empty","empty"]
    ],
    "clamp_1": "open-cap",
    "clamp_2": "empty"
  },
  "timing_ms": 170.4,
  "warnings": [],
  "history_id": 219
}
```

| field | meaning |
|---|---|
| `states.vial_rack` | 8 rows × 6 columns. `states.vial_rack[row][col]`. |
| `states.clamp_1`, `states.clamp_2` | single state string for each clamp. |
| state values | `"empty"`, `"open-cap"`, `"closed-cap"` |
| `null` instead of a grid/string | that labware was **not found** in the frame (camera blocked, moved, etc.). See `warnings`. |
| `timing_ms` | server-side processing time. |
| `warnings` | human-readable problems, e.g. `"rack clamp_2 (clamp_2) not detected"`. Empty list = all good. |
| `history_id` | id of this call in the dashboard history (the frame and full result are stored on the server). |

### Slot numbering (vial rack)

Rows are lettered A–H, columns 1–6; `vial_rack[0][0]` is **A1**, `vial_rack[7][5]` is **H6**.

| index | row | col |
|---|---|---|
| `vial_rack[r][c]` | `"ABCDEFGH"[r]` | `c + 1` |

This is the same numbering as the robot's `main_8mL_rack` `location_index`:
`location_index = row * 6 + col` (A1 = 0, A2 = 1, … B1 = 6, … H6 = 47).
To flatten: `flat = [s for row in states["vial_rack"] for s in row]` → `flat[location_index]`.

### Errors

| status | when |
|---|---|
| `400` | `image` missing, empty, or not an image content type |
| `404` | unknown app id |
| `503` | models not installed on the server (should not happen in production) |
| `422` | malformed request (FastAPI validation) |

Labware not being found is **not** an error: you still get `200` with `null` for that labware and a warning.
Treat any non-empty `warnings` as "don't trust this frame blindly".

### Latency

About 0.2–0.5 s per call on the server's GPU. The first call after a server restart takes ~4 s while models load.
Calls are processed one at a time; there is no need to batch.

---

## Examples

### curl (Windows PowerShell)

```powershell
curl.exe -F "image=@C:\path\to\frame.jpg" http://sdl4-not-for-gaming:8010/api/apps/north_deck/infer
```

### Python

```python
import requests

URL = "http://sdl4-not-for-gaming:8010/api/apps/north_deck/infer"
ROWS = "ABCDEFGH"

def read_deck(jpg_path: str) -> dict:
    with open(jpg_path, "rb") as f:
        r = requests.post(URL, files={"image": ("frame.jpg", f, "image/jpeg")}, timeout=30)
    r.raise_for_status()
    return r.json()

res = read_deck("frame.jpg")
if res["warnings"]:
    print("warning:", res["warnings"])

rack = res["states"]["vial_rack"]
if rack is not None:
    for r, row in enumerate(rack):
        for c, state in enumerate(row):
            if state != "empty":
                print(f"{ROWS[r]}{c + 1} (index {r * 6 + c}): {state}")

print("clamp_1:", res["states"]["clamp_1"], "clamp_2:", res["states"]["clamp_2"])
```

Sending an in-memory frame (e.g. from OpenCV) instead of a file:

```python
import cv2
ok, buf = cv2.imencode(".jpg", frame)
r = requests.post(URL, files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}, timeout=30)
```

---

## Full response (`?detail=full`)

Adds geometry for drawing overlays (what the dashboard uses). Most integrations don't need it.

```
POST /api/apps/north_deck/infer?detail=full
```

```jsonc
{
  "app_id": "north_deck",
  "labware": [
    {
      "id": "vial_rack", "name": "North vial rack", "type": "north_vial_rack_8x6",
      "missing": false,
      "conf": 0.95,                    // how sure the model is that it found the rack
      "bbox": [x1, y1, x2, y2],        // pixels
      "quad": [[x, y], ...4],          // corners: A1, end of row A, opposite, end of column 1
      "rows": 8, "cols": 6,
      "counts": {"empty": 38, "open-cap": 7, "closed-cap": 3},
      "slots": [
        {"name": "A1", "row": 0, "col": 0, "state": "open-cap", "conf": 0.99,
         "center": [x, y], "cell": [[x, y], ...4], "crop": [[x, y], ...4]}
        // ... one entry per slot
      ]
    }
    // clamp_1, clamp_2 (one slot each, named "clamp")
  ],
  "timing_ms": 170.4,
  "warnings": [],
  "history_id": 220
}
```

`slots[].conf` (0–1) is the per-slot confidence; it is only available here.

---

## Other endpoints

| method | path | use |
|---|---|---|
| `GET` | `/api/health` | liveness check → `{"status": "ok"}` |
| `GET` | `/api/apps/north_deck/detections?limit=200` | recent calls (summary per call) |
| `GET` | `/api/apps/north_deck/detections/{history_id}` | stored full result of one call |
| `GET` | `/api/apps/north_deck/detections/{history_id}/image` | the frame that was sent |
| `GET` | `/api/apps/north_deck/detections/export` | zip of all stored frames + per-frame CSV labels |

---

## Things to know

- **Every call is stored.** The frame and result are kept on sdl4 so they can be reviewed in the dashboard,
  corrected, and used to retrain the model. Don't send images that shouldn't be kept.
- **The model can be updated without changing the API.** Retraining happens from the dashboard; the response
  format stays the same.
- **Camera mode.** The models are trained mostly on the camera's IR/night mode. Daytime colour frames are less
  reliable until more of them are labelled.
- **Camera position.** The camera must keep roughly its current view of the deck. If it moves a lot, labware may
  come back as `null` until the locator is retrained.
