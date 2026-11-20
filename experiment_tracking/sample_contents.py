"""Stored contents: what every container holds right now, with its uncertainty.

Agreed design (Owen, 2026-11-18; see Integration_Plan/SAMPLE_CONTENTS_DESIGN.md):
- Contents travel with the sample. Every container (vial, well, reservoir or cartridge load, and
  the pipette tip) has a record: total volume, each material's concentration, the stocks it came
  from, and the uncertainty of each concentration. The tip picks up the source's record when it
  aspirates; a dispense mixes it into the destination's record.
- The uncertainty travels too, split by the run each part came from, so an interrupted run still
  leaves every sample with its contents and its ± up to the last step.
- A stock vial needs no record: it holds its stock (from stock_id), at whatever volume the vial
  file says. A record is written once the robot changes a container.
- Correcting a volume by hand changes only the amount, never the composition.
- status/container_contents.csv holds the records. It is live state, like robot_status.yaml:
  written only by the robot during live runs (a scheduler simulation uses its own copy) and kept
  out of git. Snapshots go to output/ with each run's data. The transfer log is kept as history.

The uncertainty maths is the same as the log-based calculation (sample_tracking.concentration_sd).
For c = Σ cₖvₖ / V, with each dispense volume vₖ having error σₖ:

    var(c) = [Σ (cₖ − c)² σₖ²  +  Σ vₖ² var(cₖ)] / V²
           = [A − 2cB + c²C + D] / V²,   A = Σ cₖ²σₖ², B = Σ cₖσₖ², C = Σ σₖ², D = Σ vₖ² var(cₖ)

so each record keeps A, B, C and D, split by run and (for A, B, C) into the random
(dispense-to-dispense) and shared (day-to-day) parts of the dispense error. That gives:
- sd_absolute: everything combined.
- sd_within_run for run R: only the random error of run R's dispenses into this container
  (for comparing containers filled in the same run).
Simplification: when liquid is taken out of a container, the record keeps its concentrations and
their uncertainty (the sums are rescaled); later additions then treat it as one mixed component.

Nothing here raises into a workflow; callers catch and warn.
"""

import copy
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENTS_FILE = REPO_ROOT / "status" / "container_contents.csv"
COLUMNS = ["container_id", "volume_mL", "material", "units", "concentration", "sd_absolute", "from_stocks",
           "complete", "updated", "state"]
KINDS = ("random", "shared")


def _key(material, units):
    return f"{material}|{units}"


def split_key(key):
    material, _, units = key.partition("|")
    return material, units


# --- records ------------------------------------------------------------------------------------

def empty_record(volume_mL=0.0):
    """A container's record. ``materials`` maps 'material|units' to concentration (in those units).
    ``known`` is False once anything went in whose uncertainty can't be estimated; ``complete`` is
    False once anything went in whose contents aren't known."""
    return {"volume_mL": float(volume_mL), "materials": {}, "stocks": [], "complete": True, "known": True,
            "C": {}, "AB": {}, "D": {}, "solids_mg": {}}


def stock_record(stock_id, stocks, volume_mL):
    """The record of a container that simply holds a stock (treated as exact)."""
    record = empty_record(volume_mL)
    rows = [r for r in stocks if r["stock_id"] == stock_id]
    if not rows:
        record["complete"] = False  # a stock_id that isn't in the stock list
        return record
    for row in rows:
        if row.get("concentration") not in (None, ""):
            record["materials"][_key(row["material"], row.get("units", ""))] = float(row["concentration"])
    record["stocks"] = [stock_id]
    return record


def unknown_record(volume_mL):
    """A container holding liquid we know nothing about (no stock, no record)."""
    record = empty_record(volume_mL)
    record["complete"] = record["volume_mL"] <= 0
    record["known"] = record["volume_mL"] <= 0
    return record


def variance_by_run(record, key):
    """{run: variance of this material's concentration} or None when it can't be estimated."""
    if not record["known"]:
        return None
    volume = record["volume_mL"]
    if volume <= 0 or key not in record["materials"]:
        return None
    c = record["materials"][key]
    runs = set(record["C"]) | set(record["AB"].get(key, {})) | set(record["D"].get(key, {}))
    result = {}
    for run in runs:
        total = record["D"].get(key, {}).get(run, 0.0)
        for kind in KINDS:
            A, B = record["AB"].get(key, {}).get(run, {}).get(kind, (0.0, 0.0))
            C = record["C"].get(run, {}).get(kind, 0.0)
            total += A - 2 * c * B + c * c * C
        result[run] = max(total, 0.0) / volume ** 2
    return result


def sd_absolute(record, key):
    parts = variance_by_run(record, key)
    return None if parts is None else math.sqrt(sum(parts.values()))


def sd_within_run(record, key, run_id):
    """The random error of run_id's own dispenses into this container (None if unknown)."""
    if variance_by_run(record, key) is None:
        return None
    c = record["materials"][key]
    A, B = record["AB"].get(key, {}).get(run_id, {}).get("random", (0.0, 0.0))
    C = record["C"].get(run_id, {}).get("random", 0.0)
    return math.sqrt(max(A - 2 * c * B + c * c * C, 0.0)) / record["volume_mL"]


def rescale(record, volume_mL):
    """Set the volume without changing what's in it (a hand correction, or liquid taken out)."""
    old = record["volume_mL"]
    if old > 0 and volume_mL > 0 and abs(old - volume_mL) > 1e-12:
        f = (volume_mL / old) ** 2
        for run in record["C"].values():
            for kind in run:
                run[kind] *= f
        for runs in record["AB"].values():
            for run in runs.values():
                for kind in run:
                    run[kind] = [run[kind][0] * f, run[kind][1] * f]
        for runs in record["D"].values():
            for run in runs:
                runs[run] *= f
    record["volume_mL"] = max(float(volume_mL), 0.0)
    return record


def take(record, volume_mL):
    """Take volume_mL out: returns (portion, what's left). The portion has the same contents."""
    portion = {"materials": dict(record["materials"]), "stocks": list(record["stocks"]),
               "complete": record["complete"],
               "variance": {key: variance_by_run(record, key) for key in record["materials"]},
               "solids_mg": {}}
    left = copy.deepcopy(record)
    if record["solids_mg"] and record["volume_mL"] <= 0:
        portion["complete"] = False  # undissolved solid with no liquid: can't say how much was taken
    rescale(left, max(record["volume_mL"] - volume_mL, 0.0))
    return portion, left


def add(record, portion, volume_mL, error_mL, run_id):
    """Mix volume_mL of ``portion`` into ``record`` (returns the new record).

    error_mL is (random, shared) for this dispense, or None when it can't be estimated.
    """
    record = copy.deepcopy(record)
    run = run_id or ""
    v0 = record["volume_mL"]
    v1 = v0 + volume_mL
    if v1 <= 0:
        return record
    keys = set(record["materials"]) | set(portion["materials"])
    record["materials"] = {k: (record["materials"].get(k, 0.0) * v0 + portion["materials"].get(k, 0.0) * volume_mL) / v1
                           for k in keys}
    if error_mL is None:
        record["known"] = False
    else:
        for kind, sigma in zip(KINDS, error_mL):
            s2 = float(sigma) ** 2
            slot = record["C"].setdefault(run, {})
            slot[kind] = slot.get(kind, 0.0) + s2
            for k in keys:
                ck = portion["materials"].get(k, 0.0)
                ab = record["AB"].setdefault(k, {}).setdefault(run, {}).setdefault(kind, [0.0, 0.0])
                record["AB"][k][run][kind] = [ab[0] + ck * ck * s2, ab[1] + ck * s2]
    for k in portion["materials"]:
        inherited = portion["variance"].get(k)
        if inherited is None:
            record["known"] = False
            continue
        for source_run, var in inherited.items():
            slot = record["D"].setdefault(k, {})
            slot[source_run] = slot.get(source_run, 0.0) + volume_mL ** 2 * var
    record["stocks"] = sorted(set(record["stocks"]) | set(portion["stocks"]))
    record["complete"] = record["complete"] and portion["complete"]
    record["volume_mL"] = v1
    if record["solids_mg"]:  # powder that was waiting for liquid
        for material, mg in record["solids_mg"].items():
            k = _key(material, "mg/mL")
            record["materials"][k] = record["materials"].get(k, 0.0) + mg / v1
        record["solids_mg"] = {}
        record["known"] = False
    return record


def add_solid(record, materials, mg, stock_id=None):
    """Powder into a container: mg of each named material (uncertainty not estimated)."""
    record = copy.deepcopy(record)
    for material in materials or ["unknown powder"]:
        if record["volume_mL"] > 0:
            k = _key(material, "mg/mL")
            record["materials"][k] = record["materials"].get(k, 0.0) + mg / record["volume_mL"]
        else:
            record["solids_mg"][material] = record["solids_mg"].get(material, 0.0) + mg
    if not materials:
        record["complete"] = False
    if stock_id:
        record["stocks"] = sorted(set(record["stocks"]) | {stock_id})
    record["known"] = False
    return record


def same_contents(a, b, tolerance=1e-9):
    keys = set(a["materials"]) | set(b["materials"])
    return all(abs(a["materials"].get(k, 0.0) - b["materials"].get(k, 0.0))
               <= tolerance * max(1.0, abs(a["materials"].get(k, 0.0))) for k in keys)


def describe(record, run_id=None):
    """One line for people: 'SDS 2.00 ± 0.045 mM (0.25 mL)'."""
    if record is None:
        return "Nothing recorded"
    parts = []
    for key, c in sorted(record["materials"].items()):
        material, units = split_key(key)
        sd = sd_absolute(record, key)
        parts.append(f"{material} {c:.3g}" + (f" ± {sd:.2g}" if sd is not None else "") + f" {units}".rstrip())
    for material, mg in record["solids_mg"].items():
        parts.append(f"{material} {mg:.3g} mg (no liquid yet)")
    text = ", ".join(parts) if parts else "no recorded materials"
    text += f" ({record['volume_mL']:.3g} mL)"
    if not record["complete"]:
        text += "; partly unknown"
    return text


# --- the store --------------------------------------------------------------------------------------

def to_json(record):
    return json.dumps(record, sort_keys=True)


def from_json(text):
    record = empty_record()
    record.update(json.loads(text))
    return record


class ContentsStore:
    """The records in one contents file, kept in memory and saved after every change."""

    def __init__(self, path=CONTENTS_FILE):
        self.path = Path(path)
        self.records = {}
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["container_id"] not in self.records and row.get("state"):
                        self.records[row["container_id"]] = from_json(row["state"])

    def get(self, container_id):
        record = self.records.get(container_id)
        return copy.deepcopy(record) if record is not None else None

    def put(self, container_id, record, now=None):
        record = copy.deepcopy(record)
        record["updated"] = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        self.records[container_id] = record
        self.save()

    def save(self, path=None):
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for container_id, record in sorted(self.records.items()):
                writer.writerows(rows_for(container_id, record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)


def rows_for(container_id, record):
    """Readable rows (one per material) for one container; the full record rides in 'state'."""
    base = {"container_id": container_id, "volume_mL": f"{record['volume_mL']:.6g}",
            "from_stocks": ";".join(record["stocks"]), "complete": record["complete"],
            "updated": record.get("updated", ""), "state": to_json(record)}
    rows = []
    for key, c in sorted(record["materials"].items()):
        material, units = split_key(key)
        sd = sd_absolute(record, key)
        rows.append({**base, "material": material, "units": units, "concentration": f"{c:.6g}",
                     "sd_absolute": "" if sd is None else f"{sd:.3g}"})
    for material, mg in record["solids_mg"].items():
        rows.append({**base, "material": material, "units": "mg", "concentration": f"{mg:.6g}", "sd_absolute": ""})
    if not rows:
        rows.append({**base, "material": "", "units": "", "concentration": "", "sd_absolute": ""})
    return rows


def contents_path(controller):
    """Where this controller's records live, or None when it must not write (plain simulation)."""
    root = getattr(controller, "_scheduler_state_root", None)
    if root is not None:
        return Path(root) / "container_contents.csv"
    if getattr(controller, "simulate", True):
        return None
    return CONTENTS_FILE
