"""Sample tracking: stocks, container IDs and the transfer log.

Design (agreed 2026-11-18, see the Integration Plan notes):
- status/stocks.csv: one row per stock component (a multi-component stock repeats its
  stock_id). Stock IDs are never reused.
- Every container has one ID: vials ``vial-<code>``, plates ``plate-<code>`` (from the
  plate inventory), wells ``plate-<code>:<well index>``, reservoirs ``reservoir-<n>``.
- status/transfers.csv: append-only, one row per transfer, written only by the robot
  during live runs (scheduler simulations write to their own sandbox copy).
- A container's contents are worked out from the log and the stocks.

Tracking only records. Nothing here raises into a workflow: a failure logs a warning and
the run carries on.
"""

import csv
import logging
import math
import os
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STOCKS_FILE = REPO_ROOT / "status" / "stocks.csv"
TRANSFERS_FILE = REPO_ROOT / "status" / "transfers.csv"

STOCK_COLUMNS = ["stock_id", "material", "concentration", "units", "solvent", "lot", "made_on", "notes"]
TRANSFER_COLUMNS = ["time", "run_id", "source", "destination", "amount", "units", "method", "liquid"]

_fallback_logger = logging.getLogger(__name__)


def new_container_id(prefix="vial"):
    """A new container ID in the same style as plate IDs (prefix plus 12 hex characters)."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def well_id(plate_id, well_index):
    return f"{plate_id}:{int(well_index)}"


def reservoir_id(reservoir_index):
    """The reservoir's current load (see record_load), or reservoir-<n> if none recorded."""
    return source_container(reservoir_slot(reservoir_index))


# --- stocks -----------------------------------------------------------------------------

def load_stocks(path=None):
    """List of stock component rows (dicts). Missing file means no stocks yet."""
    path = Path(path or STOCKS_FILE)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def next_stock_id(material, stocks):
    """``<MATERIAL>-NNN``, one more than the highest number already used for that prefix."""
    prefix = "".join(ch for ch in str(material).upper() if ch.isalnum())[:12] or "STOCK"
    numbers = []
    for row in stocks:
        head, _, tail = str(row.get("stock_id", "")).rpartition("-")
        if head == prefix and tail.isdigit():
            numbers.append(int(tail))
    return f"{prefix}-{max(numbers, default=0) + 1:03d}"


def add_stock(components, path=None, made_on=None):
    """Add one stock. ``components`` is a list of dicts with material, concentration, units,
    and optionally solvent, lot and notes. Returns the new stock_id."""
    path = Path(path or STOCKS_FILE)
    if not components:
        raise ValueError("A stock needs at least one component")
    stocks = load_stocks(path)
    stock_id = next_stock_id(components[0]["material"], stocks)
    made_on = made_on or datetime.now().strftime("%Y-%m-%d")
    rows = [{"stock_id": stock_id, "material": c["material"], "concentration": c.get("concentration", ""),
             "units": c.get("units", ""), "solvent": c.get("solvent", ""), "lot": c.get("lot", ""),
             "made_on": made_on, "notes": c.get("notes", "")} for c in components]
    _append_rows(path, STOCK_COLUMNS, rows)
    return stock_id


def stock_choices(stocks):
    """[(stock_id, label)] in file order, one per stock, e.g. 'TTAB-001: TTAB 50 mM in water'."""
    grouped = {}
    for row in stocks:
        grouped.setdefault(row["stock_id"], []).append(row)
    choices = []
    for stock_id, rows in grouped.items():
        parts = [" ".join(str(v) for v in (r["material"], r.get("concentration", ""), r.get("units", "")) if str(v))
                 for r in rows]
        solvent = rows[0].get("solvent") or ""
        lot = rows[0].get("lot") or ""
        label = f"{stock_id}: " + " + ".join(parts) + (f" in {solvent}" if solvent else "") + (f" (lot {lot})" if lot else "")
        choices.append((stock_id, label))
    return choices


def update_stock_notes(stock_id, notes, path=None):
    """Change only the notes of one stock (all its component rows). Everything else is history."""
    path = Path(path or STOCKS_FILE)
    rows = load_stocks(path)
    if not any(r["stock_id"] == stock_id for r in rows):
        raise ValueError(f"No stock {stock_id}")
    for row in rows:
        if row["stock_id"] == stock_id:
            row["notes"] = notes
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STOCK_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stock_usage(status_dir=None, sources_path=None):
    """{stock_id: ["Reservoir 1", "TTAB_stock (surfactant_vials.csv)", ...]} for what holds each
    stock now: the current reservoir and cartridge loads, and vials in the vial files."""
    from vial_csv import read_vial_csv
    usage = {}
    loads = {}
    for row in load_sources(sources_path):
        loads[row["slot"]] = row  # later rows replace earlier ones: the current load
    for slot, row in sorted(loads.items()):
        if row["stock_id"]:
            kind, _, number = slot.partition("-")
            name = f"Reservoir {number}" if kind == "reservoir" else f"Cartridge {number}"
            usage.setdefault(row["stock_id"], []).append(name)
    for path in sorted(Path(status_dir or STOCKS_FILE.parent).glob("*.csv")):
        try:
            _, records = read_vial_csv(path)
        except Exception:
            continue  # not a vial file (stocks, transfers, sources...) or unreadable
        for record in records:
            stock_id = (record.get("stock_id") or "").strip()
            if stock_id:
                usage.setdefault(stock_id, []).append(f"{record['vial_name']} ({path.name})")
    return usage


def usage_summary(places):
    """Short text for a table cell: 'Reservoir 1; 2 vials'."""
    if not places:
        return "—"
    named = [p for p in places if p.startswith(("Reservoir", "Cartridge"))]
    vials = len(places) - len(named)
    return "; ".join(named + ([f"{vials} vial{'s' if vials != 1 else ''}"] if vials else []))


def vial_after_stock_choice(vial_data, stock_id):
    """The vial's data with its stock set. Changing a vial's stock makes it a new vial (new ID)."""
    updated = dict(vial_data)
    old = (vial_data.get("stock_id") or "").strip() if isinstance(vial_data.get("stock_id"), str) else ""
    new = (stock_id or "").strip()
    updated["stock_id"] = new
    if new != old or not (updated.get("container_id") or "").strip():
        updated["container_id"] = new_container_id("vial")
    return updated


# --- reservoirs and powder cartridges ---------------------------------------------------------

SOURCES_FILE = REPO_ROOT / "status" / "sources.csv"
SOURCE_COLUMNS = ["container_id", "slot", "stock_id", "amount_loaded", "units", "loaded_on"]


def reservoir_slot(index):
    return f"reservoir-{int(index)}"


def cartridge_slot(channel):
    return f"powder-{int(channel)}"


def load_sources(path=None):
    """Every load ever recorded (oldest first); the latest per slot is what's in it now."""
    path = Path(path or SOURCES_FILE)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def current_load(slot, path=None):
    loads = [row for row in load_sources(path) if row["slot"] == slot]
    return loads[-1] if loads else None


def record_load(slot, stock_id, amount, units, path=None, now=None):
    """Load or replace a reservoir or cartridge. Each load is a new container (new ID), so
    transfers before and after a refill are told apart. Returns the new container ID."""
    container_id = f"{slot}-{uuid.uuid4().hex[:12]}"
    row = {"container_id": container_id, "slot": slot, "stock_id": stock_id or "",
           "amount_loaded": f"{float(amount):.6g}", "units": units,
           "loaded_on": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}
    _append_rows(Path(path or SOURCES_FILE), SOURCE_COLUMNS, [row])
    return container_id


def source_container(slot, path=None):
    """The ID transfers from this slot are recorded under: its current load, else the slot."""
    try:
        load = current_load(slot, path)
        return load["container_id"] if load else slot
    except Exception:
        return slot


def amount_left(slot, sources_path=None, transfers_file=None):
    """(amount, units) left in a slot's current load: amount loaded minus logged dispenses.
    An estimate: nobody measures it. None if nothing has been loaded."""
    load = current_load(slot, sources_path)
    if load is None:
        return None
    used = sum(float(t["amount"]) for t in load_transfers(transfers_file)
               if t["source"] == load["container_id"] and t["units"] == load["units"])
    return float(load["amount_loaded"]) - used, load["units"]


PUMPS_FILE = REPO_ROOT / "robot_state" / "syringe_pumps.yaml"


def reservoir_status(pumps_file=None, sources_path=None, transfers_file=None):
    """One dict per reservoir pump (pumps with a carousel position) for the Reservoirs tab:
    index, description, liquid, capacity_mL, low_warning_mL, load (or None), left_mL, low."""
    import yaml
    with open(pumps_file or PUMPS_FILE, encoding="utf-8") as handle:
        pumps = yaml.safe_load(handle) or {}
    rows = []
    for index, pump in sorted(pumps.items()):
        if not isinstance(pump, dict) or pump.get("carousel_angle") is None:
            continue  # the pipetting pump has no reservoir
        slot = reservoir_slot(index)
        load = current_load(slot, sources_path)
        left = amount_left(slot, sources_path, transfers_file) if load else None
        warning = pump.get("low_warning_mL")
        rows.append({"index": int(index), "slot": slot, "description": pump.get("description", ""),
                     "liquid": pump.get("liquid") or "", "capacity_mL": pump.get("capacity_mL"),
                     "low_warning_mL": warning, "load": load, "left_mL": left[0] if left else None,
                     "low": bool(left and warning is not None and left[0] < float(warning))})
    return rows


POWDER_CHANNELS_FILE = REPO_ROOT / "robot_state" / "powder_channels.yaml"


def cartridge_status(channels_file=None, sources_path=None, transfers_file=None):
    """One dict per cartridge slot in powder_channels.yaml for the Powders tab:
    channel, slot, description, load (or None), left_mg, empty."""
    import yaml
    with open(channels_file or POWDER_CHANNELS_FILE, encoding="utf-8") as handle:
        channels = yaml.safe_load(handle) or {}
    rows = []
    for channel, settings in sorted(channels.items()):
        slot = cartridge_slot(channel)
        load = current_load(slot, sources_path)
        empty = load is None or not load["stock_id"]
        left = None if empty else amount_left(slot, sources_path, transfers_file)
        rows.append({"channel": int(channel), "slot": slot,
                     "description": (settings or {}).get("description", ""),
                     "load": None if empty else load, "left_mg": left[0] if left else None,
                     "emptied_on": load["loaded_on"] if (load and empty) else None})
    return rows


def record_empty(slot, path=None, now=None):
    """Record that a cartridge was taken out or used up (a load with no stock and 0 left)."""
    return record_load(slot, "", 0, "mg", path, now)


def reservoir_warnings(pumps_file=None, sources_path=None, transfers_file=None):
    """Plain-language warnings for reservoirs estimated to be running low. Never raises."""
    try:
        return [f"Reservoir {r['index']} ({r['liquid'] or r['description']}) is estimated to have about "
                f"{max(r['left_mL'], 0):.0f} mL left (warning below {r['low_warning_mL']} mL). "
                "Refill it and use Load/replace in the Reservoirs tab."
                for r in reservoir_status(pumps_file, sources_path, transfers_file) if r["low"]]
    except Exception as error:
        return [f"Could not check reservoir levels: {error}"]


def source_stocks(path=None):
    """container_id -> stock_id for every load ever recorded."""
    return {row["container_id"]: row["stock_id"] or None for row in load_sources(path)}


# --- transfer log -------------------------------------------------------------------------

def transfers_path(controller):
    """Where this controller's transfers go, or None when it must not write.

    Live runs write the real log. A scheduler simulation writes inside its sandbox. A plain
    simulation writes nothing, as with vial files.
    """
    root = getattr(controller, "_scheduler_state_root", None)
    if root is not None:
        return Path(root) / "transfers.csv"
    if getattr(controller, "simulate", True):
        return None
    return TRANSFERS_FILE


def record_transfer(controller, source, destination, amount, units, method, liquid=None, path=None, now=None):
    """Append one transfer row. Never raises; returns True if a row was written."""
    logger = getattr(controller, "logger", None) or _fallback_logger
    try:
        target = Path(path) if path is not None else transfers_path(controller)
        if target is None or source is None or destination is None or source == destination:
            return False
        if not amount or float(amount) <= 0:
            return False
        row = {"time": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
               "run_id": getattr(controller, "run_id", None) or "",
               "source": source, "destination": destination, "amount": f"{float(amount):.6g}",
               "units": units, "method": method or "", "liquid": liquid or ""}
        _append_rows(target, TRANSFER_COLUMNS, [row])
        return True
    except Exception as error:
        logger.warning(f"Sample tracking: could not record transfer {source} -> {destination}: {error}")
        return False


def load_transfers(path=None):
    path = Path(path or TRANSFERS_FILE)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# --- container IDs in vial files ------------------------------------------------------------

EMPTY_ML = 1e-6


def prepare_vial_file(path, retire_emptied=False, transfers_file=None, logger=None):
    """Give every vial in a vial file a container ID; returns a list of changes made.

    - Adds the optional ``container_id`` and ``stock_id`` columns if the file has none.
    - A blank container_id gets a new ID; a duplicated one (a copied row) keeps the first.
    - With ``retire_emptied`` (the start of a live run): a vial at 0 mL whose ID already has
      contents in the transfer log was emptied or replaced by a person, so it gets a new
      ID. The old ID and its history stay in the log.
    Saves only if something changed. Never raises: a problem logs a warning.
    """
    logger = logger or _fallback_logger
    try:
        from vial_csv import read_vial_csv, save_vial_csv
        fields, records = read_vial_csv(path)
        fields = list(fields)
        changes = []
        for column in ("container_id", "stock_id"):
            if column not in fields:
                fields.append(column)
                for record in records:
                    record[column] = ""
                changes.append(f"added column {column}")
        filled = set()
        if retire_emptied:
            log = transfers_file if transfers_file is not None else TRANSFERS_FILE
            filled = {row["destination"] for row in load_transfers(log)}
        seen = set()
        for record in records:
            current = (record.get("container_id") or "").strip()
            reason = None
            if not current:
                reason = "had no ID"
            elif current in seen:
                reason = f"duplicated {current}"
            elif current in filled and _volume(record) <= EMPTY_ML:
                reason = f"is empty; retired {current}"
            if reason:
                record["container_id"] = new_container_id("vial")
                changes.append(f"{record['vial_name']} {reason}: now {record['container_id']}")
            seen.add(record["container_id"])
        if changes:
            save_vial_csv(path, fields, records)
            logger.info(f"Sample tracking: updated container IDs in {path}: " + "; ".join(changes))
        return changes
    except Exception as error:
        logger.warning(f"Sample tracking: could not prepare container IDs in {path}: {error}")
        return []


def _volume(record):
    try:
        return float(record.get("vial_volume") or 0)
    except (TypeError, ValueError):
        return float("inf")  # unreadable volume: never treat as empty


# --- the tip ----------------------------------------------------------------------------

class TipContents:
    """What the pipette tip holds: a list of [source container ID, mL].

    A dispense takes from every part in proportion, so a tip that aspirated from two
    containers splits each dispense between them. Unknown sources (None) are kept so the
    proportions stay right, but no row is written for them.
    """

    def __init__(self):
        self.parts = []

    def clear(self):
        self.parts = []

    def total(self):
        return sum(amount for _, amount in self.parts)

    def aspirate(self, source, amount_mL):
        if amount_mL and amount_mL > 0:
            self.parts.append([source, float(amount_mL)])

    def dispense(self, amount_mL):
        """Remove amount_mL in proportion; returns [(source, mL), ...] for what left the tip."""
        total = self.total()
        if total <= 0 or not amount_mL or amount_mL <= 0:
            return []
        fraction = min(1.0, float(amount_mL) / total)
        taken = []
        for part in self.parts:
            share = part[1] * fraction
            part[1] -= share
            taken.append((part[0], share))
        self.parts = [p for p in self.parts if p[1] > 1e-12]
        merged = {}
        for source, share in taken:
            merged[source] = merged.get(source, 0.0) + share
        return list(merged.items())


# --- contents ------------------------------------------------------------------------------

def contents(container_id, transfers, stocks, container_stocks, _seen=None):
    """What one container holds, worked out from the log.

    Returns {"total_mL": float, "materials": {(material, units): concentration}}, or None
    if the container holds a stock (use the stock directly) or nothing is known.
    ``container_stocks`` maps container IDs (vials, reservoirs) to their stock_id.
    Concentrations are per volume, so mixing keeps the stock's units; volumes are assumed
    to add, with no evaporation. Powders (mg) are reported as mg/mL of the liquid volume.
    """
    _seen = set() if _seen is None else _seen
    if container_id in _seen:
        return None  # a loop in the log (should not happen); stop rather than recurse forever
    _seen = _seen | {container_id}
    stock_rows = {}
    for row in stocks:
        stock_rows.setdefault(row["stock_id"], []).append(row)

    def stock_materials(stock_id):
        return {(r["material"], r["units"]): float(r["concentration"])
                for r in stock_rows.get(stock_id, []) if r.get("concentration") not in (None, "")}

    if container_id in container_stocks and container_stocks[container_id]:
        return {"total_mL": None, "materials": stock_materials(container_stocks[container_id]),
                "stock_id": container_stocks[container_id], "stocks": {container_stocks[container_id]}}

    incoming = [t for t in transfers if t["destination"] == container_id]
    if not incoming:
        return None
    total_mL, amounts, solids_mg, used_stocks = 0.0, {}, {}, set()
    for t in incoming:
        amount = float(t["amount"])
        if t["units"] == "mg":
            solids_mg[t["source"]] = solids_mg.get(t["source"], 0.0) + amount
            continue
        total_mL += amount
        source = container_stocks.get(t["source"])
        if source:
            materials = stock_materials(source)
            used_stocks.add(source)
        else:
            inner = contents(t["source"], transfers, stocks, container_stocks, _seen)
            materials = inner["materials"] if inner else {}
            used_stocks |= inner["stocks"] if inner else set()
        for key, concentration in materials.items():
            amounts[key] = amounts.get(key, 0.0) + concentration * amount
    result = {key: value / total_mL for key, value in amounts.items()} if total_mL > 0 else {}
    for source, mg in solids_mg.items():
        stock = container_stocks.get(source)
        if stock:
            used_stocks.add(stock)
        names = [r["material"] for r in stock_rows.get(stock, [])] if stock else [source]
        for name in names:
            if total_mL > 0:
                result[(name, "mg/mL")] = result.get((name, "mg/mL"), 0.0) + mg / total_mL
    return {"total_mL": total_mL, "materials": result, "stock_id": None, "stocks": used_stocks}


def concentration_sd(container_id, transfers, stocks, container_stocks, estimator, _seen=None):
    """{(material, units): (sd_within_run, sd_absolute)} for one container, from the log.

    For c = Σ cₖVₖ / ΣV, each dispense k moves c by (cₖ − c)/V per mL, and each source's own
    concentration error moves it by Vₖ/V. Errors are combined as root-sum-of-squares.
    - sd_within_run: only the random (dispense-to-dispense) error of the dispenses into this
      container; use it to compare containers filled in the same run.
    - sd_absolute: random plus shared (day-to-day) error, plus each robot-made source's
      own absolute error. Stocks are taken as exact (no stock uncertainty is recorded yet).
    A material gets (None, None) when any dispense into the container can't be estimated
    (reservoir, powder, unvalidated liquid or volume, unknown source). Never raises.
    """
    try:
        _seen = set() if _seen is None else _seen
        if container_id in _seen:
            return {}
        _seen = _seen | {container_id}
        mine = contents(container_id, transfers, stocks, container_stocks)
        if not mine or mine["stock_id"] is not None:
            return {key: (0.0, 0.0) for key in (mine or {}).get("materials", {})}
        incoming = [t for t in transfers if t["destination"] == container_id]
        total = mine["total_mL"]
        unknown = {key: (None, None) for key in mine["materials"]}
        if not total or any(t["units"] != "mL" for t in incoming):
            return unknown
        parts = []
        for t in incoming:
            error = estimator.lookup(t.get("liquid"), t.get("method"), float(t["amount"]) * 1000)
            if error is None:
                return unknown
            source = contents(t["source"], transfers, stocks, container_stocks)
            if source is None:
                return unknown
            source_sd = (concentration_sd(t["source"], transfers, stocks, container_stocks, estimator, _seen)
                         if source["stock_id"] is None else {})
            parts.append((float(t["amount"]), error[0] / 1000, error[1] / 1000, source["materials"],
                          source_sd, source["stock_id"] is not None))
        result = {}
        for key, c in mine["materials"].items():
            within = absolute = 0.0
            for volume, random_mL, shared_mL, source_materials, source_sd, from_stock in parts:
                slope = (source_materials.get(key, 0.0) - c) / total
                within += (slope * random_mL) ** 2
                absolute += slope ** 2 * (random_mL ** 2 + shared_mL ** 2)
                if key in source_materials and not from_stock:
                    inherited = source_sd.get(key, (None, None))[1]
                    if inherited is None:
                        within = absolute = None
                        break
                    absolute += (volume / total * inherited) ** 2
            result[key] = (None, None) if within is None else (math.sqrt(within), math.sqrt(absolute))
        return result
    except Exception as error:
        _fallback_logger.warning(f"Sample tracking: could not estimate uncertainty for {container_id}: {error}")
        return {}


def describe_contents(container_id, stock_id=None, transfers_file=None, stocks_file=None, sources_file=None,
                      contents_file=None):
    """One line for the vial dialog: its stored record, else its stock, else what the log says."""
    from experiment_tracking import sample_contents
    stocks = load_stocks(stocks_file)
    container_id = (container_id or "").strip() if isinstance(container_id, str) else ""
    stored = sample_contents.ContentsStore(contents_file or sample_contents.CONTENTS_FILE).get(container_id) \
        if container_id else None
    if stored is not None:
        return sample_contents.describe(stored)
    if stock_id:
        labels = dict(stock_choices(stocks))
        return labels.get(stock_id, f"{stock_id} (not in the stock list)")
    if not container_id:
        return "No ID yet (given at the next live run)"
    holds = source_stocks(sources_file)
    found = contents(container_id, load_transfers(transfers_file), stocks, holds)
    if not found:
        return "Nothing recorded"
    if not found["materials"]:
        return f"{found['total_mL']:.3g} mL transferred in; materials unknown"
    return ", ".join(f"{material} {value:.3g} {units}"
                     for (material, units), value in sorted(found["materials"].items()))


WELL_CONTENTS_DIR = REPO_ROOT / "output" / "sample_tracking"  # one folder per run: output/sample_tracking/<run_id>/
WELL_CONTENTS_COLUMNS = ["container", "material", "concentration", "units", "total_mL", "from_stocks",
                         "sd_within_run", "sd_absolute", "complete"]


def snapshot(run_id, stage, vial_file=None, output_folder=None, contents_file=None, logger=None):
    """Save the vial file and the stored contents as they are now, with the run's data:
    output/sample_tracking/<run_id>/<stage>_vials.csv and <stage>_contents.csv (stage is
    'start' or 'end'), also copied into the workflow's output folder when it recorded one."""
    import shutil
    from experiment_tracking import sample_contents
    logger = logger or _fallback_logger
    try:
        if not run_id:
            return None
        folders = [WELL_CONTENTS_DIR / run_id] + ([Path(output_folder) / "sample_tracking"] if output_folder else [])
        contents = Path(contents_file or sample_contents.CONTENTS_FILE)
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            if vial_file and Path(vial_file).exists():
                shutil.copyfile(vial_file, folder / f"{stage}_vials.csv")
            if contents.exists():
                shutil.copyfile(contents, folder / f"{stage}_contents.csv")
            else:
                (folder / f"{stage}_contents.csv").write_text(",".join(sample_contents.COLUMNS) + "\n", encoding="utf-8")
        return folders[0]
    except Exception as error:
        logger.warning(f"Sample tracking: could not save the {stage} snapshot for run {run_id}: {error}")
        return None


def container_stocks_from_vial_file(vial_file):
    """container_id -> stock_id for the vials in one vial file (blank stock_id -> None)."""
    from vial_csv import read_vial_csv
    _, records = read_vial_csv(vial_file)
    return {r["container_id"].strip(): (r.get("stock_id") or "").strip() or None
            for r in records if (r.get("container_id") or "").strip()}


def write_well_contents(run_id, vial_file=None, output_folder=None, transfers_file=None, stocks_file=None,
                        logger=None, estimator=None, contents_file=None):
    """At the end of a live run: what each container this run filled now holds.

    Read from each container's stored record (contents and uncertainty travel with the sample).
    A container with no stored record (from before stored contents existed) is worked out from
    the log instead. Written to output/sample_tracking/<run_id>/well_contents.csv, and also into
    the run's output folder when the workflow recorded one. SDs are blank where they can't be
    estimated. Returns the main path, or None. Never raises.
    """
    from experiment_tracking import sample_contents
    logger = logger or _fallback_logger
    try:
        transfers = load_transfers(transfers_file)
        filled = []
        for t in transfers:
            if t.get("run_id") == run_id and t["destination"] not in filled:
                filled.append(t["destination"])
        if not run_id or not filled:
            return None
        store = sample_contents.ContentsStore(contents_file or sample_contents.CONTENTS_FILE)
        stocks = load_stocks(stocks_file)
        holds = source_stocks()  # every reservoir and cartridge load
        holds.update(container_stocks_from_vial_file(vial_file) if vial_file else {})
        fmt = lambda value: "" if value is None else f"{value:.3g}"
        rows = []
        for container in filled:
            record = store.get(container)
            if record is not None:
                base = {"container": container, "total_mL": f"{record['volume_mL']:.6g}",
                        "from_stocks": ";".join(record["stocks"]), "complete": record["complete"]}
                if not record["materials"]:
                    rows.append({**base, "material": "", "concentration": "", "units": "",
                                 "sd_within_run": "", "sd_absolute": ""})
                for key, concentration in sorted(record["materials"].items()):
                    material, units = sample_contents.split_key(key)
                    rows.append({**base, "material": material, "concentration": f"{concentration:.6g}", "units": units,
                                 "sd_within_run": fmt(sample_contents.sd_within_run(record, key, run_id)),
                                 "sd_absolute": fmt(sample_contents.sd_absolute(record, key))})
                continue
            found = contents(container, transfers, stocks, holds)  # older containers: from the log
            if not found:
                continue
            if estimator is None:
                from pipetting_data.dispense_uncertainty import DispenseUncertainty
                estimator = DispenseUncertainty(logger=logger)
            errors = concentration_sd(container, transfers, stocks, holds, estimator)
            base = {"container": container, "total_mL": "" if found["total_mL"] is None else f"{found['total_mL']:.6g}",
                    "from_stocks": ";".join(sorted(found["stocks"])), "sd_within_run": "", "sd_absolute": "",
                    "complete": ""}
            if not found["materials"]:
                rows.append({**base, "material": "", "concentration": "", "units": ""})
            for (material, units), concentration in sorted(found["materials"].items()):
                within, absolute = errors.get((material, units), (None, None))
                rows.append({**base, "material": material, "concentration": f"{concentration:.6g}", "units": units,
                             "sd_within_run": fmt(within), "sd_absolute": fmt(absolute)})
        targets = [WELL_CONTENTS_DIR / run_id / "well_contents.csv"]
        if output_folder:
            targets.append(Path(output_folder) / "sample_tracking" / "well_contents.csv")
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=WELL_CONTENTS_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        return targets[0]
    except Exception as error:
        logger.warning(f"Sample tracking: could not write well contents for run {run_id}: {error}")
        return None


# --- helpers ------------------------------------------------------------------------------

def _append_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
