"""
Make a list of target surfactant samples in wellplates, then measure them.

Plan: docs/plans/surfactant_target_samples_plan.md.

The target list is decided elsewhere (TARGET_FILE, one row per sample): no grid, no
recommender, no iterations. One pass: plan -> make substocks -> make samples -> measure.

Target file columns:
  sample_id       unique; carried to every output
  <S>_mM          final in-well mM of surfactant S (a key of SURFACTANTS); blank or 0 = absent
  <A>             one column per additive in ADDITIVES; true -> its fixed volume is added
  replicates      optional; the row is made this many times (default 1)
  anything else   metadata, passed through to the results
Controls are ordinary rows (a blank is every surfactant 0).

Steps:
1. Inventory (sample tracking): every vial's contents come from its stored record, else
   its stock_id, else unknown. A vial holding only surfactant S in mM, with known
   contents, is a source of S; vials with identical contents form one group (the first
   8 mL vial is drawn from, the others refill it). A stock_id missing from the stock
   list, or a surfactant with no source, stops the run before anything moves.
2. Plan (workflows/helpers/substock_planning.py), per surfactant: every sample within
   MAX_ERROR_PCT where possible, with the fewest new substocks; samples that can't meet it
   are made by their lowest-error route and flagged. Samples that can't physically be made
   (above the stock, or components overflow the well) are skipped and reported. Wells with
   several surfactants share the volume left after additives.
3. Layout (workflows/helpers/plate_blocks.py): BATCH_SIZE wells at a time, each batch in
   the next block of wells, randomized within the block; a new plate when one is full.
4. Make every new substock (reservoir water, then the parent; vortex), most concentrated
   first. A substock that runs low is topped up with the same recipe.
5. Per batch: dispense in DISPENSE_ORDER (water, surfactants, additives), refilling vials
   in chunks, wait EQUILIBRATION_MIN, then read the batch (every MEASUREMENT_PROTOCOL_FILES protocol).

Dry run (no hardware): python workflows/surfactant_target_samples.py --plan
prints the pre-run report and saves it with the planned wells and substocks.

Not measured, assumed: dilution errors along a chain are independent; reservoir water and
stock concentrations are exact (as in sample tracking).
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from workflows.helpers.plate_blocks import assign_blocks
from workflows.helpers.substock_planning import make_transfer_error, plan_substocks

_WORKFLOW_NAME = Path(__file__).stem
_REPO = Path(__file__).resolve().parents[1]

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/surfactant_target_samples_vials.csv"
TARGET_FILE = "inputs/surfactant_target_samples_example.csv"
WELLPLATE_TYPE = "96 WELL PLATE"
WELL_VOLUME_UL = 250.0
WELL_MIN_UL = 20.0                 # smallest surfactant volume pipetted into a well
WELL_MAX_UL = 240.0                # largest surfactant volume into a well
SUBSTOCK_VOLUME_ML = 4.0           # every new substock is made to this volume
SUBSTOCK_MIN_TRANSFER_UL = 100.0   # smallest parent transfer when making a substock
VIAL_DEAD_ML = 2.0                 # left unused in every source vial (North_Safe warns below 2 mL for small-tip draws)
MAX_ERROR_PCT = 3.0                # target error (1 SD) of each sample's surfactant concentration
SURFACTANTS = {"SDS": {"liquid": "SDS"}}  # material name in sample tracking -> liquids.yaml name
ADDITIVES = {}                     # e.g. {"pyrene": {"vial": "pyrene_stock", "volume_ul": 5.0, "liquid": "ethanol"}}
WATER_VIAL = "water"
WATER_RESERVOIR = 1
WATER_VIAL_FILL_ML = 7.5
BATCH_SIZE = 24
EQUILIBRATION_MIN = 0.0            # wait after each batch is dispensed, before it is read
RANDOMIZED_ORDER = True
RANDOMIZATION_SEED = 42
PREPARATION_PROTOCOL_FILE = r"C:\Protocols\shake_5_wait_5.prt"  # run once before the reads; None = no shake
MEASUREMENT_PROTOCOL_FILES = [     # read after each batch (as multidim: turbidity, then fluorescence)
    r"C:\Protocols\CMC_Absorbance_96.prt",
    r"C:\Protocols\CMC_Fluorescence_96.prt",
]
DISPENSE_ORDER = ["water", "surfactants", "additives"]
VORTEX_SECONDS = 5

_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "TARGET_FILE", "WELLPLATE_TYPE", "WELL_VOLUME_UL",
    "WELL_MIN_UL", "WELL_MAX_UL", "SUBSTOCK_VOLUME_ML", "SUBSTOCK_MIN_TRANSFER_UL", "VIAL_DEAD_ML",
    "MAX_ERROR_PCT", "SURFACTANTS", "ADDITIVES", "WATER_VIAL", "WATER_RESERVOIR", "WATER_VIAL_FILL_ML",
    "BATCH_SIZE", "EQUILIBRATION_MIN", "RANDOMIZED_ORDER", "RANDOMIZATION_SEED", "PREPARATION_PROTOCOL_FILE",
    "MEASUREMENT_PROTOCOL_FILES",
    "DISPENSE_ORDER", "VORTEX_SECONDS",
]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV: surfactant stocks (with stock_id), water, "
                                              "empty <surfactant>_sub_<n> vials for new substocks, additive vials.",
                               "unit": None, "affects": ["vials", "liquids"], "shares": "vial_layout"},
    "TARGET_FILE": {"description": "CSV of target samples (sample_id, <surfactant>_mM, additive flags, "
                                   "optional replicates, metadata).",
                    "unit": None, "affects": ["duration", "tips", "plates", "vials", "liquids"], "ask": True,
                    "shares": "input_file"},
    "WELLPLATE_TYPE": {"description": "Wellplate type (robot_state/wellplates.yaml).", "unit": None,
                       "affects": ["plates"], "shares": "plate_type"},
    "WELL_VOLUME_UL": {"description": "Total liquid volume per well.", "unit": "uL", "affects": ["liquids"]},
    "WELL_MIN_UL": {"description": "Smallest surfactant volume pipetted into a well.", "unit": "uL",
                    "affects": ["vials", "liquids"], "effect": "higher -> more substocks"},
    "WELL_MAX_UL": {"description": "Largest surfactant volume pipetted into a well.", "unit": "uL",
                    "affects": ["vials", "liquids"]},
    "SUBSTOCK_VOLUME_ML": {"description": "Volume every new substock is made to.", "unit": "mL",
                           "affects": ["vials", "liquids"]},
    "SUBSTOCK_MIN_TRANSFER_UL": {"description": "Smallest parent volume pipetted when making a substock.",
                                 "unit": "uL", "affects": ["vials", "liquids"],
                                 "effect": "max dilution per step = SUBSTOCK_VOLUME_ML / this"},
    "VIAL_DEAD_ML": {"description": "Volume left unused in every source vial.", "unit": "mL",
                     "affects": ["liquids"]},
    "MAX_ERROR_PCT": {"description": "Target error (1 SD) of each sample's surfactant concentration; samples "
                                     "that can't meet it are still made and flagged.",
                      "unit": "%", "affects": ["vials", "liquids", "duration"],
                      "effect": "lower -> more substocks"},
    "SURFACTANTS": {"description": "Surfactants in the target file: sample-tracking material name -> "
                                   "{liquid: liquids.yaml name}.", "unit": None, "affects": ["liquids"]},
    "ADDITIVES": {"description": "Fixed-volume additives: name -> {vial, volume_ul, liquid}; a target file "
                                 "column per additive says which samples get it.",
                  "unit": None, "affects": ["liquids", "tips"]},
    "WATER_VIAL": {"description": "Vial that water is pipetted from (refilled from WATER_RESERVOIR).",
                   "unit": None, "affects": ["vials"]},
    "WATER_RESERVOIR": {"description": "Reservoir (pump) index holding water.", "unit": None, "affects": []},
    "WATER_VIAL_FILL_ML": {"description": "WATER_VIAL is refilled to this volume.", "unit": "mL",
                           "affects": ["liquids"]},
    "BATCH_SIZE": {"description": "Wells made and then read together, so samples don't sit.", "unit": None,
                   "affects": ["duration"], "effect": "one read per batch"},
    "EQUILIBRATION_MIN": {"description": "Wait after each batch is dispensed before reading it (micelle equilibration).", "unit": "min", "affects": ["duration"], "effect": "adds EQUILIBRATION_MIN per batch"},
    "RANDOMIZED_ORDER": {"description": "Shuffle samples within each batch's block of wells.", "unit": None,
                         "affects": []},
    "RANDOMIZATION_SEED": {"description": "Seed for the shuffle.", "unit": None, "affects": []},
    "PREPARATION_PROTOCOL_FILE": {"description": "Cytation protocol run once before each batch's reads "
                                                 "(shake and wait); None for no shake.",
                                  "unit": None, "affects": ["duration"]},
    "MEASUREMENT_PROTOCOL_FILES": {"description": "Cytation protocols read after each batch; [] for none.",
                                   "unit": None,
                                   "affects": ["duration"]},
    "DISPENSE_ORDER": {"description": "Order of components into the wells: water, surfactants, additives.",
                       "unit": None, "affects": []},
    "VORTEX_SECONDS": {"description": "Vortex time after making or topping up a substock.", "unit": "s",
                       "affects": ["duration"]},
}

_PRODUCT_VIALS = ["*_sub_[0-9]*"]

_WELLPLATES_FILE = _REPO / "robot_state" / "wellplates.yaml"
_TIPS_FILE = _REPO / "robot_state" / "pipet_tips.yaml"
_COLUMNS_BY_WELLS = {96: 12, 48: 8, 24: 6}
_EMPTY_ML = 0.05  # a vial at or below this counts as empty


# ============================================================================
# PLANNING (no hardware)
# ============================================================================

def plate_layout(plate_type):
    """dict(wells, columns, max_ul) for a plate type, from robot_state/wellplates.yaml."""
    with open(_WELLPLATES_FILE) as f:
        plates = yaml.safe_load(f)
    if plate_type not in plates:
        raise ValueError(f"Unknown WELLPLATE_TYPE {plate_type!r}; see {_WELLPLATES_FILE.name}")
    wells = int(plates[plate_type]["num_wells"])
    if wells not in _COLUMNS_BY_WELLS:
        raise ValueError(f"No row layout for a {wells}-well plate")
    return dict(wells=wells, columns=_COLUMNS_BY_WELLS[wells],
                max_ul=float(plates[plate_type]["max_volume_per_well"]) * 1000)


def validate_config(c):
    layout = plate_layout(c["WELLPLATE_TYPE"])
    if not 0 < c["WELL_VOLUME_UL"] <= layout["max_ul"]:
        raise ValueError(f"WELL_VOLUME_UL must be in (0, {layout['max_ul']:g}] for {c['WELLPLATE_TYPE']}")
    if not 0 < c["WELL_MIN_UL"] < c["WELL_MAX_UL"] <= c["WELL_VOLUME_UL"]:
        raise ValueError("Need 0 < WELL_MIN_UL < WELL_MAX_UL <= WELL_VOLUME_UL")
    if not 0 < c["SUBSTOCK_MIN_TRANSFER_UL"] < c["SUBSTOCK_VOLUME_ML"] * 1000:
        raise ValueError("SUBSTOCK_MIN_TRANSFER_UL must be below SUBSTOCK_VOLUME_ML")
    if not 0 <= c["VIAL_DEAD_ML"] < c["SUBSTOCK_VOLUME_ML"]:
        raise ValueError("VIAL_DEAD_ML must be below SUBSTOCK_VOLUME_ML")
    if c["MAX_ERROR_PCT"] <= 0:
        raise ValueError("MAX_ERROR_PCT must be positive")
    if c["EQUILIBRATION_MIN"] < 0:
        raise ValueError("EQUILIBRATION_MIN must be >= 0")
    if type(c["BATCH_SIZE"]) is not int or c["BATCH_SIZE"] < 1:
        raise ValueError("BATCH_SIZE must be a positive integer")
    if sorted(c["DISPENSE_ORDER"]) != ["additives", "surfactants", "water"]:
        raise ValueError("DISPENSE_ORDER must list water, surfactants and additives once each")
    if not c["SURFACTANTS"]:
        raise ValueError("SURFACTANTS is empty")
    for name, s in c["SURFACTANTS"].items():
        if not s.get("liquid"):
            raise ValueError(f"SURFACTANTS[{name!r}] needs a liquid (liquids.yaml name)")
    for name, a in c["ADDITIVES"].items():
        missing = {"vial", "volume_ul", "liquid"} - set(a)
        if missing:
            raise ValueError(f"ADDITIVES[{name!r}] is missing {', '.join(sorted(missing))}")
        if a["volume_ul"] <= 0:
            raise ValueError(f"ADDITIVES[{name!r}] volume_ul must be positive")


def load_targets(c, path=None):
    """Target file -> one row per well (replicates expanded), in file order."""
    path = Path(path or c["TARGET_FILE"])
    if not path.is_absolute():
        path = _REPO / path
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path.name}: needs a sample_id column")
    if frame.sample_id.duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample_id {frame.sample_id[frame.sample_id.duplicated()].tolist()}")
    unknown = [col for col in frame.columns if col.endswith("_mM") and col[:-3] not in c["SURFACTANTS"]]
    if unknown:
        raise ValueError(f"{path.name}: {', '.join(unknown)} not in SURFACTANTS")
    missing = [a for a in c["ADDITIVES"] if a not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: needs a column for each additive: {', '.join(missing)}")
    for s in surfactants_in(c, frame):
        frame[f"{s}_mM"] = pd.to_numeric(frame[f"{s}_mM"]).fillna(0.0)  # blank = absent (documented)
        if (frame[f"{s}_mM"] < 0).any():
            raise ValueError(f"{path.name}: negative {s}_mM")
    for a in c["ADDITIVES"]:
        frame[a] = frame[a].map(_flag)
    reps = frame["replicates"] if "replicates" in frame.columns else pd.Series(1, index=frame.index)
    if (reps < 1).any() or (reps != reps.astype(int)).any():
        raise ValueError(f"{path.name}: replicates must be whole numbers >= 1")
    rows = frame.loc[frame.index.repeat(reps.astype(int))].copy()
    rows["replicate"] = rows.groupby("sample_id").cumcount() + 1
    rows["well_id"] = [f"{s}_r{r}" for s, r in zip(rows.sample_id, rows.replicate)]
    return rows.drop(columns=["replicates"], errors="ignore").reset_index(drop=True)


def _flag(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n", "", "nan"):
        return False
    raise ValueError(f"Additive flag must be true/false, got {value!r}")


def surfactants_in(c, targets):
    return [s for s in c["SURFACTANTS"] if f"{s}_mM" in targets.columns]


def vial_capacity_ml(vial_type):
    """'8_mL' -> 8.0 (vial_type column of the vial file)."""
    text = str(vial_type)
    if not text.endswith("_mL"):
        raise ValueError(f"Unknown vial_type {vial_type!r}")
    return float(text[:-3])


def inventory(vial_df, get_record, stocks, surfactants):
    """Sources of each surfactant from sample tracking.

    vial_df: vial file rows (vial_name, vial_volume, vial_type, container_id, stock_id).
    get_record(container_id) -> stored record or None. stocks: sample_tracking.load_stocks().
    Returns {surfactant: [group]}, group = dict(name (vial drawn from), vials, conc, error,
    available_ml). Raises if a stock_id is not in the stock list.
    """
    from experiment_tracking import sample_contents as sc
    known_ids = {s["stock_id"] for s in stocks}
    bad = [f"{r.vial_name} ({r.stock_id})" for r in vial_df.itertuples()
           if isinstance(getattr(r, "stock_id", None), str) and r.stock_id.strip()
           and r.stock_id.strip() not in known_ids]
    if bad:
        raise ValueError("Unknown stock_id (add it in the scheduler's Stocks tab first): " + ", ".join(bad))
    groups = {s: {} for s in surfactants}
    for r in vial_df.itertuples():
        volume = float(r.vial_volume)
        if volume <= _EMPTY_ML:
            continue
        cid = getattr(r, "container_id", None)
        record = get_record(cid) if isinstance(cid, str) and cid else None
        if record is not None:
            sc.rescale(record, volume)
        else:
            sid = getattr(r, "stock_id", None)
            if not (isinstance(sid, str) and sid.strip()):
                continue  # unknown contents: never a source
            record = sc.stock_record(sid.strip(), stocks, volume)
        if not (record["complete"] and record["known"]) or len(record["materials"]) != 1:
            continue
        ((key, conc),) = record["materials"].items()
        material, units = sc.split_key(key)
        if material not in groups or units != "mM" or conc <= 0:
            continue
        sd = sc.sd_absolute(record, key)
        if sd is None:
            continue
        error = sd / conc
        group = groups[material].setdefault((round(conc, 9), round(error, 9)), dict(
            name=None, vials=[], conc=conc, error=error, available_ml=0.0))
        group["vials"].append((r.vial_name, vial_capacity_ml(r.vial_type)))
        group["available_ml"] += volume
    out = {}
    for s, found in groups.items():
        out[s] = []
        for g in found.values():
            # Draw from an 8 mL vial if there is one (the others refill it).
            g["vials"].sort(key=lambda v: (v[1] != 8.0, ))
            g["vials"] = [name for name, _ in g["vials"]]
            g["name"] = g["vials"][0]
            out[s].append(g)
        out[s].sort(key=lambda g: (-g["conc"], g["error"], g["name"]))
    return out


def tip_limits():
    """[(tip, largest load in uL)] from pipet_tips.yaml, smallest tip first."""
    with open(_TIPS_FILE) as f:
        tips = yaml.safe_load(f)
    limits = [(name, (t["volume"] - t.get("capacity_margin", 0)) * 1000) for name, t in tips.items()]
    return sorted(limits, key=lambda x: x[1])


def transfer_errors(c):
    """{surfactant: transfer_error(volume_ul)} from the validation history."""
    from pipetting_data.dispense_uncertainty import DispenseUncertainty
    uncertainty, limits = DispenseUncertainty(), tip_limits()
    return {s: make_transfer_error(uncertainty, c["SURFACTANTS"][s]["liquid"], limits)
            for s in c["SURFACTANTS"]}


def _additive_ul(c, row):
    return sum(a["volume_ul"] for name, a in c["ADDITIVES"].items() if row[name])


def well_caps(c, targets, sources):
    """Per-well volume caps per surfactant, and wells that can't physically be made.

    Each surfactant needs at least max(WELL_MIN_UL, target x V / strongest source); the
    room left after additives and those minimums is shared equally."""
    surfs = surfactants_in(c, targets)
    strongest = {s: max(g["conc"] for g in sources[s]) for s in surfs if sources[s]}
    caps, skipped = {s: {} for s in surfs}, {}
    for row in targets.to_dict("records"):
        budget = c["WELL_VOLUME_UL"] - _additive_ul(c, row)
        present = [s for s in surfs if row[f"{s}_mM"] > 0]
        need = {s: max(c["WELL_MIN_UL"], row[f"{s}_mM"] * c["WELL_VOLUME_UL"] / strongest[s]) for s in present}
        if budget < 0:
            skipped[row["well_id"]] = f"additives ({_additive_ul(c, row):g} uL) exceed the well"
            continue
        over = [s for s in present if need[s] > c["WELL_MAX_UL"] + 1e-9]
        if over:
            skipped[row["well_id"]] = "; ".join(
                f"{s} {row[f'{s}_mM']:g} mM needs {need[s]:.0f} uL of the strongest source "
                f"({strongest[s]:g} mM), above WELL_MAX_UL" for s in over)
            continue
        if sum(need.values()) > budget + 1e-9:
            skipped[row["well_id"]] = (f"surfactants need {sum(need.values()):.0f} uL together, only "
                                       f"{budget:.0f} uL left in the well")
            continue
        share = (budget - sum(need.values())) / len(present) if present else 0.0
        for s in present:
            caps[s][row["well_id"]] = min(c["WELL_MAX_UL"], need[s] + share)
    return caps, skipped


def substock_vial_names(surfactant, count, vial_df):
    """Names for count new substocks: <surfactant>_sub_<n>, skipping non-empty vials."""
    full = set(vial_df.loc[vial_df.vial_volume > _EMPTY_ML, "vial_name"])
    names, n = [], 0
    while len(names) < count:
        n += 1
        name = f"{surfactant}_sub_{n}"
        if name not in full:
            names.append(name)
    return names


def build_plan(c, targets, sources, errors, vial_df):
    """Plan every well: sources and volumes, new substocks, layout, skipped wells, warnings."""
    surfs = surfactants_in(c, targets)
    missing = [s for s in surfs if (targets[f"{s}_mM"] > 0).any() and not sources.get(s)]
    if missing:
        raise ValueError("No vial holds a known stock of " + ", ".join(missing) +
                         " (in mM, with stock_id set in the vial file)")
    caps, skipped = well_caps(c, targets, sources)
    wells = targets[~targets.well_id.isin(skipped)].copy()
    substocks, usage = [], []
    settings = dict(well_ul=c["WELL_VOLUME_UL"], well_min_ul=c["WELL_MIN_UL"], well_max_ul=c["WELL_MAX_UL"],
                    substock_ml=c["SUBSTOCK_VOLUME_ML"], substock_min_ul=c["SUBSTOCK_MIN_TRANSFER_UL"],
                    dead_ml=c["VIAL_DEAD_ML"], max_error=c["MAX_ERROR_PCT"] / 100)
    for _ in range(3):  # a well one surfactant can't make is skipped; re-plan the rest
        results, dropped = {}, {}
        for s in surfs:
            targets_s = {w: t for w, t in zip(wells.well_id, wells[f"{s}_mM"]) if t > 0}
            if not targets_s:
                continue
            results[s] = plan_substocks(targets_s, [dict(g, available_ml=None) for g in sources[s]],
                                        {**settings, "name_prefix": f"{s}_new_"}, errors[s], max_ul=caps[s])
            for x in results[s]["infeasible"]:
                dropped[x["sample_id"]] = f"{s}: {x['reason']}"
        if not dropped:
            break
        skipped.update(dropped)
        wells = wells[~wells.well_id.isin(dropped)].copy()
    else:
        raise RuntimeError("Planning kept dropping wells")
    for s in surfs:
        wells[f"{s}_source"], wells[f"{s}_ul"], wells[f"{s}_error"] = None, 0.0, np.nan
        wells[f"{s}_over_limit"] = False
        if s not in results:
            continue
        p = results[s]
        rename = dict(zip([x["name"] for x in p["substocks"]],
                          substock_vial_names(s, len(p["substocks"]), vial_df)))
        by_well = {w["sample_id"]: w for w in p["wells"]}
        idx = wells.well_id.map(lambda w: w in by_well)
        wells.loc[idx, f"{s}_source"] = wells.loc[idx, "well_id"].map(
            lambda w: rename.get(by_well[w]["source"], by_well[w]["source"]))
        wells.loc[idx, f"{s}_ul"] = wells.loc[idx, "well_id"].map(lambda w: by_well[w]["volume_ul"])
        wells.loc[idx, f"{s}_error"] = wells.loc[idx, "well_id"].map(lambda w: by_well[w]["error"])
        wells.loc[idx, f"{s}_over_limit"] = wells.loc[idx, "well_id"].map(lambda w: by_well[w]["over_limit"])
        for x in p["substocks"]:
            substocks.append(dict(surfactant=s, vial=rename[x["name"]], conc_mM=x["conc"], error=x["error"],
                                  parent=rename.get(x["parent"], x["parent"]), parent_ul=x["transfer_ul"],
                                  water_ul=x["water_ul"], fills=x["fills"], used_ml=x["used_ml"]))
        groups = {g["name"]: g for g in sources[s]}
        for name, u in p["usage"].items():
            g = groups.get(name)
            usage.append(dict(surfactant=s, source=rename.get(name, name), need_ml=u["need_ml"],
                              vials=", ".join(g["vials"]) if g else rename.get(name, name),
                              available_ml=g["available_ml"] if g else None))
    for name, a in c["ADDITIVES"].items():
        wells[f"{name}_ul"] = np.where(wells[name], a["volume_ul"], 0.0)
    surf_ul = wells[[f"{s}_ul" for s in surfs]].sum(axis=1) if surfs else 0.0
    add_ul = wells[[f"{a}_ul" for a in c["ADDITIVES"]]].sum(axis=1) if c["ADDITIVES"] else 0.0
    wells["water_ul"] = c["WELL_VOLUME_UL"] - surf_ul - add_ul
    if (wells.water_ul < -1e-6).any():
        raise RuntimeError("Planned volumes overflow a well")  # caps make this impossible
    wells["water_ul"] = wells.water_ul.clip(lower=0.0)
    for s in surfs:  # achieved concentration from the planned volume and source
        conc = {**{g["name"]: g["conc"] for g in sources[s]}, **{x["vial"]: x["conc_mM"] for x in substocks
                                                                 if x["surfactant"] == s}}
        wells[f"{s}_planned_mM"] = [conc[src] * ul / c["WELL_VOLUME_UL"] if src else 0.0
                                    for src, ul in zip(wells[f"{s}_source"], wells[f"{s}_ul"])]
    layout = plate_layout(c["WELLPLATE_TYPE"])
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"]) if c["RANDOMIZED_ORDER"] else None
    wells = assign_blocks(wells, c["BATCH_SIZE"], layout["wells"], layout["columns"], rng)
    over = [(w, s, e) for s in surfs for w, e, o in zip(wells.well_id, wells[f"{s}_error"], wells[f"{s}_over_limit"])
            if o]
    warnings = [f"{w}: {s} expected error {e:.1%} is above MAX_ERROR_PCT {c['MAX_ERROR_PCT']:g}% "
                "(made by its lowest-error route)" for w, s, e in over]
    warnings += [f"{w}: skipped, {why}" for w, why in skipped.items()]
    return dict(wells=wells, substocks=pd.DataFrame(substocks), usage=pd.DataFrame(usage),
                skipped=pd.DataFrame([dict(well_id=w, reason=r) for w, r in skipped.items()]),
                warnings=warnings)


def check_vials(c, plan, vial_df):
    """Before anything moves: every vial exists, new substock vials are empty, and every
    existing source holds what the run needs (stock refills are planned upstream)."""
    volumes = vial_df.set_index("vial_name").vial_volume
    problems = []
    needed = [c["WATER_VIAL"]] + [a["vial"] for a in c["ADDITIVES"].values()]
    needed += list(plan["substocks"].vial) if not plan["substocks"].empty else []
    for vial in dict.fromkeys(needed):
        if vial not in volumes.index:
            problems.append(f"add vial {vial}")
    for x in plan["substocks"].itertuples():
        if x.vial in volumes.index and volumes[x.vial] > _EMPTY_ML:
            problems.append(f"{x.vial} must start empty (has {volumes[x.vial]:.2f} mL)")
    new = set(plan["substocks"].vial) if not plan["substocks"].empty else set()
    for u in plan["usage"].itertuples():
        if u.source in new or u.need_ml <= 0:
            continue
        have = u.available_ml - c["VIAL_DEAD_ML"] * len(u.vials.split(", "))
        if u.need_ml > have + 1e-9:
            problems.append(f"{u.vials}: run needs {u.need_ml:.2f} mL, {have:.2f} mL usable")
    for name, a in c["ADDITIVES"].items():
        need = plan["wells"][f"{name}_ul"].sum() / 1000
        vial = a["vial"]
        if vial in volumes.index and need > volumes[vial] - c["VIAL_DEAD_ML"] + 1e-9:
            problems.append(f"{vial}: run needs {need:.2f} mL of {name}, has {volumes[vial]:.2f} mL")
    if problems:
        raise ValueError(f"Fix {c['INPUT_VIAL_STATUS_FILE']} before running: " + "; ".join(problems))


def report(c, plan):
    """Pre-run report as text."""
    w, lines = plan["wells"], []
    surfs = [s for s in c["SURFACTANTS"] if f"{s}_ul" in w.columns]
    plates = w.plate.nunique() if len(w) else 0
    lines.append(f"{len(w)} wells on {plates} plate(s), {w.batch.nunique() if len(w) else 0} batches of "
                 f"<= {c['BATCH_SIZE']}; {len(plan['skipped'])} skipped")
    for s in surfs:
        used = w[w[f"{s}_ul"] > 0]
        if used.empty:
            continue
        lines.append(f"{s}: error max {used[f'{s}_error'].max():.2%}, mean {used[f'{s}_error'].mean():.2%}; "
                     f"{int(used[f'{s}_over_limit'].sum())} above {c['MAX_ERROR_PCT']:g}%")
    if not plan["substocks"].empty:
        lines.append("New substocks (make in this order):")
        for x in plan["substocks"].itertuples():
            lines.append(f"  {x.vial}: {x.conc_mM:.4g} mM = {x.parent_ul:.0f} uL {x.parent} + "
                         f"{x.water_ul:.0f} uL water; error {x.error:.2%}; fills {x.fills}")
    lines.append("Volume needed per source:")
    for u in plan["usage"].itertuples():
        have = "" if u.available_ml is None or pd.isna(u.available_ml) else f" (has {u.available_ml:.2f} mL)"
        lines.append(f"  {u.source}: {u.need_ml:.2f} mL{have}")
    for warning in plan["warnings"]:
        lines.append(f"WARNING {warning}")
    return "\n".join(lines)


def save_plan(c, plan, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plan["wells"].to_csv(folder / "planned_wells.csv", index=False)
    plan["substocks"].to_csv(folder / "planned_substocks.csv", index=False)
    plan["usage"].to_csv(folder / "planned_usage.csv", index=False)
    plan["skipped"].to_csv(folder / "skipped_wells.csv", index=False)
    (folder / "plan_report.txt").write_text(report(c, plan) + "\n", encoding="utf-8")


def plan_from_files(c, vial_df, contents_file):
    """Full plan from the config, a vial file frame and a contents file (no hardware)."""
    from experiment_tracking import sample_contents as sc, sample_tracking as st
    validate_config(c)
    targets = load_targets(c)
    store = sc.ContentsStore(contents_file) if contents_file else None
    sources = inventory(vial_df, (lambda cid: store.get(cid)) if store else (lambda cid: None),
                        st.load_stocks(), list(c["SURFACTANTS"]))
    return build_plan(c, targets, sources, transfer_errors(c), vial_df)


# ============================================================================
# EXECUTION
# ============================================================================

def _volume(lash_e, vial):
    return float(lash_e.nr_robot.get_vial_info(vial, "vial_volume"))


def refill_table(c, plan, sources):
    """How each vial is refilled before a draw: {vial: rule}."""
    table = {c["WATER_VIAL"]: dict(how="reservoir", fill_ml=c["WATER_VIAL_FILL_ML"])}
    for s, groups in sources.items():
        for g in groups:
            if len(g["vials"]) > 1:
                table[g["name"]] = dict(how="vials", refills=g["vials"][1:], liquid=c["SURFACTANTS"][s]["liquid"])
    for x in plan["substocks"].itertuples():
        table[x.vial] = dict(how="recipe", recipe=x, liquid=c["SURFACTANTS"][x.surfactant]["liquid"])
    return table


def make_substock(lash_e, c, recipe, table, label):
    """Add one recipe (reservoir water, then the parent) to a substock vial; vortex."""
    parent_ml, water_ml = recipe.parent_ul / 1000, recipe.water_ul / 1000
    ensure_volume(lash_e, c, recipe.parent, parent_ml, table)
    liquid = c["SURFACTANTS"][recipe.surfactant]["liquid"]
    lash_e.logger.info(f"{label} {recipe.vial}: {parent_ml:.3f} mL {recipe.parent} + {water_ml:.3f} mL water "
                       f"-> {recipe.conc_mM:.4g} mM")
    lash_e.nr_robot.create_substock(recipe.vial, [
        {"reservoir": c["WATER_RESERVOIR"], "volume_mL": water_ml, "liquid": "water"},
        {"source_vial": recipe.parent, "volume_mL": parent_ml, "liquid": liquid},
    ], vortex_time=c["VORTEX_SECONDS"])


def ensure_volume(lash_e, c, vial, need_ml, table):
    """Make sure vial can give need_ml and keep VIAL_DEAD_ML, refilling it as its rule says."""
    dead = c["VIAL_DEAD_ML"]
    if _volume(lash_e, vial) - need_ml >= dead - 1e-9:
        return
    rule = table.get(vial)
    if rule is None:
        raise RuntimeError(f"{vial}: needs {need_ml:.3f} mL above {dead:g} mL, has "
                           f"{_volume(lash_e, vial):.3f} mL and is never refilled")
    if rule["how"] == "reservoir":
        lash_e.nr_robot.refill_vial(vial, rule["fill_ml"], reservoir=c["WATER_RESERVOIR"], liquid="water")
    elif rule["how"] == "recipe":
        make_substock(lash_e, c, rule["recipe"], table, "Topping up")
    else:
        capacity = vial_capacity_ml(lash_e.nr_robot.get_vial_info(vial, "vial_type"))
        for source in rule["refills"]:
            current = _volume(lash_e, vial)
            add = min(capacity - 0.5 - current, _volume(lash_e, source) - dead)
            if add >= c["SUBSTOCK_MIN_TRANSFER_UL"] / 1000:
                lash_e.logger.info(f"Refilling {vial} from {source}: +{add:.3f} mL")
                lash_e.nr_robot.refill_vial(vial, current + add, source_vial=source, liquid=rule["liquid"])
            if _volume(lash_e, vial) - need_ml >= dead - 1e-9:
                break
    after = _volume(lash_e, vial)
    if after - need_ml < dead - 1e-9:
        raise RuntimeError(f"{vial}: still too little after refilling ({after:.3f} mL for a "
                           f"{need_ml:.3f} mL draw above {dead:g} mL)")


def fill_chunks(volumes, max_ml):
    """Split wells into groups whose total fits in one fill."""
    chunks, current, total = [], [], 0.0
    for index, ml in volumes.items():
        if current and total + ml > max_ml:
            chunks.append(volumes.loc[current])
            current, total = [], 0.0
        current.append(index)
        total += ml
    if current:
        chunks.append(volumes.loc[current])
    return chunks


def component_volumes(c, wells, component):
    """[(vial, liquid, Series of mL by well_index)] for one DISPENSE_ORDER component."""
    w = wells.set_index("well_index")
    if component == "water":
        return [(c["WATER_VIAL"], "water", w.water_ul / 1000)]
    if component == "additives":
        return [(a["vial"], a["liquid"], w[f"{name}_ul"] / 1000) for name, a in c["ADDITIVES"].items()]
    out = []
    for s in surfactants_in(c, wells):
        for source, group in w.dropna(subset=[f"{s}_source"]).groupby(f"{s}_source"):
            out.append((source, c["SURFACTANTS"][s]["liquid"], group[f"{s}_ul"] / 1000))
    return out


def dispense_batch(lash_e, c, wells, table):
    for component in c["DISPENSE_ORDER"]:
        for vial, liquid, volumes in component_volumes(c, wells, component):
            volumes = volumes[volumes > 0]
            if volumes.empty:
                continue
            capacity = vial_capacity_ml(lash_e.nr_robot.get_vial_info(vial, "vial_type"))
            per_fill = (c["SUBSTOCK_VOLUME_ML"] if table.get(vial, {}).get("how") == "recipe"
                        else c["WATER_VIAL_FILL_ML"] if vial == c["WATER_VIAL"] else capacity) - c["VIAL_DEAD_ML"]
            for chunk in fill_chunks(volumes, per_fill):
                ensure_volume(lash_e, c, vial, float(chunk.sum()), table)
                lash_e.nr_robot.dispense_from_vials_into_wellplate(
                    pd.DataFrame({vial: chunk}), liquid={vial: liquid}, well_plate_type=c["WELLPLATE_TYPE"])


def equilibrate(lash_e, c, plate, batch):
    """Wait EQUILIBRATION_MIN after dispensing (logged for duration estimates; not waited in simulation)."""
    if c["EQUILIBRATION_MIN"] <= 0:
        return
    import time
    from workflows.helpers._planning_markers import mark_wait
    lash_e.logger.info(f"Plate {plate}, batch {batch}: equilibrating {c['EQUILIBRATION_MIN']:g} min before reading")
    mark_wait(lash_e.logger, f"plate{plate}_batch{batch}_equilibration", c["EQUILIBRATION_MIN"] * 60)
    if not lash_e.simulate:
        time.sleep(c["EQUILIBRATION_MIN"] * 60)


def _well_label(value):
    text = str(value).strip()
    return f"{text[0].upper()}{int(text[1:])}"


def raw_to_columns(raw):
    """Cytation frame -> well_position plus <protocol>_<channel> columns."""
    frame = raw.dropna(how="all").copy()
    frame.columns = ["_".join(str(p) for p in col).removeprefix("rep1_") if isinstance(col, tuple) else str(col)
                     for col in frame.columns]
    frame.index = [_well_label(p) for p in frame.index]
    return frame.rename_axis("well_position").reset_index()


def measure_batch(lash_e, c, wells, output_dir, plate, batch):
    if not c["MEASUREMENT_PROTOCOL_FILES"]:
        return wells.assign(read_at=None)
    raw = lash_e.measure_wellplate(list(c["MEASUREMENT_PROTOCOL_FILES"]), wells.well_index.tolist(),
                                   plate_type=c["WELLPLATE_TYPE"],
                                   preparation_protocol=c["PREPARATION_PROTOCOL_FILE"])
    read_at = datetime.now().isoformat(timespec="seconds")
    if raw is None:
        if not lash_e.simulate:
            raise RuntimeError(f"No plate data for plate {plate}, batch {batch}")
        return wells.assign(read_at=read_at)
    raw.to_csv(output_dir / f"raw_plate{plate}_batch{batch}.csv")
    return wells.merge(raw_to_columns(raw), on="well_position", how="left", validate="one_to_one").assign(
        read_at=read_at)


def _send_workflow_slack(lash_e, message):
    """Send live-run updates without letting Slack failures fail the experiment."""
    if lash_e.simulate:
        return
    try:
        import slack_agent
        if not slack_agent.safe_send_slack_message(message):
            lash_e.logger.warning("Slack notification was not delivered (non-fatal)")
    except Exception as error:
        lash_e.logger.warning(f"Slack notification failed (non-fatal): {error}")


def execute(config=None, show_gui=True):
    """Plan, make substocks, then make and read the samples batch by batch."""
    from master_usdl_coordinator import Lash_E
    from experiment_tracking import sample_tracking as st
    lash_e = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME, config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return None
    logger, c = lash_e.logger, lash_e.workflow_config
    try:
        lash_e.nr_robot.home_robot_components()
        if c["SIMULATE"] != lash_e.simulate:
            raise ValueError("Workflow and controller simulation modes do not match.")
        validate_config(c)
        robot = lash_e.nr_robot
        vial_df = robot.VIAL_DF
        store = robot._contents_store()
        sources = inventory(vial_df, (lambda cid: store.get(cid)) if store else (lambda cid: None),
                            st.load_stocks(), list(c["SURFACTANTS"]))
        plan = build_plan(c, load_targets(c), sources, transfer_errors(c), vial_df)
        check_vials(c, plan, vial_df)
        if not lash_e.simulate:
            for path in list(c["MEASUREMENT_PROTOCOL_FILES"]) + [c["PREPARATION_PROTOCOL_FILE"]] * bool(
                    c["PREPARATION_PROTOCOL_FILE"]):
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
        for line in report(c, plan).splitlines():
            (logger.warning if line.startswith("WARNING") else logger.info)(line)
        wells = plan["wells"]
        if wells.empty:
            raise ValueError("No wells to make (all skipped)")

        mode = "SIMULATION" if lash_e.simulate else "LIVE"
        output_dir = (_REPO / "output" / _WORKFLOW_NAME /
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode}_wells-{len(wells)}")
        output_dir.mkdir(parents=True, exist_ok=False)
        lash_e.save_run_info(output_dir)
        save_plan(c, plan, output_dir)
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started | {Path(c['TARGET_FILE']).name} | {len(wells)} wells, "
                                     f"{wells.plate.nunique()} plate(s), {len(plan['substocks'])} new substocks, "
                                     f"{len(plan['skipped'])} skipped, {len(plan['warnings'])} warnings")

        table = refill_table(c, plan, sources)
        for recipe in plan["substocks"].itertuples():
            make_substock(lash_e, c, recipe, table, "Making")

        results, plate = [], None
        for (p, batch), batch_wells in wells.groupby(["plate", "batch"], sort=True):
            if p != plate:
                if plate is not None:
                    lash_e.discard_used_wellplate()
                lash_e.grab_new_wellplate()
                plate = p
            logger.info(f"Plate {p}, batch {batch}: {len(batch_wells)} wells")
            dispense_batch(lash_e, c, batch_wells, table)
            equilibrate(lash_e, c, p, batch)
            results.append(measure_batch(lash_e, c, batch_wells, output_dir, p, batch))
            pd.concat(results, ignore_index=True).to_csv(output_dir / "results.csv", index=False)
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME}: plate {p} batch {batch} done "
                                         f"({sum(len(r) for r in results)}/{len(wells)} wells)")
        lash_e.discard_used_wellplate()
        logger.info(f"Workflow complete; results in {output_dir}")
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} complete; results in {output_dir.name}")
        return output_dir
    except (Exception, KeyboardInterrupt) as e:
        logger.exception(f"Workflow failed with error: {e}")
        from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
        cleanup_after_keyboard_interrupt(lash_e)
        if isinstance(e, KeyboardInterrupt):
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted by operator; cleanup attempted, "
                                         "review experiment log")
        raise


def _dry_run():
    """--plan: pre-run report from the saved config (or these defaults), no hardware."""
    from experiment_tracking import sample_contents as sc
    c = {k: globals()[k] for k in _CONFIG_KEYS}
    saved = _REPO / "workflow_configs" / f"{_WORKFLOW_NAME}.yaml"
    if saved.exists():
        with open(saved) as f:
            c.update(yaml.safe_load(f) or {})
    vial_df = pd.read_csv(_REPO / c["INPUT_VIAL_STATUS_FILE"])
    plan = plan_from_files(c, vial_df, sc.CONTENTS_FILE)
    folder = (_REPO / "output" / _WORKFLOW_NAME / "plans" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_plan(c, plan, folder)
    print(report(c, plan))
    print(f"\nSaved to {folder}")


if __name__ == "__main__":
    if "--plan" in sys.argv:
        _dry_run()
    else:
        execute()
