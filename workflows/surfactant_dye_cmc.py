"""
Single-surfactant plate with one dye (current run: SDS + coumarin-6), one well plate.

Plan: docs/plans/dye_surfactant_grid_plan.md (sections 4. Workflow and First run).

Steps:
1. Targets: what each well should contain (surfactant mM, dye uM, matrix), built
   from the BLOCKS config. Each block is a grid (every surfactant level x every
   dye level, some pairs skipped) in water or in pure dye solvent; a CMC series is
   a grid with one dye level, a stock check a solvent grid with surfactant 0.
   extra_block_targets() is the hook for blocks planned from earlier results
   (adaptive runs); empty now.
2. Plan: workflows/helpers/well_plan.py picks a source and volume for every
   component of every well and the fewest substocks needed. Water wells all get
   DYE_VOLUME_UL of dye solution, so the solvent fraction is the same everywhere.
3. Make the batch's new substocks (create_substock: reservoir water first, then
   the pipetted parent; vortex).
4. Dispense the batch (one dispense_from_vials_into_wellplate call per component,
   in DISPENSE_ORDER), then read it on its schedule (fluorescence + absorbance).

Batches: each block has a batch number. Batches run in order on the SAME plate;
substocks made for one batch are reused by later ones. A later batch can be
planned from earlier results through extra_block_targets().

Refills: before each draw, a vial that would go below its refill level is
refilled as its refill_table() row says (substocks in their recipe's ratio, water
from the reservoir, SDS stock and solvent from their refill vials). The table is
saved with the plan as refill_table.csv.

Reuse across runs: each new substock is recorded (what it holds, concentration,
recipe, when made) in <vial file>_substocks.yaml next to the vial file. A later run
reuses any non-empty substock vial the record matches (same surfactant / dye and
solvent, at most SUBSTOCK_MAX_AGE_HOURS old) before making new ones, and refuses to
start while a substock vial is non-empty but unrecorded, too old or holds
something else. Simulations write the record only inside a scheduler session.

build_plan(config, reused) runs steps 1-2 for every configured batch without hardware.

Pipetting validation is not part of this workflow: run validate_all_calibrations
(LIQUIDS water, SDS, ethanol) before and after it.
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from workflows.helpers.well_plan import after_batch, plan_wells, stock_sources, top_up

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/surfactant_dye_cmc_vials.csv"
WELLPLATE_TYPE = "96 WELL PLATE"

# Liquids and stocks. Stock concentrations will come from status/stocks.csv once
# it exists; until then they are set here.
SURFACTANT = "SDS"                  # liquids.yaml name; also the substock name prefix
SURFACTANT_STOCK_VIAL = "SDS_stock"
SURFACTANT_REFILL_VIAL = "SDS_refill"  # same stock; tops up SURFACTANT_STOCK_VIAL (one vial or a list, used in order)
SURFACTANT_STOCK_MM = 50.0
SURFACTANT_CMC_MM = 8.6
DYE = "coumarin-6"                  # key in surfactant_grid_ailsa._DYE_PROTOCOLS
DYE_STOCK_VIAL = "coumarin6_stock"
DYE_STOCK_UM = 300.0                # coumarin-6 in ethanol (most that dissolves); the stock check measures it
DYE_SOLVENT = "ethanol"             # liquids.yaml name
DYE_SOLVENT_VIAL = "ethanol"
SOLVENT_REFILL_VIAL = "ethanol_refill"  # pure solvent in a large vial; tops up DYE_SOLVENT_VIAL
STOCK_REFILL_TO_ML = 7.5            # SDS stock and solvent vials are topped up to this...
STOCK_REFILL_BELOW_ML = 4.0         # ...when a draw would take them below this
WATER_VIAL = "water"
WATER_RESERVOIR = 1
WATER_VIAL_FILL_ML = 7.5

# Volumes
TOTAL_VOLUME_UL = 200.0             # includes the dye
DYE_VOLUME_UL = 25.0                # fixed dye-solution volume in every water well (substocks at 8x final)
DYE_SOLVENT_EVAPORATES = True       # volatile dye solvent (ethanol): water + SDS fill TOTAL_VOLUME_UL; False for DMSO
WELL_MIN_UL = 5.0                   # smallest variable-volume transfer into a well
WELL_MAX_UL = 195.0                 # largest (small tip 200 uL less its 5 uL margin)
SURFACTANT_PREFERRED_MIN_UL = 75.0  # surfactant wells aim for at least this (more substocks if higher)
SUBSTOCK_MIN_TRANSFER_UL = 200.0    # smallest parent transfer when making a substock
SURFACTANT_SUBSTOCK_ML = 6.0        # each SDS substock is made (and topped up) to this...
SURFACTANT_SUBSTOCK_REFILL_ML = 4.0 # ...and topped up in its recipe's ratio below this
DYE_SUBSTOCK_ML = 4.0
DYE_SUBSTOCK_REFILL_ML = 2.0
STOCK_RESERVE_ML = 0.5              # left in the dye stock and the refill vials
# Substocks left from earlier runs are reused when this record says what they hold
# and they are younger than this (ethanol substocks concentrate as they evaporate).
SUBSTOCK_MAX_AGE_HOURS = 72.0

# What goes on the plate. Each block is a grid: every surfactant level x every dye
# level, `replicates` independently prepared wells each, final well concentrations.
#   matrix: "water" (water + surfactant + DYE_VOLUME_UL of dye solution) or
#           "solvent" (dye in pure DYE_SOLVENT; surfactant must be 0)
#   surfactant_mM, or surfactant_x_cmc (multiples of SURFACTANT_CMC_MM): one of the two
#   skip (optional): [surfactant, dye] pairs to leave out, in the block's own units
#   batch: batches are dispensed and read in order, each on its own schedule
BLOCKS = [
    # Block 3 of the spec: dye in ethanol (stock concentration, linearity); read once.
    {"name": "stock_check", "batch": 1, "matrix": "solvent", "replicates": 1,
     "surfactant_mM": [0.0], "dye_uM": [0.0, 1.0, 4.0, 12.0, 20.0, 32.0]},
    # Block 1: CMC series at 1 uM dye.
    {"name": "cmc", "batch": 2, "matrix": "water", "replicates": 3,
     "surfactant_mM": [0.0, 2.0, 5.0, 7.0, 7.6, 8.0, 8.4, 8.8, 9.5, 10.0, 11.0, 16.0, 22.0, 32.0],
     "dye_uM": [1.0]},
    # Block 2: dye x SDS matrix; the two pairs already in the CMC block are skipped.
    {"name": "dye_matrix", "batch": 2, "matrix": "water", "replicates": 2,
     "surfactant_mM": [0.0, 6.0, 10.0, 16.0, 25.0], "dye_uM": [0.0, 1.0, 4.0, 12.0, 32.0],
     "skip": [[0.0, 1.0], [16.0, 1.0]]},
]

# Reads: one schedule (minutes after that batch is dispensed) per batch, in batch order.
# Batch 2 follows Study 3 of the CMC/EE studies draft (0, 1, 2, 4, 8, 16 h).
BATCH_SCHEDULES_MIN = [[0], [0, 60, 120, 240, 480, 960]]
FLUORESCENCE_PROTOCOL_FILE = None   # None: the DYE's protocol in _DYE_PROTOCOLS
ABSORBANCE_PROTOCOL_FILE = None     # not created yet; reads are skipped while None

DISPENSE_ORDER = ["dye", "water", "surfactant", "solvent"]
# Waits between components, per batch: [{"batch": 2, "after": "dye", "minutes": 30}]
# (e.g. dye-in-ethanol first, let the ethanol evaporate, then the aqueous parts:
# also put "dye" first in DISPENSE_ORDER). Empty = no pauses.
DISPENSE_PAUSES = [{"batch": 2, "after": "dye", "minutes": 30}]
RANDOMIZED_ORDER = True
RANDOMIZATION_SEED = 42
VORTEX_SECONDS = 5

_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "WELLPLATE_TYPE",
    "SURFACTANT", "SURFACTANT_STOCK_VIAL", "SURFACTANT_REFILL_VIAL", "SURFACTANT_STOCK_MM",
    "SURFACTANT_CMC_MM",
    "DYE", "DYE_STOCK_VIAL", "DYE_STOCK_UM", "DYE_SOLVENT", "DYE_SOLVENT_VIAL", "SOLVENT_REFILL_VIAL",
    "STOCK_REFILL_TO_ML", "STOCK_REFILL_BELOW_ML",
    "WATER_VIAL", "WATER_RESERVOIR", "WATER_VIAL_FILL_ML",
    "TOTAL_VOLUME_UL", "DYE_VOLUME_UL", "DYE_SOLVENT_EVAPORATES", "WELL_MIN_UL", "WELL_MAX_UL", "SURFACTANT_PREFERRED_MIN_UL",
    "SUBSTOCK_MIN_TRANSFER_UL",
    "SURFACTANT_SUBSTOCK_ML", "SURFACTANT_SUBSTOCK_REFILL_ML", "DYE_SUBSTOCK_ML",
    "DYE_SUBSTOCK_REFILL_ML", "STOCK_RESERVE_ML", "SUBSTOCK_MAX_AGE_HOURS",
    "BLOCKS", "BATCH_SCHEDULES_MIN", "FLUORESCENCE_PROTOCOL_FILE", "ABSORBANCE_PROTOCOL_FILE",
    "DISPENSE_ORDER", "DISPENSE_PAUSES", "RANDOMIZED_ORDER", "RANDOMIZATION_SEED", "VORTEX_SECONDS",
]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV (repository-relative) for this run.",
                               "unit": None, "affects": ["vials", "liquids"], "shares": "vial_layout"},
    "WELLPLATE_TYPE": {"description": "Plate type (robot_state/wellplates.yaml); well count and volume limit come from there.",
                       "unit": None, "choices": ["96 WELL PLATE", "48 WELL PLATE", "24 WELL PLATE"],
                       "affects": ["plates"], "shares": "plate_type"},
    "SURFACTANT": {"description": "Surfactant liquid name (liquids.yaml) and substock name prefix.",
                   "unit": None, "affects": ["liquids"], "ask": True},
    "SURFACTANT_STOCK_VIAL": {"description": "Vial holding the surfactant stock.", "unit": None,
                              "affects": ["vials"]},
    "SURFACTANT_STOCK_MM": {"description": "Surfactant stock concentration.", "unit": "mM",
                            "affects": ["vials", "liquids"],
                            "effect": "weaker stock -> larger well volumes; error if the top level needs more than WELL_MAX_UL"},
    "SURFACTANT_REFILL_VIAL": {"description": "Vial(s) of the same surfactant stock that top up SURFACTANT_STOCK_VIAL: one name or a list, used in order (e.g. two closed 8 mL vials).",
                               "unit": None, "affects": ["vials", "liquids"]},
    "SURFACTANT_CMC_MM": {"description": "Expected CMC; CMC block levels are multiples of it.",
                          "unit": "mM", "affects": ["liquids"], "ask": True},
    "DYE": {"description": "Dye; selects the fluorescence protocol.", "unit": None,
            "choices": ["pyrene", "coumarin-6", "nile-red"], "affects": [], "shares": "dye"},
    "DYE_STOCK_VIAL": {"description": "Vial holding the concentrated dye stock (in DYE_SOLVENT).",
                       "unit": None, "affects": ["vials"]},
    "DYE_STOCK_UM": {"description": "Dye stock concentration (nominal; the stock check measures it).",
                     "unit": "uM", "affects": ["vials", "liquids"]},
    "DYE_SOLVENT": {"description": "Dye solvent liquid name (liquids.yaml).", "unit": None,
                    "affects": ["liquids"]},
    "DYE_SOLVENT_VIAL": {"description": "Pure dye-solvent vial: dye substock diluent, dye blank, stock-check make-up.",
                         "unit": None, "affects": ["vials", "liquids"]},
    "SOLVENT_REFILL_VIAL": {"description": "Large vial of pure dye solvent; tops up DYE_SOLVENT_VIAL.",
                            "unit": None, "affects": ["vials", "liquids"]},
    "STOCK_REFILL_TO_ML": {"description": "Surfactant stock and solvent vials are topped up to this from their refill vials.",
                           "unit": "mL", "affects": ["duration"]},
    "STOCK_REFILL_BELOW_ML": {"description": "Top up the surfactant stock or solvent vial when a draw would take it below this.",
                              "unit": "mL", "affects": ["duration"],
                              "effect": "each fill gives STOCK_REFILL_TO_ML - this to use"},
    "WATER_VIAL": {"description": "Water vial used for well make-up; refilled from WATER_RESERVOIR.",
                   "unit": None, "affects": ["vials"]},
    "WATER_RESERVOIR": {"description": "Reservoir (pump) index for water.", "unit": None, "affects": []},
    "WATER_VIAL_FILL_ML": {"description": "Water vial is refilled to this from WATER_RESERVOIR.",
                           "unit": "mL", "affects": ["duration"],
                           "effect": "refilled when a draw would take it below 2 mL"},
    "TOTAL_VOLUME_UL": {"description": "Final volume per well; includes the dye solution unless DYE_SOLVENT_EVAPORATES.",
                        "unit": "uL", "affects": ["liquids"]},
    "DYE_SOLVENT_EVAPORATES": {"description": "Dye solvent evaporates before reading: water + surfactant fill TOTAL_VOLUME_UL "
                                              "and the dye solution goes on top; concentrations are over TOTAL_VOLUME_UL. "
                                              "Volumes only; independent of DISPENSE_ORDER and DISPENSE_PAUSES.",
                               "unit": None, "affects": ["liquids", "duration"]},
    "DYE_VOLUME_UL": {"description": "Dye-solution volume in every water well (sets the solvent fraction).",
                      "unit": "uL", "affects": ["liquids", "vials"],
                      "effect": "one dye substock per dye level, at dye x TOTAL/DYE_VOLUME"},
    "WELL_MIN_UL": {"description": "Smallest variable-volume transfer into a well.", "unit": "uL",
                    "affects": ["vials"], "effect": "higher -> more substocks"},
    "WELL_MAX_UL": {"description": "Largest variable-volume transfer into a well.", "unit": "uL",
                    "affects": ["vials"], "effect": "lower -> more substocks"},
    "SURFACTANT_PREFERRED_MIN_UL": {"description": "Surfactant volume each well aims for at least; a substock is added when no source reaches it.",
                                    "unit": "uL", "affects": ["vials", "liquids", "duration"],
                                    "effect": "higher -> larger, more accurate volumes but more substocks (CMC block, 50 mM stock: 20 -> 1, 75 -> 2, 100 -> 4)"},
    "SUBSTOCK_MIN_TRANSFER_UL": {"description": "Smallest parent transfer when making a substock.",
                                 "unit": "uL", "affects": ["vials"],
                                 "effect": "a substock needing less from every source gets an intermediate"},
    "SURFACTANT_SUBSTOCK_ML": {"description": "Each surfactant substock is made, and topped up, to this.",
                               "unit": "mL", "affects": ["liquids", "duration"]},
    "SURFACTANT_SUBSTOCK_REFILL_ML": {"description": "Top up a surfactant substock (same recipe ratio) when a draw would take it below this.",
                                      "unit": "mL", "affects": ["liquids", "duration"],
                                      "effect": "each fill gives SURFACTANT_SUBSTOCK_ML - this to use"},
    "DYE_SUBSTOCK_ML": {"description": "Each dye substock is made, and topped up, to this.", "unit": "mL",
                        "affects": ["liquids", "duration"]},
    "DYE_SUBSTOCK_REFILL_ML": {"description": "Top up a dye substock (same recipe ratio) when a draw would take it below this.",
                               "unit": "mL", "affects": ["liquids", "duration"],
                               "effect": "each fill gives DYE_SUBSTOCK_ML - this to use"},
    "STOCK_RESERVE_ML": {"description": "Volume left in the dye stock (never refilled) and in the refill vials.",
                         "unit": "mL", "affects": []},
    "SUBSTOCK_MAX_AGE_HOURS": {"description": "Substocks left from earlier runs are reused up to this age (from when first made); older ones must be emptied.",
                               "unit": "h", "affects": ["vials", "liquids"],
                               "effect": "reused substocks need no new stock; a run refuses to start while an older or unrecorded substock vial is not empty"},
    "BLOCKS": {"description": "Plate contents: list of blocks, each a grid of surfactant x dye levels "
                              "(name, batch, matrix water/solvent, replicates, surfactant_mM or "
                              "surfactant_x_cmc, dye_uM, optional skip pairs). Final well concentrations.",
               "unit": None, "affects": ["liquids", "vials", "tips", "duration"],
               "effect": "wells per block = (surfactant levels x dye levels - skipped) x replicates"},
    "BATCH_SCHEDULES_MIN": {"description": "Read times per batch (minutes after it is dispensed), one list per batch.",
                            "unit": "min", "affects": ["duration"],
                            "effect": "each batch waits until its last read before the next batch"},
    "FLUORESCENCE_PROTOCOL_FILE": {"description": "Cytation fluorescence protocol; null = the DYE's default.",
                                   "unit": None, "affects": ["duration"]},
    "ABSORBANCE_PROTOCOL_FILE": {"description": "Cytation absorbance protocol (dye peak + 750 nm); null skips absorbance.",
                                 "unit": None, "affects": ["duration"]},
    "DISPENSE_ORDER": {"description": "Order of components: water, solvent, surfactant, dye.", "unit": None,
                       "affects": []},
    "DISPENSE_PAUSES": {"description": "Waits between components: list of {batch, after (water/solvent/surfactant/dye), minutes}. "
                                       "The wait starts after the last well gets that component.",
                        "unit": "min", "affects": ["duration"],
                        "effect": "each pause adds its minutes to the batch, before the first read"},
    "RANDOMIZED_ORDER": {"description": "Place each batch's wells at random free positions.", "unit": None,
                         "affects": []},
    "RANDOMIZATION_SEED": {"description": "Seed for well placement.", "unit": None, "affects": []},
    "VORTEX_SECONDS": {"description": "Vortex time after making each substock.", "unit": "s",
                       "affects": ["duration"]},
}

# Substocks this workflow makes: <SURFACTANT>_sub_<n>, dye_sub_<n>.
_PRODUCT_VIALS = ["*_sub_[0-9]*"]

_WELLPLATES_FILE = Path(__file__).resolve().parents[1] / "robot_state" / "wellplates.yaml"
# Wells per row for each standard plate size (same layout as biotek_new.well_index_to_label).
_COLUMNS_BY_WELLS = {96: 12, 48: 8, 24: 6}
_BLOCK_KEYS = {"name", "batch", "matrix", "replicates", "surfactant_mM", "surfactant_x_cmc", "dye_uM", "skip"}
# The small tip warns below ~2 mL in a vial, so the water vial is refilled above this.
_WATER_VIAL_RESERVE_ML = 2.0
_VIAL_CAPACITY_ML = 8.0  # substock vials (8 mL)
_COMPONENTS = ("water", "solvent", "surfactant", "dye")


# ============================================================================
# PLANNING (no hardware)
# ============================================================================

def plate_layout(plate_type):
    """dict(wells, columns, max_ul) for a plate type, from robot_state/wellplates.yaml."""
    import yaml
    with open(_WELLPLATES_FILE) as f:
        plates = yaml.safe_load(f)
    if plate_type not in plates:
        raise ValueError(f"Unknown WELLPLATE_TYPE {plate_type!r}; see {_WELLPLATES_FILE.name}")
    wells = int(plates[plate_type]["num_wells"])
    if wells not in _COLUMNS_BY_WELLS:
        raise ValueError(f"No row layout for a {wells}-well plate")
    return dict(wells=wells, columns=_COLUMNS_BY_WELLS[wells],
                max_ul=float(plates[plate_type]["max_volume_per_well"]) * 1000)


def well_position(index, columns):
    return f"{chr(65 + index // columns)}{index % columns + 1}"


def planning_rules(c):
    """Config -> rules for well_plan.plan_wells."""
    return dict(
        total_ul=c["TOTAL_VOLUME_UL"], dye_ul=c["DYE_VOLUME_UL"],
        dye_solvent_evaporates=c["DYE_SOLVENT_EVAPORATES"],
        well_min_ul=c["WELL_MIN_UL"], well_max_ul=c["WELL_MAX_UL"],
        preferred_min_ul={"surfactant": c["SURFACTANT_PREFERRED_MIN_UL"]},
        substock_min_transfer_ul=c["SUBSTOCK_MIN_TRANSFER_UL"],
        stocks={"surfactant": {"vial": c["SURFACTANT_STOCK_VIAL"], "conc": c["SURFACTANT_STOCK_MM"]},
                "dye": {"vial": c["DYE_STOCK_VIAL"], "conc": c["DYE_STOCK_UM"]}},
        water_vial=c["WATER_VIAL"], solvent_vial=c["DYE_SOLVENT_VIAL"],
        solvent_diluent_vial=c["SOLVENT_REFILL_VIAL"],  # bulk solvent for dye substocks (large tip)
        substock_ml={"surfactant": c["SURFACTANT_SUBSTOCK_ML"], "dye": c["DYE_SUBSTOCK_ML"]},
        substock_refill_ml={"surfactant": c["SURFACTANT_SUBSTOCK_REFILL_ML"],
                            "dye": c["DYE_SUBSTOCK_REFILL_ML"]},
        vial_capacity_ml=_VIAL_CAPACITY_ML,
        substock_prefix={"surfactant": f"{c['SURFACTANT']}_sub", "dye": "dye_sub"},
    )


def validate_blocks(c):
    """Check BLOCKS; error messages name the block."""
    blocks = c["BLOCKS"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("BLOCKS must be a non-empty list")
    names = [b.get("name") if isinstance(b, dict) else None for b in blocks]
    if None in names or len(set(names)) != len(names):
        raise ValueError("Every block needs a unique name")
    batches = len(c["BATCH_SCHEDULES_MIN"])
    for b in blocks:
        where = f"Block {b['name']!r}"
        unknown = set(b) - _BLOCK_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        if ("surfactant_mM" in b) == ("surfactant_x_cmc" in b):
            raise ValueError(f"{where}: give exactly one of surfactant_mM or surfactant_x_cmc")
        if b.get("matrix") not in ("water", "solvent"):
            raise ValueError(f"{where}: matrix must be water or solvent")
        if type(b.get("batch")) is not int or not 1 <= b["batch"] <= batches:
            raise ValueError(f"{where}: batch must be a batch with a schedule (1..{batches})")
        if type(b.get("replicates")) is not int or b["replicates"] < 1:
            raise ValueError(f"{where}: replicates must be an integer >= 1")
        surfactant = b.get("surfactant_mM", b.get("surfactant_x_cmc"))
        for key, levels in (("surfactant", surfactant), ("dye_uM", b.get("dye_uM"))):
            if not isinstance(levels, list) or not levels:
                raise ValueError(f"{where}: {key} levels must be a non-empty list")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in levels):
                raise ValueError(f"{where}: {key} levels must be numbers >= 0")
            if len(set(levels)) != len(levels):
                raise ValueError(f"{where}: repeated {key} level (use replicates instead)")
        if b["matrix"] == "solvent" and any(v > 0 for v in surfactant):
            raise ValueError(f"{where}: solvent wells cannot contain surfactant")
        grid = {(float(s), float(d)) for s in surfactant for d in b["dye_uM"]}
        for pair in b.get("skip", []):
            if not isinstance(pair, list) or len(pair) != 2 or tuple(map(float, pair)) not in grid:
                raise ValueError(f"{where}: skip entry {pair} is not a [surfactant, dye] pair of the block")


def validate_refill_vials(c):
    value = c["SURFACTANT_REFILL_VIAL"]
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) \
            or len(set(names)) != len(names):
        raise ValueError("SURFACTANT_REFILL_VIAL must be a vial name or a list of different vial names")
    # Dye substocks draw solvent straight from SOLVENT_REFILL_VIAL, so it stays one vial.
    if not isinstance(c["SOLVENT_REFILL_VIAL"], str) or not c["SOLVENT_REFILL_VIAL"]:
        raise ValueError("SOLVENT_REFILL_VIAL must be one vial name")


def validate_pauses(c):
    pauses = c["DISPENSE_PAUSES"]
    if not isinstance(pauses, list):
        raise ValueError("DISPENSE_PAUSES must be a list")
    batches, seen = len(c["BATCH_SCHEDULES_MIN"]), set()
    for p in pauses:
        if not isinstance(p, dict) or set(p) != {"batch", "after", "minutes"}:
            raise ValueError(f"DISPENSE_PAUSES entries are {{batch, after, minutes}}: {p}")
        if type(p["batch"]) is not int or not 1 <= p["batch"] <= batches:
            raise ValueError(f"DISPENSE_PAUSES batch must be a batch with a schedule (1..{batches}): {p}")
        if p["after"] not in _COMPONENTS:
            raise ValueError(f"DISPENSE_PAUSES after must be one of {', '.join(_COMPONENTS)}: {p}")
        if not isinstance(p["minutes"], (int, float)) or not math.isfinite(p["minutes"]) or p["minutes"] <= 0:
            raise ValueError(f"DISPENSE_PAUSES minutes must be positive: {p}")
        if (p["batch"], p["after"]) in seen:
            raise ValueError(f"Two DISPENSE_PAUSES for batch {p['batch']} after {p['after']}")
        seen.add((p["batch"], p["after"]))


def validate_config(c):
    layout = plate_layout(c["WELLPLATE_TYPE"])
    # With an evaporating dye solvent, the dye goes on top of the total until it evaporates.
    most = c["TOTAL_VOLUME_UL"] + (c["DYE_VOLUME_UL"] if c["DYE_SOLVENT_EVAPORATES"] else 0.0)
    if most > layout["max_ul"]:
        raise ValueError(f"Wells would hold up to {most:g} uL (TOTAL_VOLUME_UL"
                         f"{' + DYE_VOLUME_UL' if c['DYE_SOLVENT_EVAPORATES'] else ''}), above the "
                         f"{layout['max_ul']:g} uL limit of a {c['WELLPLATE_TYPE']}")
    for key in ("SURFACTANT_STOCK_MM", "SURFACTANT_CMC_MM", "DYE_STOCK_UM", "TOTAL_VOLUME_UL",
                "DYE_VOLUME_UL", "WELL_MIN_UL", "WELL_MAX_UL", "SURFACTANT_PREFERRED_MIN_UL",
                "SUBSTOCK_MIN_TRANSFER_UL",
                "SURFACTANT_SUBSTOCK_ML", "DYE_SUBSTOCK_ML", "WATER_VIAL_FILL_ML",
                "SURFACTANT_SUBSTOCK_REFILL_ML", "DYE_SUBSTOCK_REFILL_ML",
                "STOCK_REFILL_TO_ML", "STOCK_REFILL_BELOW_ML", "SUBSTOCK_MAX_AGE_HOURS"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if c["DYE_VOLUME_UL"] >= c["TOTAL_VOLUME_UL"] or c["WELL_MIN_UL"] > c["WELL_MAX_UL"]:
        raise ValueError("Need DYE_VOLUME_UL < TOTAL_VOLUME_UL and WELL_MIN_UL <= WELL_MAX_UL")
    for fill, below in (("SURFACTANT_SUBSTOCK_ML", "SURFACTANT_SUBSTOCK_REFILL_ML"),
                        ("DYE_SUBSTOCK_ML", "DYE_SUBSTOCK_REFILL_ML"),
                        ("STOCK_REFILL_TO_ML", "STOCK_REFILL_BELOW_ML")):
        if not c[below] < c[fill] <= _VIAL_CAPACITY_ML:
            raise ValueError(f"Need {below} < {fill} <= {_VIAL_CAPACITY_ML:g} mL")
    if not c["WELL_MIN_UL"] <= c["SURFACTANT_PREFERRED_MIN_UL"] <= c["WELL_MAX_UL"]:
        raise ValueError("SURFACTANT_PREFERRED_MIN_UL must be between WELL_MIN_UL and WELL_MAX_UL")
    for schedule in c["BATCH_SCHEDULES_MIN"]:
        if not schedule or schedule[0] != 0 or list(schedule) != sorted(set(schedule)):
            raise ValueError("Each BATCH_SCHEDULES_MIN list must start at 0 and increase")
    validate_blocks(c)
    validate_pauses(c)
    validate_refill_vials(c)
    if sorted(c["DISPENSE_ORDER"]) != sorted(_COMPONENTS):
        raise ValueError(f"DISPENSE_ORDER must list {', '.join(_COMPONENTS)} once each")


def configured_targets(c):
    """Target rows for every configured block (batch, block, replicate, matrix, concentrations)."""
    rows = []
    for b in c["BLOCKS"]:
        if "surfactant_mM" in b:
            surfactant, scale = b["surfactant_mM"], 1.0
        else:
            surfactant, scale = b["surfactant_x_cmc"], c["SURFACTANT_CMC_MM"]
        skip = {tuple(map(float, pair)) for pair in b.get("skip", [])}
        for s in surfactant:
            for d in b["dye_uM"]:
                if (float(s), float(d)) in skip:
                    continue
                for replicate in range(1, b["replicates"] + 1):
                    rows.append(dict(batch=b["batch"], block=b["name"], replicate=replicate,
                                     matrix=b["matrix"], surfactant_mM=float(s) * scale, dye_uM=float(d)))
    return pd.DataFrame(rows)


def extra_block_targets(c, batch, results):
    """Hook for a further block (e.g. capacity), planned per batch, optionally from
    earlier results. Return rows like configured_targets(); none yet."""
    return pd.DataFrame()


def assign_wells(targets, used, rng, layout):
    """Give each target a free well: random free wells if rng, else the next free ones."""
    free = [i for i in range(layout["wells"]) if i not in used]
    if len(targets) > len(free):
        raise ValueError(f"{len(targets)} wells needed, {len(free)} free on the plate")
    chosen = list(rng.permutation(free)[:len(targets)]) if rng is not None else free[:len(targets)]
    targets = targets.reset_index(drop=True).copy()
    targets["well_index"] = [int(i) for i in chosen]
    targets["well_position"] = targets.well_index.map(lambda i: well_position(i, layout["columns"]))
    return targets


def plan_batch(c, batch, sources, used_wells, rng, results=None):
    """Plan one batch: (wells, sources after it, recipes to make first)."""
    every = pd.concat([t for t in (configured_targets(c), extra_block_targets(c, batch, results))
                       if not t.empty], ignore_index=True)
    targets = every[every.batch == batch]
    if targets.empty:
        return targets, sources, pd.DataFrame()
    targets = assign_wells(targets, used_wells, rng, plate_layout(c["WELLPLATE_TYPE"]))
    wells, planned, recipes = plan_wells(targets, planning_rules(c), existing=sources,
                                         later_targets=every[every.batch > batch])
    recipes["batch"] = batch
    return wells, planned, recipes


# ============================================================================
# SUBSTOCK REUSE (leftovers from earlier runs)
# ============================================================================

_RECORD_COLUMNS = ["vial", "component", "material", "solvent", "conc", "units", "stock_conc",
                   "parent", "parent_ml", "diluent", "diluent_ml", "volume_ml", "made_at"]


def substock_record_path(vial_file):
    """The substock record sits next to the vial file: <vial file>_substocks.yaml.
    YAML, not CSV, so tools that scan for vial CSV files skip it."""
    vial_file = Path(vial_file)
    return vial_file.with_name(f"{vial_file.stem}_substocks.yaml")


def load_substock_record(path):
    import yaml
    path = Path(path)
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    return pd.DataFrame(rows or [], columns=_RECORD_COLUMNS)


def save_substock_record(record, path):
    import yaml
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in record[_RECORD_COLUMNS].to_dict("records")]
    Path(path).write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")


def _material(c, component):
    """(material, solvent, stock concentration, units) a component's substocks hold."""
    if component == "surfactant":
        return c["SURFACTANT"], "water", c["SURFACTANT_STOCK_MM"], "mM"
    return c["DYE"], c["DYE_SOLVENT"], c["DYE_STOCK_UM"], "uM"


def record_rows(c, recipes, made_at):
    """Record rows for substocks just made."""
    rows = []
    for r in recipes.itertuples():
        material, solvent, stock_conc, units = _material(c, r.component)
        rows.append(dict(vial=r.name, component=r.component, material=material, solvent=solvent,
                         conc=r.conc, units=units, stock_conc=stock_conc, parent=r.parent,
                         parent_ml=r.parent_ml, diluent=r.diluent, diluent_ml=r.diluent_ml,
                         volume_ml=r.volume_ml, made_at=made_at.isoformat(timespec="seconds")))
    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)


def update_substock_record(record, rows):
    """Replace the rows of re-made vials, keep the rest."""
    kept = record[~record.vial.isin(rows.vial)] if not record.empty else record
    return pd.concat([kept, rows], ignore_index=True)[_RECORD_COLUMNS]


def reusable_substocks(c, inventory, record, now):
    """Substock vials left from earlier runs: (sources for the planner, problems).

    A non-empty substock vial is reused when the record says it holds this run's
    surfactant (in water) or dye (in DYE_SOLVENT) and it is at most
    SUBSTOCK_MAX_AGE_HOURS old. Its concentration is rescaled if the stock
    concentration has changed since it was made. Any other non-empty substock
    vial is a problem (the run cannot know what is in it)."""
    import fnmatch
    prefixes = planning_rules(c)["substock_prefix"]
    rows = {r.vial: r for r in record.itertuples()} if not record.empty else {}
    refill_ml = {"surfactant": c["SURFACTANT_SUBSTOCK_REFILL_ML"], "dye": c["DYE_SUBSTOCK_REFILL_ML"]}
    sources, problems = [], []
    for vial in sorted(inventory.index):
        if not any(fnmatch.fnmatch(vial, p) for p in _PRODUCT_VIALS):
            continue
        volume = float(inventory.loc[vial, "vial_volume"])
        if volume <= 0.05:
            continue
        r = rows.get(vial)
        if r is None:
            problems.append(f"{vial} has {volume:.2f} mL but no record of what it holds")
            continue
        material, solvent, stock_conc, units = _material(c, r.component)
        if (r.material, r.solvent, r.units) != (material, solvent, units) or not vial.startswith(prefixes[r.component] + "_"):
            problems.append(f"{vial} holds {r.material} in {r.solvent}, not this run's {material} in {solvent}")
            continue
        age_h = (now - datetime.fromisoformat(r.made_at)).total_seconds() / 3600
        if age_h > c["SUBSTOCK_MAX_AGE_HOURS"]:
            problems.append(f"{vial} was made {age_h:.0f} h ago (limit SUBSTOCK_MAX_AGE_HOURS = "
                            f"{c['SUBSTOCK_MAX_AGE_HOURS']:g} h)")
            continue
        sources.append(dict(name=vial, component=r.component, conc=r.conc * stock_conc / r.stock_conc,
                            kind="substock", volume_ml=float(r.volume_ml), parent=r.parent,
                            parent_ml=float(r.parent_ml), diluent=r.diluent, diluent_ml=float(r.diluent_ml),
                            new=False, left_ml=max(0.0, volume - refill_ml[r.component]), reused=True))
    # Top-ups need the parent; one that is not here now is replaced by the stock.
    stock_vial = {"surfactant": c["SURFACTANT_STOCK_VIAL"], "dye": c["DYE_STOCK_VIAL"]}
    present = set(stock_vial.values()) | {s["name"] for s in sources}
    for s in sources:
        if s["parent"] not in present:
            stock_conc = _material(c, s["component"])[2]
            s["parent"] = stock_vial[s["component"]]
            s["parent_ml"] = s["volume_ml"] * s["conc"] / stock_conc
            s["diluent_ml"] = s["volume_ml"] - s["parent_ml"]
    return sources, problems


def starting_sources(c, reused):
    """Planner sources before the first batch: stocks plus reused substocks (None = stocks only)."""
    if not reused:
        return None
    return stock_sources(planning_rules(c)) + [dict(s) for s in reused]


def reused_recipes(reused):
    """Recipes of reused substocks, so they can be topped up like new ones."""
    return pd.DataFrame([dict(name=s["name"], component=s["component"], conc=s["conc"], parent=s["parent"],
                              parent_ml=s["parent_ml"], diluent=s["diluent"], diluent_ml=s["diluent_ml"],
                              volume_ml=s["volume_ml"], batch=0) for s in reused or []],
                        columns=["name", "component", "conc", "parent", "parent_ml", "diluent",
                                 "diluent_ml", "volume_ml", "batch"])


def build_plan(config, reused=None):
    """Plan every configured batch without hardware.

    reused: leftover substocks from reusable_substocks() (None = none).
    Returns dict(wells=all wells, recipes=all substocks with the batch that makes
    them, sources=per-batch source lists with demand_ml).
    """
    c = config
    validate_config(c)
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"]) if c["RANDOMIZED_ORDER"] else None
    sources, used = starting_sources(c, reused), set()
    all_wells, all_recipes, batch_sources = [], [], {}
    for batch in range(1, len(c["BATCH_SCHEDULES_MIN"]) + 1):
        wells, planned, recipes = plan_batch(c, batch, sources, used, rng)
        if wells.empty:
            continue
        used |= set(wells.well_index)
        all_wells.append(wells)
        all_recipes.append(recipes)
        batch_sources[batch] = planned
        sources = after_batch(planned)
    if not all_wells:
        raise ValueError("No wells configured")
    return dict(wells=pd.concat(all_wells, ignore_index=True),
                recipes=pd.concat([r for r in all_recipes if not r.empty] or all_recipes[:1],
                                  ignore_index=True),
                sources=batch_sources)


def total_demand(plan):
    """mL drawn from each vial over the whole plan."""
    demand = {}
    for sources in plan["sources"].values():
        for s in sources:
            demand[s["name"]] = demand.get(s["name"], 0.0) + s["demand_ml"]
    return demand


# ============================================================================
# MEASUREMENT
# ============================================================================

def read_protocols(c):
    """[(label, protocol path)] for each read, skipping a protocol set to None."""
    from workflows.surfactant_grid_ailsa import _DYE_PROTOCOLS, _PLATE_PROTOCOL_SUFFIX
    if c["DYE"] not in _DYE_PROTOCOLS:
        raise ValueError(f"Unknown DYE {c['DYE']}")
    fluorescence = (c["FLUORESCENCE_PROTOCOL_FILE"]
                    or _DYE_PROTOCOLS[c["DYE"]]["fluorescence_protocol_file"].format(
                        plate=_PLATE_PROTOCOL_SUFFIX[c["WELLPLATE_TYPE"]]))
    reads = [("fluorescence", fluorescence)]
    if c["ABSORBANCE_PROTOCOL_FILE"]:
        reads.append(("absorbance", c["ABSORBANCE_PROTOCOL_FILE"]))
    return reads


def simulate_read(wells, label, c, rng):
    """Synthetic plate data in Cytation layout (index A1.., columns (protocol, channel)).

    Shapes only, not real values: dye binding K = 2 /mM to micellised surfactant
    (smooth over 5% of the CMC), and absorbance at 5e4 /M/cm over a 0.6 cm path."""
    from workflows.surfactant_grid_ailsa import _DYE_PROTOCOLS
    cmc, dye = c["SURFACTANT_CMC_MM"], wells.dye_uM.to_numpy()
    scale = 0.05 * cmc
    micellar = scale * np.logaddexp(0.0, (wells.surfactant_mM.to_numpy() - cmc) / scale)
    bound = np.where(wells.matrix == "solvent", 1.0, 2.0 * micellar / (1 + 2.0 * micellar))
    noise = lambda n: 1 + 0.03 * rng.standard_normal(n)
    if label == "fluorescence":
        channel = next(iter(_DYE_PROTOCOLS[c["DYE"]]["raw_column_mapping"]))  # e.g. 485_530
        data = {channel: (50 + dye * (100 + 2000 * bound)) * noise(len(wells))}
    else:
        data = {"peak": (0.04 + dye * 0.05 * 0.6 * noise(len(wells))),
                "750": np.full(len(wells), 0.04)}
    frame = pd.DataFrame(data, index=wells.well_position.to_numpy())
    frame.columns = pd.MultiIndex.from_tuples([(f"rep1_{label}", col) for col in frame.columns])
    return frame


def raw_to_columns(raw, label):
    """Cytation frame -> well_position plus <label>_<channel> columns."""
    frame = raw.copy()
    frame.columns = [f"{label}_{col[1] if isinstance(col, tuple) else col}" for col in frame.columns]
    frame.index = [f"{str(p)[0].upper()}{int(str(p)[1:])}" for p in frame.index]
    return frame.rename_axis("well_position").reset_index()


# ============================================================================
# EXECUTION
# ============================================================================

def _vial_list(value):
    """A refill setting as a list: one vial name or a list of them (used in order)."""
    return [value] if isinstance(value, str) else list(value)


def refillable(c):
    """{working vial: [refill vials, used in order]} for vials topped up from refill vials."""
    return {c["SURFACTANT_STOCK_VIAL"]: _vial_list(c["SURFACTANT_REFILL_VIAL"]),
            c["DYE_SOLVENT_VIAL"]: _vial_list(c["SOLVENT_REFILL_VIAL"])}


def refill_table(c, recipes):
    """How each vial is refilled: {vial: dict(fill_ml, below_ml, how, recipe)}.
    how: "recipe" (top up in the recipe's ratio), "reservoir", or a list of refill vials.
    A vial not in the table is never refilled and keeps STOCK_RESERVE_ML."""
    table = {c["WATER_VIAL"]: dict(fill_ml=c["WATER_VIAL_FILL_ML"], below_ml=_WATER_VIAL_RESERVE_ML,
                                   how="reservoir", recipe=None)}
    for vial, refill in refillable(c).items():
        table[vial] = dict(fill_ml=c["STOCK_REFILL_TO_ML"], below_ml=c["STOCK_REFILL_BELOW_ML"],
                           how=refill, recipe=None)
    for r in recipes.itertuples():
        below = c["SURFACTANT_SUBSTOCK_REFILL_ML"] if r.component == "surfactant" else c["DYE_SUBSTOCK_REFILL_ML"]
        table[r.name] = dict(fill_ml=r.volume_ml, below_ml=below, how="recipe", recipe=r)
    return table


def _rule(c, table, vial):
    return table.get(vial, dict(fill_ml=None, below_ml=c["STOCK_RESERVE_ML"], how=None, recipe=None))


def refill_table_frame(table):
    """The refill table as rows, saved with the plan."""
    return pd.DataFrame([dict(vial=vial, fill_ml=r["fill_ml"], refill_below_ml=r["below_ml"],
                              refilled_from=(f"{r['recipe'].parent} + {r['recipe'].diluent}, recipe ratio"
                                             if r["how"] == "recipe" else
                                             ", ".join(r["how"]) if isinstance(r["how"], list) else r["how"]))
                         for vial, r in table.items()])


def check_vials(c, plan, inventory):
    """Before anything moves: vials exist, substock vials are empty, and the vials
    that are never refilled (dye stock, refill vials) hold enough for the whole run."""
    demand = total_demand(plan)
    refills = refillable(c)
    missing = sorted((set(demand) | {v for vials in refills.values() for v in vials}) - set(inventory.index))
    if missing:
        raise ValueError(f"Add these vials to {c['INPUT_VIAL_STATUS_FILE']}: {', '.join(missing)}")
    for recipe in plan["recipes"].itertuples():
        if inventory.loc[recipe.name, "vial_volume"] > 0.05:
            raise ValueError(f"Substock vial {recipe.name} must start empty "
                             f"(has {inventory.loc[recipe.name, 'vial_volume']:.2f} mL)")
    reserve = c["STOCK_RESERVE_ML"]
    have = inventory.loc[c["DYE_STOCK_VIAL"], "vial_volume"]
    need = demand.get(c["DYE_STOCK_VIAL"], 0.0)
    if need > 0 and need + reserve > have + 1e-9:
        raise ValueError(f"{c['DYE_STOCK_VIAL']}: needs {need:.2f} mL + {reserve:g} mL reserve, has {have:.2f} mL")
    for vial, refills_for in refills.items():
        # The refill vials cover what the working vial can't, plus anything drawn from them directly.
        from_working = inventory.loc[vial, "vial_volume"] - c["STOCK_REFILL_BELOW_ML"]
        need = max(0.0, demand.get(vial, 0.0) - from_working) + sum(demand.get(r, 0.0) for r in refills_for)
        have = sum(max(0.0, inventory.loc[r, "vial_volume"] - reserve) for r in refills_for)
        if need > have + 1e-9:
            raise ValueError(f"{', '.join(refills_for)}: the run needs {need:.2f} mL from them, they have "
                             f"{have:.2f} mL above their {reserve:g} mL reserve")


def _volume(lash_e, vial):
    return float(lash_e.nr_robot.get_vial_info(vial, "vial_volume"))


def _add_to_substock(lash_e, c, recipe, parent_ml, diluent_ml, table, label):
    """Add parent + diluent to a substock (make or top up), refilling the sources first."""
    ensure_volume(lash_e, c, recipe.parent, parent_ml, table)
    components = []
    if recipe.diluent == c["WATER_VIAL"]:  # water for substocks comes from the reservoir
        components.append({"reservoir": c["WATER_RESERVOIR"], "volume_mL": diluent_ml, "liquid": "water"})
    else:
        ensure_volume(lash_e, c, recipe.diluent, diluent_ml, table)
        components.append({"source_vial": recipe.diluent, "volume_mL": diluent_ml, "liquid": c["DYE_SOLVENT"]})
    liquid = c["SURFACTANT"] if recipe.component == "surfactant" else c["DYE_SOLVENT"]
    components.append({"source_vial": recipe.parent, "volume_mL": parent_ml, "liquid": liquid})
    lash_e.logger.info(f"{label} {recipe.name}: {parent_ml:.3f} mL {recipe.parent} + "
                       f"{diluent_ml:.3f} mL {recipe.diluent}")
    lash_e.nr_robot.create_substock(recipe.name, components, vortex_time=c["VORTEX_SECONDS"])


def ensure_volume(lash_e, c, vial, need_ml, table):
    """Make sure vial can give need_ml without going below its refill level,
    refilling it the way its refill_table row says."""
    rule = _rule(c, table, vial)
    fill, below = rule["fill_ml"], rule["below_ml"]
    current = _volume(lash_e, vial)
    if current - need_ml >= below - 1e-9:
        return
    if fill is None:
        raise RuntimeError(f"{vial}: needs {need_ml:.3f} mL above {below:g} mL, has {current:.3f} mL "
                           "and is never refilled")
    if fill - need_ml < below - 1e-9:
        raise RuntimeError(f"{vial}: one draw of {need_ml:.3f} mL is more than a fill can give")
    if rule["how"] == "recipe":
        r = rule["recipe"]
        parent_ml, diluent_ml = top_up(fill, r.parent_ml / r.volume_ml, current,
                                       c["SUBSTOCK_MIN_TRANSFER_UL"] / 1000, _VIAL_CAPACITY_ML)
        _add_to_substock(lash_e, c, r, parent_ml, diluent_ml, table, "Topping up")
    elif rule["how"] == "reservoir":
        lash_e.nr_robot.refill_vial(vial, fill, reservoir=c["WATER_RESERVOIR"], liquid="water")
    else:
        # Refill vials in order: empty the first (down to its reserve) before the next.
        liquid = c["SURFACTANT"] if vial == c["SURFACTANT_STOCK_VIAL"] else c["DYE_SOLVENT"]
        for source in rule["how"]:
            current = _volume(lash_e, vial)
            add = min(fill - current, _volume(lash_e, source) - c["STOCK_RESERVE_ML"])
            if add > 0:
                lash_e.logger.info(f"Refilling {vial} from {source}: +{add:.3f} mL")
                lash_e.nr_robot.refill_vial(vial, current + add, source_vial=source, liquid=liquid)
            if _volume(lash_e, vial) >= fill - 1e-9:
                break
    after = _volume(lash_e, vial)
    if after - need_ml < below - 1e-9:
        raise RuntimeError(f"{vial}: still too little after refilling ({after:.3f} mL "
                           f"for a {need_ml:.3f} mL draw above {below:g} mL)")


def make_substocks(lash_e, c, recipes, table):
    for r in recipes.itertuples():
        _add_to_substock(lash_e, c, r, r.parent_ml, r.diluent_ml, table, "Making")


def component_volumes(c, wells, component):
    """{vial: Series of mL by well_index} for one component."""
    if component == "water":
        return {c["WATER_VIAL"]: wells.set_index("well_index").water_ul / 1000}
    if component == "solvent":
        return {c["DYE_SOLVENT_VIAL"]: wells.set_index("well_index").solvent_ul / 1000}
    out = {}
    for source, group in wells.dropna(subset=[f"{component}_source"]).groupby(f"{component}_source"):
        out[source] = group.set_index("well_index")[f"{component}_ul"] / 1000
    return out


def batch_pauses(c, batch):
    """{component: minutes} to wait after that component in this batch (DISPENSE_PAUSES)."""
    return {p["after"]: float(p["minutes"]) for p in c["DISPENSE_PAUSES"] if p["batch"] == batch}


def pause(lash_e, batch, component, minutes):
    """Wait `minutes` from now (logged for duration estimates; not waited in simulation)."""
    from workflows.helpers._planning_markers import mark_wait
    lash_e.logger.info(f"Batch {batch}: waiting {minutes:g} min after {component} before the next component")
    mark_wait(lash_e.logger, f"batch{batch}_after_{component}", minutes * 60)
    if not lash_e.simulate:
        time.sleep(minutes * 60)


def dispense_batch(lash_e, c, wells, table, batch):
    """Dispense each component in DISPENSE_ORDER. Each source vial's wells go in
    chunks that one fill can give; the vial is refilled before a chunk if needed.
    After a component listed in DISPENSE_PAUSES for this batch, wait before the
    next one (timed from the last well, so every well waits at least that long)."""
    liquids = {c["WATER_VIAL"]: "water", c["DYE_SOLVENT_VIAL"]: c["DYE_SOLVENT"]}
    pauses = batch_pauses(c, batch)
    for component in c["DISPENSE_ORDER"]:
        dispensed = False
        for vial, volumes in component_volumes(c, wells, component).items():
            volumes = volumes[volumes > 0]
            if volumes.empty:
                continue
            dispensed = True
            liquid = liquids.get(vial, c["SURFACTANT"] if component == "surfactant" else c["DYE_SOLVENT"])
            rule = _rule(c, table, vial)
            chunks = ([volumes] if rule["fill_ml"] is None
                      else fill_chunks(volumes, rule["fill_ml"] - rule["below_ml"]))
            for chunk in chunks:
                ensure_volume(lash_e, c, vial, float(chunk.sum()), table)
                lash_e.nr_robot.dispense_from_vials_into_wellplate(
                    pd.DataFrame({vial: chunk}), liquid={vial: liquid}, well_plate_type=c["WELLPLATE_TYPE"])
        if component in pauses and dispensed:
            pause(lash_e, batch, component, pauses[component])


def fill_chunks(volumes, max_ml):
    """Split wells into groups whose total fits in one vial fill."""
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


def measure_batch(lash_e, c, wells, batch, reads, rng, output_dir, earlier=None):
    """Read the batch on its schedule; return long results (one row per well x timepoint).

    results.csv (earlier batches + this batch so far) is rewritten after every read,
    so a run that stops partway keeps every read it made."""
    from workflows.helpers._planning_markers import mark_clock_start, mark_wait_until
    results = []
    t0 = time.monotonic()
    clock = f"batch{batch}"
    mark_clock_start(lash_e.logger, clock)
    for minutes in c["BATCH_SCHEDULES_MIN"][batch - 1]:
        mark_wait_until(lash_e.logger, clock, minutes * 60)
        wait = t0 + minutes * 60 - time.monotonic()
        if wait > 0 and not lash_e.simulate:
            time.sleep(wait)
        raw = lash_e.measure_wellplate([path for _, path in reads], wells.well_index.tolist(),
                                       plate_type=c["WELLPLATE_TYPE"])
        merged = wells.copy()
        for label, path in reads:
            if raw is None:
                if not lash_e.simulate:
                    raise RuntimeError(f"No plate data for batch {batch}, t={minutes} min")
                part = simulate_read(wells, label, c, rng)
            else:
                stem = Path(path).stem
                part = raw[[col for col in raw.columns if col[0].endswith(stem)]]
            merged = merged.merge(raw_to_columns(part, label), on="well_position", how="left",
                                  validate="one_to_one")
        merged["timepoint_min"] = minutes
        merged["elapsed_actual_min"] = (time.monotonic() - t0) / 60
        merged["read_at"] = datetime.now().isoformat(timespec="seconds")
        results.append(merged)
        pd.concat(list(earlier or []) + results, ignore_index=True).to_csv(output_dir / "results.csv", index=False)
        if raw is not None:
            raw.to_csv(output_dir / f"raw_batch{batch}_t{minutes}.csv")
        lash_e.logger.info(f"Batch {batch}: read at t={minutes} min")
    return pd.concat(results, ignore_index=True)


def _send_workflow_slack(lash_e, message):
    if lash_e.simulate:
        return
    try:
        import slack_agent
        if not slack_agent.safe_send_slack_message(message):
            lash_e.logger.warning("Slack notification was not delivered (non-fatal)")
    except Exception as error:
        lash_e.logger.warning(f"Slack notification failed (non-fatal): {error}")


def execute(config=None, show_gui=True):
    """Plan, make substocks, dispense and read each batch on one plate."""
    from master_usdl_coordinator import Lash_E
    lash_e = Lash_E(workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
                    config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return None
    logger, c = lash_e.logger, lash_e.workflow_config
    try:
        lash_e.nr_robot.home_robot_components()
        if c["SIMULATE"] != lash_e.simulate:
            raise ValueError("Workflow and controller simulation modes do not match.")
        inventory = lash_e.nr_robot.VIAL_DF.set_index("vial_name")
        record_path = substock_record_path(lash_e.nr_robot.VIAL_FILE)
        record = load_substock_record(record_path)
        reused, problems = reusable_substocks(c, inventory, record, datetime.now())
        if problems:
            raise ValueError("Empty these substock vials (and set them to 0 mL in "
                             f"{c['INPUT_VIAL_STATUS_FILE']}) or fix {record_path.name}: " + "; ".join(problems))
        for s in reused:
            logger.info(f"Reusing {s['name']}: {s['conc']:.4g} {_material(c, s['component'])[3]}, "
                        f"{inventory.loc[s['name'], 'vial_volume']:.2f} mL")
        plan = build_plan(c, reused)
        reads = read_protocols(c)
        if not lash_e.simulate:
            for _, path in reads:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            if not c["ABSORBANCE_PROTOCOL_FILE"] and (plan["wells"].matrix == "solvent").any():
                raise ValueError("Solvent (stock-check) wells need ABSORBANCE_PROTOCOL_FILE")
        if not c["ABSORBANCE_PROTOCOL_FILE"]:
            logger.warning("ABSORBANCE_PROTOCOL_FILE not set: absorbance reads skipped")
        check_vials(c, plan, inventory)
        try:
            lash_e.nr_track.inventory.plate_at("source_stack", c["WELLPLATE_TYPE"])
        except ValueError:
            raise ValueError(f"Load a {c['WELLPLATE_TYPE']} in the source stack before starting") from None


        mode = "SIMULATION" if lash_e.simulate else "LIVE"
        output_dir = (Path(__file__).resolve().parents[1] / "output" / _WORKFLOW_NAME /
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{mode}_wells-{len(plan['wells'])}")
        output_dir.mkdir(parents=True, exist_ok=False)
        lash_e.save_run_info(output_dir)
        plan["wells"].to_csv(output_dir / "planned_wells.csv", index=False)
        plan["recipes"].to_csv(output_dir / "planned_substocks.csv", index=False)
        reused_frame = reused_recipes(reused)
        reused_frame.to_csv(output_dir / "reused_substocks.csv", index=False)
        refill_table_frame(refill_table(c, pd.concat([reused_frame, plan["recipes"]], ignore_index=True))
                           ).to_csv(output_dir / "refill_table.csv", index=False)
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started | {c['SURFACTANT']} + {c['DYE']} | "
                                     f"{len(plan['wells'])} wells | {len(plan['recipes'])} new substocks, "
                                     f"{len(reused)} reused")

        # Re-plan each batch as it runs, so a batch can use earlier results (extra_block_targets).
        rng = np.random.default_rng(c["RANDOMIZATION_SEED"]) if c["RANDOMIZED_ORDER"] else None
        read_rng = np.random.default_rng(0)
        sources, used, results = starting_sources(c, reused), set(), []
        made = [reused_frame] if not reused_frame.empty else []
        from scheduler.simulation_state import can_save
        save_record = can_save(lash_e.nr_robot, lash_e.nr_robot.VIAL_FILE)
        lash_e.grab_new_wellplate()
        for batch in range(1, len(c["BATCH_SCHEDULES_MIN"]) + 1):
            so_far = pd.concat(results, ignore_index=True) if results else None
            wells, planned, recipes = plan_batch(c, batch, sources, used, rng, so_far)
            if wells.empty:
                continue
            logger.info(f"Batch {batch}: {len(wells)} wells, {len(recipes)} new substocks")
            if not recipes.empty:
                made.append(recipes)
            all_recipes = (pd.concat(made, ignore_index=True) if made  # every substock so far, for top-ups
                           else recipes)
            table = refill_table(c, all_recipes)
            make_substocks(lash_e, c, recipes, table)
            if not recipes.empty:
                # Record what each new substock holds, so later runs can reuse it.
                record = update_substock_record(record, record_rows(c, recipes, datetime.now()))
                if save_record:
                    save_substock_record(record, record_path)
            dispense_batch(lash_e, c, wells, table, batch)
            results.append(measure_batch(lash_e, c, wells, batch, reads, read_rng, output_dir, earlier=results))
            used |= set(wells.well_index)
            sources = after_batch(planned)
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME}: batch {batch} read ({len(wells)} wells)")
        lash_e.discard_used_wellplate()
        logger.info(f"Workflow complete; results in {output_dir}")
        return output_dir
    except (Exception, KeyboardInterrupt) as e:
        logger.exception(f"Workflow failed with error: {e}")
        from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
        cleanup_after_keyboard_interrupt(lash_e)
        if isinstance(e, KeyboardInterrupt):
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted by operator; cleanup attempted, review experiment log")
        raise


if __name__ == "__main__":
    execute()
