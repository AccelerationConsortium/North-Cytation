"""
Single-surfactant CMC plate with one dye (first run: SDS + Nile Red), one 96-well plate.

Plan: docs/plans/dye_surfactant_grid_plan.md (sections 4. Workflow and First run).

Steps:
1. Targets: what each well should contain (surfactant mM, dye uM, matrix), built
   from config blocks. Blocks: "cmc" (surfactant series at one dye level, in water)
   and "stock_check" (dye stock diluted in pure solvent, read by absorbance).
   extra_block_targets() is the hook for a further block (e.g. capacity); empty now.
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

build_plan(config) runs steps 1-2 for every configured batch without hardware.

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

from workflows.helpers.well_plan import after_batch, plan_wells, top_up

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/surfactant_dye_cmc_vials.csv"
WELLPLATE_TYPE = "96 WELL PLATE"

# Liquids and stocks. Stock concentrations will come from status/stocks.csv once
# it exists; until then they are set here.
SURFACTANT = "SDS"                  # liquids.yaml name; also the substock name prefix
SURFACTANT_STOCK_VIAL = "SDS_stock"
SURFACTANT_REFILL_VIAL = "SDS_refill"  # same stock in a large vial; tops up SURFACTANT_STOCK_VIAL
SURFACTANT_STOCK_MM = 50.0
SURFACTANT_CMC_MM = 8.6
DYE = "nile-red"                    # key in surfactant_grid_ailsa._DYE_PROTOCOLS
DYE_STOCK_VIAL = "nilered_stock"
DYE_STOCK_UM = 500.0                # assumed until measured
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
DYE_VOLUME_UL = 5.0                 # fixed dye-solution volume in every water well
WELL_MIN_UL = 20.0                  # smallest variable-volume transfer into a well
WELL_MAX_UL = 195.0                 # largest (small tip 200 uL less its 5 uL margin)
SURFACTANT_PREFERRED_MIN_UL = 75.0  # surfactant wells aim for at least this (more substocks if higher)
SUBSTOCK_MIN_TRANSFER_UL = 200.0    # smallest parent transfer when making a substock
SURFACTANT_SUBSTOCK_ML = 6.0        # each SDS substock is made (and topped up) to this...
SURFACTANT_SUBSTOCK_REFILL_ML = 4.0 # ...and topped up in its recipe's ratio below this
DYE_SUBSTOCK_ML = 4.0
DYE_SUBSTOCK_REFILL_ML = 2.0
STOCK_RESERVE_ML = 0.5              # left in the dye stock and the refill vials

# CMC block: [multiple of CMC, replicates], all at CMC_DYE_UM in water.
CMC_LEVELS_X_CMC = (
    [[0.0, 3], [0.3, 3], [0.6, 3]]
    + [[round(0.75 + i * 0.6 / 11, 4), 3] for i in range(12)]
    + [[1.5, 3], [2.0, 3], [2.5, 3], [3.5, 3], [5.0, 3]]
)
CMC_DYE_UM = 1.0
CMC_BATCH = 1

# Stock check: dye stock diluted in pure solvent; fractions of DYE_STOCK_UM in the well.
STOCK_CHECK_FRACTIONS = [0.01, 0.02, 0.04, 0.06, 0.1, 0.15]  # 5-75 uM at 500 uM stock
STOCK_CHECK_REPLICATES = 1
STOCK_CHECK_BLANKS = 2
STOCK_CHECK_BATCH = 1

# Reads: one schedule (minutes after that batch is dispensed) per batch, in batch order.
BATCH_SCHEDULES_MIN = [[0, 30, 60, 120]]
FLUORESCENCE_PROTOCOL_FILE = None   # None: the DYE's protocol in _DYE_PROTOCOLS
ABSORBANCE_PROTOCOL_FILE = None     # not created yet; reads are skipped while None

DISPENSE_ORDER = ["water", "solvent", "surfactant", "dye"]
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
    "TOTAL_VOLUME_UL", "DYE_VOLUME_UL", "WELL_MIN_UL", "WELL_MAX_UL", "SURFACTANT_PREFERRED_MIN_UL",
    "SUBSTOCK_MIN_TRANSFER_UL",
    "SURFACTANT_SUBSTOCK_ML", "SURFACTANT_SUBSTOCK_REFILL_ML", "DYE_SUBSTOCK_ML",
    "DYE_SUBSTOCK_REFILL_ML", "STOCK_RESERVE_ML",
    "CMC_LEVELS_X_CMC", "CMC_DYE_UM", "CMC_BATCH",
    "STOCK_CHECK_FRACTIONS", "STOCK_CHECK_REPLICATES", "STOCK_CHECK_BLANKS", "STOCK_CHECK_BATCH",
    "BATCH_SCHEDULES_MIN", "FLUORESCENCE_PROTOCOL_FILE", "ABSORBANCE_PROTOCOL_FILE",
    "DISPENSE_ORDER", "RANDOMIZED_ORDER", "RANDOMIZATION_SEED", "VORTEX_SECONDS",
]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV (repository-relative) for this run.",
                               "unit": None, "affects": ["vials", "liquids"], "shares": "vial_layout"},
    "WELLPLATE_TYPE": {"description": "Plate type; only 96 WELL PLATE is set up.", "unit": None,
                       "choices": ["96 WELL PLATE"], "affects": ["plates"], "shares": "plate_type"},
    "SURFACTANT": {"description": "Surfactant liquid name (liquids.yaml) and substock name prefix.",
                   "unit": None, "affects": ["liquids"], "ask": True},
    "SURFACTANT_STOCK_VIAL": {"description": "Vial holding the surfactant stock.", "unit": None,
                              "affects": ["vials"]},
    "SURFACTANT_STOCK_MM": {"description": "Surfactant stock concentration.", "unit": "mM",
                            "affects": ["vials", "liquids"],
                            "effect": "weaker stock -> larger well volumes; error if the top level needs more than WELL_MAX_UL"},
    "SURFACTANT_REFILL_VIAL": {"description": "Large vial of the same surfactant stock; tops up SURFACTANT_STOCK_VIAL.",
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
    "TOTAL_VOLUME_UL": {"description": "Total volume per well, including the dye.", "unit": "uL",
                        "affects": ["liquids"]},
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
    "CMC_LEVELS_X_CMC": {"description": "CMC block: list of [multiple of CMC, replicates].",
                         "unit": None, "affects": ["liquids", "tips", "duration"],
                         "effect": "wells = sum of replicates"},
    "CMC_DYE_UM": {"description": "Dye concentration in every CMC-block well.", "unit": "uM",
                   "affects": ["liquids"]},
    "CMC_BATCH": {"description": "Batch the CMC block runs in (1 = first).", "unit": None,
                  "affects": ["duration"]},
    "STOCK_CHECK_FRACTIONS": {"description": "Stock check: dye levels as fractions of DYE_STOCK_UM, in pure solvent.",
                              "unit": None, "affects": ["liquids", "tips"],
                              "effect": "wells = fractions x replicates + blanks"},
    "STOCK_CHECK_REPLICATES": {"description": "Replicates per stock-check level.", "unit": None,
                               "affects": ["liquids", "tips"]},
    "STOCK_CHECK_BLANKS": {"description": "Pure-solvent wells in the stock check (0 to skip the block: also set FRACTIONS []).",
                           "unit": None, "affects": ["liquids"]},
    "STOCK_CHECK_BATCH": {"description": "Batch the stock check runs in.", "unit": None, "affects": ["duration"]},
    "BATCH_SCHEDULES_MIN": {"description": "Read times per batch (minutes after it is dispensed), one list per batch.",
                            "unit": "min", "affects": ["duration"],
                            "effect": "each batch waits until its last read before the next batch"},
    "FLUORESCENCE_PROTOCOL_FILE": {"description": "Cytation fluorescence protocol; null = the DYE's default.",
                                   "unit": None, "affects": ["duration"]},
    "ABSORBANCE_PROTOCOL_FILE": {"description": "Cytation absorbance protocol (dye peak + 750 nm); null skips absorbance.",
                                 "unit": None, "affects": ["duration"]},
    "DISPENSE_ORDER": {"description": "Order of components: water, solvent, surfactant, dye.", "unit": None,
                       "affects": []},
    "RANDOMIZED_ORDER": {"description": "Place each batch's wells at random free positions.", "unit": None,
                         "affects": []},
    "RANDOMIZATION_SEED": {"description": "Seed for well placement.", "unit": None, "affects": []},
    "VORTEX_SECONDS": {"description": "Vortex time after making each substock.", "unit": "s",
                       "affects": ["duration"]},
}

# Substocks this workflow makes: <SURFACTANT>_sub_<n>, dye_sub_<n>.
_PRODUCT_VIALS = ["*_sub_[0-9]*"]

_PLATE_WELLS, _PLATE_COLUMNS = 96, 12
# The small tip warns below ~2 mL in a vial, so the water vial is refilled above this.
_WATER_VIAL_RESERVE_ML = 2.0
_VIAL_CAPACITY_ML = 8.0  # substock vials (8 mL)
_COMPONENTS = ("water", "solvent", "surfactant", "dye")


# ============================================================================
# PLANNING (no hardware)
# ============================================================================

def well_position(index):
    return f"{chr(65 + index // _PLATE_COLUMNS)}{index % _PLATE_COLUMNS + 1}"


def planning_rules(c):
    """Config -> rules for well_plan.plan_wells."""
    return dict(
        total_ul=c["TOTAL_VOLUME_UL"], dye_ul=c["DYE_VOLUME_UL"],
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


def validate_config(c):
    if c["WELLPLATE_TYPE"] != "96 WELL PLATE":
        raise ValueError("Only 96 WELL PLATE is set up")
    for key in ("SURFACTANT_STOCK_MM", "SURFACTANT_CMC_MM", "DYE_STOCK_UM", "TOTAL_VOLUME_UL",
                "DYE_VOLUME_UL", "WELL_MIN_UL", "WELL_MAX_UL", "SURFACTANT_PREFERRED_MIN_UL",
                "SUBSTOCK_MIN_TRANSFER_UL",
                "SURFACTANT_SUBSTOCK_ML", "DYE_SUBSTOCK_ML", "WATER_VIAL_FILL_ML",
                "SURFACTANT_SUBSTOCK_REFILL_ML", "DYE_SUBSTOCK_REFILL_ML",
                "STOCK_REFILL_TO_ML", "STOCK_REFILL_BELOW_ML"):
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
    for level, replicates in c["CMC_LEVELS_X_CMC"]:
        if level < 0 or type(replicates) is not int or replicates < 1:
            raise ValueError(f"CMC_LEVELS_X_CMC entries are [multiple >= 0, replicates >= 1]: {[level, replicates]}")
    if any(not 0 < f <= 1 for f in c["STOCK_CHECK_FRACTIONS"]):
        raise ValueError("STOCK_CHECK_FRACTIONS must be in (0, 1]")
    schedules = c["BATCH_SCHEDULES_MIN"]
    for schedule in schedules:
        if not schedule or schedule[0] != 0 or list(schedule) != sorted(set(schedule)):
            raise ValueError("Each BATCH_SCHEDULES_MIN list must start at 0 and increase")
    for key in ("CMC_BATCH", "STOCK_CHECK_BATCH"):
        if type(c[key]) is not int or not 1 <= c[key] <= len(schedules):
            raise ValueError(f"{key} must be a batch with a schedule (1..{len(schedules)})")
    if sorted(c["DISPENSE_ORDER"]) != sorted(_COMPONENTS):
        raise ValueError(f"DISPENSE_ORDER must list {', '.join(_COMPONENTS)} once each")


def configured_targets(c):
    """Target rows for every configured block (batch, block, replicate, matrix, concentrations)."""
    rows = []
    for level, replicates in c["CMC_LEVELS_X_CMC"]:
        for replicate in range(1, replicates + 1):
            rows.append(dict(batch=c["CMC_BATCH"], block="cmc", replicate=replicate, matrix="water",
                             level=level, surfactant_mM=level * c["SURFACTANT_CMC_MM"],
                             dye_uM=c["CMC_DYE_UM"]))
    for fraction in [0.0] * c["STOCK_CHECK_BLANKS"] + list(c["STOCK_CHECK_FRACTIONS"]) * c["STOCK_CHECK_REPLICATES"]:
        rows.append(dict(batch=c["STOCK_CHECK_BATCH"], block="stock_check", replicate=None,
                         matrix="solvent", level=fraction, surfactant_mM=0.0,
                         dye_uM=fraction * c["DYE_STOCK_UM"]))
    targets = pd.DataFrame(rows)
    if not targets.empty:
        targets["replicate"] = targets.groupby(["block", "level"]).cumcount() + 1
    return targets


def extra_block_targets(c, batch, results):
    """Hook for a further block (e.g. capacity), planned per batch, optionally from
    earlier results. Return rows like configured_targets(); none yet."""
    return pd.DataFrame()


def assign_wells(targets, used, rng):
    """Give each target a free well: random free wells if rng, else the next free ones."""
    free = [i for i in range(_PLATE_WELLS) if i not in used]
    if len(targets) > len(free):
        raise ValueError(f"{len(targets)} wells needed, {len(free)} free on the plate")
    chosen = list(rng.permutation(free)[:len(targets)]) if rng is not None else free[:len(targets)]
    targets = targets.reset_index(drop=True).copy()
    targets["well_index"] = [int(i) for i in chosen]
    targets["well_position"] = targets.well_index.map(well_position)
    return targets


def plan_batch(c, batch, sources, used_wells, rng, results=None):
    """Plan one batch: (wells, sources after it, recipes to make first)."""
    targets = pd.concat([t for t in (configured_targets(c), extra_block_targets(c, batch, results))
                         if not t.empty], ignore_index=True)
    targets = targets[targets.batch == batch]
    if targets.empty:
        return targets, sources, pd.DataFrame()
    targets = assign_wells(targets, used_wells, rng)
    wells, planned, recipes = plan_wells(targets, planning_rules(c), existing=sources)
    recipes["batch"] = batch
    return wells, planned, recipes


def build_plan(config):
    """Plan every configured batch without hardware.

    Returns dict(wells=all wells, recipes=all substocks with the batch that makes
    them, sources=per-batch source lists with demand_ml).
    """
    c = config
    validate_config(c)
    rng = np.random.default_rng(c["RANDOMIZATION_SEED"]) if c["RANDOMIZED_ORDER"] else None
    sources, used = None, set()
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
    """Synthetic plate data in Cytation layout (index A1.., columns (protocol, channel))."""
    cmc, dye = c["SURFACTANT_CMC_MM"], wells.dye_uM.to_numpy()
    micellar = np.clip(wells.surfactant_mM.to_numpy() - cmc, 0, None)
    bound = np.where(wells.matrix == "solvent", 1.0, 2.0 * micellar / (1 + 2.0 * micellar))
    noise = lambda n: 1 + 0.03 * rng.standard_normal(n)
    if label == "fluorescence":
        data = {"550_648": (50 + dye * (100 + 2000 * bound)) * noise(len(wells))}
    else:
        # ~3.4e4 /M/cm, ~0.56 cm path for 200 uL in a 96-well plate
        data = {"579": (0.04 + dye * 0.034 * 0.56 * noise(len(wells))),
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

def refillable(c):
    """{working vial: refill vial} for vials topped up from a large refill vial."""
    return {c["SURFACTANT_STOCK_VIAL"]: c["SURFACTANT_REFILL_VIAL"],
            c["DYE_SOLVENT_VIAL"]: c["SOLVENT_REFILL_VIAL"]}


def refill_table(c, recipes):
    """How each vial is refilled: {vial: dict(fill_ml, below_ml, how, recipe)}.
    how: "recipe" (top up in the recipe's ratio), "reservoir", or a refill vial name.
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
                                             if r["how"] == "recipe" else r["how"]))
                         for vial, r in table.items()])


def check_vials(c, plan, inventory):
    """Before anything moves: vials exist, substock vials are empty, and the vials
    that are never refilled (dye stock, refill vials) hold enough for the whole run."""
    demand = total_demand(plan)
    refills = refillable(c)
    missing = sorted((set(demand) | set(refills.values())) - set(inventory.index))
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
    for vial, refill in refills.items():
        # The refill vial covers what the working vial can't, plus anything drawn from it directly.
        from_working = inventory.loc[vial, "vial_volume"] - c["STOCK_REFILL_BELOW_ML"]
        need = max(0.0, demand.get(vial, 0.0) - from_working) + demand.get(refill, 0.0)
        have = inventory.loc[refill, "vial_volume"] - reserve
        if need > have + 1e-9:
            raise ValueError(f"{refill}: the run needs {need:.2f} mL from it, it has {have:.2f} mL "
                             f"above its {reserve:g} mL reserve")


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
        source = rule["how"]
        add = min(fill - current, _volume(lash_e, source) - c["STOCK_RESERVE_ML"])
        if add > 0:
            liquid = c["SURFACTANT"] if vial == c["SURFACTANT_STOCK_VIAL"] else c["DYE_SOLVENT"]
            lash_e.logger.info(f"Refilling {vial} from {source}: +{add:.3f} mL")
            lash_e.nr_robot.refill_vial(vial, current + add, source_vial=source, liquid=liquid)
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


def dispense_batch(lash_e, c, wells, table):
    """Dispense each component in DISPENSE_ORDER. Each source vial's wells go in
    chunks that one fill can give; the vial is refilled before a chunk if needed."""
    liquids = {c["WATER_VIAL"]: "water", c["DYE_SOLVENT_VIAL"]: c["DYE_SOLVENT"]}
    for component in c["DISPENSE_ORDER"]:
        for vial, volumes in component_volumes(c, wells, component).items():
            volumes = volumes[volumes > 0]
            if volumes.empty:
                continue
            liquid = liquids.get(vial, c["SURFACTANT"] if component == "surfactant" else c["DYE_SOLVENT"])
            rule = _rule(c, table, vial)
            chunks = ([volumes] if rule["fill_ml"] is None
                      else fill_chunks(volumes, rule["fill_ml"] - rule["below_ml"]))
            for chunk in chunks:
                ensure_volume(lash_e, c, vial, float(chunk.sum()), table)
                lash_e.nr_robot.dispense_from_vials_into_wellplate(
                    pd.DataFrame({vial: chunk}), liquid={vial: liquid}, well_plate_type=c["WELLPLATE_TYPE"])


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


def measure_batch(lash_e, c, wells, batch, reads, rng, output_dir):
    """Read the batch on its schedule; return long results (one row per well x timepoint)."""
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
        results.append(merged)
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
        plan = build_plan(c)
        reads = read_protocols(c)
        if not lash_e.simulate:
            for _, path in reads:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            if not c["ABSORBANCE_PROTOCOL_FILE"] and (plan["wells"].block == "stock_check").any():
                raise ValueError("The stock check needs ABSORBANCE_PROTOCOL_FILE")
        if not c["ABSORBANCE_PROTOCOL_FILE"]:
            logger.warning("ABSORBANCE_PROTOCOL_FILE not set: absorbance reads skipped")
        inventory = lash_e.nr_robot.VIAL_DF.set_index("vial_name")
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
        refill_table_frame(refill_table(c, plan["recipes"])).to_csv(output_dir / "refill_table.csv", index=False)
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started | {c['SURFACTANT']} + {c['DYE']} | "
                                     f"{len(plan['wells'])} wells | {len(plan['recipes'])} substocks")

        # Re-plan each batch as it runs, so a batch can use earlier results (extra_block_targets).
        rng = np.random.default_rng(c["RANDOMIZATION_SEED"]) if c["RANDOMIZED_ORDER"] else None
        read_rng = np.random.default_rng(0)
        sources, used, results, made = None, set(), [], []
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
            dispense_batch(lash_e, c, wells, table)
            results.append(measure_batch(lash_e, c, wells, batch, reads, read_rng, output_dir))
            pd.concat(results, ignore_index=True).to_csv(output_dir / "results.csv", index=False)
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
