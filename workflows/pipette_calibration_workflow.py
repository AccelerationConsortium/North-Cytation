"""Single-vial calibration followed by independent validation, without promotion."""
import sys
sys.path.append("../utoronto_demo")

import importlib
import math
from copy import deepcopy
from numbers import Real
from pathlib import Path
from uuid import uuid4

import yaml
import pandas as pd

from master_usdl_coordinator import Lash_E
from pipetting_data.liquid_registry import get_liquid

_WORKFLOW_NAME = Path(__file__).stem
_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_NAME = "calibration_protocol_northrobot"

SIMULATE = True
# Internal development switches; excluded from GUI/YAML configuration.
# Hardware always saves files and uses the configured Bayesian optimizer.
_SAVE_FILES_ON_SIMULATE = False  # Legacy switch; simulated data is never saved.
_USE_BAYES_OPT_ON_SIMULATE = False
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
TARGET_VIAL = "acetone"
CALIBRATION_CONFIG_FILE = "sdl_pipette_calibration/experiment_config_workflow.yaml"
LIQUID = "acetone"
TIP_TYPE = "small_tip"
VOLUME_TARGETS_ML = [0.02, 0.01, 0.005]
RANDOM_SEED = 30
MAX_TOTAL_MEASUREMENTS = 300
MAX_MEASUREMENTS_FIRST_VOLUME = 200
MAX_REPLICATES_PER_TRIAL = 3
NUM_SCREENING_TRIALS = 8
TWO_POINT_CALIBRATION_REPLICATES = 3
MIN_GOOD_TRIALS = 5
VALIDATION_VOLUMES_ML = [0.02, 0.01, 0.005]
REPLICATES_PER_VOLUME = 5
ADJUST_VOLUME = True
CONTINUOUS_MONITORING = True
MAX_RETRIES_PER_MEASUREMENT = 0
QUALITY_STD_THRESHOLD_G = 0.1
OVERASPIRATE_VOL_MIN = 0.0
OVERASPIRATE_VOL_MAX = 0.025
OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET = 1.0
ASPIRATE_SPEED_FIXED = False
ASPIRATE_SPEED_FIXED_VALUE = 10
ASPIRATE_SPEED_MIN = 2
ASPIRATE_SPEED_MAX = 30
DISPENSE_SPEED_FIXED = False
DISPENSE_SPEED_FIXED_VALUE = 10
DISPENSE_SPEED_MIN = 2
DISPENSE_SPEED_MAX = 30
ASPIRATE_WAIT_TIME_FIXED = False
ASPIRATE_WAIT_TIME_FIXED_VALUE = 10.0
ASPIRATE_WAIT_TIME_MIN = 0.0
ASPIRATE_WAIT_TIME_MAX = 30.0
PRE_ASP_AIR_VOL_FIXED = False
PRE_ASP_AIR_VOL_FIXED_VALUE = 0.0
PRE_ASP_AIR_VOL_MIN = 0.0
PRE_ASP_AIR_VOL_MAX = 0.5
BLOWOUT_VOL_FIXED = False
BLOWOUT_VOL_FIXED_VALUE = 0.0
BLOWOUT_VOL_MIN = 0.0
BLOWOUT_VOL_MAX = 0.5
ASP_DISP_CYCLES_FIXED = True
ASP_DISP_CYCLES_FIXED_VALUE = 0
ASP_DISP_CYCLES_MIN = 0
ASP_DISP_CYCLES_MAX = 5
POST_ASP_AIR_VOL_FIXED = True
POST_ASP_AIR_VOL_FIXED_VALUE = 0.0
POST_ASP_AIR_VOL_MIN = 0.0
POST_ASP_AIR_VOL_MAX = 0.1
POST_RETRACT_WAIT_TIME_FIXED = True
POST_RETRACT_WAIT_TIME_FIXED_VALUE = 5.0
POST_RETRACT_WAIT_TIME_MIN = 0.0
POST_RETRACT_WAIT_TIME_MAX = 15.0
RETRACT_SPEED_FIXED = True
RETRACT_SPEED_FIXED_VALUE = 5
RETRACT_SPEED_MIN = 1
RETRACT_SPEED_MAX = 99
DISPENSE_WAIT_TIME_FIXED = True
DISPENSE_WAIT_TIME_FIXED_VALUE = 3.0
DISPENSE_WAIT_TIME_MIN = 0.0
DISPENSE_WAIT_TIME_MAX = 15.0

_HARDWARE_PARAMETERS = (
    "aspirate_speed", "dispense_speed", "aspirate_wait_time", "pre_asp_air_vol",
    "blowout_vol", "asp_disp_cycles", "post_asp_air_vol", "post_retract_wait_time",
    "retract_speed", "dispense_wait_time",
)
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "TARGET_VIAL", "CALIBRATION_CONFIG_FILE",
    "LIQUID", "TIP_TYPE", "VOLUME_TARGETS_ML", "RANDOM_SEED", "MAX_TOTAL_MEASUREMENTS",
    "MAX_MEASUREMENTS_FIRST_VOLUME", "MAX_REPLICATES_PER_TRIAL",
    "NUM_SCREENING_TRIALS", "TWO_POINT_CALIBRATION_REPLICATES", "MIN_GOOD_TRIALS",
    "VALIDATION_VOLUMES_ML", "REPLICATES_PER_VOLUME", "ADJUST_VOLUME",
    "CONTINUOUS_MONITORING", "MAX_RETRIES_PER_MEASUREMENT", "QUALITY_STD_THRESHOLD_G",
    "OVERASPIRATE_VOL_MIN", "OVERASPIRATE_VOL_MAX",
    "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET",
    "ASPIRATE_SPEED_FIXED", "ASPIRATE_SPEED_FIXED_VALUE", "ASPIRATE_SPEED_MIN", "ASPIRATE_SPEED_MAX",
    "DISPENSE_SPEED_FIXED", "DISPENSE_SPEED_FIXED_VALUE", "DISPENSE_SPEED_MIN", "DISPENSE_SPEED_MAX",
    "ASPIRATE_WAIT_TIME_FIXED", "ASPIRATE_WAIT_TIME_FIXED_VALUE", "ASPIRATE_WAIT_TIME_MIN", "ASPIRATE_WAIT_TIME_MAX",
    "PRE_ASP_AIR_VOL_FIXED", "PRE_ASP_AIR_VOL_FIXED_VALUE", "PRE_ASP_AIR_VOL_MIN", "PRE_ASP_AIR_VOL_MAX",
    "BLOWOUT_VOL_FIXED", "BLOWOUT_VOL_FIXED_VALUE", "BLOWOUT_VOL_MIN", "BLOWOUT_VOL_MAX",
    "ASP_DISP_CYCLES_FIXED", "ASP_DISP_CYCLES_FIXED_VALUE", "ASP_DISP_CYCLES_MIN", "ASP_DISP_CYCLES_MAX",
    "POST_ASP_AIR_VOL_FIXED", "POST_ASP_AIR_VOL_FIXED_VALUE", "POST_ASP_AIR_VOL_MIN", "POST_ASP_AIR_VOL_MAX",
    "POST_RETRACT_WAIT_TIME_FIXED", "POST_RETRACT_WAIT_TIME_FIXED_VALUE", "POST_RETRACT_WAIT_TIME_MIN", "POST_RETRACT_WAIT_TIME_MAX",
    "RETRACT_SPEED_FIXED", "RETRACT_SPEED_FIXED_VALUE", "RETRACT_SPEED_MIN", "RETRACT_SPEED_MAX",
    "DISPENSE_WAIT_TIME_FIXED", "DISPENSE_WAIT_TIME_FIXED_VALUE", "DISPENSE_WAIT_TIME_MIN", "DISPENSE_WAIT_TIME_MAX",
]  # Literal (read by planning tools with ast); hardware keys follow _HARDWARE_PARAMETERS.
# Planning annotations, one per _CONFIG_KEYS entry (plain literals read with ast;
# field meanings in workflows/workflow_template.py). Hardware-parameter dwell
# times are skipped in simulation, so planners add them per measurement.
_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.", "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV (repository-relative); must be the file the robot loaded and contain TARGET_VIAL.", "unit": None, "affects": ["vials"], "shares": "vial_layout"},
    "TARGET_VIAL": {"description": "Name of the single vial holding the liquid; it is both the pipetting source and the weighed destination on the clamp scale.", "unit": None, "affects": ["vials", "liquids"], "effect": "one vial; liquid is drawn from and dispensed back into it, so net use is mainly evaporation/tip losses"},
    "CALIBRATION_CONFIG_FILE": {"description": "Base calibration YAML (sdl_pipette_calibration); supplies optimizer objectives, tolerances, adaptive-replicate thresholds and other defaults not overridden here.", "unit": None, "affects": ["duration", "tips"], "effect": "its adaptive_measurement and first_volume_final_calibration sections change how many measurements run", "shares": "input_file"},
    "LIQUID": {"description": "Liquid to calibrate; must be in the calibration liquid registry and be what TARGET_VIAL holds.", "unit": None, "affects": ["liquids", "tips", "duration"], "effect": "refill_pipets liquids get a new tip after every hardware measurement (not reproduced in simulation); others use one tip per calibration/validation phase", "ask": True},
    "TIP_TYPE": {"description": "Pipette tip size for the whole run; every calibration and validation volume must fit it.", "unit": None, "choices": ["small_tip", "large_tip"], "affects": ["tips"]},
    "VOLUME_TARGETS_ML": {"description": "Volumes to calibrate; the first gets screening plus optimization, later ones reuse its results.", "unit": "mL", "affects": ["duration", "tips"], "effect": "first volume: screening + two-point + optimization within MAX_MEASUREMENTS_FIRST_VOLUME; each later volume: two-point + optimization sharing the remaining budget", "ask": True},
    "RANDOM_SEED": {"description": "Seed for reproducible screening and simulated measurements.", "unit": None, "affects": []},
    "MAX_TOTAL_MEASUREMENTS": {"description": "Measurement budget for calibration across all volumes (validation not included).", "unit": None, "affects": ["duration", "tips"], "effect": "caps calibration measurements; later volumes split (total - used) equally"},
    "MAX_MEASUREMENTS_FIRST_VOLUME": {"description": "Measurement budget for the first calibration volume.", "unit": None, "affects": ["duration", "tips"], "effect": "caps first-volume screening + two-point + optimization measurements"},
    "MAX_REPLICATES_PER_TRIAL": {"description": "Most replicate measurements one trial can take when its accuracy looks promising.", "unit": None, "affects": ["duration", "tips"], "effect": "trials within the adaptive deviation threshold get extra replicates up to this count"},
    "NUM_SCREENING_TRIALS": {"description": "Exploratory trials on the first volume before optimization.", "unit": None, "affects": ["duration", "tips"], "effect": "first volume only: this many trials of 1 to MAX_REPLICATES_PER_TRIAL measurements each"},
    "TWO_POINT_CALIBRATION_REPLICATES": {"description": "Replicates at each of the two overaspirate points used to fit a volume correction.", "unit": None, "affects": ["duration", "tips"], "effect": "2 x this many measurements per volume; the first volume may repeat it in a final calibration"},
    "MIN_GOOD_TRIALS": {"description": "Optimization stops early once this many trials meet the accuracy/precision tolerance.", "unit": None, "affects": ["duration", "tips"], "effect": "lower values end first-volume optimization sooner; otherwise it runs to the budget"},
    "VALIDATION_VOLUMES_ML": {"description": "Volumes checked in the independent validation after calibration (must fit the tip's safe range).", "unit": "mL", "affects": ["duration", "tips"], "effect": "validation measurements = len(VALIDATION_VOLUMES_ML) x REPLICATES_PER_VOLUME"},
    "REPLICATES_PER_VOLUME": {"description": "Validation replicates at each validation volume.", "unit": None, "affects": ["duration", "tips"], "effect": "validation measurements = len(VALIDATION_VOLUMES_ML) x REPLICATES_PER_VOLUME"},
    "ADJUST_VOLUME": {"description": "Weigh the vial before and after each hardware measurement to correct its tracked volume.", "unit": None, "affects": ["duration"], "effect": "adds two steady scale reads per hardware measurement; skipped in simulation"},
    "CONTINUOUS_MONITORING": {"description": "Record the scale throughout each dispense and use its stability to accept or retry the measurement.", "unit": None, "affects": ["duration"], "effect": "adds about 5 s of scale settling/baseline per hardware dispense; skipped in simulation"},
    "MAX_RETRIES_PER_MEASUREMENT": {"description": "How many times a hardware measurement is repeated when the scale reading is unstable.", "unit": None, "affects": ["duration", "tips"], "effect": "each retry repeats the aspirate/dispense and uses budget; hardware only (simulation never retries)"},
    "QUALITY_STD_THRESHOLD_G": {"description": "Scale standard deviation (grams) above which a monitored measurement counts as unstable.", "unit": None, "affects": ["duration"], "effect": "lower values trigger more retries when MAX_RETRIES_PER_MEASUREMENT > 0; hardware only"},
    "OVERASPIRATE_VOL_MIN": {"description": "Lower search bound for the extra volume aspirated to correct delivery.", "unit": "mL", "affects": []},
    "OVERASPIRATE_VOL_MAX": {"description": "Upper search bound for the extra volume aspirated to correct delivery.", "unit": "mL", "affects": []},
    "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET": {"description": "Cap on the overaspirate volume as a fraction of each target volume.", "unit": None, "affects": []},
    "ASPIRATE_SPEED_FIXED": {"description": "Hold the syringe-pump aspirate speed (1-40, higher = slower) at ASPIRATE_SPEED_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "ASPIRATE_SPEED_FIXED_VALUE": {"description": "Fixed value of the syringe-pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"], "effect": "higher values slow every aspirate"},
    "ASPIRATE_SPEED_MIN": {"description": "Lower optimizer bound for the syringe-pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "ASPIRATE_SPEED_MAX": {"description": "Upper optimizer bound for the syringe-pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_FIXED": {"description": "Hold the syringe-pump dispense speed (1-40, higher = slower) at DISPENSE_SPEED_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_FIXED_VALUE": {"description": "Fixed value of the syringe-pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"], "effect": "higher values slow every dispense"},
    "DISPENSE_SPEED_MIN": {"description": "Lower optimizer bound for the syringe-pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_MAX": {"description": "Upper optimizer bound for the syringe-pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_FIXED": {"description": "Hold the wait in the liquid after aspirating at ASPIRATE_WAIT_TIME_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_FIXED_VALUE": {"description": "Fixed value of the wait in the liquid after aspirating.", "unit": "s", "affects": ["duration"], "effect": "dwell per hardware aspirate/dispense; North_Safe skips it in simulation, so add it per measurement"},
    "ASPIRATE_WAIT_TIME_MIN": {"description": "Lower optimizer bound for the wait in the liquid after aspirating.", "unit": "s", "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_MAX": {"description": "Upper optimizer bound for the wait in the liquid after aspirating.", "unit": "s", "affects": ["duration"]},
    "PRE_ASP_AIR_VOL_FIXED": {"description": "Hold the air drawn before aspirating liquid at PRE_ASP_AIR_VOL_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": []},
    "PRE_ASP_AIR_VOL_FIXED_VALUE": {"description": "Fixed value of the air drawn before aspirating liquid.", "unit": "mL", "affects": []},
    "PRE_ASP_AIR_VOL_MIN": {"description": "Lower optimizer bound for the air drawn before aspirating liquid.", "unit": "mL", "affects": []},
    "PRE_ASP_AIR_VOL_MAX": {"description": "Upper optimizer bound for the air drawn before aspirating liquid.", "unit": "mL", "affects": []},
    "BLOWOUT_VOL_FIXED": {"description": "Hold the air blown out after dispensing at BLOWOUT_VOL_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "BLOWOUT_VOL_FIXED_VALUE": {"description": "Fixed value of the air blown out after dispensing.", "unit": "mL", "affects": ["duration"], "effect": "nonzero adds an air aspirate/dispense stroke to every dispense"},
    "BLOWOUT_VOL_MIN": {"description": "Lower optimizer bound for the air blown out after dispensing.", "unit": "mL", "affects": ["duration"]},
    "BLOWOUT_VOL_MAX": {"description": "Upper optimizer bound for the air blown out after dispensing.", "unit": "mL", "affects": ["duration"]},
    "ASP_DISP_CYCLES_FIXED": {"description": "Hold the number of aspirate/dispense mixing cycles before the measured aspirate at ASP_DISP_CYCLES_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "ASP_DISP_CYCLES_FIXED_VALUE": {"description": "Fixed value of the number of aspirate/dispense mixing cycles before the measured aspirate.", "unit": None, "affects": ["duration"], "effect": "each cycle adds one aspirate/dispense per measurement"},
    "ASP_DISP_CYCLES_MIN": {"description": "Lower optimizer bound for the number of aspirate/dispense mixing cycles before the measured aspirate.", "unit": None, "affects": ["duration"]},
    "ASP_DISP_CYCLES_MAX": {"description": "Upper optimizer bound for the number of aspirate/dispense mixing cycles before the measured aspirate.", "unit": None, "affects": ["duration"]},
    "POST_ASP_AIR_VOL_FIXED": {"description": "Hold the air gap drawn after aspirating liquid at POST_ASP_AIR_VOL_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": []},
    "POST_ASP_AIR_VOL_FIXED_VALUE": {"description": "Fixed value of the air gap drawn after aspirating liquid.", "unit": "mL", "affects": []},
    "POST_ASP_AIR_VOL_MIN": {"description": "Lower optimizer bound for the air gap drawn after aspirating liquid.", "unit": "mL", "affects": []},
    "POST_ASP_AIR_VOL_MAX": {"description": "Upper optimizer bound for the air gap drawn after aspirating liquid.", "unit": "mL", "affects": []},
    "POST_RETRACT_WAIT_TIME_FIXED": {"description": "Hold the wait after retracting the tip from the liquid (drip-off) at POST_RETRACT_WAIT_TIME_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "POST_RETRACT_WAIT_TIME_FIXED_VALUE": {"description": "Fixed value of the wait after retracting the tip from the liquid (drip-off).", "unit": "s", "affects": ["duration"], "effect": "dwell per hardware aspirate/dispense; North_Safe skips it in simulation, so add it per measurement"},
    "POST_RETRACT_WAIT_TIME_MIN": {"description": "Lower optimizer bound for the wait after retracting the tip from the liquid (drip-off).", "unit": "s", "affects": ["duration"]},
    "POST_RETRACT_WAIT_TIME_MAX": {"description": "Upper optimizer bound for the wait after retracting the tip from the liquid (drip-off).", "unit": "s", "affects": ["duration"]},
    "RETRACT_SPEED_FIXED": {"description": "Hold the tip retract speed out of the liquid (1-99) at RETRACT_SPEED_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "RETRACT_SPEED_FIXED_VALUE": {"description": "Fixed value of the tip retract speed out of the liquid (1-99).", "unit": None, "affects": ["duration"]},
    "RETRACT_SPEED_MIN": {"description": "Lower optimizer bound for the tip retract speed out of the liquid (1-99).", "unit": None, "affects": ["duration"]},
    "RETRACT_SPEED_MAX": {"description": "Upper optimizer bound for the tip retract speed out of the liquid (1-99).", "unit": None, "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_FIXED": {"description": "Hold the wait after dispensing before the tip moves at DISPENSE_WAIT_TIME_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_FIXED_VALUE": {"description": "Fixed value of the wait after dispensing before the tip moves.", "unit": "s", "affects": ["duration"], "effect": "dwell per hardware aspirate/dispense; North_Safe skips it in simulation, so add it per measurement"},
    "DISPENSE_WAIT_TIME_MIN": {"description": "Lower optimizer bound for the wait after dispensing before the tip moves.", "unit": "s", "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_MAX": {"description": "Upper optimizer bound for the wait after dispensing before the tip moves.", "unit": "s", "affects": ["duration"]},
}

# Solutions this workflow makes: none (it weighs dispenses of an existing liquid).
_PRODUCT_VIALS = []


def _number(value, label, minimum=0, integer=False, positive=False):
    """Validate a finite number; minimum=None permits signed corrections."""
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (minimum is not None and value < minimum)
            or (positive and value <= 0) or (integer and type(value) is not int)):
        bound = " greater than zero" if positive else (
            f" at least {minimum}" if minimum is not None else "")
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}"
                         f"{bound}")
    return value


def _volumes(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list")
    for value in values:
        _number(value, label, positive=True)
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must contain distinct volumes")


def validate_experiment(config, lash_e):
    """Check confirmed settings against the already-created controller."""
    for key in _CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing workflow setting: {key}")
    for key in ("SIMULATE", "ADJUST_VOLUME", "CONTINUOUS_MONITORING"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes do not match")
    vial_file = config["INPUT_VIAL_STATUS_FILE"]
    if not isinstance(vial_file, str) or not Path(vial_file).is_file():
        raise ValueError("INPUT_VIAL_STATUS_FILE must name an existing vial CSV")
    if lash_e.nr_robot.VIAL_FILE is None or Path(vial_file).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during review; restart with the confirmed file")
    if not isinstance(config["TARGET_VIAL"], str) or not config["TARGET_VIAL"].strip():
        raise ValueError("TARGET_VIAL must name a single vial")
    lash_e.nr_robot.normalize_vial_index(config["TARGET_VIAL"])
    for key in ("VOLUME_TARGETS_ML", "VALIDATION_VOLUMES_ML"):
        _volumes(config[key], key)
    tip_type = config["TIP_TYPE"]
    tips = lash_e.nr_robot.PIPET_TIPS
    if tip_type not in ("small_tip", "large_tip") or tip_type not in tips:
        raise ValueError("TIP_TYPE must be a configured small_tip or large_tip")
    capacity = float(tips[tip_type]["volume"])
    minimum = float(tips[tip_type].get("min_suggested_volume", 0))
    oav_max = _number(config["OVERASPIRATE_VOL_MAX"], "OVERASPIRATE_VOL_MAX", positive=True)
    fraction = _number(config["OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET"],
                       "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET", positive=True)
    post_max = (config["POST_ASP_AIR_VOL_FIXED_VALUE"] if config["POST_ASP_AIR_VOL_FIXED"]
                else config["POST_ASP_AIR_VOL_MAX"])
    _number(post_max, "maximum post-aspiration air volume")
    for key in ("VOLUME_TARGETS_ML", "VALIDATION_VOLUMES_ML"):
        for volume in config[key]:
            if volume < minimum or volume + min(oav_max, volume * fraction) + post_max >= capacity:
                raise ValueError(
                    f"{key} contains {volume * 1000:g} uL, outside the safe {tip_type} "
                    f"calibration range. Lower the maximum volume or run separate "
                    f"small-tip and large-tip calibrations."
                )
    for key in ("MAX_TOTAL_MEASUREMENTS", "MAX_MEASUREMENTS_FIRST_VOLUME",
                "MAX_REPLICATES_PER_TRIAL", "NUM_SCREENING_TRIALS",
                "TWO_POINT_CALIBRATION_REPLICATES", "MIN_GOOD_TRIALS", "REPLICATES_PER_VOLUME"):
        _number(config[key], key, integer=True, positive=True)
    for key in ("RANDOM_SEED", "MAX_RETRIES_PER_MEASUREMENT"):
        _number(config[key], key, integer=True)
    _number(config["QUALITY_STD_THRESHOLD_G"], "QUALITY_STD_THRESHOLD_G", positive=True)
    first = config["MAX_MEASUREMENTS_FIRST_VOLUME"]
    total = config["MAX_TOTAL_MEASUREMENTS"]
    required = config["NUM_SCREENING_TRIALS"] + 2 * config["TWO_POINT_CALIBRATION_REPLICATES"]
    if first < max(required, config["MAX_REPLICATES_PER_TRIAL"], config["MIN_GOOD_TRIALS"]):
        raise ValueError("First-volume budget cannot support screening and two-point calibration")
    subsequent = 2 * config["TWO_POINT_CALIBRATION_REPLICATES"] * (len(config["VOLUME_TARGETS_ML"]) - 1)
    if total < first + subsequent:
        raise ValueError("Total budget must cover first-volume budget and subsequent two-point measurements")


def _base_path(value, base_path):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_path.parent / path).resolve()


def _effective_config(config, base_path):
    """Copy the scientific source and apply only reviewed workflow overrides."""
    with base_path.open(encoding="utf-8") as stream:
        raw = deepcopy(yaml.safe_load(stream))
    experiment = raw["experiment"]
    if "overaspirate_vol" in experiment["fixed_parameters"]:
        raise ValueError("Base config must not fix overaspirate_vol")
    for key in ("LIQUID", "VOLUME_TARGETS_ML", "SIMULATE", "RANDOM_SEED",
                "MAX_TOTAL_MEASUREMENTS", "MAX_MEASUREMENTS_FIRST_VOLUME",
                "MAX_REPLICATES_PER_TRIAL", "NUM_SCREENING_TRIALS",
                "TWO_POINT_CALIBRATION_REPLICATES", "ADJUST_VOLUME", "CONTINUOUS_MONITORING",
                "MAX_RETRIES_PER_MEASUREMENT", "QUALITY_STD_THRESHOLD_G"):
        experiment[key.lower()] = deepcopy(config[key])
    experiment["protocol_override"] = _PROTOCOL_NAME
    experiment["north_tip_type"] = config["TIP_TYPE"]
    raw["optimization"]["stopping_criteria"]["min_good_trials"] = config["MIN_GOOD_TRIALS"]
    overaspirate = raw["calibration_parameters"]["overaspirate_vol"]
    lower = _number(config["OVERASPIRATE_VOL_MIN"], "OVERASPIRATE_VOL_MIN")
    upper = _number(config["OVERASPIRATE_VOL_MAX"], "OVERASPIRATE_VOL_MAX", positive=True)
    fraction = _number(config["OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET"],
                       "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET", positive=True)
    if lower >= upper or any(lower >= min(upper, volume * fraction)
                             for volume in config["VOLUME_TARGETS_ML"]):
        raise ValueError("Overaspirate bounds/cap leave no variable search range")
    overaspirate["bounds"] = [lower, upper]
    overaspirate["max_fraction_of_target"] = fraction
    hardware = raw["hardware_parameters"]
    for name in _HARDWARE_PARAMETERS:
        prefix = name.upper()
        fixed = config[prefix + "_FIXED"]
        if type(fixed) is not bool:
            raise ValueError(f"{prefix}_FIXED must be boolean")
        if name not in hardware:
            if name not in ("retract_speed", "asp_disp_cycles"):
                raise ValueError(f"Missing base parameter definition: {name}")
            hardware[name] = {"type": "integer", "round_to_nearest": 1,
                              "default": config[prefix + "_FIXED_VALUE"]}
        definition = hardware[name]
        if definition["type"] not in ("integer", "float"):
            raise ValueError(f"Unsupported parameter type for {name}")
        integer = definition["type"] == "integer"
        lower = _number(config[prefix + "_MIN"], prefix + "_MIN", integer=integer)
        upper = _number(config[prefix + "_MAX"], prefix + "_MAX", integer=integer)
        value = _number(config[prefix + "_FIXED_VALUE"], prefix + "_FIXED_VALUE", integer=integer)
        if lower >= upper or not lower <= value <= upper:
            raise ValueError(f"Invalid bounds/fixed value for {name}")
        if name in ("aspirate_speed", "dispense_speed") and not 1 <= lower < upper <= 40:
            raise ValueError(f"{name} bounds must be within North pump speed limits 1-40")
        if name == "retract_speed" and not 1 <= lower < upper <= 99:
            raise ValueError("retract_speed bounds must be within 1-99")
        definition["bounds"] = [lower, upper]
        if fixed:
            experiment["fixed_parameters"][name] = value
        else:
            experiment["fixed_parameters"].pop(name, None)
    for section, key in ((raw["optimization"]["llm_optimization"], "config_path"),
                         (raw["screening"], "llm_config_path"),
                         (raw["screening"]["external_data"], "data_path")):
        if section[key] is not None:
            section[key] = str(_base_path(section[key], base_path))
    raw["validation"]["volumes_ml"] = deepcopy(config["VALIDATION_VOLUMES_ML"])
    raw["validation"]["replicates_per_volume"] = config["REPLICATES_PER_VOLUME"]
    if raw["output"]["export_optimal_conditions"] is not True:
        raise ValueError("Base config must enable optimal-conditions export")
    raw["output"]["save_files"] = not config["SIMULATE"]
    raw["optimization"]["use_bayesian"] = not config["SIMULATE"] or _USE_BAYES_OPT_ON_SIMULATE
    return raw


def _snapshot(path, raw):
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=False)


def _preflight_artifact(path, raw):
    from sdl_pipette_calibration.pipetting_wizard import PipettingWizard

    if not path.is_file():
        raise FileNotFoundError(f"Current run final exporter CSV missing: {path}")
    wizard = PipettingWizard()
    data = wizard.load_calibration_data(path)
    if data is None:
        raise ValueError("Current run CSV is not usable wizard calibration data")
    _preflight_conditions(data, raw)


def _preflight_conditions(data, raw):
    """Validate the same calibration table for disk and in-memory handoffs."""
    from sdl_pipette_calibration.pipetting_wizard import PipettingWizard
    from sdl_pipette_calibration.parameter_constraints import constrain_parameters

    wizard = PipettingWizard()
    required = set(raw["hardware_parameters"]) | set(raw["experiment"]["fixed_parameters"])
    columns = ["volume_target_ul", "volume_measured_ml", "calibration_overaspirate_vol"]
    columns.extend("hardware_parameters_" + name for name in sorted(required))
    for column in columns:
        if column not in data:
            raise ValueError(f"Final CSV missing mandatory column: {column}")
        for value in data[column]:
            _number(value, column,
                    minimum=None if column == "calibration_overaspirate_vol" else 0,
                    positive=column in ("volume_target_ul", "volume_measured_ml"))
    if data["volume_target_ul"].duplicated().any():
        raise ValueError("Final CSV contains duplicate calibrated volumes")
    expected = sorted(volume * 1000 for volume in raw["experiment"]["volume_targets_ml"])
    actual = sorted(data["volume_target_ul"].tolist())
    if len(actual) != len(expected) or any(not math.isclose(found, wanted, rel_tol=1e-9)
                                         for found, wanted in zip(actual, expected)):
        raise ValueError("Final CSV does not cover every calibrated volume")
    compensated = wizard.apply_overvolume_compensation(data.copy())
    protocol = importlib.import_module(_PROTOCOL_NAME).protocol_instance
    tip_type = raw['experiment'].get('north_tip_type')
    for volume in raw["validation"]["volumes_ml"]:
        parameters = wizard.interpolate_parameters(compensated, volume)
        parameters = constrain_parameters(
            parameters, protocol.get_parameter_constraints(volume, tip_type=tip_type),
            {'target_volume_ml': volume})
        for name in required | {"overaspirate_vol", "volume_ml"}:
            _number(parameters[name], f"Interpolated {name}",
                    minimum=None if name == "overaspirate_vol" else 0)
        protocol.validate_workflow_capacity(volume, parameters, tip_type=tip_type)


def _check_validation(results, config):
    measurements = results["validation_results"]
    expected = len(config["VALIDATION_VOLUMES_ML"]) * config["REPLICATES_PER_VOLUME"]
    if len(measurements) != expected:
        raise RuntimeError("Validation returned an incomplete measurement set")
    for measurement in measurements:
        if "error" in measurement:
            raise RuntimeError(f"Validation measurement failed: {measurement['error']}")
        _number(measurement["volume_measured_ml"], "Validation measured volume", positive=True)
        _number(measurement["duration_s"], "Validation duration")
    summary = results["analysis"]["summary"]
    if summary["total_volumes_tested"] != len(config["VALIDATION_VOLUMES_ML"]):
        raise RuntimeError("Validation analysis is incomplete")
    rate = _number(summary["overall_pass_rate"], "Validation pass rate")
    if rate > 1:
        raise RuntimeError("Invalid validation pass rate")
    return summary


def _send_workflow_slack(lash_e, message):
    if lash_e.simulate:
        return
    try:
        import slack_agent
        if not slack_agent.safe_send_slack_message(message):
            lash_e.logger.warning("Slack notification was not delivered (non-fatal)")
    except Exception as error:
        lash_e.logger.warning("Slack notification failed (non-fatal): %s", error)


def _cleanup(lash_e, config, physical):
    actions = []
    if physical:
        actions = [("remove_pipet", lambda: lash_e.nr_robot.remove_pipet()),
                   ("return_target", lambda: lash_e.nr_robot.return_vial_home(config["TARGET_VIAL"])),
                   ("move_home", lambda: lash_e.nr_robot.move_home())]
    actions.append(("disconnect", lambda: lash_e.nr_robot.c9.network.disconnect()))
    for label, action in actions:
        try:
            action()
        except (Exception, KeyboardInterrupt):
            lash_e.logger.exception("Best-effort workflow cleanup failed: %s", label)


def execute(config=None, show_gui=True):
    """Use GUI-confirmed settings or a complete supplied config without GUI."""
    lash_e = Lash_E(initialize_biotek=False, workflow_globals=globals(),
                    workflow_name=_WORKFLOW_NAME, config=config, show_gui=show_gui)
    if not lash_e._workflow_should_continue:
        return None
    confirmed = lash_e.workflow_config
    failed = True
    try:
        validate_experiment(confirmed, lash_e)
        get_liquid(confirmed["LIQUID"])
        from sdl_pipette_calibration.config_manager import ExperimentConfig
        from sdl_pipette_calibration.experiment import CalibrationExperiment
        from sdl_pipette_calibration.protocol_loader import load_hardware_protocol
        from sdl_pipette_calibration.run_validation import ValidationRunner

        base_path = Path(confirmed["CALIBRATION_CONFIG_FILE"])
        if not base_path.is_absolute():
            base_path = _ROOT / base_path
        base_path = base_path.resolve(strict=True)
        raw = _effective_config(confirmed, base_path)
        protocol = load_hardware_protocol(_PROTOCOL_NAME)
        liquids = importlib.import_module(_PROTOCOL_NAME).LIQUIDS
        if confirmed["LIQUID"] not in liquids:
            raise ValueError(f"Unknown calibration liquid: {confirmed['LIQUID']}")
        save_files = raw["output"]["save_files"]
        output = None
        if save_files:
            output = _base_path(raw["output"]["base_directory"], base_path) / (
                "pc_" + uuid4().hex[:12])
            output.mkdir(parents=True, exist_ok=False)
            raw["output"]["base_directory"] = str(output / "calibration")
            raw["validation"]["output_directory"] = str(output / "validation")
            _snapshot(output / "calibration_effective_config.yaml", raw)
            _snapshot(output / "workflow_confirmed_config.yaml", confirmed)
        calibration_config = ExperimentConfig(raw, str(base_path))
        lash_e.logger.info("Calibration workflow output: %s", output or "in-memory only")
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started | liquid={confirmed['LIQUID']}")
        experiment = CalibrationExperiment(calibration_config)
        with protocol.workflow_session(lash_e, raw, confirmed["TARGET_VIAL"]):
            calibration_results = experiment.run()
        artifact = None
        conditions = None
        if save_files:
            artifact = Path(experiment.output_dir).resolve() / f"optimal_conditions_{confirmed['LIQUID']}.csv"
            _preflight_artifact(artifact, raw)
            table = pd.read_csv(artifact)
            table['tip_type'] = confirmed['TIP_TYPE']
            table.to_csv(artifact, index=False)
        else:
            conditions = experiment.optimal_conditions_table(calibration_results)
            _preflight_conditions(conditions, raw)
            conditions['tip_type'] = confirmed['TIP_TYPE']
        validation_raw = deepcopy(raw)
        validation_raw["experiment"]["volume_targets_ml"] = deepcopy(confirmed["VALIDATION_VOLUMES_ML"])
        validation_raw["validation"]["optimal_conditions_file"] = str(artifact) if artifact is not None else None
        validation_config = ExperimentConfig(validation_raw, str(base_path))
        if save_files:
            _snapshot(output / "validation_effective_config.yaml", validation_raw)
            validator = ValidationRunner(validation_config, base_path)
        else:
            validator = ValidationRunner(validation_config, base_path, optimal_conditions=conditions)
        with protocol.workflow_session(lash_e, validation_raw, confirmed["TARGET_VIAL"]):
            validation_results = validator.run_validation()
        summary = _check_validation(validation_results, confirmed)
        message = (f"{_WORKFLOW_NAME} completed | liquid={confirmed['LIQUID']} | validation "
                   f"{summary['volumes_passed']}/{summary['total_volumes_tested']} volumes passed "
                   f"({summary['overall_pass_rate']:.1%}); parameters not installed | "
                   f"output_dir={output} | optimal_conditions_file={artifact}")
        lash_e.logger.info(message)
        _send_workflow_slack(lash_e, message)
        failed = False
        return {"calibration_results": calibration_results, "validation_results": validation_results,
                "optimal_conditions_file": str(artifact) if artifact is not None else None,
                "output_dir": str(output) if output is not None else None}
    except (Exception, KeyboardInterrupt) as error:
        lash_e.logger.exception("Pipette calibration workflow failed")
        if isinstance(error, KeyboardInterrupt):
            from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
            cleanup_after_keyboard_interrupt(lash_e)
        else:
            _cleanup(lash_e, confirmed, physical=True)
        # The scheduler live runner reports failures; it does not report interrupts.
        if isinstance(error, KeyboardInterrupt):
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted; cleanup attempted")
        raise
    finally:
        if not failed:
            _cleanup(lash_e, confirmed, physical=False)


if __name__ == "__main__":
    execute()
