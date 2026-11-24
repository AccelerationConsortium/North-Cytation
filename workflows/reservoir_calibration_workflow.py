"""Reservoir calibration followed by independent validation, without promotion.

Copy of pipette_calibration_workflow.py for a reservoir pump: the calibration program and its
base config are the same; the hardware parameters are the reservoir's five, and dispenses go
into the measurement vials of INPUT_VIAL_STATUS_FILE (used in order as each fills). The protocol
(calibration_protocol_reservoir) primes the line at the start of calibration and of validation.
"""
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
_PROTOCOL_NAME = "calibration_protocol_reservoir"
_OUTPUT_DIRECTORY = "../output/reservoir_calibration/runs/"  # relative to CALIBRATION_CONFIG_FILE

SIMULATE = True
# Internal development switch; excluded from GUI/YAML configuration.
_USE_BAYES_OPT_ON_SIMULATE = False
INPUT_VIAL_STATUS_FILE = "status/reservoir_calibration_vials.csv"
CALIBRATION_CONFIG_FILE = "sdl_pipette_calibration/experiment_config_workflow.yaml"
LIQUID = "water"
RESERVOIR_INDEX = 1
VOLUME_TARGETS_ML = [1.0, 0.5, 2.4]
RANDOM_SEED = 30
MAX_TOTAL_MEASUREMENTS = 120
MAX_MEASUREMENTS_FIRST_VOLUME = 60
MAX_REPLICATES_PER_TRIAL = 3
NUM_SCREENING_TRIALS = 6
TWO_POINT_CALIBRATION_REPLICATES = 3
MIN_GOOD_TRIALS = 5
VALIDATION_VOLUMES_ML = [0.5, 1.0, 1.5, 2.0, 2.4]
REPLICATES_PER_VOLUME = 5
MAX_RETRIES_PER_MEASUREMENT = 3
QUALITY_STD_THRESHOLD_G = 0.004
OVERASPIRATE_VOL_MIN = 0.0
OVERASPIRATE_VOL_MAX = 0.05
OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET = 0.1
ASPIRATE_SPEED_FIXED = False
ASPIRATE_SPEED_FIXED_VALUE = 15
ASPIRATE_SPEED_MIN = 5
ASPIRATE_SPEED_MAX = 30
DISPENSE_SPEED_FIXED = False
DISPENSE_SPEED_FIXED_VALUE = 15
DISPENSE_SPEED_MIN = 5
DISPENSE_SPEED_MAX = 30
ASPIRATE_WAIT_TIME_FIXED = False
ASPIRATE_WAIT_TIME_FIXED_VALUE = 0.0
ASPIRATE_WAIT_TIME_MIN = 0.0
ASPIRATE_WAIT_TIME_MAX = 5.0
DISPENSE_WAIT_TIME_FIXED = False
DISPENSE_WAIT_TIME_FIXED_VALUE = 1.0
DISPENSE_WAIT_TIME_MIN = 0.0
DISPENSE_WAIT_TIME_MAX = 5.0
VALVE_SWITCH_DELAY_FIXED = True
VALVE_SWITCH_DELAY_FIXED_VALUE = 0.1
VALVE_SWITCH_DELAY_MIN = 0.0
VALVE_SWITCH_DELAY_MAX = 2.0

# name -> type; the protocol requires all five
_HARDWARE_PARAMETERS = {
    "aspirate_speed": "integer", "dispense_speed": "integer", "aspirate_wait_time": "float",
    "dispense_wait_time": "float", "valve_switch_delay": "float",
}
_CONFIG_KEYS = [
    "SIMULATE", "INPUT_VIAL_STATUS_FILE", "CALIBRATION_CONFIG_FILE", "LIQUID", "RESERVOIR_INDEX",
    "VOLUME_TARGETS_ML", "RANDOM_SEED", "MAX_TOTAL_MEASUREMENTS", "MAX_MEASUREMENTS_FIRST_VOLUME",
    "MAX_REPLICATES_PER_TRIAL", "NUM_SCREENING_TRIALS", "TWO_POINT_CALIBRATION_REPLICATES",
    "MIN_GOOD_TRIALS", "VALIDATION_VOLUMES_ML", "REPLICATES_PER_VOLUME",
    "MAX_RETRIES_PER_MEASUREMENT", "QUALITY_STD_THRESHOLD_G",
    "OVERASPIRATE_VOL_MIN", "OVERASPIRATE_VOL_MAX", "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET",
    "ASPIRATE_SPEED_FIXED", "ASPIRATE_SPEED_FIXED_VALUE", "ASPIRATE_SPEED_MIN", "ASPIRATE_SPEED_MAX",
    "DISPENSE_SPEED_FIXED", "DISPENSE_SPEED_FIXED_VALUE", "DISPENSE_SPEED_MIN", "DISPENSE_SPEED_MAX",
    "ASPIRATE_WAIT_TIME_FIXED", "ASPIRATE_WAIT_TIME_FIXED_VALUE", "ASPIRATE_WAIT_TIME_MIN", "ASPIRATE_WAIT_TIME_MAX",
    "DISPENSE_WAIT_TIME_FIXED", "DISPENSE_WAIT_TIME_FIXED_VALUE", "DISPENSE_WAIT_TIME_MIN", "DISPENSE_WAIT_TIME_MAX",
    "VALVE_SWITCH_DELAY_FIXED", "VALVE_SWITCH_DELAY_FIXED_VALUE", "VALVE_SWITCH_DELAY_MIN", "VALVE_SWITCH_DELAY_MAX",
]  # Literal (read by planning tools with ast); hardware keys follow _HARDWARE_PARAMETERS.
# Planning annotations, one per _CONFIG_KEYS entry (field meanings in workflows/workflow_template.py).
_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.", "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial CSV (repository-relative) of measurement vials; each is filled in file order until it has no room, then the next is used.", "unit": None, "affects": ["vials"], "shares": "vial_layout"},
    "CALIBRATION_CONFIG_FILE": {"description": "Base calibration YAML (sdl_pipette_calibration); supplies optimizer objectives, tolerances, adaptive-replicate thresholds and other defaults not overridden here. Its hardware_parameters are replaced by the reservoir's.", "unit": None, "affects": ["duration"], "shares": "input_file"},
    "LIQUID": {"description": "Liquid to calibrate; must be in the liquid registry and be the liquid of RESERVOIR_INDEX in syringe_pumps.yaml.", "unit": None, "affects": ["liquids", "duration"], "ask": True},
    "RESERVOIR_INDEX": {"description": "Reservoir pump to calibrate (syringe_pumps.yaml); its prime_volume_mL is dispensed at the start of calibration and of validation.", "unit": None, "affects": ["liquids"]},
    "VOLUME_TARGETS_ML": {"description": "Volumes to calibrate; the first gets screening plus optimization, later ones reuse its results.", "unit": "mL", "affects": ["duration", "liquids"], "effect": "first volume: screening + two-point + optimization within MAX_MEASUREMENTS_FIRST_VOLUME; each later volume: two-point + optimization sharing the remaining budget", "ask": True},
    "RANDOM_SEED": {"description": "Seed for reproducible screening and simulated measurements.", "unit": None, "affects": []},
    "MAX_TOTAL_MEASUREMENTS": {"description": "Measurement budget for calibration across all volumes (validation not included).", "unit": None, "affects": ["duration", "liquids"]},
    "MAX_MEASUREMENTS_FIRST_VOLUME": {"description": "Measurement budget for the first calibration volume.", "unit": None, "affects": ["duration", "liquids"]},
    "MAX_REPLICATES_PER_TRIAL": {"description": "Most replicate measurements one trial can take when its accuracy looks promising.", "unit": None, "affects": ["duration", "liquids"]},
    "NUM_SCREENING_TRIALS": {"description": "Exploratory trials on the first volume before optimization.", "unit": None, "affects": ["duration", "liquids"]},
    "TWO_POINT_CALIBRATION_REPLICATES": {"description": "Replicates at each of the two overaspirate points used to fit a volume correction.", "unit": None, "affects": ["duration", "liquids"]},
    "MIN_GOOD_TRIALS": {"description": "Optimization stops early once this many trials meet the accuracy/precision tolerance.", "unit": None, "affects": ["duration", "liquids"]},
    "VALIDATION_VOLUMES_ML": {"description": "Volumes checked in the independent validation after calibration.", "unit": "mL", "affects": ["duration", "liquids"], "effect": "validation measurements = len(VALIDATION_VOLUMES_ML) x REPLICATES_PER_VOLUME"},
    "REPLICATES_PER_VOLUME": {"description": "Validation replicates at each validation volume.", "unit": None, "affects": ["duration", "liquids"]},
    "MAX_RETRIES_PER_MEASUREMENT": {"description": "How many times a hardware measurement is repeated when the scale reading is unstable.", "unit": None, "affects": ["duration", "liquids"], "effect": "each retry dispenses again; hardware only"},
    "QUALITY_STD_THRESHOLD_G": {"description": "Scale baseline standard deviation (grams) below which a reading counts as stable.", "unit": None, "affects": ["duration"]},
    "OVERASPIRATE_VOL_MIN": {"description": "Lower search bound for the extra volume drawn and dispensed to correct delivery.", "unit": "mL", "affects": []},
    "OVERASPIRATE_VOL_MAX": {"description": "Upper search bound for the extra volume drawn and dispensed to correct delivery.", "unit": "mL", "affects": []},
    "OVERASPIRATE_VOL_MAX_FRACTION_OF_TARGET": {"description": "Cap on the overaspirate volume as a fraction of each target volume.", "unit": None, "affects": []},
    "ASPIRATE_SPEED_FIXED": {"description": "Hold the pump aspirate speed (1-40, higher = slower) at ASPIRATE_SPEED_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "ASPIRATE_SPEED_FIXED_VALUE": {"description": "Fixed value of the pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "ASPIRATE_SPEED_MIN": {"description": "Lower optimizer bound for the pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "ASPIRATE_SPEED_MAX": {"description": "Upper optimizer bound for the pump aspirate speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_FIXED": {"description": "Hold the pump dispense speed (1-40, higher = slower) at DISPENSE_SPEED_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_FIXED_VALUE": {"description": "Fixed value of the pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_MIN": {"description": "Lower optimizer bound for the pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "DISPENSE_SPEED_MAX": {"description": "Upper optimizer bound for the pump dispense speed (1-40, higher = slower).", "unit": None, "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_FIXED": {"description": "Hold the wait after each aspirate stroke at ASPIRATE_WAIT_TIME_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_FIXED_VALUE": {"description": "Fixed value of the wait after each aspirate stroke.", "unit": "s", "affects": ["duration"], "effect": "dwell per stroke; North_Safe skips it in simulation"},
    "ASPIRATE_WAIT_TIME_MIN": {"description": "Lower optimizer bound for the wait after each aspirate stroke.", "unit": "s", "affects": ["duration"]},
    "ASPIRATE_WAIT_TIME_MAX": {"description": "Upper optimizer bound for the wait after each aspirate stroke.", "unit": "s", "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_FIXED": {"description": "Hold the wait after each dispense stroke at DISPENSE_WAIT_TIME_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_FIXED_VALUE": {"description": "Fixed value of the wait after each dispense stroke.", "unit": "s", "affects": ["duration"], "effect": "dwell per stroke; North_Safe skips it in simulation"},
    "DISPENSE_WAIT_TIME_MIN": {"description": "Lower optimizer bound for the wait after each dispense stroke.", "unit": "s", "affects": ["duration"]},
    "DISPENSE_WAIT_TIME_MAX": {"description": "Upper optimizer bound for the wait after each dispense stroke.", "unit": "s", "affects": ["duration"]},
    "VALVE_SWITCH_DELAY_FIXED": {"description": "Hold the wait after each valve switch at VALVE_SWITCH_DELAY_FIXED_VALUE instead of optimizing it.", "unit": None, "affects": ["duration"]},
    "VALVE_SWITCH_DELAY_FIXED_VALUE": {"description": "Fixed value of the wait after each valve switch.", "unit": "s", "affects": ["duration"], "effect": "two waits per stroke; North_Safe skips it in simulation"},
    "VALVE_SWITCH_DELAY_MIN": {"description": "Lower optimizer bound for the wait after each valve switch.", "unit": "s", "affects": ["duration"]},
    "VALVE_SWITCH_DELAY_MAX": {"description": "Upper optimizer bound for the wait after each valve switch.", "unit": "s", "affects": ["duration"]},
}

# Solutions this workflow makes: none (it weighs dispenses of the reservoir liquid).
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
    if type(config["SIMULATE"]) is not bool:
        raise ValueError("SIMULATE must be boolean")
    if config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes do not match")
    vial_file = config["INPUT_VIAL_STATUS_FILE"]
    if not isinstance(vial_file, str) or not Path(vial_file).is_file():
        raise ValueError("INPUT_VIAL_STATUS_FILE must name an existing vial CSV")
    robot = lash_e.nr_robot
    if robot.VIAL_FILE is None or Path(vial_file).resolve() != Path(robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during review; restart with the confirmed file")
    reservoir = config["RESERVOIR_INDEX"]
    pump = robot.PUMP_CONFIG.get(reservoir) if type(reservoir) is int else None
    if not pump or pump.get("carousel_angle") is None:
        raise ValueError("RESERVOIR_INDEX must be a reservoir pump in syringe_pumps.yaml")
    if pump.get("liquid") != config["LIQUID"]:
        raise ValueError(f"Reservoir {reservoir} holds {pump.get('liquid')!r}, not {config['LIQUID']!r}")
    _number(pump.get("prime_volume_mL"), f"prime_volume_mL of reservoir {reservoir}", positive=True)
    for key in ("VOLUME_TARGETS_ML", "VALIDATION_VOLUMES_ML"):
        _volumes(config[key], key)
    oav_max = _number(config["OVERASPIRATE_VOL_MAX"], "OVERASPIRATE_VOL_MAX", positive=True)
    capacity = float(robot.get_config_parameter("vial_positions", "clamp", "vial_volume", error_on_missing=True))
    for key in ("VOLUME_TARGETS_ML", "VALIDATION_VOLUMES_ML"):
        for volume in config[key]:
            if volume + oav_max > capacity:
                raise ValueError(f"{key} contains {volume:g} mL; with overaspirate it does not fit "
                                 f"an empty clamp vial ({capacity:g} mL)")
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
    """Copy the scientific source, replace its hardware parameters with the reservoir's,
    and apply only reviewed workflow overrides."""
    with base_path.open(encoding="utf-8") as stream:
        raw = deepcopy(yaml.safe_load(stream))
    experiment = raw["experiment"]
    for key in ("LIQUID", "VOLUME_TARGETS_ML", "SIMULATE", "RANDOM_SEED",
                "MAX_TOTAL_MEASUREMENTS", "MAX_MEASUREMENTS_FIRST_VOLUME",
                "MAX_REPLICATES_PER_TRIAL", "NUM_SCREENING_TRIALS",
                "TWO_POINT_CALIBRATION_REPLICATES", "MAX_RETRIES_PER_MEASUREMENT",
                "QUALITY_STD_THRESHOLD_G", "RESERVOIR_INDEX"):
        experiment[key.lower()] = deepcopy(config[key])
    experiment["protocol_override"] = _PROTOCOL_NAME
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
    # The base config's hardware parameters are the pipette's; the reservoir has its own five
    hardware = raw["hardware_parameters"] = {}
    experiment["fixed_parameters"] = {}
    for name, kind in _HARDWARE_PARAMETERS.items():
        prefix = name.upper()
        fixed = config[prefix + "_FIXED"]
        if type(fixed) is not bool:
            raise ValueError(f"{prefix}_FIXED must be boolean")
        integer = kind == "integer"
        lower = _number(config[prefix + "_MIN"], prefix + "_MIN", integer=integer)
        upper = _number(config[prefix + "_MAX"], prefix + "_MAX", integer=integer)
        value = _number(config[prefix + "_FIXED_VALUE"], prefix + "_FIXED_VALUE", integer=integer)
        if lower >= upper or not lower <= value <= upper:
            raise ValueError(f"Invalid bounds/fixed value for {name}")
        if name in ("aspirate_speed", "dispense_speed") and not 1 <= lower < upper <= 40:
            raise ValueError(f"{name} bounds must be within North pump speed limits 1-40")
        hardware[name] = {"type": kind, "bounds": [lower, upper], "default": value,
                          "round_to_nearest": 1 if integer else 0.1,
                          "description": _CONFIG_DOCS[prefix + "_FIXED_VALUE"]["description"]}
        if fixed:
            experiment["fixed_parameters"][name] = value
    for section, key in ((raw["optimization"]["llm_optimization"], "config_path"),
                         (raw["screening"], "llm_config_path"),
                         (raw["screening"]["external_data"], "data_path")):
        if section[key] is not None:
            section[key] = str(_base_path(section[key], base_path))
    if raw["screening"]["external_data"]["enabled"]:
        raise ValueError("Base config enables external screening data, which is pipetting data")
    # Volumes above the base config's tolerance ranges (pipette volumes) get its top range's tolerance
    ranges = raw["tolerances"]["volume_ranges"]
    top = max(ranges, key=lambda entry: entry["volume_max_ul"])
    largest_ul = 1000 * max(config["VOLUME_TARGETS_ML"] + config["VALIDATION_VOLUMES_ML"])
    top["volume_max_ul"] = max(top["volume_max_ul"], largest_ul)
    raw["validation"]["volumes_ml"] = deepcopy(config["VALIDATION_VOLUMES_ML"])
    raw["validation"]["replicates_per_volume"] = config["REPLICATES_PER_VOLUME"]
    if raw["output"]["export_optimal_conditions"] is not True:
        raise ValueError("Base config must enable optimal-conditions export")
    raw["output"]["base_directory"] = _OUTPUT_DIRECTORY
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
    data = PipettingWizard().load_calibration_data(path)
    if data is None:
        raise ValueError("Current run CSV is not usable wizard calibration data")
    _preflight_conditions(data, raw)


def _preflight_conditions(data, raw):
    """Validate the same calibration table for disk and in-memory handoffs."""
    from sdl_pipette_calibration.pipetting_wizard import PipettingWizard

    wizard = PipettingWizard()
    required = set(raw["hardware_parameters"])
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
    for volume in raw["validation"]["volumes_ml"]:
        parameters = wizard.interpolate_parameters(compensated, volume)
        for name in required | {"overaspirate_vol", "volume_ml"}:
            _number(parameters[name], f"Interpolated {name}",
                    minimum=None if name == "overaspirate_vol" else 0)


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


def _disconnect(lash_e):
    try:
        lash_e.nr_robot.c9.network.disconnect()
    except (Exception, KeyboardInterrupt):
        lash_e.logger.exception("Best-effort workflow cleanup failed: disconnect")


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
        save_files = raw["output"]["save_files"]
        output = None
        if save_files:
            output = _base_path(raw["output"]["base_directory"], base_path) / ("rc_" + uuid4().hex[:12])
            output.mkdir(parents=True, exist_ok=False)
            raw["output"]["base_directory"] = str(output / "calibration")
            raw["validation"]["output_directory"] = str(output / "validation")
            _snapshot(output / "calibration_effective_config.yaml", raw)
            _snapshot(output / "workflow_confirmed_config.yaml", confirmed)
            lash_e.save_run_info(output)
        calibration_config = ExperimentConfig(raw, str(base_path))
        lash_e.logger.info("Reservoir calibration workflow output: %s", output or "in-memory only")
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} started | liquid={confirmed['LIQUID']} | "
                                     f"reservoir={confirmed['RESERVOIR_INDEX']} | "
                                     f"volumes_mL={confirmed['VOLUME_TARGETS_ML']}")
        experiment = CalibrationExperiment(calibration_config)
        with protocol.workflow_session(lash_e, raw):
            calibration_results = experiment.run()
        artifact = None
        conditions = None
        if save_files:
            artifact = Path(experiment.output_dir).resolve() / f"optimal_conditions_{confirmed['LIQUID']}.csv"
            _preflight_artifact(artifact, raw)
            table = pd.read_csv(artifact)
            table['reservoir_index'] = confirmed['RESERVOIR_INDEX']
            table.to_csv(artifact, index=False)
        else:
            conditions = experiment.optimal_conditions_table(calibration_results)
            _preflight_conditions(conditions, raw)
            conditions['reservoir_index'] = confirmed['RESERVOIR_INDEX']
        validation_raw = deepcopy(raw)
        validation_raw["experiment"]["volume_targets_ml"] = deepcopy(confirmed["VALIDATION_VOLUMES_ML"])
        validation_raw["validation"]["optimal_conditions_file"] = str(artifact) if artifact is not None else None
        validation_config = ExperimentConfig(validation_raw, str(base_path))
        if save_files:
            _snapshot(output / "validation_effective_config.yaml", validation_raw)
            validator = ValidationRunner(validation_config, base_path)
        else:
            validator = ValidationRunner(validation_config, base_path, optimal_conditions=conditions)
        with protocol.workflow_session(lash_e, validation_raw):
            validation_results = validator.run_validation()
        summary = _check_validation(validation_results, confirmed)
        message = (f"{_WORKFLOW_NAME} completed | liquid={confirmed['LIQUID']} | reservoir="
                   f"{confirmed['RESERVOIR_INDEX']} | validation {summary['volumes_passed']}/"
                   f"{summary['total_volumes_tested']} volumes passed ({summary['overall_pass_rate']:.1%}); "
                   f"parameters not installed | output_dir={output} | optimal_conditions_file={artifact}")
        lash_e.logger.info(message)
        _send_workflow_slack(lash_e, message)
        failed = False
        return {"calibration_results": calibration_results, "validation_results": validation_results,
                "optimal_conditions_file": str(artifact) if artifact is not None else None,
                "output_dir": str(output) if output is not None else None}
    except (Exception, KeyboardInterrupt) as error:
        lash_e.logger.exception("Reservoir calibration workflow failed")
        from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
        cleanup_after_keyboard_interrupt(lash_e)
        if not isinstance(error, KeyboardInterrupt):
            _disconnect(lash_e)
        if isinstance(error, KeyboardInterrupt):
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted; cleanup attempted")
        raise
    finally:
        if not failed:
            _disconnect(lash_e)


if __name__ == "__main__":
    execute()
