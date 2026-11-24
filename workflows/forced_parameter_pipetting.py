"""Forced-parameter pipetting: pipette exactly the given parameters and weigh every dispense.

An idle-time data-gathering workflow (plan: docs/plans/archive/forced_parameter_pipetting_plan.md).
PARAMETER_FILE lists rows of liquid, volume and a full parameter set. Each row is pipetted
REPLICATES times (or its own replicates column) from the liquid's source vial on the clamp and back into the same vial, with
exactly those parameters (liquid=None: no calibration, corrections or defaults can fill a gap),
and every dispense is weighed. Mass -> volume with the liquid's density (liquids.yaml).

- Tip: chosen from the row's volume and parameters by North_Safe's own rule, then fixed.
  Rows share a tip by liquid and tip type (liquids that refill_pipets get a tip per
  measurement). An optional tip_group column splits that further: a new value is a new tip
  (for tip-to-tip drift). A fresh tip is conditioned when the liquid's condition_tip is on.
- Saving: one line per measurement in output/forced_parameter_pipetting/<CAMPAIGN_ID>/
  measurements.csv, written the moment it is taken. Running the same CAMPAIGN_ID again resumes:
  (param_set_id, replicate) pairs already measured are skipped. A row whose parameters changed
  since it was measured is skipped and listed; new rows run.
- Time: MAX_RUN_MINUTES > 0 stops starting measurements when the next one (the slowest so far)
  would not finish in time; the rest stays pending for the next run.
- Faults: an unstable scale is recorded (measurement_acceptable False), never retried, as in the
  embedded validation; a scale that stays unstable stops the run. Three measurements in a row
  under 10% of target (empty vial, no tip) stop it. Off-target volumes are the data: no warnings.
- The master pipetting database is never written; this campaign is its own dataset, with the
  master's column names where they overlap.

Simulation runs every move; masses are nominal (volume x density) and saved under
<CAMPAIGN_ID>_simulate.
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import yaml

from master_usdl_coordinator import Lash_E

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
PARAMETER_FILE = "pipetting_data/forced_parameters/example.csv"
CAMPAIGN_ID = "example"
REPLICATES = 1
SOURCE_VIALS = {}  # liquid -> [vial names], used in turn; a liquid not listed uses the vial named after it
VIAL_LOW_THRESHOLD_ML = 1.0
MAX_RUN_MINUTES = 0.0  # 0 = no time limit
MAX_ROWS_FOR_SIMULATION = 6
RESERVOIR_REFILLS = {}  # liquid -> reservoir index: top its source vials up to REFILL_TO_ML before measuring
REFILL_TO_ML = 8.0
APPLY_BLOWOUT_SPEED = False  # True: push blowout air at each row's blowout_speed (off = today's robot)

_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "PARAMETER_FILE", "CAMPAIGN_ID", "REPLICATES",
                "SOURCE_VIALS", "VIAL_LOW_THRESHOLD_ML", "MAX_RUN_MINUTES", "MAX_ROWS_FOR_SIMULATION",
                "RESERVOIR_REFILLS", "REFILL_TO_ML", "APPLY_BLOWOUT_SPEED"]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV (repository-relative) for this run.",
                               "unit": None, "affects": ["vials", "liquids"], "shares": "vial_layout"},
    "PARAMETER_FILE": {"description": "CSV of rows to pipette: liquid, volume_ul and every pipetting parameter "
                                      "(optional param_set_id, tip_group).",
                       "unit": None, "affects": ["duration", "tips", "liquids"], "ask": True,
                       "effect": "one measurement per row and replicate"},
    "CAMPAIGN_ID": {"description": "Names the output folder; the same ID resumes, a new ID starts fresh.",
                    "unit": None, "affects": [], "ask": True},
    "REPLICATES": {"description": "Measurements per row.", "unit": None, "affects": ["duration", "tips"],
                   "effect": "measurements = rows x REPLICATES"},
    "SOURCE_VIALS": {"description": "Per liquid, the vials holding it, used in turn as each runs low. "
                                    "A liquid not listed uses the vial named after it.",
                     "unit": None, "affects": ["vials"]},
    "VIAL_LOW_THRESHOLD_ML": {"description": "Move to the liquid's next source vial below this volume.",
                              "unit": "mL", "affects": ["vials"]},
    "MAX_RUN_MINUTES": {"description": "Stop starting measurements after this long (0 = no limit); "
                                       "the rest stays pending for the next run.",
                        "unit": "min", "affects": ["duration"]},
    "MAX_ROWS_FOR_SIMULATION": {"description": "Rows measured in a simulation.", "unit": None,
                                "affects": ["duration"]},
    "RESERVOIR_REFILLS": {"description": "Per liquid, the reservoir that holds it: its source vials are topped up "
                                         "to REFILL_TO_ML before measuring (start full).",
                          "unit": None, "affects": ["duration", "liquids"]},
    "REFILL_TO_ML": {"description": "Volume reservoir-refilled source vials are topped up to.", "unit": "mL",
                     "affects": ["liquids"]},
    "APPLY_BLOWOUT_SPEED": {"description": "Push the blowout air at each row's blowout_speed instead of its "
                                           "dispense_speed. Off (default) matches every other workflow.",
                            "unit": None, "affects": []},
}

_PRODUCT_VIALS = []

# Parameter file columns (all required, no blanks). uL in the file, mL to the robot.
PARAMETER_COLUMNS = {  # file column -> PipettingParameters field, mL factor (None: not a volume)
    "aspirate_speed": ("aspirate_speed", None), "dispense_speed": ("dispense_speed", None),
    "retract_speed": ("retract_speed", None), "blowout_speed": ("blowout_speed", None),
    "aspirate_wait_time": ("aspirate_wait_time", None), "dispense_wait_time": ("dispense_wait_time", None),
    "post_retract_wait_time": ("post_retract_wait_time", None),
    "overaspirate_vol_ul": ("overaspirate_vol", 1000.0), "pre_asp_air_vol_ul": ("pre_asp_air_vol", 1000.0),
    "post_asp_air_vol_ul": ("post_asp_air_vol", 1000.0), "blowout_vol_ul": ("blowout_vol", 1000.0),
    "asp_disp_cycles": ("asp_disp_cycles", None),
}
WHOLE = {"aspirate_speed", "dispense_speed", "retract_speed", "blowout_speed", "asp_disp_cycles"}
REQUIRED = ["liquid", "volume_ul", *PARAMETER_COLUMNS]
OUTPUT_COLUMNS = [
    "param_set_id", "replicate", "timestamp", "status", "measurement_acceptable", "liquid_type",
    "density_used", "source_vial", "tip_type", "tip_number", "new_tip", "conditioned", "target_ul",
    "target_volume_ml", "mass_difference_g", "measured_ul", "measured_volume_ml", "accuracy_pct",
    "elapsed_s", *[field for field, _ in PARAMETER_COLUMNS.values()], "pre_baseline_std",
    "post_baseline_std", "pre_stable_count", "pre_total_count", "post_stable_count", "post_total_count",
    "temp_c", "humidity_pct", "pressure_pa", "validation_type", "row_hash", "simulated", "run_started",
]
BROKEN_RUN = 3          # measurements in a row under BROKEN_FRACTION of target stop the run
BROKEN_FRACTION = 0.10
FAILED_RUN = 3          # measurements in a row that returned no mass stop the run
END_RESERVE_S = 120.0   # kept free before MAX_RUN_MINUTES for removing the tip and returning the vial


# ── rows ──────────────────────────────────────────────────────────────────────

def row_hash(row):
    """Identity of a row's liquid, volume and parameters (what a resumed campaign must match)."""
    key = {k: str(row[k]).strip() for k in REQUIRED}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def read_rows(path):
    """Rows of the parameter file, each with param_set_id (the row number when the file has none)."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = []
        for number, row in enumerate(reader, start=1):
            row = {k: (v or "").strip() for k, v in row.items() if k}
            row["param_set_id"] = row.get("param_set_id") or str(number)
            rows.append(row)
    ids = [r["param_set_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: param_set_id values must be unique")
    return rows


def parse_row(row):
    """(volume_mL, {field: value}) from a file row; ValueError naming the first bad value."""
    from pipetting_data.pipetting_parameters import PipettingParameters
    values = {}
    for column in REQUIRED[1:]:
        text = row.get(column, "")
        if text == "":
            raise ValueError(f"{column} is blank")
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"{column} {text!r} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{column} {text!r} is not finite")
        if column in WHOLE:
            if number != int(number):
                raise ValueError(f"{column} {text!r} must be a whole number")
            number = int(number)
        values[column] = number
    volume_ml = values.pop("volume_ul") / 1000.0
    if volume_ml <= 0:
        raise ValueError("volume_ul must be positive")
    fields = {}
    for column, (field, factor) in PARAMETER_COLUMNS.items():
        fields[field] = values[column] / factor if factor else values[column]
    return volume_ml, PipettingParameters(**fields)


def check_rows(rows, robot, liquids, source_vials, threshold_ml):
    """Every row checked before anything moves. Returns (plan rows, [rejected (id, reason)]).

    A plan row adds volume_ml, params, tip, density, refill, condition, vials. A row is
    rejected if a value is missing or outside robot_hardware.yaml parameter_limits, no tip
    fits it, air + liquid is more than the pipetting pump holds, its liquid isn't in liquids.yaml, or the liquid has no usable source vial.
    The echo check confirms North_Safe resolves exactly the row's parameters (liquid=None).
    """
    from North_Safe import _check_parameter_limits
    plan, rejected = [], []
    for row in rows:
        set_id = row["param_set_id"]
        try:
            liquid = row["liquid"]
            if liquid not in liquids:
                raise ValueError(f"liquid {liquid!r} is not in robot_state/liquids.yaml")
            volume_ml, params = parse_row(row)
            replicates_n = None
            if row.get("replicates", "") != "":
                replicates_n = float(row["replicates"])
                if replicates_n != int(replicates_n) or not 1 <= replicates_n <= 20:
                    raise ValueError("replicates must be a whole number from 1 to 20")
                replicates_n = int(replicates_n)
            _check_parameter_limits(robot, "Forced-parameter row", params)
            # The pump only warns and fills to its maximum when asked for more: refuse instead.
            pump_ml = float(robot.get_config_parameter("pumps", 0, "volume"))
            stroke = (volume_ml + max(0.0, params.overaspirate_vol) + params.pre_asp_air_vol
                      + params.post_asp_air_vol)
            if stroke > pump_ml + 1e-9:
                raise ValueError(f"air + liquid {stroke * 1000:.0f} uL is more than the pipetting pump's "
                                 f"{pump_ml * 1000:.0f} uL (syringe_pumps.yaml)")
            resolved, tip = robot.resolve_pipetting_setup(volume_ml, None, params)
            mismatch = [f for f, _ in PARAMETER_COLUMNS.values()
                        if getattr(resolved, f) != getattr(params, f)]
            if mismatch:
                raise ValueError(f"North_Safe changed {', '.join(mismatch)} (rejected_override_mismatch)")
            vials = list(source_vials.get(liquid) or [liquid])
            usable = [v for v in vials if _vial_volume(robot, v) >= threshold_ml]
            if not usable:
                raise ValueError(f"no source vial for {liquid} with at least {threshold_ml} mL "
                                 f"(tried {', '.join(vials)})")
        except (ValueError, KeyError, TypeError) as error:
            rejected.append((set_id, str(error)))
            continue
        props = liquids[liquid]
        extra = {"replicates_n": replicates_n} if replicates_n else {}
        plan.append({**row, **extra, "volume_ml": volume_ml, "params": params, "tip": tip,
                     "density": float(props["density"]), "refill": bool(props["refill_pipets"]),
                     "condition": bool(props.get("condition_tip", False)), "vials": vials,
                     "hash": row_hash(row)})
    return plan, rejected


def _vial_volume(robot, vial):
    """Tracked volume of a vial in the vial file; KeyError-like ValueError if it isn't there."""
    try:
        return float(robot.get_vial_info(vial, "vial_volume"))
    except Exception as error:
        raise ValueError(f"source vial {vial!r} is not in the vial file ({error})") from None


def pending_groups(plan, done, replicates):
    """Tip groups of what is left to measure, in run order.

    A group shares one tip and one source liquid: (liquid, tip type, tip_group). Groups run
    liquid by liquid (in first-appearance order), rows in file order. A row's own replicates
    column overrides REPLICATES. done is the set of
    (param_set_id, replicate) already measured. Returns [(key, [(row, replicate)])].
    """
    groups, liquid_order = {}, []
    for row in plan:
        if row["liquid"] not in liquid_order:
            liquid_order.append(row["liquid"])
        key = (row["liquid"], row["tip"], row.get("tip_group", ""))
        for replicate in range(1, row.get("replicates_n", replicates) + 1):
            if (row["param_set_id"], replicate) not in done:
                groups.setdefault(key, []).append((row, replicate))
    return sorted(groups.items(), key=lambda item: liquid_order.index(item[0][0]))


# ── campaign files ────────────────────────────────────────────────────────────

def campaign_dir(campaign_id, simulate, root=None):
    base = Path(root or Path(__file__).resolve().parents[1]) / "output" / _WORKFLOW_NAME
    return base / (f"{campaign_id}_simulate" if simulate else campaign_id)


def load_done(folder):
    """{(param_set_id, replicate): row_hash} of measurements already saved with status ok."""
    path = Path(folder) / "measurements.csv"
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as stream:
        return {(r["param_set_id"], int(r["replicate"])): r.get("row_hash", "")
                for r in csv.DictReader(stream) if r.get("status") == "ok"}


def append_measurement(folder, record):
    """One line, flushed to disk before the next measurement starts."""
    path = Path(folder) / "measurements.csv"
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerow(record)
        stream.flush()
        os.fsync(stream.fileno())


def changed_rows(plan, done):
    """Rows measured before with different parameters: skipped, never mixed into one set."""
    hashes = {}
    for (set_id, _), saved in done.items():
        hashes.setdefault(set_id, saved)
    return [r["param_set_id"] for r in plan if hashes.get(r["param_set_id"]) not in (None, "", r["hash"])]


def out_of_time(started, limit_s, slowest_s):
    """True when the next measurement (as slow as the slowest so far) would pass the limit."""
    if not limit_s:
        return False
    return time.monotonic() - started + slowest_s + END_RESERVE_S > limit_s


def validate_config(config, lash_e):
    if config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes do not match.")
    vial_file = config["INPUT_VIAL_STATUS_FILE"]
    if not vial_file or Path(vial_file).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during review. Restart with the new vial file.")
    if not Path(config["PARAMETER_FILE"]).is_file():
        raise FileNotFoundError(config["PARAMETER_FILE"])
    if not str(config["CAMPAIGN_ID"]).strip() or any(c in str(config["CAMPAIGN_ID"]) for c in r'\/:*?"<>|'):
        raise ValueError("CAMPAIGN_ID must be a plain folder name.")
    if type(config["REPLICATES"]) is not int or config["REPLICATES"] < 1:
        raise ValueError("REPLICATES must be a positive integer.")
    if not isinstance(config["SOURCE_VIALS"], dict):
        raise ValueError("SOURCE_VIALS must map liquid -> [vial names].")
    for key in ("VIAL_LOW_THRESHOLD_ML", "MAX_RUN_MINUTES"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number.")
    if not isinstance(config["RESERVOIR_REFILLS"], dict) or not all(
            type(v) is int and v >= 0 for v in config["RESERVOIR_REFILLS"].values()):
        raise ValueError("RESERVOIR_REFILLS must map liquid -> reservoir index.")
    value = config["REFILL_TO_ML"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("REFILL_TO_ML must be a positive number.")
    if not isinstance(config["APPLY_BLOWOUT_SPEED"], bool):
        raise ValueError("APPLY_BLOWOUT_SPEED must be true or false.")
    if type(config["MAX_ROWS_FOR_SIMULATION"]) is not int or config["MAX_ROWS_FOR_SIMULATION"] < 1:
        raise ValueError("MAX_ROWS_FOR_SIMULATION must be a positive integer.")


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


def refill_sources(robot, plan, groups, reservoirs, target_ml, logger):
    """Top up the source vials of liquids that have a reservoir, so they start full. {vial: mL added}."""
    pending = {key[0] for key, _ in groups}
    added = {}
    for row in plan:
        liquid = row["liquid"]
        if liquid not in reservoirs or liquid not in pending:
            continue
        for vial in row["vials"]:
            if vial in added:
                continue
            added[vial] = robot.refill_vial(vial, target_ml, reservoir=reservoirs[liquid], liquid=liquid) or 0.0
            logger.info(f"{vial}: topped up {added[vial]:.2f} mL from reservoir {reservoirs[liquid]}")
    return added


# ── the run ───────────────────────────────────────────────────────────────────

class _Stop(Exception):
    """Stop the campaign cleanly (everything measured stays saved)."""


class _Campaign:
    def __init__(self, lash_e, config, folder):
        from pipetting_data import embedded_calibration_validation as validation
        self.lash_e, self.robot, self.logger = lash_e, lash_e.nr_robot, lash_e.logger
        self.c, self.folder, self.validation = config, folder, validation
        self.started, self.slowest = time.monotonic(), 0.0
        self.run_started = datetime.now().isoformat(timespec="seconds")
        self.tips_used, self.saved, self.unstable = 0, 0, 0
        self.low_run = self.failed_run = 0
        self.current_vial = {}  # liquid -> index into its vials
        self.placed = set()     # vials this run put on the clamp

    def source(self, row):
        """The liquid's current source vial; moves on (vial home) below the threshold."""
        vials = row["vials"]
        i = self.current_vial.get(row["liquid"], 0)
        while i < len(vials) and _vial_volume(self.robot, vials[i]) < self.c["VIAL_LOW_THRESHOLD_ML"]:
            self.logger.info(f"{vials[i]} is below {self.c['VIAL_LOW_THRESHOLD_ML']} mL; next source vial")
            if self.robot.get_vial_info(vials[i], "location") == "clamp":
                if self.robot.HELD_PIPET_TYPE:
                    self.robot.remove_pipet()
                self.robot.return_vial_home(vials[i])
            i += 1
        self.current_vial[row["liquid"]] = i
        if i >= len(vials):
            raise _Stop(f"no {row['liquid']} left above {self.c['VIAL_LOW_THRESHOLD_ML']} mL "
                        f"in {', '.join(vials)}")
        return vials[i]

    def scale_ok(self, row):
        threshold = self.validation.calculate_quality_threshold(row["volume_ml"])
        return self.validation._scale_ready(self.lash_e, row["liquid"], row["tip"], threshold)

    def place(self, vial):
        """Put the vial on the clamp (the scale), as the embedded validation does; True if it moved.
        Any tip is dropped first (a vial can't move under a held tip) and another source vial
        on the clamp goes home."""
        if self.robot.get_vial_info(vial, "location") == "clamp":
            return False
        if self.robot.HELD_PIPET_TYPE:
            self.robot.remove_pipet()
        for other in self.placed - {vial}:
            if self.robot.get_vial_info(other, "location") == "clamp":
                self.robot.return_vial_home(other)
        self.robot.move_vial_to_location(vial, "clamp", 0)
        self.placed.add(vial)
        return True

    def run_group(self, key, items):
        liquid, tip, _ = key
        if self.robot.HELD_PIPET_TYPE:
            self.robot.remove_pipet()
        fresh, conditioned, checked = True, False, False
        largest = max(row["volume_ml"] for row, _ in items)
        for row, replicate in items:
            if out_of_time(self.started, self.c["MAX_RUN_MINUTES"] * 60, self.slowest):
                raise _Stop("out of time (MAX_RUN_MINUTES)")
            vial = self.source(row)
            if self.place(vial):
                fresh, conditioned = True, False
            if not checked:  # scale check with the vial on it, before a tip is used
                if not self.scale_ok(row):
                    raise _Stop("the scale is unstable")
                checked = True
            if fresh:
                self.tips_used += 1
                if row["condition"] and not row["refill"]:
                    spec = self.robot.PIPET_TIPS[tip]
                    volume = min(1.2 * largest, spec["volume"] - spec.get("capacity_margin", 0))
                    self.robot.condition_tip(vial, volume_per_cycle_mL=volume, specified_tip=tip)
                    conditioned = True
            record = self.measure(row, replicate, vial, tip, fresh, conditioned)
            append_measurement(self.folder, record)
            self.saved += 1
            fresh = row["refill"]
            if row["refill"]:
                self.robot.remove_pipet()
            self.check_faults(row, record)
            if self.saved % 20 == 0:
                _send_workflow_slack(self.lash_e, f"{_WORKFLOW_NAME} {self.c['CAMPAIGN_ID']}: "
                                                  f"{self.saved} measurements saved")
        if self.robot.HELD_PIPET_TYPE:
            self.robot.remove_pipet()

    def measure(self, row, replicate, vial, tip, fresh, conditioned):
        params, volume = row["params"], row["volume_ml"]
        t0 = time.monotonic()
        self.robot.aspirate_from_vial(vial, volume, parameters=params, liquid=None, specified_tip=tip)
        result = self.robot.dispense_into_vial(
            vial, volume, parameters=params, liquid=None, measure_weight=True,
            continuous_mass_monitoring=True, save_mass_data=not self.lash_e.simulate,
            record_measurement=False)
        elapsed = time.monotonic() - t0
        self.slowest = max(self.slowest, elapsed)
        mass, stability = result if isinstance(result, tuple) else (result, None)
        stability = stability or {}
        if self.lash_e.simulate and not mass:
            mass = volume * row["density"]  # nominal: simulation has no balance
        status = "ok" if mass is not None else "measurement_failed"
        acceptable = (True if self.lash_e.simulate or not stability else
                      bool(self.validation._evaluate_measurement(
                          stability, self.validation.calculate_quality_threshold(volume), log_result=False)))
        measured_ml = mass / row["density"] if mass is not None else None
        env = self.validation._get_latest_environmental_data() or {}
        record = {
            "param_set_id": row["param_set_id"], "replicate": replicate,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "status": status,
            "measurement_acceptable": acceptable, "liquid_type": row["liquid"], "density_used": row["density"],
            "source_vial": vial, "tip_type": tip, "tip_number": self.tips_used, "new_tip": fresh,
            "conditioned": conditioned, "target_ul": round(volume * 1000, 4), "target_volume_ml": volume,
            "mass_difference_g": mass,
            "measured_ul": round(measured_ml * 1000, 4) if measured_ml is not None else "",
            "measured_volume_ml": measured_ml if measured_ml is not None else "",
            "accuracy_pct": round((measured_ml / volume - 1) * 100, 3) if measured_ml is not None else "",
            "elapsed_s": round(elapsed, 2),
            **{field: getattr(params, field) for field, _ in PARAMETER_COLUMNS.values()},
            **{k: stability.get(k, "") for k in ("pre_baseline_std", "post_baseline_std", "pre_stable_count",
                                                 "pre_total_count", "post_stable_count", "post_total_count")},
            "temp_c": env.get("temp_c", ""), "humidity_pct": env.get("humidity_pct", ""),
            "pressure_pa": env.get("pressure_pa", ""), "validation_type": "forced_parameters",
            "row_hash": row["hash"], "simulated": self.lash_e.simulate, "run_started": self.run_started,
        }
        self.logger.info(f"{row['param_set_id']} rep {replicate}: target {volume * 1000:.1f} uL -> "
                         f"{record['measured_ul']} uL ({tip}, {elapsed:.1f} s)")
        return record

    def check_faults(self, row, record):
        if record["status"] != "ok":
            self.failed_run += 1
            if self.failed_run >= FAILED_RUN:
                raise _Stop(f"{FAILED_RUN} measurements in a row returned no mass")
            return
        self.failed_run = 0
        if not record["measurement_acceptable"]:
            self.unstable += 1
            self.validation._notify(self.lash_e, f"{_WORKFLOW_NAME} {self.c['CAMPAIGN_ID']}: scale unstable at "
                                                 f"{row['param_set_id']} (recorded, not retried)")
            if not self.scale_ok(row):
                raise _Stop("the scale stayed unstable")
        tiny = record["measured_ul"] != "" and record["measured_ul"] < BROKEN_FRACTION * record["target_ul"]
        self.low_run = self.low_run + 1 if tiny else 0
        if self.low_run >= BROKEN_RUN:
            raise _Stop(f"{BROKEN_RUN} measurements in a row under {BROKEN_FRACTION:.0%} of target "
                        "(empty vial, missing tip or clog?)")


def execute(config=None, show_gui=True):
    """Pipette every pending row of PARAMETER_FILE with exactly its parameters and weigh it."""
    lash_e = Lash_E(
        initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    logger = lash_e.logger
    c = lash_e.workflow_config
    robot = lash_e.nr_robot

    try:
        validate_config(c, lash_e)
        robot.APPLY_BLOWOUT_SPEED = c["APPLY_BLOWOUT_SPEED"]
        from pipetting_data.liquid_registry import load_liquids
        rows = read_rows(c["PARAMETER_FILE"])
        if lash_e.simulate:
            rows = rows[:c["MAX_ROWS_FOR_SIMULATION"]]
        plan, rejected = check_rows(rows, robot, load_liquids(), c["SOURCE_VIALS"], c["VIAL_LOW_THRESHOLD_ML"])
        for set_id, reason in rejected:
            logger.warning(f"Row {set_id} skipped: {reason}")
        if not plan:
            raise ValueError("No usable rows: " + "; ".join(f"{i}: {r}" for i, r in rejected))

        folder = campaign_dir(c["CAMPAIGN_ID"], lash_e.simulate)
        folder.mkdir(parents=True, exist_ok=True)
        run_dir = folder / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        lash_e.save_run_info(run_dir)
        file_hash = hashlib.sha256(Path(c["PARAMETER_FILE"]).read_bytes()).hexdigest()[:16]
        campaign_file = folder / "campaign.yaml"
        info = yaml.safe_load(campaign_file.read_text(encoding="utf-8")) if campaign_file.is_file() else {}
        info.setdefault("campaign_id", c["CAMPAIGN_ID"])
        info.setdefault("runs", []).append({"started": datetime.now().isoformat(timespec="seconds"),
                                             "parameter_file": str(c["PARAMETER_FILE"]), "file_sha256": file_hash,
                                             "vial_file": str(c["INPUT_VIAL_STATUS_FILE"]),
                                             "apply_blowout_speed": c["APPLY_BLOWOUT_SPEED"],
                                             "rejected": dict(rejected)})
        campaign_file.write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")

        done = load_done(folder)
        changed = changed_rows(plan, done)
        if changed:
            logger.warning(f"Rows changed since they were measured, skipped: {', '.join(changed)}")
            plan = [r for r in plan if r["param_set_id"] not in changed]
        groups = pending_groups(plan, set(done), c["REPLICATES"])
        refilled = refill_sources(robot, plan, groups, c["RESERVOIR_REFILLS"], c["REFILL_TO_ML"], logger)
        total = sum(len(items) for _, items in groups)
        logger.info(f"Campaign {c['CAMPAIGN_ID']}: {total} measurements pending in {len(groups)} tip groups "
                    f"({len(done)} already saved, {len(rejected)} rows rejected); output {folder}")
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} {c['CAMPAIGN_ID']} started | {total} measurements "
                                     f"pending | limit {c['MAX_RUN_MINUTES'] or 'none'} min")

        campaign = _Campaign(lash_e, c, folder)
        stopped = None
        try:
            for key, items in groups:
                campaign.run_group(key, items)
        except _Stop as why:
            stopped = str(why)
            logger.warning(f"Campaign stopped: {stopped}")
        if robot.HELD_PIPET_TYPE:
            robot.remove_pipet()
        for row in plan:
            for vial in row["vials"]:
                if robot.get_vial_info(vial, "location") == "clamp":
                    robot.return_vial_home(vial)
        robot.move_home()
        left = total - campaign.saved
        summary = {"campaign_id": c["CAMPAIGN_ID"], "folder": str(folder), "saved": campaign.saved,
                   "pending": left, "tips_used": campaign.tips_used, "unstable": campaign.unstable,
                   "rejected": dict(rejected), "changed_rows_skipped": changed, "stopped": stopped,
                   "refilled": refilled}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} {c['CAMPAIGN_ID']}: {campaign.saved} saved, {left} pending, "
                                     f"{campaign.tips_used} tips" + (f" | stopped: {stopped}" if stopped else ""))
        return summary

    except (Exception, KeyboardInterrupt) as e:
        logger.exception(f"Forced-parameter pipetting failed: {e}")
        from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
        cleanup_after_keyboard_interrupt(lash_e)
        if isinstance(e, KeyboardInterrupt):
            _send_workflow_slack(lash_e, f"{_WORKFLOW_NAME} interrupted by operator; measurements so far are saved")
        raise
    finally:
        robot.APPLY_BLOWOUT_SPEED = False  # never carries into the next job in the same process


if __name__ == "__main__":
    execute()
