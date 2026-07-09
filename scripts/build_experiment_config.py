"""One-time script to rewrite experiment_config.yaml with logical section order
and human-readable comments using ruamel.yaml round-trip mode.

Run from the workspace root:
    python scripts/build_experiment_config.py
"""
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
yaml.width = 120


def cm(**kwargs):
    d = CommentedMap()
    for k, v in kwargs.items():
        d[k] = v
    return d


def cs(items):
    s = CommentedSeq(items)
    s.fa.set_flow_style()
    return s


cfg = CommentedMap()

# ── experiment ───────────────────────────────────────────────────────────────
exp = CommentedMap()
exp["liquid"] = "ethanol"
exp["volume_targets_ml"] = [0.050]
exp["simulate"] = False
exp["hardware_protocol"] = "calibration_protocol_northrobot"
exp["simulation_protocol"] = "calibration_protocol_simulated"
exp["name"] = "multi_volume_calibration_test"
exp["description"] = "Testing calibration across 4 volumes"
exp["random_seed"] = 30
exp["max_total_measurements"] = 96
exp["max_measurements_first_volume"] = 96
exp["max_replicates_per_trial"] = 3
exp["num_screening_trials"] = 8
exp["two_point_calibration_replicates"] = 0

fp = CommentedMap()
fp["asp_disp_cycles"] = 0
fp.yaml_set_comment_before_after_key(
    "asp_disp_cycles",
    before=(
        "Parameters listed here are held constant and excluded from optimization.\n"
        "Their bounds/defaults in hardware_parameters are kept for reference.\n"
        "Remove a key from this block to let the optimizer tune it."
    ),
    indent=4,
)
fp["post_asp_air_vol"] = 0
fp["post_retract_wait_time"] = 5.0
fp["retract_speed"] = 5.0
fp["dispense_wait_time"] = 3.0
exp["fixed_parameters"] = fp

cfg["experiment"] = exp
cfg.yaml_set_comment_before_after_key(
    "experiment",
    before="\n── Main experiment settings ───────────────────────────────────────────────\nEdit liquid, volume_targets_ml, simulate, and hardware_protocol for each run.",
)

# ── calibration_parameters ───────────────────────────────────────────────────
oa = CommentedMap()
oa["bounds"] = cs([0.0, 0.025])
oa["default"] = 0.004
oa["type"] = "float"
oa["volume_dependent"] = True
oa["max_fraction_of_target"] = 1.0
oa["round_to_nearest"] = 0.0001
oa["description"] = (
    "Extra volume drawn by the syringe pump, to compensate for lack of volume aspired "
    "or dispensed for any reason. Can be used to crudely correct accuracy at minimal "
    "cost to time. May reduce precision if overused."
)

cp = CommentedMap()
cp["overaspirate_vol"] = oa
cfg["calibration_parameters"] = cp
cfg.yaml_set_comment_before_after_key(
    "calibration_parameters",
    before=(
        "\n── Calibration target parameter ───────────────────────────────────────────\n"
        "overaspirate_vol is treated separately from hardware_parameters and is always tuned."
    ),
)

# ── hardware_parameters ──────────────────────────────────────────────────────
def make_param(bounds, default, ptype, description, round_to, time_affecting, **extras):
    p = CommentedMap()
    p["bounds"] = cs(bounds)
    p["default"] = default
    p["type"] = ptype
    p["description"] = description
    p["round_to_nearest"] = round_to
    p["time_affecting"] = time_affecting
    for k, v in extras.items():
        p[k] = v
    return p


hp = CommentedMap()
hp["aspirate_speed"] = make_param(
    [2, 30], 10, "integer",
    "Aspiration speed (relative units) - Higher values = SLOWER operation. "
    "Increasing aspirate_speed can help with thicker liquids for accuracy and precision "
    "but increases operation time. Slower aspiration reduces cavitation and bubble "
    "formation in viscous liquids.",
    1, True,
)
hp["dispense_speed"] = make_param(
    [2, 30], 10, "integer",
    "Dispense speed (relative units) - Higher values = SLOWER operation. "
    "Increasing dispense_speed can help with thicker liquids for accuracy, but making "
    "it low (fast) may help expel liquids quickly which can be helpful since it is "
    "simpler to push out than pull in. Fast dispense can reduce dripping time.",
    1, True,
)
hp["aspirate_wait_time"] = make_param(
    [0.0, 30.0], 10.0, "float",
    "Wait time after aspiration in liquid (seconds) - Allows pressure equilibration "
    "and meniscus stabilization after liquid is drawn into tip. Critical for viscous "
    "liquids >1000 cP to prevent incomplete aspiration and ensure accurate volume transfer.",
    0.1, True,
)
hp["dispense_wait_time"] = make_param(
    [0, 15.0], 1.5, "float",
    "Wait time after dispense (seconds) - Allows complete liquid evacuation from tip "
    "and pressure equilibration. Longer waits help ensure all liquid is expelled, "
    "especially important for viscous liquids that flow slowly.",
    0.1, True,
)
hp["post_asp_air_vol"] = make_param(
    [0.0, 0.1], 0.0, "float",
    "Post-aspiration air gap volume (mL) - Creates a gas bubble after your liquid. "
    "Seems to help more for larger volumes with large tips but can hurt with thick liquids. "
    "Provides barrier against backflow during transport.",
    0.001, False,
)
hp["pre_asp_air_vol"] = make_param(
    [0.0, 0.5], 0.0, "float",
    "Pre-aspiration air volume (mL) - A faster way to achieve blowout and push liquid "
    "from the tip, but it is more constrained and can cause liquid to get stuck in the tip. "
    "Creates air cushion before liquid aspiration.",
    0.001, False, volume_dependent=True,
)
hp["blowout_vol"] = make_param(
    [0.0, 0.5], 0.0, "float",
    "Blowout volume after dispense (mL) - A neat and simple way to push out extra liquid "
    "from the tip afterwards. If dispense_speed is high (slow) then this blowout will take "
    "a long time. More effective for viscous liquids that stick to tips.",
    0.001, False,
)
hp["post_retract_wait_time"] = make_param(
    [0.0, 15.0], 5.0, "float",
    "Wait time after aspiration in liquid (seconds) - Allows thick liquid to drip off the "
    "outside of tip. Only relevant for quite viscous liquids that need time for external "
    "dripping, e.g. ~5s for glycerol-level viscosity.",
    0.1, True,
)

cfg["hardware_parameters"] = hp
cfg.yaml_set_comment_before_after_key(
    "hardware_parameters",
    before=(
        "\n── Hardware parameter search space ────────────────────────────────────────\n"
        "Parameters in experiment.fixed_parameters are excluded from optimization\n"
        "but their bounds/defaults stay here for reference."
    ),
)

# ── optimization ─────────────────────────────────────────────────────────────
opt = CommentedMap()

obj = CommentedMap()
obj["accuracy_weight"] = 0.4
obj["precision_weight"] = 0.5
obj["time_weight"] = 0.1
obj.yaml_set_comment_before_after_key("accuracy_weight", before="weights must sum to 1.0", indent=4)
opt["objectives"] = obj

opt_thresh = CommentedMap()
opt_thresh["deviation_threshold_pct"] = 50.0
opt_thresh["variability_threshold_pct"] = 25.0
opt_thresh["time_threshold_s"] = 120.0
opt["objective_thresholds"] = opt_thresh

optimizer = CommentedMap()
optimizer["type"] = "multi_objective"
optimizer["backend"] = "qNEHVI"
optimizer["backend_subsequent"] = "GPEI"
opt["optimizer"] = optimizer

stop = CommentedMap()
stop["min_good_trials"] = 100
opt["stopping_criteria"] = stop

inherit = CommentedMap()
inherit["enabled"] = True
inherit["carry_optimizer_state"] = False
opt["parameter_inheritance"] = inherit

opt["use_range_based_variability"] = False

fvc = CommentedMap()
fvc["enabled"] = False
fvc["skip_if_good_trial"] = True
fvc["description"] = (
    "Performs final 2-point overaspirate calibration on first volume "
    "before proceeding to next volume"
)
opt["first_volume_final_calibration"] = fvc

llm_opt = CommentedMap()
llm_opt["enabled"] = False
llm_opt["config_path"] = "calibration_screening_llm_template.json"
opt["llm_optimization"] = llm_opt
opt.yaml_set_comment_before_after_key(
    "llm_optimization",
    before="LLM-assisted optimization (experimental - requires API key)",
    indent=2,
)

cfg["optimization"] = opt
cfg.yaml_set_comment_before_after_key(
    "optimization",
    before="\n── Optimizer settings ─────────────────────────────────────────────────────",
)

# ── output ────────────────────────────────────────────────────────────────────
out = CommentedMap()
out["base_directory"] = "output/"
out["export_optimal_conditions"] = True
out["generate_plots"] = True
out["save_raw_measurements"] = True
cfg["output"] = out
cfg.yaml_set_comment_before_after_key(
    "output",
    before="\n── Output ──────────────────────────────────────────────────────────────────",
)

# ── validation ────────────────────────────────────────────────────────────────
val = CommentedMap()
val["optimal_conditions_file"] = "optimized_parameters/optimal_conditions_example.csv"
val.yaml_set_comment_before_after_key(
    "optimal_conditions_file",
    before="Set to your calibration output path before running run_validation.py",
    indent=2,
)
val["volumes_ml"] = [0.05]
val["replicates_per_volume"] = 5
val["output_directory"] = "validation/"
val["generate_plots"] = True
val["save_raw_data"] = True
sc = CommentedMap()
sc["use_volume_tolerances"] = True
val["success_criteria"] = sc
cfg["validation"] = val
cfg.yaml_set_comment_before_after_key(
    "validation",
    before="\n── Validation settings ─────────────────────────────────────────────────────",
)

# ── screening ─────────────────────────────────────────────────────────────────
scr = CommentedMap()
scr["use_llm_suggestions"] = False
scr["llm_config_path"] = "calibration_screening_llm_template.json"
ed = CommentedMap()
ed["enabled"] = False
ed["data_path"] = "input_data/external_calibration_data.csv"
ed["volume_filter_ml"] = None
ed["liquid_filter"] = None
ed["required_columns"] = []
scr["external_data"] = ed
cfg["screening"] = scr
cfg.yaml_set_comment_before_after_key(
    "screening",
    before=(
        "\n── Advanced: screening options ────────────────────────────────────────────\n"
        "LLM suggestions and external data injection (both disabled by default)."
    ),
)

# ── tolerances ────────────────────────────────────────────────────────────────
tol = CommentedMap()
sim_tol = CommentedMap()
sim_tol["deviation_multiplier"] = 1.5
sim_tol["variability_multiplier"] = 1.5
tol["simulation"] = sim_tol

ranges = CommentedSeq()
for vmin, vmax, pct in [
    (200, 1000, 1.0),
    (60, 199, 2.0),
    (20, 59, 3.0),
    (2.5, 19, 10.0),
    (0, 2.499, 10.0),
]:
    r = CommentedMap()
    r["volume_min_ul"] = vmin
    r["volume_max_ul"] = vmax
    r["tolerance_pct"] = pct
    ranges.append(r)
tol["volume_ranges"] = ranges
cfg["tolerances"] = tol
cfg.yaml_set_comment_before_after_key(
    "tolerances",
    before="\n── Pass/fail tolerances by volume range ────────────────────────────────────",
)

# ── adaptive_measurement ──────────────────────────────────────────────────────
adp = CommentedMap()
adp["enabled"] = True
adp["base_replicates"] = 1
adp["deviation_threshold_pct"] = 100.0
adp["penalty_variability_pct"] = 100.0
cfg["adaptive_measurement"] = adp
cfg.yaml_set_comment_before_after_key(
    "adaptive_measurement",
    before=(
        "\n── Advanced: adaptive replication ──────────────────────────────────────────\n"
        "Automatically run extra replicates when a measurement looks noisy."
    ),
)

out_path = Path("sdl_pipette_calibration/experiment_config.yaml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print(f"Written: {out_path}")
