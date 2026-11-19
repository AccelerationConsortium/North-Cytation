"""Validate every calibrated condition for a list of liquids.

For each liquid in LIQUIDS, every (tip, volume) row in its calibration file
(pipetting_data/optimal_conditions_<liquid>.csv) gets an embedded gravimetric
check (validate_pipetting_accuracy). The liquid is aspirated from and returned
to its own vial on the balance, so no receiving vial is needed. Surprising
checks update overaspirate_corrections.csv on live runs; base calibration CSVs
are never written. Simulation runs the same calls without saving data.

VIALS names the source vial for each liquid, in the same order as LIQUIDS.
Leave it empty when each vial is named after its liquid.
"""
import sys
sys.path.append("../utoronto_demo")  # Always first, before any local imports

import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from master_usdl_coordinator import Lash_E

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
LIQUIDS = ["water"]
VIALS = []  # Same order as LIQUIDS; empty = vial named after each liquid
REPLICATES = 4

_CONFIG_KEYS = ["SIMULATE", "INPUT_VIAL_STATUS_FILE", "LIQUIDS", "VIALS", "REPLICATES"]

_CONFIG_DOCS = {
    "SIMULATE": {"description": "Run without hardware; the scheduler sets this per job.",
                 "unit": None, "affects": []},
    "INPUT_VIAL_STATUS_FILE": {"description": "Vial layout CSV (repository-relative) for this run.",
                               "unit": None, "affects": ["vials", "liquids"], "shares": "vial_layout"},
    "LIQUIDS": {"description": "Liquids (liquids.yaml names) whose calibrations are all validated.",
                "unit": None, "affects": ["duration", "tips", "liquids"], "ask": True,
                "effect": "one validation series per calibrated tip of each liquid, covering every calibrated volume"},
    "VIALS": {"description": "Source vial for each liquid, same order as LIQUIDS; empty uses vials named after the liquids.",
              "unit": None, "affects": ["vials"]},
    "REPLICATES": {"description": "Check replicates per calibrated volume.",
                   "unit": None, "affects": ["duration", "liquids"],
                   "effect": "time and liquid scale with replicates x calibrated volumes"},
}

_PRODUCT_VIALS = []


def calibrated_conditions(liquid):
    """Return {tip: sorted calibrated volumes in uL} from the liquid's calibration files."""
    from pipetting_data.pipetting_wizard import PipettingWizard

    wizard = PipettingWizard()
    conditions = {}
    for path in wizard.find_calibration_files(liquid):
        df = wizard.load_calibration_data(path)
        if df is None or "tip_type" not in df.columns:
            continue
        volumes = pd.to_numeric(df["volume_target"], errors="coerce")
        for tip, volume in zip(df["tip_type"], volumes):
            if isinstance(tip, str) and math.isfinite(volume) and volume > 0:
                conditions.setdefault(tip, set()).add(float(volume))
    return {tip: sorted(volumes) for tip, volumes in sorted(conditions.items())}


def plan_validations(config, lash_e):
    """Check inputs and return [(vial, liquid, tip, volumes_ul)] before anything moves."""
    from pipetting_data.liquid_registry import get_liquid

    if config["SIMULATE"] != lash_e.simulate:
        raise ValueError("Workflow and controller simulation modes do not match.")
    if Path(config["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(lash_e.nr_robot.VIAL_FILE).resolve():
        raise ValueError("Vial file changed during review. Restart with the new vial file.")
    if type(config["REPLICATES"]) is not int or config["REPLICATES"] < 1:
        raise ValueError("REPLICATES must be a positive integer.")
    liquids = config["LIQUIDS"]
    if not liquids:
        raise ValueError("LIQUIDS is empty.")
    vials = config["VIALS"] or liquids
    if len(vials) != len(liquids):
        raise ValueError("VIALS must be empty or list one vial per liquid, in LIQUIDS order.")

    plan = []
    for vial, liquid in zip(vials, liquids):
        get_liquid(liquid)
        lash_e.nr_robot.get_vial_info(vial, "location")
        conditions = calibrated_conditions(liquid)
        if not conditions:
            raise ValueError(f"No tip-specific calibration rows for {liquid}.")
        for tip, volumes in conditions.items():
            plan.append((vial, liquid, tip, volumes))
    return plan


def execute(config=None, show_gui=True):
    """Validate every calibrated (tip, volume) condition of each liquid in LIQUIDS."""
    lash_e = Lash_E(
        initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    logger = lash_e.logger
    c = lash_e.workflow_config

    try:
        plan = plan_validations(c, lash_e)
        from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = "SIMULATION" if lash_e.simulate else "LIVE"
        output_dir = Path(__file__).resolve().parents[1] / "output" / _WORKFLOW_NAME / f"{timestamp}_{mode}"
        output_dir.mkdir(parents=True, exist_ok=False)
        lash_e.save_run_info(output_dir)
        logger.info(f"Calibration validation output: {output_dir}")

        results = {}
        for vial, liquid, tip, volumes in plan:
            logger.info(f"Validating {liquid} from {vial} ({tip}): {volumes} uL")
            results[(vial, liquid, tip)] = validate_pipetting_accuracy(
                lash_e=lash_e,
                source_vial=vial,
                destination_vial=vial,
                liquid_type=liquid,
                volumes_ml=[v / 1000 for v in volumes],
                replicates=c["REPLICATES"],
                output_folder=str(output_dir / f"{liquid}_{tip}"),
                plot_title=f"{liquid} ({tip}): all calibrated volumes",
                switch_pipet=False,
                save_raw_data=not lash_e.simulate,
                condition_tip_enabled=True,
                conditioning_volume_ul=min(150 if tip == "small_tip" else 800, max(volumes)),
                tip_type=tip,
            )
        lash_e.nr_robot.move_home()
        logger.info(f"Validated {len(plan)} liquid/tip series")
        return results

    except (Exception, KeyboardInterrupt) as e:
        logger.exception(f"Calibration validation failed: {e}")
        from workflows.helpers.interrupt_cleanup import cleanup_after_keyboard_interrupt
        cleanup_after_keyboard_interrupt(lash_e)
        raise


if __name__ == "__main__":
    execute()
