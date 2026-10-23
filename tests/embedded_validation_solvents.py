"""Run four embedded calibration validations using the existing calibration vials.

From the repository root: python tests/embedded_validation_solvents.py
Startup uses the template's configuration and vial review GUI. Confirm SIMULATE
there before running hardware. Each liquid is aspirated and returned to its own
vial on the balance, so no separate receiving vial is needed.

Stage 1 tests existing liquid calibration parameters. When outside tolerance,
adaptive correction runs Stage 2 (crossing) and Stage 3 (interpolation), selects
the best stage, and updates the liquid calibration CSV on live runs. Simulation
executes the same calls without saving measurements or calibration changes.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from master_usdl_coordinator import Lash_E
from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy

_WORKFLOW_NAME = Path(__file__).stem

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "status/calibration_vials_short.csv"
LIQUIDS = ["acetone", "ethyl_acetate"]
SMALL_VOLUMES_ML = [0.010, 0.020, 0.050]
LARGE_VOLUMES_ML = [0.200, 0.500, 0.800]
REPLICATES = 5
SMALL_CONDITIONING_UL = 150
LARGE_CONDITIONING_UL = 900


def execute(config=None, show_gui=True):
    """Run acetone small/large, then ethyl acetate small/large validation."""
    lash_e = Lash_E(
        initialize_biotek=False,
        workflow_globals=globals(), workflow_name=_WORKFLOW_NAME,
        config=config, show_gui=show_gui,
    )
    if not lash_e._workflow_should_continue:
        return None
    c = lash_e.workflow_config
    robot = lash_e.nr_robot

    try:
        if c["SIMULATE"] != lash_e.simulate:
            raise ValueError("Workflow and controller simulation modes do not match.")
        if Path(c["INPUT_VIAL_STATUS_FILE"]).resolve() != Path(robot.VIAL_FILE).resolve():
            raise ValueError("Vial file changed during review. Restart with the new vial file.")
        if type(c["REPLICATES"]) is not int or c["REPLICATES"] < 1:
            raise ValueError("REPLICATES must be a positive integer.")
        # Check all sources before starting the first validation.
        for liquid in c["LIQUIDS"]:
            robot.get_vial_info(liquid, "location")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = "SIMULATION" if lash_e.simulate else "LIVE"
        output_dir = Path(__file__).resolve().parents[1] / "output" / _WORKFLOW_NAME / f"{timestamp}_{mode}"
        output_dir.mkdir(parents=True, exist_ok=False)
        lash_e.logger.info(f"Embedded validation output: {output_dir}")
        results = {}
        for liquid in c["LIQUIDS"]:
            for size, volumes, conditioning_ul in (
                ("small", c["SMALL_VOLUMES_ML"], c["SMALL_CONDITIONING_UL"]),
                ("large", c["LARGE_VOLUMES_ML"], c["LARGE_CONDITIONING_UL"]),
            ):
                name = f"{liquid}_{size}"
                lash_e.logger.info(f"Starting {name}: {[v * 1000 for v in volumes]} uL")
                if robot.HELD_PIPET_TYPE is not None:
                    robot.remove_pipet()
                # Rack positions 36/38 cannot be reached while holding a tip.
                # Position the vial before pickup; validation then keeps it in clamp.
                robot.move_vial_to_location(liquid, "clamp", 0)
                robot.get_pipet(f"{size}_tip")
                try:
                    results[name] = validate_pipetting_accuracy(
                        lash_e=lash_e,
                        source_vial=liquid,
                        destination_vial=liquid,
                        liquid_type=liquid,
                        volumes_ml=volumes,
                        replicates=c["REPLICATES"],
                        output_folder=str(output_dir / name),
                        plot_title=f"{liquid} ({size} tip): embedded validation",
                        switch_pipet=False,
                        save_raw_data=not lash_e.simulate,
                        condition_tip_enabled=True,
                        conditioning_volume_ul=conditioning_ul,
                        adaptive_correction=True,
                        compensate_overvolume=False,
                        tip_type=f"{size}_tip",
                    )
                finally:
                    if robot.HELD_PIPET_TYPE is not None:
                        robot.remove_pipet()
                robot.return_vial_home(liquid)
        robot.move_home()
        lash_e.logger.info("All four embedded validations completed")
        return results
    except (Exception, KeyboardInterrupt):
        lash_e.logger.exception("Embedded validation interrupted or failed")
        # Preserve the original failure even if cleanup also fails.
        for label, action in (
            ("remove_pipet", lambda: robot.remove_pipet() if robot.HELD_PIPET_TYPE is not None else None),
            ("move_home", robot.move_home),
        ):
            try:
                action()
            except Exception:
                lash_e.logger.exception(f"Cleanup failed: {label}")
        raise


if __name__ == "__main__":
    execute()
