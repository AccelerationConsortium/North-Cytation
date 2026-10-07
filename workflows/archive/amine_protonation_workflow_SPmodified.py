"""
Amine Protonation Workflow

Concept:
    Protonate amines (dissolved in ethanol) by adding HCl to a stirring solution
    in the photoreactor, then add a metal salt, vortex, and finally deposit a
    droplet onto a slide.

Per-experiment inputs (workflow function args):
    - reactor_vial:         Vial that holds the reaction. Assumed to be
                            pre-loaded with 5 mL of ethanol prior to workflow
                            start. Uses an open cap for now so it can be
                            pipetted into while sitting in the photoreactor.
    - amine_vial + volume:  Source vial and volume of amine-in-ethanol stock.
    - metal_salt_vial + volume: Source vial and volume of metal salt stock.

Fixed constants (not per-experiment parameters):
    - HCl vial identity and HCl addition volume (5 uL of 6 M HCl).
    - Photoreactor slot used for stirring.
    - Stirring RPM.

Notes:
    - Slide dispensing is left as a stub (locations not yet defined).
    - No measurement or data saving is performed here; Slack messages are used
      to communicate progress.
    - Run home_robot_components() once before the first experiment, and call
      move_home() after the last one. Multiple experiments can share a single
      Lash_E object.
"""

import sys
sys.path.append("../utoronto_demo")

try:
    import slack_agent
    _SLACK_AVAILABLE = True
except Exception:
    _SLACK_AVAILABLE = False

from master_usdl_coordinator import Lash_E
import robot_state.Locator as Locator


# ================================================================================
# FIXED CONFIGURATION (not workflow parameters)
# ================================================================================

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/amine_protonation_vials.csv"

SIMULATE = False  # Set to False for hardware execution

# Ethanol stock vial — 20 mL vial in the large_vial_rack, shared across all experiments.
# Volume is tracked automatically by the robot after each dispense.
# Update ETHANOL_VIAL to match the vial_name in the CSV.
ETHANOL_VIAL = "ethanol_stock"       # must be a large_vial_rack entry in the CSV
ETHANOL_LOW_VOL_ML = 2.0            # log a warning when remaining volume drops below this

# HCl addition
HCL_VIAL = "hcl_stock"
HCL_VOLUME_ML = 0.005
HCL_LIQUID = "6M_HCl"  # calibrated liquid class in master_pipetting_measurements.csv

# Pipette mixing — done at safe rack positions (43-47) on the 8 mL rack
# MIX_CYCLES applies to both the amine+ethanol mix and the post-HCl mix
USE_PIPETTE_MIX = False   # pipette up/down mixing
USE_VORTEX = True       # vortex after pipette mix (or instead of, if USE_PIPETTE_MIX=False)
MIX_VOLUME_ML = 1.000   # volume per aspirate/dispense cycle (mL)
MIX_CYCLES = 4          # number of pipette cycles (3-6 recommended)
VORTEX_TIME_S = 10      # vortex duration in seconds
VORTEX_SPEED = 70       # vortex speed (default 70)

# Slide heater temperature
SLIDE_HEATER_TEMP_C = 60
SLIDE_HEAT_SOAK_MIN = 10   # minutes to hold temperature after all droplets are dispensed

# Slide drop grid configuration
# Define the four physical corners of the slide array in robot units [gripper, elbow, shoulder, z].
# Positions are interpolated using bilinear interpolation so the grid fits the actual slide geometry
# even if the slide is slightly skewed or tilted.
# Corners must be calibrated in robot_state/Locator.py.
#
#   TOP_LEFT -------- TOP_RIGHT
#      |                  |
#   BOT_LEFT -------- BOT_RIGHT
#
# Change SLIDE_GRID_ROWS / SLIDE_GRID_COLS to resize the grid without re-calibrating corners.
SLIDE_GRID_ROWS = 3
SLIDE_GRID_COLS = 3
DROPLET_VOLUME_ML = 0.015          # 15 uL per drop
DROPLET_SAFE_RACK_INDEX = 43       # main_8mL_rack position used for small-tip droplet aspiration (safe range: 43-47)
DISPENSE_TO_SLIDE = False          # Set True to execute real robot dispense; False logs only
TEST_SLIDE_GRID = False            # Set True to run dispense_droplet_onto_slide across ALL grid positions (skips full workflow)
TEST_SLIDE_VIAL = "reactor_vial_1" # Vial to aspirate from during grid test


# ================================================================================
# EXPERIMENT LIST
# ================================================================================
# Add one dict per sample. Each dict maps directly to the keyword args of
# amine_protonation_workflow(). reactor_vial must match a vial_name in the CSV.
# vortex_time_s is optional (defaults to VORTEX_TIME_S if omitted).
#
# To run more samples:
#   1. Add rows to status/amine_protonation_vials.csv for the new reactor vials.
#   2. Add a corresponding dict here.

EXPERIMENTS = [
    {
        "reactor_vial":         "reactor_vial_1",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.154,
        "ethanol_volume_mL":    0.008,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.615,
        "hcl_volume_mL":        0.000,   # Vial_02: no separate HCl (excess from BiCl3 stock)
        "slide_position_index": 0,
    },
    {
        "reactor_vial":         "reactor_vial_2",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.272,
        "ethanol_volume_mL":    0.990,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.495,
        "hcl_volume_mL":        0.243,   # Vial_04: 242.57 uL
        "slide_position_index": 1,
    },
    {
        "reactor_vial":         "reactor_vial_3",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.364,
        "ethanol_volume_mL":    0.730,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.365,
        "hcl_volume_mL":        0.540,   # Vial_07: 540.15 uL
        "slide_position_index": 2,
    },
    {
        "reactor_vial":         "reactor_vial_4",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.311,
        "ethanol_volume_mL":    0.621,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.311,
        "hcl_volume_mL":        0.665,   # Vial_09: 664.60 uL
        "slide_position_index": 3,
    },
    {
        "reactor_vial":         "reactor_vial_5",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.270,
        "ethanol_volume_mL":    0.541,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.270,
        "hcl_volume_mL":        0.757,   # Vial_11: 756.76 uL
        "slide_position_index": 4,
    },
    {
        "reactor_vial":         "reactor_vial_6",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.226,
        "ethanol_volume_mL":    0.453,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.226,
        "hcl_volume_mL":        0.857,   # Vial_14: 857.47 uL
        "slide_position_index": 5,
    },
    {
        "reactor_vial":         "reactor_vial_7",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.195,
        "ethanol_volume_mL":    0.389,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.195,
        "hcl_volume_mL":        0.930,   # Vial_17: 929.96 uL
        "slide_position_index": 6,
    },
    {
        "reactor_vial":         "reactor_vial_8",
        "amine_vial":           "amine_1",
        "amine_volume_mL":      0.170,
        "ethanol_volume_mL":    0.341,
        "metal_salt_vial":      "metal_salt_1",
        "metal_salt_volume_mL": 0.171,
        "hcl_volume_mL":        0.985,   # Vial_20: 984.64 uL
        "slide_position_index": 7,
    }
]


# ================================================================================
# WORKFLOW
# ================================================================================

def _log_ethanol_volume(lash_e):
    """Log remaining ethanol volume and warn if below ETHANOL_LOW_VOL_ML."""
    vial_idx = lash_e.nr_robot.normalize_vial_index(ETHANOL_VIAL)
    remaining = float(lash_e.nr_robot.get_vial_info(vial_idx, 'vial_volume'))
    if remaining < ETHANOL_LOW_VOL_ML:
        lash_e.logger.warning(
            f"Ethanol stock low: {remaining*1000:.0f}uL remaining in {ETHANOL_VIAL} "
            f"(large_vial_rack) - consider refilling"
        )
    else:
        lash_e.logger.info(f"Ethanol stock: {remaining*1000:.0f}uL remaining in {ETHANOL_VIAL}")


def _release_tip_if_held(lash_e):
    """Remove any held pipette tip so the gripper is free to move vials."""
    if lash_e.nr_robot.HELD_PIPET_TYPE is not None:
        lash_e.logger.info("Releasing held tip before vial move")
        lash_e.nr_robot.remove_pipet()


def _mix_reactor_vial(lash_e, reactor_vial: str):
    """Mix reactor_vial using pipette cycling, vortex, or both, depending on config flags."""
    if USE_PIPETTE_MIX:
        lash_e.logger.info(
            f"Pipette mixing {reactor_vial}: {MIX_VOLUME_ML*1000:.0f}uL x {MIX_CYCLES} cycles"
        )
        # Move vial to clamp once before mixing (no tip held at this point).
        # All mix cycles then run with use_safe_location=False since the vial
        # is already at the clamp, avoiding the tip-held + position-24 safety error.
        lash_e.nr_robot.move_vial_to_location(reactor_vial, 'clamp', 0)
        for i in range(MIX_CYCLES):
            last_cycle = (i == MIX_CYCLES - 1)
            lash_e.nr_robot.dispense_from_vial_into_vial(
                reactor_vial, reactor_vial, MIX_VOLUME_ML,
                liquid="ethanol",
                use_safe_location=False,
                remove_tip=last_cycle,
                return_vial_home=last_cycle,
            )

    if USE_VORTEX:
        lash_e.logger.info(
            f"Vortexing {reactor_vial} for {VORTEX_TIME_S}s at speed {VORTEX_SPEED}"
        )
        lash_e.nr_robot.vortex_vial(vial_name=reactor_vial, vortex_time=VORTEX_TIME_S, vortex_speed=VORTEX_SPEED)

    if not USE_PIPETTE_MIX and not USE_VORTEX:
        lash_e.logger.warning(f"No mixing performed for {reactor_vial}: both USE_PIPETTE_MIX and USE_VORTEX are False")


def prepare_reactor_vial(
    lash_e,
    reactor_vial: str,
    amine_vial: str,
    amine_volume_mL: float,
    ethanol_volume_mL: float,
    metal_salt_vial: str,
    metal_salt_volume_mL: float,
    hcl_volume_mL: float = HCL_VOLUME_ML,
    slide_position_index: int = 0,   # kept for __main__ unpacking; not used here
):
    """
    Prepare one reactor vial (steps 1-6). Does NOT dispense onto the slide.
    Call dispense_droplet_onto_slide() separately after all vials are prepared.

    Steps:
        1. Ethanol (ETHANOL_VIAL) -> reactor vial.
        2. Amine -> reactor vial.
        3. Mix ethanol + amine.
        4. HCl -> reactor vial.
        5. Mix after HCl.
        6. Metal salt -> reactor vial.
        7. Mix after metal salt.
    """
    lash_e.logger.info(
        f"Preparing reactor vial: {reactor_vial}, "
        f"ethanol={ethanol_volume_mL:.3f}mL, amine={amine_vial} {amine_volume_mL:.3f}mL, "
        f"metal={metal_salt_vial} {metal_salt_volume_mL:.3f}mL, "
        f"HCl={hcl_volume_mL*1000:.1f}uL"
    )

    if not lash_e.simulate and _SLACK_AVAILABLE:
        slack_agent.send_slack_message(
            f"Preparing {reactor_vial}\n"
            f"Ethanol: {ethanol_volume_mL:.3f}mL | Amine: {amine_vial} {amine_volume_mL:.3f}mL\n"
            f"HCl: {hcl_volume_mL*1000:.1f}uL | Metal salt: {metal_salt_vial} {metal_salt_volume_mL:.3f}mL"
        )

    # ---- 1. Ethanol -> reactor vial ----
    lash_e.logger.info(f"Dispensing {ethanol_volume_mL:.3f}mL ethanol from {ETHANOL_VIAL} -> {reactor_vial}")
    lash_e.nr_robot.dispense_from_vial_into_vial(
        ETHANOL_VIAL, reactor_vial, ethanol_volume_mL, liquid="ethanol", use_safe_location=False
    )
    _log_ethanol_volume(lash_e)

    # ---- 2. Amine -> reactor vial ----
    lash_e.logger.info(f"Dispensing {amine_volume_mL:.3f}mL amine from {amine_vial} -> {reactor_vial}")
    _release_tip_if_held(lash_e)
    lash_e.nr_robot.dispense_from_vial_into_vial(
        amine_vial, reactor_vial, amine_volume_mL, liquid="ethanol", use_safe_location=True
    )

    # ---- 3. Vortex ethanol + amine ----
    _release_tip_if_held(lash_e)
    _mix_reactor_vial(lash_e, reactor_vial)

    # ---- 4. HCl -> reactor vial ----
    lash_e.logger.info(f"Adding {hcl_volume_mL*1000:.1f}uL HCl from {HCL_VIAL} -> {reactor_vial}")
    _release_tip_if_held(lash_e)
    lash_e.nr_robot.dispense_from_vial_into_vial(
        HCL_VIAL, reactor_vial, hcl_volume_mL, liquid=HCL_LIQUID, use_safe_location=True
    )

    # ---- 5. Vortex after HCl ----
    _release_tip_if_held(lash_e)
    _mix_reactor_vial(lash_e, reactor_vial)

    # ---- 6. Metal salt -> reactor vial ----
    lash_e.logger.info(f"Dispensing {metal_salt_volume_mL:.3f}mL metal salt from {metal_salt_vial} -> {reactor_vial}")
    _release_tip_if_held(lash_e)
    lash_e.nr_robot.dispense_from_vial_into_vial(
        metal_salt_vial, reactor_vial, metal_salt_volume_mL, liquid="ethanol", use_safe_location=True
    )

    # ---- 7. Vortex after metal salt ----
    _release_tip_if_held(lash_e)
    _mix_reactor_vial(lash_e, reactor_vial)

    lash_e.logger.info(f"Preparation complete for {reactor_vial}")
    if not lash_e.simulate and _SLACK_AVAILABLE:
        slack_agent.send_slack_message(f"Preparation complete for {reactor_vial}")


def _compute_slide_grid():
    """Return a flat list of (SLIDE_GRID_ROWS * SLIDE_GRID_COLS) robot positions
    computed by bilinear interpolation of the four slide corners defined in Locator.py.

    Corner layout:
        slide_corner_TL  slide_corner_TR
        slide_corner_BL  slide_corner_BR

    Position index i -> row = i // SLIDE_GRID_COLS, col = i % SLIDE_GRID_COLS.
    t = row / (ROWS-1) runs 0 (top) -> 1 (bottom).
    s = col / (COLS-1) runs 0 (left) -> 1 (right).
    """
    TL = Locator.slide_corner_TL
    TR = Locator.slide_corner_TR
    BL = Locator.slide_corner_BL
    BR = Locator.slide_corner_BR

    grid = []
    for row in range(SLIDE_GRID_ROWS):
        t = row / (SLIDE_GRID_ROWS - 1) if SLIDE_GRID_ROWS > 1 else 0.0
        for col in range(SLIDE_GRID_COLS):
            s = col / (SLIDE_GRID_COLS - 1) if SLIDE_GRID_COLS > 1 else 0.0
            pos = [
                round((1-t)*(1-s)*TL[i] + (1-t)*s*TR[i] + t*(1-s)*BL[i] + t*s*BR[i])
                for i in range(4)
            ]
            grid.append(pos)
    return grid


def dispense_droplet_onto_slide(lash_e, reactor_vial: str, position_index: int):
    """
    Deposit a droplet from reactor_vial onto the slide at the given grid position.

    The grid is (SLIDE_GRID_ROWS x SLIDE_GRID_COLS) positions computed from
    Locator.slide_drop_origin + SLIDE_ROW_OFFSET / SLIDE_COL_OFFSET.
    position_index is row-major: index i -> row = i // SLIDE_GRID_COLS, col = i % SLIDE_GRID_COLS.

    Fill in robot motion and dispense calls below once slide_drop_origin is calibrated.
    """
    n_positions = SLIDE_GRID_ROWS * SLIDE_GRID_COLS
    if not (0 <= position_index < n_positions):
        raise ValueError(
            f"position_index {position_index} out of range for a "
            f"{SLIDE_GRID_ROWS}x{SLIDE_GRID_COLS} grid (0-{n_positions - 1})"
        )

    grid = _compute_slide_grid()
    target_pos = grid[position_index]
    row = position_index // SLIDE_GRID_COLS
    col = position_index % SLIDE_GRID_COLS

    lash_e.logger.info(
        f"Dispensing droplet from {reactor_vial} -> slide position {position_index} "
        f"(row={row}, col={col}), volume={DROPLET_VOLUME_ML*1000:.0f}uL: {target_pos}"
    )

    # Move reactor vial to a safe small-tip position before aspirating.
    # Positions 43-47 on main_8mL_rack are the only safe locations for small-tip aspiration.
    # No tip is held at this point so the gripper move is safe.
    lash_e.logger.info(
        f"Moving {reactor_vial} to safe small-tip position (main_8mL_rack[{DROPLET_SAFE_RACK_INDEX}])"
    )
    lash_e.nr_robot.move_vial_to_location(reactor_vial, 'main_8mL_rack', DROPLET_SAFE_RACK_INDEX)
◘
    # Aspirate the droplet from the reactor vial
    lash_e.nr_robot.aspirate_from_vial(reactor_vial, DROPLET_VOLUME_ML, liquid="ethanol")

    # Move to the target slide position
    lash_e.nr_robot.c9.goto_xy_safe(target_pos, vel=lash_e.nr_robot.get_speed('standard_xy'))
    lash_e.nr_robot.c9.move_z(lash_e.nr_robot.get_height_at_location(target_pos))

    # Dispense all aspirated liquid onto the slide (PIPET_FLUID_VOLUME includes overaspirate)
    lash_e.nr_robot.pipet_dispense(lash_e.nr_robot.PIPET_FLUID_VOLUME)

    # Remove tip, then return the reactor vial to its home position
    lash_e.nr_robot.remove_pipet()
    lash_e.nr_robot.return_vial_home(
        lash_e.nr_robot.normalize_vial_index(reactor_vial)
    )


# ================================================================================
# WORKFLOW EXECUTION
# ================================================================================

if __name__ == "__main__":
    # Validate that no two experiments share a slide position before any hardware moves
    _positions = [exp["slide_position_index"] for exp in EXPERIMENTS]
    _duplicates = [idx for idx in _positions if _positions.count(idx) > 1]
    if _duplicates:
        raise ValueError(
            f"Duplicate slide_position_index values detected: {sorted(set(_duplicates))}. "
            f"Each reactor must map to a unique grid position."
        )

    lash_e = Lash_E(
        INPUT_VIAL_STATUS_FILE,
        simulate=SIMULATE,
        initialize_t8=True,
        workflow_globals=globals(),
        workflow_name="amine_protonation_workflow",
    )

    # Home robot once before the first experiment
    lash_e.nr_robot.home_robot_components()

    if TEST_SLIDE_GRID:
        # Walk every grid position in row-major order to verify spacing.
        # Set DISPENSE_TO_SLIDE=True to actually move the robot; False just logs positions.
        n_positions = SLIDE_GRID_ROWS * SLIDE_GRID_COLS
        lash_e.logger.info(
            f"TEST_SLIDE_GRID: testing {n_positions} positions "
            f"({SLIDE_GRID_ROWS}x{SLIDE_GRID_COLS}) using vial '{TEST_SLIDE_VIAL}'"
        )
        for idx in range(n_positions):
            dispense_droplet_onto_slide(lash_e, TEST_SLIDE_VIAL, idx)
        lash_e.logger.info("TEST_SLIDE_GRID: complete")
        lash_e.nr_robot.move_home()
    else:
        try:
            # ---- Phase 1: Prepare all solutions ----
            lash_e.logger.info(f"=== Phase 1: Preparing {len(EXPERIMENTS)} reactor vials ===")
            for i, exp in enumerate(EXPERIMENTS):
                lash_e.logger.info(f"Preparing vial {i+1}/{len(EXPERIMENTS)}: {exp['reactor_vial']}")
                prepare_reactor_vial(lash_e, **exp)
            lash_e.logger.info("=== Phase 1 complete: all vials prepared ===")

            # ---- Phase 2: Dispense droplets onto slide ----
            if DISPENSE_TO_SLIDE:
                lash_e.logger.info(f"=== Phase 2: Dispensing {len(EXPERIMENTS)} droplets onto slide ===")
                for i, exp in enumerate(EXPERIMENTS):
                    lash_e.logger.info(
                        f"Dispensing droplet {i+1}/{len(EXPERIMENTS)}: "
                        f"{exp['reactor_vial']} -> slide position {exp['slide_position_index']}"
                    )
                    dispense_droplet_onto_slide(lash_e, exp['reactor_vial'], exp['slide_position_index'])
                lash_e.logger.info("=== Phase 2 complete: all droplets dispensed ===")
                if not lash_e.simulate and _SLACK_AVAILABLE:
                    slack_agent.send_slack_message("All droplets dispensed onto slide.")

                # Start heating only after every droplet has been placed on the slide
                lash_e.logger.info(f"Setting slide heater to {SLIDE_HEATER_TEMP_C} C")
                lash_e.temp_controller.set_temp(SLIDE_HEATER_TEMP_C)

                # Hold heater temperature for defined soak time before turning off
                lash_e.logger.info(
                    f"Heat soak: holding {SLIDE_HEATER_TEMP_C}C for {SLIDE_HEAT_SOAK_MIN} minutes"
                )
                if not lash_e.simulate and _SLACK_AVAILABLE:
                    slack_agent.send_slack_message(
                        f"Heat soak started: {SLIDE_HEAT_SOAK_MIN} min at {SLIDE_HEATER_TEMP_C}C"
                    )
                import time
                time.sleep(SLIDE_HEAT_SOAK_MIN * 60)
                lash_e.logger.info("Heat soak complete")
            else:
                lash_e.logger.info("DISPENSE_TO_SLIDE=False: skipping slide dispensing phase")
        finally:
            # Turn off slide heater and return robot home after all experiments
            lash_e.logger.info("Turning off slide heater")
            lash_e.temp_controller.turn_off_heating()
            lash_e.nr_robot.move_home()
