#!/usr/bin/env python3
"""
ZIF-8@BSA Synthesis Workflow - Scheduled
==========================================
Automated synthesis of ZIF-8 Metal-Organic Framework with BSA protein encapsulation.

SYNTHESIS PROTOCOL (scaled to 6 mL / vial):
  1. Dispense HmIm stock + BSA stock + water into reaction vial  (3 mL)
  2. Age HmIm/BSA mixture for AGING_TIME_MIN (default 60 min)
  3. Add Zn(OAc)2 stock + water                                  (3 mL)
  4. React at room temperature, unstirred, for REACTION_TIME_MIN
  5. Collect vials for offline centrifugation:
       Spin 1:  5k rpm / 30 min -> keep supernatant, discard pellet
       Spin 2: 14k rpm / 30 min -> keep pellet (ZIF-8@BSA), redisperse in water

SCALE NOTE:
  Original: 10 mL HmIm/BSA + 10 mL Zn = 20 mL total
  This:      3 mL HmIm/BSA +  3 mL Zn =  6 mL total (0.3x scale)

SCHEDULING:
  The workflow builds an event list upfront (bsa_water_himim + zn_addition per vial)
  and executes them via a time-polling loop - no blocking sleep() calls.
  Each vial's zn_addition is scheduled exactly AGING_TIME_MIN after its
  bsa_water_himim event, so per-vial aging is precise regardless of robot busy time.

  With 3 dispenses/vial at 30 s each, one vial takes ~90 s of robot time.
  24 vials fill the 60-min aging window -> zero idle time with enough vials.
  With 6 physical reaction vials you run 6 per batch (sequential batches).

VARIABLE PARAMETERS (per-experiment, in EXPERIMENT_FILE):
  himim_conc_mM     - HmIm concentration in final 6 mL (mM)
  zn_conc_mM        - Zn(OAc)2 concentration in final 6 mL (mM)
  bsa_mg_per_mL     - BSA concentration in final 6 mL (mg/mL)
  reaction_time_min - Wait time after Zn addition before collecting vial (min)

REQUIRED STOCK VIALS (see status/zif8_bsa_vials.csv):
  himim_stock   aqueous HmIm at HIMIM_STOCK_CONC_MM
  zn_stock      aqueous Zn(OAc)2 at ZN_STOCK_CONC_MM
  bsa_stock     aqueous BSA at BSA_STOCK_CONC_MG_PER_ML
  water         DI water for dilution
  reaction_vial_1 ... reaction_vial_6: empty 8 mL vials
"""

import sys
sys.path.append("../utoronto_demo")
import time
import pandas as pd
import slack_agent
from master_usdl_coordinator import Lash_E

# ================================================================================
# CONFIGURATION - UPDATE THESE TO MATCH YOUR PREPARED STOCK SOLUTIONS
# ================================================================================

INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/zif8_bsa_vials.csv"
EXPERIMENT_FILE        = "../utoronto_demo/status/zif8_bsa_experiments.csv"

# Stock solution concentrations (must match what is physically in the vials)
HIMIM_STOCK_CONC_MM       = 160.0  # HmIm aqueous stock (mM)
ZN_STOCK_CONC_MM          = 40.0   # Zn(OAc)2 aqueous stock (mM)
BSA_STOCK_CONC_MG_PER_ML  = 10.0  # BSA aqueous stock (mg/mL)

# Protocol timing (fixed by chemistry)
AGING_TIME_MIN    = 60    # HmIm/BSA complexation aging (min) - do not change
REACTION_TIME_MIN = 60    # Default reaction time after Zn addition (min)
                          # Override per-experiment via 'reaction_time_min' column in EXPERIMENT_FILE

# Scheduling estimate parameters (used for ETA only, not for actual timing)
# HmIm slow dispense speed (lower = gentler addition onto BSA solution)
HIMIM_DISPENSE_SPEED = 5  # mm/s - slower than default

SECONDS_PER_DISPENSE   = 30  # Estimated time per pipetting operation
DISPENSES_PER_VIAL_BSA_WATER_HIMIM = 3   # BSA, water, HmIm (slow)
DISPENSES_PER_VIAL_ZN              = 1   # Zn only

# Pipetting limits
MIN_PIPETTE_VOLUME_ML = 0.010  # 10 uL minimum
TOTAL_REACTION_VOL_ML = 6.0   # Final volume per vial (mL)
MAX_VIALS_PER_BATCH   = 6     # Physical reaction vials available in rack

SIMULATE = True  # Set to False for hardware execution

# ================================================================================
# VOLUME CALCULATION
# ================================================================================

def calculate_volumes(himim_conc_mM: float, zn_conc_mM: float, bsa_mg_per_mL: float) -> dict:
    """
    Compute stock dispense volumes for one vial (approach 2: direct stock addition).

    All stocks are added directly to the 6 mL reaction vial:
      V_stock = C_target * V_total / C_stock
      V_water = V_total - V_bsa - V_himim - V_zn

    Addition order:
      Step 1: BSA stock  -> reaction vial
      Step 2: water      -> reaction vial
      Step 3: HmIm stock -> reaction vial  (slow)
      Step 4: [wait 60 min]
      Step 5: Zn stock   -> reaction vial

    Raises ValueError if any volume is negative or below pipette minimum.
    """
    himim_vol = himim_conc_mM * TOTAL_REACTION_VOL_ML / HIMIM_STOCK_CONC_MM
    zn_vol    = zn_conc_mM    * TOTAL_REACTION_VOL_ML / ZN_STOCK_CONC_MM
    bsa_vol   = bsa_mg_per_mL * TOTAL_REACTION_VOL_ML / BSA_STOCK_CONC_MG_PER_ML
    water_vol = TOTAL_REACTION_VOL_ML - himim_vol - bsa_vol - zn_vol

    errors = []
    for name, vol in [
        ("himim_vol", himim_vol), ("zn_vol",    zn_vol),
        ("bsa_vol",   bsa_vol),   ("water_vol", water_vol),
    ]:
        if vol < 0:
            errors.append(f"{name} = {vol*1000:.1f} uL (negative: stock concentrations sum exceeds total volume)")
        elif 0 < vol < MIN_PIPETTE_VOLUME_ML:
            errors.append(
                f"{name} = {vol*1000:.1f} uL (below min {MIN_PIPETTE_VOLUME_ML*1000:.0f} uL)"
            )
    if errors:
        raise ValueError(
            f"Volume error for HmIm={himim_conc_mM}mM / "
            f"Zn={zn_conc_mM}mM / BSA={bsa_mg_per_mL}mg/mL:\n  " + "\n  ".join(errors)
        )
    return {
        "bsa_vol_mL":   bsa_vol,
        "water_vol_mL": water_vol,
        "himim_vol_mL": himim_vol,
        "zn_vol_mL":    zn_vol,
    }


# ================================================================================
# DISPENSE HELPERS
# ================================================================================

def _dispense(lash_e, source: str, dest: str, volume_mL: float, liquid: str = "water"):
    if volume_mL < MIN_PIPETTE_VOLUME_ML:
        lash_e.logger.info(f"  Skip {source}->{dest}: {volume_mL*1000:.1f} uL (below min)")
        return
    lash_e.logger.info(f"  {source} -> {dest}: {volume_mL*1000:.1f} uL")
    lash_e.nr_robot.dispense_from_vial_into_vial(source, dest, volume=volume_mL, liquid=liquid)


def add_bsa_water_himim(lash_e, vial: str, vols: dict):
    """Steps 1-3: BSA stock -> water -> HmIm stock (slow) into reaction vial."""
    lash_e.logger.info(f"BSA + water + HmIm addition -> {vial}")
    _dispense(lash_e, "bsa_stock", vial, vols["bsa_vol_mL"],   liquid="water")
    _dispense(lash_e, "water",     vial, vols["water_vol_mL"], liquid="water")
    lash_e.logger.info(
        f"  himim_stock -> {vial}: {vols['himim_vol_mL']*1000:.0f} uL (slow)"
    )
    lash_e.nr_robot.dispense_from_vial_into_vial(
        "himim_stock", vial, volume=vols["himim_vol_mL"], liquid="water",
        dispense_speed=HIMIM_DISPENSE_SPEED,
    )
    total = vols["bsa_vol_mL"] + vols["water_vol_mL"] + vols["himim_vol_mL"]
    lash_e.logger.info(
        f"  {vial}: {total*1000:.0f} uL added "
        f"(BSA {vols['bsa_vol_mL']*1000:.0f} + "
        f"water {vols['water_vol_mL']*1000:.0f} + "
        f"HmIm {vols['himim_vol_mL']*1000:.0f} uL slow)"
    )


def add_zn(lash_e, vial: str, vols: dict):
    """Step 5: Zn(OAc)2 stock into aged vial. No water - all water was added in step 2."""
    lash_e.logger.info(f"Zn addition -> {vial}")
    _dispense(lash_e, "zn_stock", vial, vols["zn_vol_mL"], liquid="water")
    lash_e.logger.info(
        f"  {vial}: {TOTAL_REACTION_VOL_ML*1000:.0f} uL total "
        f"(Zn {vols['zn_vol_mL']*1000:.0f} uL added)"
    )


# ================================================================================
# SCHEDULE BUILDER
# ================================================================================

def build_schedule(runs: list) -> list:
    """
    Build a sorted event list for all runs.

    Actions per vial:
      bsa_water_himim - BSA + water + HmIm (slow) into reaction vial
      zn_addition     - Zn stock into vial (exactly AGING_TIME_MIN * 60 s after bsa_water_himim)
      collect         - vial ready for centrifugation (log only, no robot action)

    The robot cursor advances sequentially: bsa_water_himim events are assigned
    in order, Zn events are scheduled at max(target, cursor) to avoid conflicts.
    Zn additions naturally interleave with later bsa_water_himim events.

    Returns: sorted list of event dicts with keys:
        start_time_s  - seconds from T0 when the event should fire
        action        - 'bsa_water_himim' | 'zn_addition' | 'collect'
        vial          - reaction vial name
        exp           - original experiment dict
        vols          - pre-computed volume dict (None for 'collect')
    """
    cursor_s        = 0.0
    step1_s_per_vial = DISPENSES_PER_VIAL_BSA_WATER_HIMIM * SECONDS_PER_DISPENSE
    zn_s_per_vial    = DISPENSES_PER_VIAL_ZN * SECONDS_PER_DISPENSE
    aging_s          = AGING_TIME_MIN * 60.0

    # First pass: assign bsa_water_himim times sequentially
    step1_events = []
    for exp in runs:
        vols = calculate_volumes(
            himim_conc_mM = exp["himim_conc_mM"],
            zn_conc_mM    = exp["zn_conc_mM"],
            bsa_mg_per_mL = exp["bsa_mg_per_mL"],
        )
        step1_events.append({
            "start_time_s": cursor_s,
            "action":       "bsa_water_himim",
            "vial":         exp["reaction_vial"],
            "exp":          exp,
            "vols":         vols,
        })
        cursor_s += step1_s_per_vial

    # Second pass: schedule zn_addition events at max(target, cursor)
    # so robot finishes any previous task first. Target = step1_time + aging_s.
    zn_events = []
    for evt in sorted(step1_events, key=lambda e: e["start_time_s"] + aging_s):
        target = evt["start_time_s"] + aging_s
        earliest = max(target, cursor_s)
        zn_events.append({
            "start_time_s": earliest,
            "action":       "zn_addition",
            "vial":         evt["vial"],
            "exp":          evt["exp"],
            "vols":         evt["vols"],
        })
        cursor_s = earliest + zn_s_per_vial

    # Third pass: collect events (no robot action, just a log/Slack notification)
    collect_events = []
    for evt in zn_events:
        collect_events.append({
            "start_time_s": evt["start_time_s"] + evt["exp"]["reaction_time_min"] * 60.0,
            "action":       "collect",
            "vial":         evt["vial"],
            "exp":          evt["exp"],
            "vols":         None,
        })

    all_events = step1_events + zn_events + collect_events
    all_events.sort(key=lambda e: e["start_time_s"])
    return all_events


def print_schedule(events: list, logger):
    """Log the full event schedule before execution starts."""
    logger.info("=== ZIF-8@BSA SYNTHESIS SCHEDULE ===")
    for evt in events:
        t_min = evt["start_time_s"] / 60.0
        logger.info(f"  T+{t_min:6.1f} min  {evt['action']:12s}  {evt['vial']}")
    total_min = max(e["start_time_s"] for e in events) / 60.0
    logger.info(f"  Estimated finish: T+{total_min:.0f} min ({total_min/60:.1f} h)")
    logger.info("=====================================")


# ================================================================================
# SCHEDULED WORKFLOW EXECUTOR
# ================================================================================

def zif8_bsa_scheduled(lash_e, runs: list):
    """
    Execute ZIF-8@BSA synthesis for all runs using a time-based event schedule.

    Builds the event list upfront, logs the schedule, then polls time.time()
    in a loop - same pattern as peroxide_serena_v4.py and Degradation_serena.py.
    No blocking sleep() calls: events fire when elapsed time reaches their target.

    Arguments:
        lash_e: Initialized Lash_E coordinator
        runs (list of dict): Each entry must have:
            reaction_vial    (str)   - vial name (reaction_vial_1 ... reaction_vial_6)
            himim_conc_mM    (float) - HmIm final concentration in 6 mL (mM)
            zn_conc_mM       (float) - Zn(OAc)2 final concentration in 6 mL (mM)
            bsa_mg_per_mL    (float) - BSA final concentration in 6 mL (mg/mL)
            reaction_time_min(float) - wait time after Zn addition (min)

    Returns:
        list of completed event dicts
    """
    logger = lash_e.logger

    logger.info(
        f"Stock concentrations: HmIm {HIMIM_STOCK_CONC_MM} mM | "
        f"Zn {ZN_STOCK_CONC_MM} mM | BSA {BSA_STOCK_CONC_MG_PER_ML} mg/mL"
    )
    logger.info(f"Aging time: {AGING_TIME_MIN} min (fixed by protocol)")

    # Build and validate full schedule before touching hardware
    events = build_schedule(runs)
    print_schedule(events, logger)

    if not lash_e.simulate:
        last_event_min = max(e["start_time_s"] for e in events) / 60.0
        slack_agent.send_slack_message(
            f"ZIF-8@BSA Synthesis Started\n"
            f"Vials: {', '.join(r['reaction_vial'] for r in runs)}\n"
            f"Events: {len(events)} | Est. {last_event_min:.0f} min"
        )

    # --- Time-polling loop (same pattern as peroxide / degradation workflows) ---
    items_completed  = 0
    heartbeat_next_s = 300  # log alive message every 5 min
    start_time       = time.time()

    while items_completed < len(events):
        evt       = events[items_completed]
        target_s  = evt["start_time_s"]
        action    = evt["action"]
        vial      = evt["vial"]
        elapsed_s = time.time() - start_time

        if lash_e.simulate:
            elapsed_s = target_s + 1  # fast-forward in simulation

        if elapsed_s >= target_s:
            logger.info(
                f"Event: {action} | {vial} | "
                f"T+{elapsed_s/60.0:.1f} min (target T+{target_s/60.0:.1f} min)"
            )

            if action == "bsa_water_himim":
                add_bsa_water_himim(lash_e, vial, evt["vols"])

            elif action == "zn_addition":
                add_zn(lash_e, vial, evt["vols"])

            elif action == "collect":
                logger.info(
                    f"  {vial} READY for centrifugation "
                    f"(reaction_time={evt['exp']['reaction_time_min']} min complete)"
                )
                logger.info(
                    "  Centrifuge: 5k rpm 30 min -> keep supernatant; "
                    "14k rpm 30 min -> keep pellet, redisperse in water"
                )
                if not lash_e.simulate:
                    slack_agent.send_slack_message(
                        f"ZIF-8@BSA: {vial} ready for centrifugation\n"
                        f"HmIm={evt['exp']['himim_conc_mM']} mM | "
                        f"Zn={evt['exp']['zn_conc_mM']} mM | "
                        f"BSA={evt['exp']['bsa_mg_per_mL']} mg/mL"
                    )

            items_completed += 1

        else:
            if elapsed_s > heartbeat_next_s:
                logger.info(
                    f"  [alive] T+{elapsed_s/60:.1f} min - "
                    f"waiting for '{action}' on {vial} at T+{target_s/60.0:.1f} min"
                )
                heartbeat_next_s += 300
            if not lash_e.simulate:
                time.sleep(1)

    logger.info("All synthesis events complete.")
    if not lash_e.simulate:
        slack_agent.send_slack_message(
            f"ZIF-8@BSA: All {len(runs)} vials done. Collect for centrifugation."
        )
    return events


# ================================================================================
# MULTI-BATCH RUNNER (sequential batches if > MAX_VIALS_PER_BATCH experiments)
# ================================================================================

def run_all_experiments(lash_e, all_experiments: list, vials_per_batch: int = MAX_VIALS_PER_BATCH):
    """
    Run all experiments in sequential batches of up to vials_per_batch.

    Each batch reuses the same 6 physical reaction vials. Between batches the
    operator removes completed vials (for centrifugation) and loads fresh empty
    vials. The robot pauses and waits for Enter before starting the next batch.

    Arguments:
        lash_e:           Initialized Lash_E coordinator
        all_experiments:  Full list of experiment dicts (from EXPERIMENT_FILE)
        vials_per_batch:  Max concurrent vials (default MAX_VIALS_PER_BATCH = 6)
    """
    logger = lash_e.logger
    batches = [
        all_experiments[i : i + vials_per_batch]
        for i in range(0, len(all_experiments), vials_per_batch)
    ]

    vial_names = [f"reaction_vial_{j+1}" for j in range(vials_per_batch)]
    for batch in batches:
        for k, run in enumerate(batch):
            run["reaction_vial"] = vial_names[k]

    logger.info(
        f"Running {len(all_experiments)} experiments "
        f"in {len(batches)} batch(es) of up to {vials_per_batch}"
    )

    for i, batch in enumerate(batches):
        logger.info(f"=== BATCH {i+1} / {len(batches)} ({len(batch)} vials) ===")
        zif8_bsa_scheduled(lash_e, batch)

        if i < len(batches) - 1:
            msg = (
                f"Batch {i+1} done. Remove {len(batch)} completed vials for centrifugation "
                f"and load {len(batches[i+1])} fresh empty vials, then press Enter."
            )
            logger.info(msg)
            if not lash_e.simulate:
                input(f"\n{msg}\n> ")

    logger.info(f"All {len(all_experiments)} experiments complete.")


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    experiments_df = pd.read_csv(EXPERIMENT_FILE, comment="#")
    for col in ["himim_conc_mM", "zn_conc_mM", "bsa_mg_per_mL", "reaction_time_min"]:
        experiments_df[col] = pd.to_numeric(experiments_df[col])
    runs = experiments_df.to_dict(orient="records")

    with Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE) as lash_e:
        lash_e.nr_robot.check_input_file()
        run_all_experiments(lash_e, runs, vials_per_batch=MAX_VIALS_PER_BATCH)
