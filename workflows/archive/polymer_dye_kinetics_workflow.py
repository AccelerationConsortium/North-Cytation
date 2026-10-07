# -*- coding: utf-8 -*-
"""
Polymer + Dye Kinetics Workflow (1D)
====================================

One polymer + one dye per invocation. Chain multiple invocations in one script
to sweep polymer/dye pairs sequentially.

Flow per invocation:
  1. Plan polymer substocks so every requested well concentration is dispensable
     within a per-well polymer-volume envelope (V_min, V_max in uL).
  2. Physically create (or top-up / skip) those substocks.
  3. Build well recipes: log-spaced polymer targets x n_replicates,
     with optional fixed-conc salt + fixed-uL buffer + fixed-uL dye + water.
  4. Dispense wells (water -> polymer substocks -> salt -> buffer -> dye).
  5. Kinetics: measure at t=0 immediately after last dispense, then every
     kinetics_interval_min for kinetics_duration_min. Uses time.time()
     polling (no time.sleep through intervals).

Blank/reference saving, real analysis, and plotting are intentionally left out.
Measurement is a placeholder that returns a stub DataFrame; the real Cytation
protocol wiring is a TODO for later.
"""
from __future__ import annotations

import sys
sys.path.append("../utoronto_demo")

import os
import time
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from master_usdl_coordinator import Lash_E


# =============================================================================
# CONSTANTS (module-level only where physical/architectural, not experiment-level)
# =============================================================================

SIMULATE = True
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/polymer_dye_vials.csv"

# Substock creation
MIN_SUBSTOCK_PIPETTE_UL = 200.0        # Minimum accurate substock pipette volume
FINAL_SUBSTOCK_VOLUME_ML = 6.0         # Target volume for each substock vial
SUBSTOCK_REFILL_THRESHOLD_ML = 4.0     # Above this -> skip; below -> top up / remake
SUBSTOCK_VIAL_CAPACITY_ML = 8.0        # Physical vial capacity

# Well dispensing
MIN_WELL_PIPETTE_UL = 10.0             # Minimum per-well pipetting volume
WATER_RESERVOIR_INDEX = 1              # Reservoir index used for water refill
POLYMER_SAFE_SLOTS = [43, 44, 45, 46, 47]  # main_8mL_rack slots safe for small-tip aspirate

# Kinetics scheduler
SCHEDULER_POLL_S = 5.0                 # Poll interval; NOT a full-interval sleep
HEARTBEAT_INTERVAL_S = 300.0           # Print alive message to terminal every 5 min

# Wellplate
WELLPLATE_CAPACITY = 96


# =============================================================================
# SUBSTOCK PLANNING
# =============================================================================

def plan_polymer_substocks(
    polymer_stock_conc_mm: float,
    targets_mm: np.ndarray,
    well_volume_ul: float,
    min_polymer_vol_ul: float,
    max_polymer_vol_ul: float,
    polymer_vial: str,
    logger: logging.Logger,
) -> dict:
    """
    Choose a set of polymer substocks so every target concentration in `targets_mm`
    can be dispensed with a polymer volume in [min_polymer_vol_ul, max_polymer_vol_ul].

    Each source of concentration c can produce well concentrations in the window
    [c * V_min / W, c * V_max / W]. We build substocks geometrically spaced by
    r = V_max / V_min so consecutive windows abut, starting from the stock and
    descending until the lowest target is coverable.

    Fails loudly (raises ValueError) if the stock cannot reach the highest target
    or if any target falls outside every window.

    Returns:
        {
          'substocks': [{'vial_name': str, 'concentration_mm': float}, ...],
              # ordered highest -> lowest concentration; excludes the stock itself
          'target_map': {target_conc_mm: {'source_vial': str,
                                          'source_conc_mm': float,
                                          'polymer_volume_ul': float}}
        }
    """
    W = well_volume_ul
    Vmin = min_polymer_vol_ul
    Vmax = max_polymer_vol_ul

    if Vmin <= 0 or Vmax <= Vmin:
        raise ValueError(f"Invalid polymer volume window: [{Vmin}, {Vmax}]")

    r = Vmax / Vmin  # geometric step between abutting windows

    upper = lambda c: c * Vmax / W   # highest well conc achievable from source c
    lower = lambda c: c * Vmin / W   # lowest  well conc achievable from source c

    t_max = float(np.max(targets_mm))
    t_min = float(np.min(targets_mm))

    if t_max > upper(polymer_stock_conc_mm):
        raise ValueError(
            f"Highest target {t_max:.4g} mM exceeds stock capability "
            f"(stock={polymer_stock_conc_mm:.4g} mM, upper={upper(polymer_stock_conc_mm):.4g} mM). "
            f"Increase max_polymer_vol_ul or use a more concentrated stock."
        )

    # Build source series: stock, then substocks stepping down by r
    # Substock names use the polymer_vial base with any trailing "_stock" stripped,
    # so `polymer_A_stock` -> `polymer_A_sub_1, polymer_A_sub_2, ...`.
    sub_prefix = polymer_vial[:-len("_stock")] if polymer_vial.endswith("_stock") else polymer_vial
    # `sources` is [(conc, vial_name), ...] highest -> lowest
    sources: list[tuple[float, str]] = [(polymer_stock_conc_mm, polymer_vial)]
    current = polymer_stock_conc_mm
    idx = 1
    # Keep adding substocks while the lowest so far cannot reach t_min
    while lower(current) > t_min:
        current = current / r
        sources.append((current, f"{sub_prefix}_sub_{idx}"))
        idx += 1
        if idx > 20:
            raise ValueError(
                f"Substock chain exceeded 20 levels for {polymer_vial}. "
                f"Check t_min={t_min:.4g} mM vs stock={polymer_stock_conc_mm:.4g} mM."
            )

    # Assign each target to the LOWEST-concentration source whose window covers it
    # (lower conc -> larger, more accurate polymer volume).
    target_map: dict[float, dict] = {}
    for t in targets_mm:
        t = float(t)
        chosen = None
        # sources sorted high->low; iterate low->high for "lowest valid"
        for conc, name in reversed(sources):
            if lower(conc) <= t <= upper(conc):
                chosen = (conc, name)
                break
        if chosen is None:
            raise ValueError(
                f"Target {t:.4g} mM cannot be covered by any planned source. "
                f"Sources: {[(f'{c:.3g}', n) for c, n in sources]}"
            )
        conc, name = chosen
        vol_ul = t * W / conc
        target_map[t] = {
            'source_vial': name,
            'source_conc_mm': conc,
            'polymer_volume_ul': vol_ul,
        }

    substocks_used_names = {info['source_vial'] for info in target_map.values()}
    substocks = [
        {'vial_name': name, 'concentration_mm': conc}
        for conc, name in sources[1:]  # skip stock
        if name in substocks_used_names
    ]

    logger.info(f"=== Substock plan for {polymer_vial} (stock={polymer_stock_conc_mm:.4g} mM) ===")
    logger.info(f"  Window per source: [{Vmin}, {Vmax}] uL -> ratio r={r:.2f}")
    logger.info(f"  Substocks needed ({len(substocks)}):")
    for s in substocks:
        logger.info(f"    {s['vial_name']}: {s['concentration_mm']:.4g} mM")
    logger.info(f"  Target assignments ({len(target_map)}):")
    for t in sorted(target_map.keys()):
        info = target_map[t]
        logger.info(
            f"    {t:.4g} mM -> {info['source_vial']} ({info['source_conc_mm']:.4g} mM), "
            f"{info['polymer_volume_ul']:.1f} uL"
        )

    return {'substocks': substocks, 'target_map': target_map}


def calculate_substock_recipes(
    plan: dict,
    polymer_vial: str,
    polymer_stock_conc_mm: float,
    logger: logging.Logger,
) -> list[dict]:
    """
    For each substock in `plan['substocks']`, choose the highest-concentration
    already-available source (stock or a higher substock) that lets the source
    pipette volume be >= MIN_SUBSTOCK_PIPETTE_UL when diluted to
    FINAL_SUBSTOCK_VOLUME_ML.

    Returns list of recipe dicts sorted highest -> lowest target concentration
    (creation order).
    """
    substocks_sorted = sorted(
        plan['substocks'], key=lambda s: s['concentration_mm'], reverse=True
    )
    # Available sources start with just the stock; substocks add themselves once created.
    available: list[tuple[float, str]] = [(polymer_stock_conc_mm, polymer_vial)]
    final_vol_ml = FINAL_SUBSTOCK_VOLUME_ML

    recipes: list[dict] = []
    for s in substocks_sorted:
        target_conc = s['concentration_mm']
        target_vial = s['vial_name']

        best = None  # (source_vol_ul, source_conc, source_name)
        for src_conc, src_name in available:
            if src_conc <= target_conc:
                continue  # cannot dilute up
            dilution_factor = src_conc / target_conc
            src_vol_ml = final_vol_ml / dilution_factor
            src_vol_ul = src_vol_ml * 1000.0
            if src_vol_ul < MIN_SUBSTOCK_PIPETTE_UL:
                continue
            if best is None or src_vol_ul < best[0]:
                best = (src_vol_ul, src_conc, src_name)

        if best is None:
            raise ValueError(
                f"No usable source for substock {target_vial} at {target_conc:.4g} mM "
                f"(min pipette {MIN_SUBSTOCK_PIPETTE_UL} uL). "
                f"Available sources: {available}"
            )

        src_vol_ul, src_conc, src_name = best
        src_vol_ml = src_vol_ul / 1000.0
        water_vol_ml = final_vol_ml - src_vol_ml

        recipes.append({
            'target_vial': target_vial,
            'target_conc_mm': target_conc,
            'source_vial': src_name,
            'source_conc_mm': src_conc,
            'source_volume_ml': src_vol_ml,
            'water_volume_ml': water_vol_ml,
            'final_volume_ml': final_vol_ml,
            'dilution_factor': src_conc / target_conc,
        })
        available.append((target_conc, target_vial))

    logger.info("=== Substock recipes ===")
    for r in recipes:
        logger.info(
            f"  {r['target_vial']} ({r['target_conc_mm']:.4g} mM): "
            f"{r['source_volume_ml']*1000:.0f} uL of {r['source_vial']} "
            f"({r['source_conc_mm']:.4g} mM) + {r['water_volume_ml']*1000:.0f} uL water "
            f"-> {r['final_volume_ml']:.1f} mL (df={r['dilution_factor']:.2f})"
        )
    return recipes


# =============================================================================
# SUBSTOCK CREATION (physical)
# =============================================================================

def create_substocks(lash_e: Lash_E, recipes: list[dict], water_vial: str) -> None:
    """
    Physically create substocks per `recipes`. For each substock vial:
      - vol >= SUBSTOCK_REFILL_THRESHOLD_ML: SKIP (already sufficient).
      - 0 < vol < threshold: TOP UP proportionally to FINAL_SUBSTOCK_VOLUME_ML.
      - vol == 0 (or missing): CREATE fresh at final volume.

    Fails loudly if any resulting source aspirate falls below MIN_WELL_PIPETTE_UL.
    """
    logger = lash_e.logger
    logger.info(f"=== Creating {len(recipes)} substocks (threshold={SUBSTOCK_REFILL_THRESHOLD_ML} mL) ===")

    for r in recipes:
        target_vial = r['target_vial']
        target_conc = r['target_conc_mm']
        src_vial = r['source_vial']
        full_src_ml = r['source_volume_ml']
        full_water_ml = r['water_volume_ml']
        final_ml = r['final_volume_ml']

        current_vol_ml = float(lash_e.nr_robot.get_vial_info(target_vial, 'vial_volume'))
        current_vol_ml = max(0.0, current_vol_ml)

        if current_vol_ml >= SUBSTOCK_REFILL_THRESHOLD_ML:
            logger.info(f"  SKIP {target_vial}: {current_vol_ml:.2f} mL already >= {SUBSTOCK_REFILL_THRESHOLD_ML} mL")
            continue

        if current_vol_ml > 0:
            # Top-up proportionally
            top_up_ml = final_ml - current_vol_ml
            ratio = top_up_ml / final_ml
            src_ml = full_src_ml * ratio
            water_ml = full_water_ml * ratio
            action = f"TOP-UP +{top_up_ml*1000:.0f} uL -> {final_ml:.1f} mL"
        else:
            src_ml = full_src_ml
            water_ml = full_water_ml
            action = f"CREATE {final_ml:.1f} mL"

        # Enforce minimum accurate pipette on the source aspirate (MIN_WELL_PIPETTE_UL
        # is the general floor; substock aspirates should really be >= 200 uL, but if
        # a top-up is small we scale up both source and water together to keep ratio).
        if src_ml * 1000.0 < MIN_SUBSTOCK_PIPETTE_UL:
            # Scale both to hit min source pipette; check capacity.
            scale = MIN_SUBSTOCK_PIPETTE_UL / (src_ml * 1000.0)
            src_ml_scaled = src_ml * scale
            water_ml_scaled = water_ml * scale
            new_total = current_vol_ml + src_ml_scaled + water_ml_scaled
            if new_total > SUBSTOCK_VIAL_CAPACITY_ML:
                # Reduce to fit capacity
                room = SUBSTOCK_VIAL_CAPACITY_ML - current_vol_ml
                combined = src_ml_scaled + water_ml_scaled
                if room < MIN_SUBSTOCK_PIPETTE_UL / 1000.0:
                    raise ValueError(
                        f"{target_vial}: insufficient room ({room*1000:.1f} uL) to add "
                        f"minimum {MIN_SUBSTOCK_PIPETTE_UL} uL pipettable source. "
                        f"Empty and remake."
                    )
                safe_ratio = room / combined
                src_ml_scaled *= safe_ratio
                water_ml_scaled *= safe_ratio
            src_ml = src_ml_scaled
            water_ml = water_ml_scaled
            action += f" (scaled to {MIN_SUBSTOCK_PIPETTE_UL:.0f} uL min source)"

        logger.info(f"  {action}: {target_vial} ({target_conc:.4g} mM)")
        logger.info(f"    {src_ml*1000:.1f} uL from {src_vial} + {water_ml*1000:.1f} uL water")

        lash_e.nr_robot.dispense_from_vial_into_vial(
            source_vial_name=src_vial,
            dest_vial_name=target_vial,
            volume=src_ml,
            liquid='water',  # polymer solutions pipetted with water params (adjust later)
        )
        if water_ml > 0:
            lash_e.nr_robot.dispense_into_vial_from_reservoir(
                reservoir_index=WATER_RESERVOIR_INDEX,
                vial_index=target_vial,
                volume=water_ml,
                return_home=True,  # return vial home so clamp is free for next operation
            )

    logger.info("=== Substock creation complete ===")


def _refill_water_vial(lash_e: Lash_E, vial_name: str, max_vol_ml: float = 8.0) -> None:
    """Fill the water working vial to max_vol_ml from the reservoir. Skips if already full."""
    current_ml = float(lash_e.nr_robot.get_vial_info(vial_name, 'vial_volume'))
    fill_ml = max_vol_ml - current_ml
    if fill_ml <= 0.1:
        lash_e.logger.info(f"  Water vial '{vial_name}' already full ({current_ml:.2f} mL)")
        return
    lash_e.logger.info(
        f"  Filling '{vial_name}': {current_ml:.2f} mL -> {max_vol_ml:.2f} mL "
        f"(+{fill_ml:.2f} mL from reservoir {WATER_RESERVOIR_INDEX})"
    )
    lash_e.nr_robot.dispense_into_vial_from_reservoir(
        WATER_RESERVOIR_INDEX, vial_name, fill_ml
    )


# =============================================================================
# WELL RECIPE BUILDER
# =============================================================================

def build_well_recipes(
    targets_mm: np.ndarray,
    n_replicates: int,
    plan: dict,
    well_volume_ul: float,
    salt_config: Optional[dict],
    buffer_config: Optional[dict],
    dye_config: dict,
    starting_well_idx: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Build per-well recipe DataFrame with columns:
      wellplate_index, replicate, polymer_target_conc_mm,
      substock_name, substock_conc_mm,
      polymer_volume_ul, salt_volume_ul, buffer_volume_ul, dye_volume_ul,
      water_volume_ul

    Args:
      salt_config: None, or {'vial': str, 'stock_conc_mm': float, 'target_conc_mm': float}
      buffer_config: None, or {'vial': str, 'volume_ul': float}
      dye_config: {'vial': str, 'volume_ul': float}  (required)

    Fails loudly if water volume is negative or below MIN_WELL_PIPETTE_UL (unless
    exactly zero, which is allowed).
    """
    W = well_volume_ul

    if dye_config is None:
        raise ValueError("dye_config is required")
    dye_vol = float(dye_config['volume_ul'])

    buffer_vol = 0.0 if buffer_config is None else float(buffer_config['volume_ul'])

    if salt_config is not None:
        salt_vol = salt_config['target_conc_mm'] * W / salt_config['stock_conc_mm']
        logger.info(f"  Salt: {salt_config['target_conc_mm']:.4g} mM target -> {salt_vol:.1f} uL from {salt_config['vial']}")
    else:
        salt_vol = 0.0

    total_wells = len(targets_mm) * n_replicates
    if total_wells > WELLPLATE_CAPACITY:
        raise ValueError(
            f"{len(targets_mm)} points x {n_replicates} replicates = {total_wells} wells "
            f"exceeds wellplate capacity {WELLPLATE_CAPACITY}"
        )
    if starting_well_idx + total_wells > WELLPLATE_CAPACITY:
        raise ValueError(
            f"Wells {starting_well_idx}..{starting_well_idx + total_wells - 1} exceed "
            f"wellplate capacity {WELLPLATE_CAPACITY}"
        )

    rows: list[dict] = []
    well_idx = starting_well_idx
    for target in targets_mm:
        info = plan['target_map'][float(target)]
        polymer_vol = info['polymer_volume_ul']
        for rep in range(n_replicates):
            water_vol = W - polymer_vol - salt_vol - buffer_vol - dye_vol
            if water_vol < 0:
                raise ValueError(
                    f"Well {well_idx} (target={target:.4g} mM): negative water "
                    f"({water_vol:.1f} uL). Reduce salt/buffer/dye or shrink polymer window."
                )
            if 0 < water_vol < MIN_WELL_PIPETTE_UL:
                raise ValueError(
                    f"Well {well_idx} (target={target:.4g} mM): water "
                    f"{water_vol:.1f} uL is below MIN_WELL_PIPETTE_UL={MIN_WELL_PIPETTE_UL} uL. "
                    f"Reduce polymer window or increase well volume."
                )
            rows.append({
                'wellplate_index': well_idx,
                'replicate': rep,
                'polymer_target_conc_mm': float(target),
                'substock_name': info['source_vial'],
                'substock_conc_mm': info['source_conc_mm'],
                'polymer_volume_ul': polymer_vol,
                'salt_volume_ul': salt_vol,
                'buffer_volume_ul': buffer_vol,
                'dye_volume_ul': dye_vol,
                'water_volume_ul': water_vol,
            })
            well_idx += 1

    df = pd.DataFrame(rows)

    # Sanity check: rows sum to well_volume_ul
    totals = df[['polymer_volume_ul', 'salt_volume_ul', 'buffer_volume_ul',
                 'dye_volume_ul', 'water_volume_ul']].sum(axis=1)
    if not np.allclose(totals, W, atol=1e-6):
        raise ValueError(f"Recipe volumes do not sum to {W} uL: {totals.values}")

    logger.info(f"Built {len(df)} well recipes (wells {starting_well_idx}..{well_idx-1})")
    return df


# =============================================================================
# WELL DISPENSING
# =============================================================================

def _condition_tip(
    lash_e: Lash_E,
    vial_name: str,
    conditioning_volume_ul: float = 30.0,
    cycles: int = 3,
    liquid_type: str = 'water',
) -> None:
    """
    Pre-wet the tip by aspirating and dispensing back into the source vial
    before the first real dispense. Primes tip walls and removes air-pocket
    bias from the first aspirate. Vial must already be at a safe position
    (e.g. clamp) before calling.
    """
    vol_ml = conditioning_volume_ul / 1000.0
    lash_e.logger.info(
        f"    Conditioning tip: {cycles}x {conditioning_volume_ul:.0f}uL into {vial_name}"
    )
    for _ in range(cycles):
        lash_e.nr_robot.aspirate_from_vial(vial_name, vol_ml, liquid=liquid_type)
        lash_e.nr_robot.dispense_into_vial(vial_name, vol_ml, liquid=liquid_type)


def _stage_polymer_substocks(
    lash_e: Lash_E,
    substock_names_dilute_to_conc: list[str],
) -> None:
    """
    Move all polymer substocks to safe rack positions before any tip is picked up.
    Safe positions are main_8mL_rack slots 43-47 (5 slots) plus clamp (slot 6).
    Raises ValueError if more substocks than safe positions.
    """
    n = len(substock_names_dilute_to_conc)
    positions = [(slot, 'main_8mL_rack') for slot in POLYMER_SAFE_SLOTS]
    if n > len(POLYMER_SAFE_SLOTS):
        positions.append((0, 'clamp'))
    if n > len(positions):
        raise ValueError(
            f"{n} polymer substocks exceed {len(positions)} available safe positions "
            f"(slots {POLYMER_SAFE_SLOTS} + clamp). Reduce concentration range or "
            f"widen the polymer volume window."
        )
    lash_e.logger.info(f"  Staging {n} polymer substocks at safe positions:")
    for vial_name, (slot, rack) in zip(substock_names_dilute_to_conc, positions):
        if rack == 'clamp':
            lash_e.logger.info(f"    {vial_name} -> clamp")
            lash_e.nr_robot.move_vial_to_location(vial_name, 'clamp', 0)
        else:
            lash_e.logger.info(f"    {vial_name} -> {rack}[{slot}]")
            lash_e.nr_robot.move_vial_to_location(vial_name, rack, slot)


def _dispense_polymer_substocks(
    lash_e: Lash_E,
    recipes_df: pd.DataFrame,
    sorted_substock_names: list[str],
) -> None:
    """
    Dispense all polymer substocks with a single shared tip.

    Pre-condition: all substocks must already be at safe positions (call
    _stage_polymer_substocks first, before any tip is held).

    Order: dilute -> concentrated so any trace carry-over into the next vial
    is negligible (small amount of less-concentrated solution enters a more
    concentrated one).

    Conditions the tip once on the first (most dilute) vial, then reuses the
    same tip for all subsequent substocks. One remove_pipet() at the end.
    """
    logger = lash_e.logger
    if not sorted_substock_names:
        return

    logger.info(
        f"  Polymer substocks (1 shared tip, dilute -> concentrated): "
        f"{sorted_substock_names}"
    )

    tip_held = False
    for sub_name in sorted_substock_names:
        sub_df = recipes_df[
            (recipes_df['substock_name'] == sub_name) &
            (recipes_df['polymer_volume_ul'] > 0)
        ].sort_values('polymer_volume_ul', ascending=False)
        if len(sub_df) == 0:
            logger.info(f"    {sub_name}: no wells, skip")
            continue
        logger.info(f"    {sub_name}: {len(sub_df)} wells")

        # Condition tip on the first substock that actually has wells.
        # This implicitly picks up a fresh tip; subsequent substocks reuse it.
        if not tip_held:
            _condition_tip(lash_e, sub_name, liquid_type='water')
            tip_held = True

        for _, row in sub_df.iterrows():
            vol_ml = row['polymer_volume_ul'] / 1000.0
            well_idx = int(row['wellplate_index'])
            lash_e.nr_robot.aspirate_from_vial(sub_name, vol_ml, liquid='water')
            lash_e.nr_robot.dispense_into_wellplate(
                dest_wp_num_array=[well_idx],
                amount_mL_array=[vol_ml],
                liquid='water',
            )

    # Drop the shared tip only if one was picked up
    if tip_held:
        lash_e.nr_robot.remove_pipet()

    # Return all substocks to their registered home positions
    logger.info(f"  Returning {len(sorted_substock_names)} polymer substocks home")
    for sub_name in sorted_substock_names:
        lash_e.nr_robot.return_vial_home(sub_name)


def _dispense_column(
    lash_e: Lash_E,
    recipes_df: pd.DataFrame,
    source_vial: str,
    volume_column: str,
    liquid_type: str,
    row_filter: Optional[pd.Series] = None,
) -> None:
    """
    Dispense one source vial into all wells needing it (largest volume first).
    Uses a fresh tip per call: move to clamp -> condition -> dispense -> drop
    tip -> return home. Used for water, salt, buffer, and dye.
    """
    logger = lash_e.logger
    df = recipes_df
    if row_filter is not None:
        df = df[row_filter]
    df = df[df[volume_column] > 0]
    if len(df) == 0:
        logger.info(f"  {source_vial} -> {volume_column}: no wells")
        return

    df = df.sort_values(volume_column, ascending=False)
    logger.info(f"  {source_vial} -> {volume_column}: {len(df)} wells")

    # Move vial to clamp (safe position for all tip sizes, avoids small-tip warning)
    lash_e.nr_robot.move_vial_to_location(source_vial, 'clamp', 0)

    # Condition tip before first real aspirate
    _condition_tip(lash_e, source_vial, liquid_type=liquid_type)

    for _, row in df.iterrows():
        vol_ml = row[volume_column] / 1000.0
        well_idx = int(row['wellplate_index'])
        lash_e.nr_robot.aspirate_from_vial(source_vial, vol_ml, liquid=liquid_type)
        lash_e.nr_robot.dispense_into_wellplate(
            dest_wp_num_array=[well_idx],
            amount_mL_array=[vol_ml],
            liquid=liquid_type,
        )
    lash_e.nr_robot.remove_pipet()

    # Return vial to its registered home position
    lash_e.nr_robot.return_vial_home(source_vial)


def dispense_wells(
    lash_e: Lash_E,
    recipes_df: pd.DataFrame,
    water_vial: str,
    plan: dict,
    salt_config: Optional[dict],
    buffer_config: Optional[dict],
    dye_config: dict,
) -> float:
    """
    Dispense every component into the wellplate. Order:
      water -> polymer substocks (dilute -> concentrated) -> salt -> buffer -> dye.
    Polymer substocks share one tip (staged at safe slots before pickup).
    Water, salt, buffer, and dye each use a fresh tip via _dispense_column.
    Returns time.time() epoch when the last dispense completed (== kinetics t=0).
    """
    logger = lash_e.logger
    logger.info("=== Dispensing wellplate ===")

    # 1) water
    _dispense_column(lash_e, recipes_df, water_vial, 'water_volume_ul', 'water')

    # 2) polymer substocks — stage all at safe slots first (no tip held),
    #    then share one tip across all substocks dilute -> concentrated.
    substock_names_in_use = recipes_df['substock_name'].unique().tolist()
    conc_lookup = {info['source_vial']: info['source_conc_mm']
                   for info in plan['target_map'].values()}
    substock_names_sorted = sorted(substock_names_in_use, key=lambda n: conc_lookup[n])  # dilute -> concentrated
    _stage_polymer_substocks(lash_e, substock_names_sorted)
    _dispense_polymer_substocks(lash_e, recipes_df, substock_names_sorted)

    # 3) salt
    if salt_config is not None:
        _dispense_column(lash_e, recipes_df, salt_config['vial'], 'salt_volume_ul', 'water')

    # 4) buffer
    if buffer_config is not None:
        _dispense_column(lash_e, recipes_df, buffer_config['vial'], 'buffer_volume_ul', 'water')

    # 5) dye (last: usually smallest volume, contamination-sensitive)
    _dispense_column(lash_e, recipes_df, dye_config['vial'], 'dye_volume_ul', 'DMSO')

    lash_e.nr_robot.move_home()
    t_zero = time.time()
    logger.info(f"=== Dispensing complete. t=0 set to {time.strftime('%H:%M:%S', time.localtime(t_zero))} ===")
    return t_zero


# =============================================================================
# MEASUREMENT (placeholder)
# =============================================================================

def measure_dummy(
    lash_e: Lash_E,
    well_indices: list[int],
    measurement_index: int,
) -> pd.DataFrame:
    """
    Placeholder measurement. Returns a DataFrame with columns
    [wellplate_index, measurement_index, value].

    Simulate mode: fabricates values without moving hardware.
    Hardware mode: currently logs and returns a stub. TODO: wire real Cytation
    protocol via lash_e.measure_wellplate(protocol_path, well_indices).
    """
    logger = lash_e.logger
    logger.info(f"[measurement #{measurement_index}] {len(well_indices)} wells")

    if lash_e.simulate:
        rng = np.random.default_rng(seed=measurement_index)
        values = rng.uniform(0.1, 1.0, size=len(well_indices))
    else:
        # TODO: replace with real protocol invocation, e.g.
        #   raw = lash_e.measure_wellplate(protocol_file_path=..., wells_to_measure=well_indices)
        #   values = flatten_cytation_data(raw, 'turbidity')['turbidity_600'].values
        logger.warning("Hardware measurement not yet wired; returning zeros as stub.")
        values = np.zeros(len(well_indices))

    return pd.DataFrame({
        'wellplate_index': well_indices,
        'measurement_index': measurement_index,
        'value': values,
    })


# =============================================================================
# KINETICS SCHEDULER
# =============================================================================

def run_kinetics_schedule(
    lash_e: Lash_E,
    well_indices: list[int],
    t_zero: float,
    interval_min: float,
    duration_min: float,
) -> pd.DataFrame:
    """
    Fire measurements at t_zero + k*interval_min for k=0..K where
    K = floor(duration_min / interval_min).

    Polls time.time() every SCHEDULER_POLL_S seconds. In simulate mode,
    fast-forwards (no real wait). This mirrors the pattern used in
    zif8_bsa_workflow / peroxide workflows.
    """
    logger = lash_e.logger
    n = int(duration_min / interval_min) + 1
    target_offsets_s = [i * interval_min * 60.0 for i in range(n)]

    logger.info(
        f"=== Kinetics: {n} measurements at t+0, {interval_min:.1f}, "
        f"{2*interval_min:.1f}, ... {duration_min:.1f} min ==="
    )

    all_data: list[pd.DataFrame] = []
    heartbeat_next_epoch = t_zero + HEARTBEAT_INTERVAL_S  # first print at t+5 min
    for k, offset_s in enumerate(target_offsets_s):
        target_epoch = t_zero + offset_s
        if lash_e.simulate:
            logger.info(f"  [sim] fast-forward to t+{offset_s/60:.1f} min")
        else:
            # Poll until the target time is reached
            while True:
                now = time.time()
                remaining = target_epoch - now
                if remaining <= 0:
                    break
                # Heartbeat: print to terminal so operator knows scheduler is alive
                if now >= heartbeat_next_epoch:
                    elapsed_min = (now - t_zero) / 60.0
                    next_meas_min = remaining / 60.0
                    print(
                        f"[kinetics] alive at t+{elapsed_min:.1f} min -- "
                        f"next measurement in {next_meas_min:.1f} min"
                    )
                    heartbeat_next_epoch += HEARTBEAT_INTERVAL_S
                if remaining > SCHEDULER_POLL_S:
                    time.sleep(SCHEDULER_POLL_S)
                else:
                    time.sleep(max(0.0, remaining))

        elapsed_min = (time.time() - t_zero) / 60.0 if not lash_e.simulate else offset_s / 60.0
        logger.info(f"  Measurement {k}/{n-1} at t+{elapsed_min:.2f} min (target t+{offset_s/60:.1f} min)")
        data = measure_dummy(lash_e, well_indices, measurement_index=k)
        data['t_min'] = elapsed_min
        all_data.append(data)

    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"=== Kinetics complete: {len(result)} rows total ===")
    return result


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_polymer_dye_kinetics(
    *,
    lash_e: Lash_E,
    polymer_vial: str,
    polymer_stock_conc_mm: float,
    dye_vial: str,
    dye_volume_ul: float,
    c_min_mm: float,
    c_max_mm: float,
    n_points: int,
    n_replicates: int,
    salt_vial: Optional[str] = None,
    salt_stock_conc_mm: Optional[float] = None,
    target_salt_conc_mm: Optional[float] = None,
    buffer_vial: Optional[str] = None,
    buffer_volume_ul: float = 0.0,
    water_vial: str = "water",
    well_volume_ul: float = 200.0,
    min_polymer_vol_ul: float = 10.0,
    max_polymer_vol_ul: float = 100.0,
    kinetics_interval_min: float = 30.0,
    kinetics_duration_min: float = 240.0,
    starting_well_idx: int = 0,
) -> dict:
    """
    Full 1D polymer + dye kinetics run for a single (polymer_vial, dye_vial) pair.
    Always grabs a fresh wellplate at the start. Discard it afterwards with
    lash_e.discard_used_wellplate().

    Returns {'recipes_df', 'measurements_df', 't_zero', 'plan', 'output_folder'}.
    """
    logger = lash_e.logger
    logger.info("=" * 70)
    logger.info(f"POLYMER+DYE KINETICS: {polymer_vial} + {dye_vial}")
    logger.info(
        f"  targets: {n_points} log-spaced pts in [{c_min_mm:.4g}, {c_max_mm:.4g}] mM, "
        f"{n_replicates} reps"
    )
    logger.info(
        f"  well_vol={well_volume_ul} uL, polymer window=[{min_polymer_vol_ul}, "
        f"{max_polymer_vol_ul}] uL, dye={dye_volume_ul} uL, buffer={buffer_volume_ul} uL"
    )
    logger.info("=" * 70)

    # Home all axes and top up water before anything else
    lash_e.nr_robot.home_robot_components()
    _refill_water_vial(lash_e, water_vial)

    # Create output folder for this run (timestamped; saving wired in later)
    # Structure: output/polymer_dye_kinetics/simulated/{run}/ (or experimental/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_subfolder = "simulated" if lash_e.simulate else "experimental"
    experiment_name = f"{polymer_vial}_{dye_vial}_{timestamp}"
    output_folder = os.path.join("output", "polymer_dye_kinetics", sim_subfolder, experiment_name)
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Validate salt config (all-or-nothing)
    salt_args = (salt_vial, salt_stock_conc_mm, target_salt_conc_mm)
    if any(a is not None for a in salt_args) and not all(a is not None for a in salt_args):
        raise ValueError(
            "Salt config must be all-or-none: provide salt_vial, salt_stock_conc_mm, "
            "and target_salt_conc_mm together (or none)."
        )
    salt_config = None
    if salt_vial is not None:
        salt_config = {
            'vial': salt_vial,
            'stock_conc_mm': float(salt_stock_conc_mm),
            'target_conc_mm': float(target_salt_conc_mm),
        }

    buffer_config = None
    if buffer_vial is not None and buffer_volume_ul > 0:
        buffer_config = {'vial': buffer_vial, 'volume_ul': float(buffer_volume_ul)}

    dye_config = {'vial': dye_vial, 'volume_ul': float(dye_volume_ul)}

    # Log-spaced targets
    targets_mm = np.logspace(np.log10(c_min_mm), np.log10(c_max_mm), n_points)

    # Plan + create substocks
    plan = plan_polymer_substocks(
        polymer_stock_conc_mm=polymer_stock_conc_mm,
        targets_mm=targets_mm,
        well_volume_ul=well_volume_ul,
        min_polymer_vol_ul=min_polymer_vol_ul,
        max_polymer_vol_ul=max_polymer_vol_ul,
        polymer_vial=polymer_vial,
        logger=logger,
    )
    recipes = calculate_substock_recipes(
        plan=plan,
        polymer_vial=polymer_vial,
        polymer_stock_conc_mm=polymer_stock_conc_mm,
        logger=logger,
    )
    create_substocks(lash_e, recipes, water_vial=water_vial)

    # Build well recipes
    recipes_df = build_well_recipes(
        targets_mm=targets_mm,
        n_replicates=n_replicates,
        plan=plan,
        well_volume_ul=well_volume_ul,
        salt_config=salt_config,
        buffer_config=buffer_config,
        dye_config=dye_config,
        starting_well_idx=starting_well_idx,
        logger=logger,
    )
    logger.info("Recipe sample:\n" + recipes_df.head(10).to_string())

    # Always use a fresh wellplate for each run
    lash_e.grab_new_wellplate()

    # Dispense
    t_zero = dispense_wells(
        lash_e=lash_e,
        recipes_df=recipes_df,
        water_vial=water_vial,
        plan=plan,
        salt_config=salt_config,
        buffer_config=buffer_config,
        dye_config=dye_config,
    )

    # Kinetics
    well_indices = recipes_df['wellplate_index'].astype(int).tolist()
    measurements_df = run_kinetics_schedule(
        lash_e=lash_e,
        well_indices=well_indices,
        t_zero=t_zero,
        interval_min=kinetics_interval_min,
        duration_min=kinetics_duration_min,
    )

    lash_e.nr_robot.move_home()
    logger.info(f"Run complete. Output folder: {output_folder}")

    return {
        'recipes_df': recipes_df,
        'measurements_df': measurements_df,
        't_zero': t_zero,
        'plan': plan,
        'output_folder': output_folder,
    }


# =============================================================================
# MAIN — example chaining two runs
# =============================================================================

def main():
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=SIMULATE, show_gui=False)

    # Run 1: polymer A + dye 1, with salt
    run_polymer_dye_kinetics(
        lash_e=lash_e,
        polymer_vial="polymer_A_stock",
        polymer_stock_conc_mm=50.0,
        dye_vial="dye_1",
        dye_volume_ul=5.0,
        c_min_mm=0.01,
        c_max_mm=10.0,
        n_points=8,
        n_replicates=3,
        salt_vial="salt_1",
        salt_stock_conc_mm=1000.0,
        target_salt_conc_mm=100.0,
        buffer_vial="buffer_1",
        buffer_volume_ul=20.0,
    )
    lash_e.discard_used_wellplate()

    # Run 2: polymer A + dye 2, no salt, no buffer
    run_polymer_dye_kinetics(
        lash_e=lash_e,
        polymer_vial="polymer_A_stock",
        polymer_stock_conc_mm=50.0,
        dye_vial="dye_2",
        dye_volume_ul=5.0,
        c_min_mm=0.05,
        c_max_mm=20.0,
        n_points=6,
        n_replicates=2,
    )
    lash_e.discard_used_wellplate()


if __name__ == "__main__":
    main()
