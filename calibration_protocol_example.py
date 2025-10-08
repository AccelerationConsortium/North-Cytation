"""Hardware example calibration protocol (post-0.7.0 minimal interface).

Unified lifecycle (no injected hardware / context objects):
    initialize() -> state(dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

All hardware acquisition, vial naming, and rotation logic is INTERNAL to this
module. Users adapting to their lab edit ONLY this file (or create a copy).

Replicate result dict REQUIRED keys:
    replicate (int), volume (mL), elapsed_s (float)
Optional keys (timestamps, echoed params) are included for analysis convenience.

Note: We intentionally DO NOT accept a config or context object because user
deployments vary. If external parameters are desired, read environment vars or
define module-level constants.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Dict, Any, List
import yaml, os

try:  # Optional dependency
    from pipetting_data.pipetting_parameters import PipettingParameters
except Exception:  # pragma: no cover
    class PipettingParameters:  # type: ignore
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


def _load_liquids_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """Lightweight loader for a `liquids.yaml` file if present.

    Expected schema (subset):
        water:
          density_g_per_ml: 0.997
          refill_pipets: false
    Missing file or parse issues -> empty dict (callers fallback to defaults).
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            # Normalize keys we care about only
            norm: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    norm[k] = dict(v)
            return norm
    except Exception:
        pass
    return {}


def initialize(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create and stage hardware for calibration.

    All hardware setup is intentionally inlined here so users see / modify a
    single function when adapting to their lab. Edit vial file path, init
    flags, or simulation detection logic as needed.
    """
    # Explicit simulation flag (edit manually if you want mock mode)
    simulate = False  # Set True for simulated (mock) hardware behavior

    # Import and construct Lash_E (user can alter flags or provide builders)
    from master_usdl_coordinator import Lash_E  # local import to avoid side effects at module import time
    vial_file = "status/calibration_vials_short.csv"  # Updated to match calibration modular reference
    lash_e = Lash_E(
        vial_file=vial_file,
        simulate=simulate,
        initialize_robot=True,
        initialize_track=True,
        initialize_biotek=False,
    )
    if not hasattr(lash_e, 'nr_robot'):
        raise RuntimeError("Lash_E instance missing 'nr_robot' – verify implementation")

    # Basic staging: ensure vial config parsed & move first measurement vial
    lash_e.nr_robot.check_input_file()
    first_name = "measurement_vial_0"
    lash_e.nr_robot.move_vial_to_location(first_name, "clamp", 0)

    # Load liquids metadata (density + refill flag) from a SINGLE canonical path.
    # Canonical default: next_gen_calibration/liquids.yaml adjacent to this repository structure.
    # Optional override: cfg['liquids_file'] if provided.
    base_dir = os.path.dirname(__file__)
    default_liquids_path = os.path.join(base_dir, 'next_gen_calibration', 'liquids.yaml')
    liquids_path = None
    if cfg and isinstance(cfg, dict) and cfg.get('liquids_file'):
        liquids_path = cfg['liquids_file']
    else:
        liquids_path = default_liquids_path
    liquids = _load_liquids_yaml(liquids_path)

    # --- Liquid selection (strict, concise) ---
    def _resolve_liquid(cfg_dict: Dict[str, Any] | None) -> str:
        if not cfg_dict:
            raise ValueError("Missing config: provide a dict with 'liquid' or protocol.liquid")
        if cfg_dict.get('liquid'):
            return cfg_dict['liquid']
        proto = cfg_dict.get('protocol')
        if isinstance(proto, dict) and proto.get('liquid'):
            return proto['liquid']
        raise ValueError("No liquid specified (expect 'liquid' at top level or protocol.liquid)")

    if not liquids:
        raise FileNotFoundError("liquids.yaml not found; cannot resolve density")
    liquid_name = _resolve_liquid(cfg)
    meta = liquids.get(liquid_name)
    if not meta:
        raise KeyError(f"Liquid '{liquid_name}' absent from liquids.yaml (available: {list(liquids)})")
    try:
        density_g_per_ml = float(meta['density_g_per_ml'])
    except KeyError as e:
        raise KeyError(f"Liquid '{liquid_name}' missing 'density_g_per_ml'") from e
    new_pipet_each_time = bool(meta.get('refill_pipets', False))

    return {
        "lash_e": lash_e,
        "measurement_vial_index": 0,
        "measurement_vial_name": first_name,
        "_simulate": simulate,
        "_liquid_name": liquid_name,
        "_density_g_per_ml": density_g_per_ml,
        "_new_pipet_each_time": new_pipet_each_time,
    }


def _build_params(params: Dict[str, Any]):
    aspirate = PipettingParameters(
        aspirate_speed=params["aspirate_speed"],
        aspirate_wait_time=params["aspirate_wait_time"],
        retract_speed=params["retract_speed"],
        pre_asp_air_vol=0.0,
        post_asp_air_vol=params.get("post_asp_air_vol", 0.0),
        blowout_vol=params.get("blowout_vol", 0.05),
    )
    dispense = PipettingParameters(
        dispense_speed=params["dispense_speed"],
        dispense_wait_time=params["dispense_wait_time"],
        air_vol=params.get("post_asp_air_vol", 0.0),
    )
    return aspirate, dispense


def _maybe_rotate_measurement_vial(state: Dict[str, Any]):
    lash_e = state.get("lash_e")
    if lash_e is None:
        raise RuntimeError("state missing 'lash_e' – cannot rotate measurement vial")
    vol = lash_e.nr_robot.get_vial_info(state["measurement_vial_name"], "vial_volume")
    if vol > 7.0:
        lash_e.nr_robot.remove_pipet()
        lash_e.nr_robot.return_vial_home(state["measurement_vial_name"])
        state["measurement_vial_index"] += 1
        new_name = f"measurement_vial_{state['measurement_vial_index']}"
        state["measurement_vial_name"] = new_name
        lash_e.nr_robot.move_vial_to_location(new_name, "clamp", 0)
        if (logger := getattr(lash_e, 'logger', None)):
            try:
                logger.info(f"[protocol] Rotated to new measurement vial: {new_name}")
            except Exception:
                pass


def measure(state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int) -> List[Dict[str, Any]]:
    lash_e = state.get("lash_e")
    if lash_e is None:
        raise RuntimeError("state missing 'lash_e' – initialize() must store hardware handle")
    over = params.get("overaspirate_vol", 0.0)
    source_vial = "liquid_source"  # Hard-coded; change here if needed
    dest_vial = state.get("measurement_vial_name", "measurement_vial_0")
    aspirate_params, dispense_params = _build_params(params)
    density = float(state.get('_density_g_per_ml', 1.0)) or 1.0
    new_pipet_each = bool(state.get('_new_pipet_each_time', False))

    results: List[Dict[str, Any]] = []
    for r in range(replicates):
        _maybe_rotate_measurement_vial(state)
        if new_pipet_each:
            # Conservative strategy: remove any existing pipet so a fresh tip/ volume is ensured
            try:
                lash_e.nr_robot.remove_pipet()
            except Exception:
                pass
        start_ts = datetime.now().isoformat()
        t0 = time.time()
        lash_e.nr_robot.aspirate_from_vial(source_vial, volume_mL + over, parameters=aspirate_params)
        mass_g = lash_e.nr_robot.dispense_into_vial(dest_vial, volume_mL + over, parameters=dispense_params, measure_weight=True)
        # If hardware returned None (e.g., vial not clamped), we fallback to target mass via density
        if mass_g is None:
            mass_g = (volume_mL) * density
        measured_volume = mass_g / density  # mL
        elapsed = time.time() - t0
        end_ts = datetime.now().isoformat()
        results.append({
            "replicate": r,
            "elapsed_s": elapsed,
            "start_time": start_ts,
            "end_time": end_ts,
            "volume": measured_volume,
            "mass_g": mass_g,
            "density_g_per_ml": density,
            "new_pipet_each_time": new_pipet_each,
            **params
        })
    return results


def wrapup(state: Dict[str, Any]) -> None:
    lash_e = state.get("lash_e")
    if lash_e is None:
        return
    for fn in ("remove_pipet", "return_vial_home", "move_home"):
        try:
            if fn == "return_vial_home":
                getattr(lash_e.nr_robot, fn)(state.get("measurement_vial_name", "measurement_vial_0"))
            else:
                getattr(lash_e.nr_robot, fn)()
        except Exception:
            pass
