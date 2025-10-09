from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import random, os, yaml

@dataclass
class PipettingResult:
    volume_target_ul: float
    measured_masses_g: list
    calculated_volumes_ul: list
    replicate_times_s: list
    params: Dict[str, Any]
    simulate: bool

class RobotAdapter:
    """Abstracts pipetting operations.

    Modes:
      - simulate=True: uses internal stochastic model
      - simulate=False: expects hardware interfaces described in hardware.yaml (partial / stubbed)

    Extend by overriding `_hardware_pipetting` and providing concrete component drivers.
    """
    def __init__(self, liquid: str = "water", simulate: bool = True, seed: int | None = None, hardware_config_path: str | None = None):
        self.liquid = liquid
        self.simulate = simulate
        if seed is not None:
            random.seed(seed)
        # Liquids loaded lazily from liquids.yaml (simple dict: name -> props)
        self._liquids_cache: Dict[str, Dict[str, Any]] | None = None
        self.hardware_cfg: Dict[str, Any] = {}
        if not simulate:
            self._load_hardware_config(hardware_config_path)
            self._extract_minimal_interfaces()

    # --- Hardware Config ---
    def _load_hardware_config(self, path: str | None):
        if path is None:
            # default to next_gen_calibration/hardware.yaml
            path = os.path.join(os.path.dirname(__file__), 'hardware.yaml')
        if os.path.exists(path):
            try:
                with open(path,'r') as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    self.hardware_cfg = data
            except Exception as e:
                print('[RobotAdapter] Failed to load hardware.yaml:', e)
        else:
            print('[RobotAdapter] hardware.yaml not found at', path)

    def _extract_minimal_interfaces(self):
        # Minimal schema expects keys: system, handler, measurement.method, density.override_g_per_ml, hints.settle_delay_s
        self._handler_ref = self.hardware_cfg.get('handler')
        meas = self.hardware_cfg.get('measurement', {})
        self._measurement_method = meas.get('method', 'mass')
        dens = self.hardware_cfg.get('density', {})
        self._density_override = dens.get('override_g_per_ml')
        hints = self.hardware_cfg.get('hints', {})
        self._settle_delay_s = hints.get('settle_delay_s', 0.5)

    def _load_liquids(self):
        if self._liquids_cache is not None:
            return
        path = os.path.join(os.path.dirname(__file__), 'liquids.yaml')
        if os.path.exists(path):
            try:
                with open(path,'r') as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    self._liquids_cache = {k: dict(v) for k,v in data.items()}
                else:
                    self._liquids_cache = {}
            except Exception:
                self._liquids_cache = {}
        else:
            self._liquids_cache = {}

    def get_density(self) -> float:
        self._load_liquids()
        props = (self._liquids_cache or {}).get(self.liquid, {})
        if not self.simulate and getattr(self, '_density_override', None) is not None:
            return float(self._density_override)
        return float(props.get('density_g_per_ml', 1.0))

    # --- Public API ---
    def pipet_and_measure(self, volume_mL: float, params: Dict[str, Any], replicates: int) -> PipettingResult:
        if self.simulate:
            return self._simulate_pipetting(volume_mL, params, replicates)
        else:
            return self._hardware_pipetting(volume_mL, params, replicates)

    # --- Simulation ---
    def _simulate_pipetting(self, volume_mL: float, params: Dict[str, Any], replicates: int) -> PipettingResult:
        density = self.get_density()
        target_ul = volume_mL * 1000.0
        # Base expected time ~ volume scale + waits
        base_time = 4 + (target_ul / 50.0) + params.get('aspirate_wait_time',0)/10 + params.get('dispense_wait_time',0)/10
        time_jitter = 0.25 * base_time
        measured_times = [max(0.5, random.gauss(base_time, time_jitter*0.25)) for _ in range(replicates)]
        # Deviation scales down with moderate speeds and waits
        speed_factor = (params.get('aspirate_speed',15) + params.get('dispense_speed',15))/60
        wait_factor = (params.get('aspirate_wait_time',5)+params.get('dispense_wait_time',5))/60
        control_quality = 0.6* (1 - abs(speed_factor-0.5)) + 0.4 * min(wait_factor,0.6)
        base_rel_dev = 0.03 + 0.08*(1-control_quality)
        # Variation a bit higher
        base_rel_var = base_rel_dev * 1.3
        masses = []
        volumes = []
        for _ in range(replicates):
            rel_error = random.gauss(0, base_rel_dev)
            vol_ul = target_ul * (1 + rel_error)
            masses.append(vol_ul/1000*density)  # g
            volumes.append(vol_ul)
        return PipettingResult(
            volume_target_ul=target_ul,
            measured_masses_g=masses,
            calculated_volumes_ul=volumes,
            replicate_times_s=measured_times,
            params=params,
            simulate=True
        )

    # --- Hardware (placeholder) ---
    def _hardware_pipetting(self, volume_mL: float, params: Dict[str, Any], replicates: int) -> PipettingResult:
        # Minimal contract: if a handler callable is available and returns volumes, use it.
        handler_path = getattr(self, '_handler_ref', None)
        if not handler_path:
            raise NotImplementedError("No hardware handler configured (handler key missing in hardware.yaml).")
        try:
            mod_name, func_name = handler_path.rsplit('.',1)
            mod = __import__(mod_name, fromlist=[func_name])
            handler = getattr(mod, func_name)
        except Exception as e:
            raise RuntimeError(f"Failed to import hardware handler '{handler_path}': {e}") from e
        start_density = self.get_density()
        volumes_ul = handler(params=params, target_volume_mL=volume_mL, replicates=replicates, liquid=self.liquid)
        if not isinstance(volumes_ul, list) or len(volumes_ul) != replicates:
            raise ValueError("Hardware handler must return list[float] of length == replicates (µL values).")
        # Stub times: handler may optionally return timing later; use settle delay heuristic
        times = [self._settle_delay_s]*replicates
        masses = [v/1000*start_density for v in volumes_ul]
        return PipettingResult(
            volume_target_ul=volume_mL*1000.0,
            measured_masses_g=masses,
            calculated_volumes_ul=volumes_ul,
            replicate_times_s=times,
            params=params,
            simulate=False
        )
