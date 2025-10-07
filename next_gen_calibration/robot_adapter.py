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
    """Abstracts pipetting operations. Replace simulate_* methods with actual hardware calls."""
    def __init__(self, liquid: str = "water", simulate: bool = True, seed: int | None = None):
        self.liquid = liquid
        self.simulate = simulate
        if seed is not None:
            random.seed(seed)
        # Liquids loaded lazily from liquids.yaml (simple dict: name -> props)
        self._liquids_cache: Dict[str, Dict[str, Any]] | None = None

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
        raise NotImplementedError("Hardware integration not implemented. Override in subclass.")
