"""Reservoir calibration protocol for the North Robotics SDL platform.

THIS FILE IS HARDWARE-SPECIFIC AND WILL NOT RUN ON OTHER SYSTEMS. It imports the North Robot
libraries (master_usdl_coordinator, North_Safe). To adapt the framework to other hardware, start
from protocols/calibration_protocol_template.py.

A reservoir pump draws liquid from a bottle and dispenses it through a line into the vial in the
clamp, which sits on the balance, so every dispense is weighed. There is no tip, so there is no
pre-wetting; instead the line is primed once at the start (a small dispense, prime_volume_mL in
syringe_pumps.yaml), which does the same job.

Settings:
    cfg['experiment']: liquid, simulate (both required), random_seed (simulated readings)
    reservoir_hardware.yaml (this folder): reservoir_index, vial_status, quality_std_threshold_g,
        max_retries_per_measurement
    Parameters per measurement (all required): overaspirate_vol, aspirate_speed, dispense_speed,
        aspirate_wait_time, dispense_wait_time, valve_switch_delay

Simulation runs every robot step (North_Safe stubs the hardware); only the measured volume is
synthetic.

Unified minimal interface:
    initialize(cfg) -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None
"""
import logging
import math
import random
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from calibration_protocol_base import CalibrationProtocolBase

# Add utoronto_demo/ (protocols/ -> sdl_pipette_calibration/ -> utoronto_demo/) for North imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from master_usdl_coordinator import Lash_E
from pipetting_data.liquid_registry import get_liquid, load_liquids
from pipetting_data.pipetting_parameters import ReservoirParameters

logger = logging.getLogger(__name__)

HARDWARE_CONFIG_PATH = Path(__file__).parent / "reservoir_hardware.yaml"
HARDWARE_PARAMETERS = ('aspirate_speed', 'dispense_speed', 'aspirate_wait_time',
                       'dispense_wait_time', 'valve_switch_delay')
LIQUIDS = load_liquids()


def _require(mapping, key, where):
    if key not in mapping:
        raise ValueError(f"Missing required '{key}' in {where}")
    return mapping[key]


class ReservoirCalibrationProtocol(CalibrationProtocolBase):
    """Reservoir calibration protocol implementing the abstract interface."""

    _workflow_binding = None

    @contextmanager
    def workflow_session(self, lash_e, config):
        """Use a workflow's controller and phase config (experiment: liquid, simulate, reservoir_index,
        quality_std_threshold_g, max_retries_per_measurement, random_seed) instead of reservoir_hardware.yaml."""
        if self._workflow_binding is not None:
            raise RuntimeError("A reservoir calibration workflow session is already active")
        if config['experiment']['simulate'] != lash_e.simulate:
            raise ValueError("Protocol and controller simulation modes do not match")
        self._workflow_binding = (lash_e, deepcopy(config))
        try:
            yield self
        finally:
            self._workflow_binding = None

    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start the robot (or use the workflow's), check the pump holds the experiment liquid, and prime the line."""
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        liquid = _require(cfg['experiment'], 'liquid', "experiment configuration")
        if self._workflow_binding is not None:
            lash_e, config = self._workflow_binding
            settings = config['experiment']
            where = "workflow calibration config"
            if settings['liquid'] != liquid:
                raise ValueError("Engine liquid does not match the bound workflow")
            simulate = lash_e.simulate
            seed = _require(settings, 'random_seed', where)
        else:
            simulate = _require(cfg['experiment'], 'simulate', "experiment configuration")
            with open(HARDWARE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                settings = yaml.safe_load(f)
            where = HARDWARE_CONFIG_PATH.name
            lash_e = Lash_E(_require(settings, 'vial_status', where), simulate=simulate,
                            initialize_biotek=False, show_gui=False)
            seed = cfg.get('random_seed')
        density = get_liquid(liquid, LIQUIDS)['density']
        reservoir_index = _require(settings, 'reservoir_index', where)
        robot = lash_e.nr_robot
        pump_liquid = (robot.PUMP_CONFIG.get(reservoir_index) or {}).get('liquid')
        if pump_liquid != liquid:
            raise ValueError(f"Reservoir {reservoir_index} holds {pump_liquid!r} (syringe_pumps.yaml), "
                             f"not the experiment liquid {liquid!r}")

        state = {
            'initialized_at': datetime.now(),
            'liquid': liquid,
            'density_g_mL': density,
            'lash_e': lash_e,
            'simulate': simulate,
            'workflow_managed': self._workflow_binding is not None,
            'reservoir_index': reservoir_index,
            'quality_std_threshold_g': _require(settings, 'quality_std_threshold_g', where),
            'max_retries_per_measurement': _require(settings, 'max_retries_per_measurement', where),
            'clamp_capacity_mL': robot.get_config_parameter('vial_positions', 'clamp', 'vial_volume',
                                                            error_on_missing=True),
            'measurement_vials': list(robot.VIAL_DF['vial_name']),
            'measurement_vial': None,
            'measurement_count': 0,
            'simulation_random': random.Random(seed),
        }
        robot.home_robot_components()
        prime_volume = robot.PUMP_CONFIG[reservoir_index].get('prime_volume_mL')
        if prime_volume is None:
            raise ValueError(f"Reservoir {reservoir_index} has no prime_volume_mL in syringe_pumps.yaml")
        vial = self._vial_with_room(state, prime_volume)
        robot.prime_reservoir_line(reservoir_index, vial, prime_volume)
        logger.info("Reservoir calibration ready: liquid=%s, reservoir=%s, primed %.3f mL into %s",
                    liquid, reservoir_index, prime_volume, vial)
        return state

    def _vial_with_room(self, state, volume_mL):
        """The measurement vial in the clamp, swapped for the next vial in the file when it has no room."""
        robot = state['lash_e'].nr_robot
        capacity = state['clamp_capacity_mL']
        current = state['measurement_vial']
        if current is not None and robot.get_vial_info(current, 'vial_volume') + volume_mL <= capacity:
            return current
        for vial in state['measurement_vials']:
            if vial != current and robot.get_vial_info(vial, 'vial_volume') + volume_mL <= capacity:
                break
        else:
            raise RuntimeError(f"No measurement vial has room for {volume_mL:.3f} mL "
                               f"(capacity {capacity:.3f} mL)")
        if current is not None:
            robot.return_vial_home(current)
        robot.move_vial_to_location(vial, "clamp", 0)
        state['measurement_vial'] = vial
        logger.info("Measurement vial now %s", vial)
        return vial

    def _reservoir_parameters(self, params):
        """ReservoirParameters from the optimizer's params; every parameter is required."""
        hardware = params['parameters'] if isinstance(params.get('parameters'), dict) else params
        values = {name: _require(hardware, name, "measurement parameters") for name in HARDWARE_PARAMETERS}
        values['overaspirate_vol'] = _require(params, 'overaspirate_vol', "measurement parameters")
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Reservoir parameter {name} must be a finite number, got {value!r}")
        return ReservoirParameters(**values)

    def _is_stable(self, stability_info, std_threshold):
        """Both mass baselines stable: more than half the readings steady, or std below the threshold."""
        def stable(prefix):
            percent = 100 * stability_info[f'{prefix}_stable_count'] / max(stability_info[f'{prefix}_total_count'], 1)
            return percent > 50.0 or stability_info[f'{prefix}_baseline_std'] < std_threshold
        return stable('pre') and stable('post')

    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any],
                replicates: int = 1) -> List[Dict[str, Any]]:
        """Dispense volume_mL from the reservoir into the measurement vial and weigh it."""
        reservoir_params = self._reservoir_parameters(params)
        robot = state['lash_e'].nr_robot
        reservoir_index = state['reservoir_index']
        dispensed_mL = volume_mL + reservoir_params.overaspirate_vol
        results = []
        for rep in range(replicates):
            start_time = datetime.now()
            rep_start = time.perf_counter()
            stability_info = None
            quality_ok = True
            if state['simulate']:
                vial = self._vial_with_room(state, dispensed_mL)
                robot.dispense_into_vial_from_reservoir(
                    reservoir_index, vial, volume_mL, reservoir_params=reservoir_params,
                    liquid=state['liquid'], return_home=False)
                noise = state['simulation_random'].uniform(-0.001, 0.001) * volume_mL
                measured_volume_mL = max(volume_mL * 0.995 + reservoir_params.overaspirate_vol + noise, 0.0)
                attempts = 1
            else:
                max_attempts = 1 + state['max_retries_per_measurement']
                for attempts in range(1, max_attempts + 1):
                    vial = self._vial_with_room(state, dispensed_mL)
                    mass_g, stability_info = robot.dispense_into_vial_from_reservoir(
                        reservoir_index, vial, volume_mL, reservoir_params=reservoir_params,
                        liquid=state['liquid'], measure_weight=True, continuous_mass_monitoring=True,
                        save_mass_data=True, return_home=False)
                    quality_ok = self._is_stable(stability_info, state['quality_std_threshold_g'])
                    if quality_ok:
                        break
                    logger.warning("Unstable mass reading (attempt %d/%d)", attempts, max_attempts)
                measured_volume_mL = mass_g / state['density_g_mL']
            elapsed_s = time.perf_counter() - rep_start
            state['measurement_count'] += 1
            result = {
                'replicate': rep + 1,
                'volume': measured_volume_mL,
                'elapsed_s': elapsed_s,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'target_volume_mL': volume_mL,
                'measured_volume_mL': measured_volume_mL,
                'reservoir_index': reservoir_index,
                'measurement_vial': state['measurement_vial'],
                'quality_ok': quality_ok,
                'measurement_budget_consumed': attempts,
                'overaspirate_vol': reservoir_params.overaspirate_vol,
                **{name: getattr(reservoir_params, name) for name in HARDWARE_PARAMETERS},
            }
            if stability_info is not None:
                result['stability_info'] = stability_info
            results.append(result)
            logger.info("Reservoir %.3f mL -> measured %.4f mL in %.1f s%s", volume_mL, measured_volume_mL,
                        elapsed_s, "" if quality_ok else " (unstable reading)")
        return results

    def wrapup(self, state: Dict[str, Any]) -> bool:
        """Return the measurement vial home, home the robot and release the controller."""
        lash_e = state.get('lash_e')
        if lash_e is None:
            return True
        robot = lash_e.nr_robot
        try:
            if state.get('measurement_vial') is not None:
                robot.return_vial_home(state['measurement_vial'])
            robot.move_home()
            logger.info("Reservoir calibration finished: %d measurements", state['measurement_count'])
            return True
        except Exception as error:
            logger.warning("Reservoir cleanup failed: %s", error)
            return False
        finally:
            if not state.get('workflow_managed'):  # a workflow keeps its controller between phases
                try:
                    robot.c9.network.disconnect()  # release the serial port for the next process
                except Exception as error:
                    logger.warning("Could not close the North Robot serial connection: %s", error)

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """None: the pump repeats strokes for volumes beyond its capacity, so there is no tip limit."""
        return []


# Export the protocol instance for clean importing
protocol_instance = ReservoirCalibrationProtocol()
