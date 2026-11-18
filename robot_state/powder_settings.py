"""Powder dispensing protocols. The values live in powder_settings.yaml (no hard-coded values);
this module only defines their shape and loads them. default_ps, sticky_ps, friendly_ps and
flowing_ps keep their old names for callers."""
from dataclasses import dataclass
from pathlib import Path

import yaml

SETTINGS_FILE = Path(__file__).resolve().parent / "powder_settings.yaml"


@dataclass
class PowderSettings:
    opening_deg: int  # (0, 90] how many degrees to open the valve during dispense
    percent_target: float  # (0, 1] dispense iter target is percent_target*remaining
    max_growth: float  # how much the dispense time can grow/shrink between iters (i.e. 2x)
    thresh: float  # settings take effect below this amount remaining - unless there is a lower one
    amplitude: int  # (0, 100] vibratory amplitude in %
    freq: int  # vibratory freq in hz
    max_shake_t: int  # max shake time in ms
    min_shake_t: int  # min shake time in ms
    shut_valve: bool  # shut valve between every iteration - needed for free-flowing powders


@dataclass
class PowderProtocol:
    tol: float  # best expected dispense tolerance in units of mg
    fast_settings: PowderSettings
    med_settings: PowderSettings
    slow_settings: PowderSettings
    scale_delay: float  # seconds before measuring powder... fall time for small amounts of solids


def load_protocols(path=SETTINGS_FILE):
    """{name: PowderProtocol} from the YAML file."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    protocols = {}
    for name, values in data.items():
        stages = {stage: PowderSettings(**values[stage]) for stage in ("fast_settings", "med_settings", "slow_settings")}
        protocols[name] = PowderProtocol(tol=values["tol"], scale_delay=values["scale_delay"], **stages)
    return protocols


_protocols = load_protocols()
default_ps = _protocols["default"]
sticky_ps = _protocols["sticky"]
friendly_ps = _protocols["friendly"]
flowing_ps = _protocols["flowing"]
