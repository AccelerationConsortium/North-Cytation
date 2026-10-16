"""Shared liquid properties, independent of controllers and working directory."""

import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Optional, TypedDict

import yaml

LIQUID_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "robot_state" / "liquids.yaml"


class LiquidProperties(TypedDict):
    density: float
    refill_pipets: bool


def load_liquids(path: Path = LIQUID_REGISTRY_PATH) -> dict[str, LiquidProperties]:
    """Validate the YAML and retain the existing Python 'density' field name."""
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not isinstance(raw.get("liquids"), dict) or not raw["liquids"]:
        raise ValueError(f"{path} must contain a nonempty 'liquids' mapping")
    liquids: dict[str, LiquidProperties] = {}
    for name, properties in raw["liquids"].items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(f"{path}: liquid identifiers must be nonempty strings without surrounding spaces")
        if not isinstance(properties, dict):
            raise ValueError(f"Liquid '{name}' in {path} must contain density_g_ml and refill_pipets")
        for field in ("density_g_ml", "refill_pipets"):
            if field not in properties:
                raise ValueError(f"Liquid '{name}' in {path} is missing required field '{field}'")
        density = properties["density_g_ml"]
        if isinstance(density, bool) or not isinstance(density, Real) or not math.isfinite(density) or density <= 0:
            raise ValueError(f"Liquid '{name}' in {path}: density_g_ml must be a finite positive number")
        if type(properties["refill_pipets"]) is not bool:
            raise ValueError(f"Liquid '{name}' in {path}: refill_pipets must be true or false")
        liquids[name] = {"density": float(density), "refill_pipets": properties["refill_pipets"]}
    return liquids


def get_liquid(
    name: str, liquids: Optional[Mapping[str, LiquidProperties]] = None,
) -> LiquidProperties:
    """Require an exact registry key; never substitute another liquid's properties."""
    if liquids is None:
        liquids = load_liquids()
    if not isinstance(name, str) or name not in liquids:
        raise ValueError(
            f"Unknown liquid '{name}': no entry in {LIQUID_REGISTRY_PATH}. "
            "Add an entry under 'liquids' with density_g_ml and refill_pipets, then rerun. "
            "LIQUID/liquid_type must match a registry key exactly. "
            "TARGET_VIAL identifies a physical vial and does not register a liquid."
        )
    return liquids[name]
