"""Plan and run gravimetric checks at volumes used by a workflow."""

import math
from numbers import Real

from pipetting_data.liquid_registry import get_liquid


def volume_groups(volumes_ul):
    """Keep the small and large pipette ranges in distinct validation sessions.

    Exactly 200 uL belongs to the large-tip range. For each occupied range,
    measure its endpoints and a representative middle volume from the plan.
    """
    values = sorted({float(v) for v in volumes_ul
                     if isinstance(v, Real) and not isinstance(v, bool)
                     and math.isfinite(v) and v > 0})
    groups = {}
    for label, selected in (("small", [v for v in values if v < 200]),
                            ("large", [v for v in values if v >= 200])):
        if selected:
            groups[label] = sorted({selected[0], selected[len(selected) // 2], selected[-1]})
    return groups


def run_liquid_validations(lash, sources, output_folder, replicates=3):
    """Run one session per source liquid and occupied tip range.

    ``sources`` contains (source vial, exact liquid registry name, expected uL).
    Validate all liquid names before moving any hardware.
    """
    from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
    from pipetting_data.pipetting_wizard import PipettingWizard

    planned = []
    for source_vial, liquid_name, volumes_ul in sources:
        values = sorted({float(v) for v in volumes_ul
                         if isinstance(v, Real) and not isinstance(v, bool)
                         and math.isfinite(v) and v > 0})
        if not values:
            continue
        get_liquid(liquid_name)
        if not PipettingWizard(logger=lash.logger).find_calibration_files(liquid_name):
            raise ValueError(f"No calibration file for {liquid_name}; cannot validate tip-specific settings")
        by_tip = {}
        for volume in values:
            _, tip = lash.nr_robot.resolve_pipetting_setup(
                volume / 1000, liquid_name, compensate_overvolume=False)
            by_tip.setdefault(tip, []).append(volume)
        for tip, selected in by_tip.items():
            volumes = sorted({selected[0], selected[len(selected) // 2], selected[-1]})
            planned.append((source_vial, liquid_name, tip, volumes))

    results = {}
    for source_vial, liquid_name, tip, volumes in planned:
        lash.logger.info("Validating %s from %s (%s): %s uL", liquid_name, source_vial, tip, volumes)
        results[(source_vial, liquid_name, tip)] = validate_pipetting_accuracy(
            lash_e=lash,
            source_vial=source_vial,
            destination_vial=source_vial,
            liquid_type=liquid_name,
            volumes_ml=[volume / 1000 for volume in volumes],
            replicates=replicates,
            output_folder=str(output_folder),
            switch_pipet=False,
            save_raw_data=not lash.simulate,
            condition_tip_enabled=True,
            conditioning_volume_ul=min(150 if tip == "small_tip" else 800, max(volumes)),
            adaptive_correction=True,
            compensate_overvolume=False,
            tip_type=tip,
        )
    return results
