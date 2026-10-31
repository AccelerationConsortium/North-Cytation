"""Shared helper to plan and run gravimetric checks at workflow volumes."""

import math
from numbers import Real

import pandas as pd

from pipetting_data.liquid_registry import get_liquid

# "stages": legacy 3-stage optimisation that rewrites the base calibration row.
# "check_confirm": check batch, confirm only if surprising, damped update into the
# separate overaspirate_corrections.csv. Set to "stages" to go back; deleting
# overaspirate_corrections.csv returns every row to its base calibration value.
VALIDATION_PROTOCOL = "check_confirm"
CHECK_CONFIRM_REPLICATES = 4


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


def calibrated_volumes_ul(liquid_name, tip):
    """Return the calibrated target volumes (uL) stored for this liquid and tip."""
    from pipetting_data.pipetting_wizard import PipettingWizard

    wizard = PipettingWizard()
    volumes = set()
    for path in wizard.find_calibration_files(liquid_name):
        df = wizard.load_calibration_data(path)
        if df is None or 'tip_type' not in df.columns:
            continue
        for value in pd.to_numeric(df.loc[df['tip_type'] == tip, 'volume_target'], errors='coerce'):
            if math.isfinite(value) and value > 0:
                volumes.add(float(value))
    return sorted(volumes)


def run_liquid_validations(lash, sources, output_folder, replicates=None, *, vial_transfers=(),
                           protocol=None):
    """Run one session per source liquid and occupied tip range.

    ``sources`` contains (source vial, exact liquid registry name, expected uL).
    ``vial_transfers`` has the same form, but its volumes are total vial-transfer
    requests. Resolve these into executable chunks before choosing check volumes.
    Validate all liquid names before moving any hardware.

    Tips are chosen exactly as the workflow chooses them. Each requested volume is then
    rounded to the nearest calibrated volume for its tip, so validation only measures
    (and corrects) existing calibration rows and never extrapolates. Volumes that cannot
    be planned are skipped with a warning; validation must not end the workflow.
    """
    from pipetting_data.embedded_calibration_validation import validate_pipetting_accuracy
    from pipetting_data.pipetting_wizard import PipettingWizard

    planned = {}
    operations = [(entry, False) for entry in sources] + [(entry, True) for entry in vial_transfers]
    for (source_vial, liquid_name, volumes_ul), split in operations:
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
            try:
                if split:
                    chunk, _, _, tip = lash.nr_robot.plan_vial_transfer(volume / 1000, liquid_name)
                    volume = chunk * 1000
                else:
                    _, tip = lash.nr_robot.resolve_pipetting_setup(volume / 1000, liquid_name)
            except ValueError as error:
                lash.logger.warning("Not validating %s at %g uL: %s", liquid_name, volume, error)
                continue
            by_tip.setdefault(tip, set()).add(volume)
        for tip, requested in by_tip.items():
            rows = calibrated_volumes_ul(liquid_name, tip)
            if not rows:
                lash.logger.warning("Not validating %s with %s: no calibrated volumes for this tip",
                                    liquid_name, tip)
                continue
            nearest = {min(rows, key=lambda row: abs(row - volume)) for volume in requested}
            planned.setdefault((source_vial, liquid_name, tip), set()).update(nearest)

    protocol = protocol or VALIDATION_PROTOCOL
    check_confirm = protocol == "check_confirm"
    if replicates is None:
        replicates = CHECK_CONFIRM_REPLICATES if check_confirm else 3
    results = {}
    for (source_vial, liquid_name, tip), selected in planned.items():
        selected = sorted(selected)
        volumes = sorted({selected[0], selected[len(selected) // 2], selected[-1]})
        lash.logger.info("Validating %s from %s (%s, %s): %s uL", liquid_name, source_vial, tip, protocol, volumes)
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
            # The check batch must dispense exactly what the workflow dispenses.
            adaptive_correction=not check_confirm,
            compensate_overvolume=check_confirm,
            tip_type=tip,
            protocol=protocol,
        )
    return results
