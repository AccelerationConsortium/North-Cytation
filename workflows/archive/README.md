# Workflow Archive

Archived on 2026-10-07 at the user's request because these scripts do not meet
the scheduler Setup convention: a matching `workflow_configs/<script_stem>.yaml`
containing `INPUT_VIAL_STATUS_FILE` that points to an existing vial CSV.

This is a configuration compatibility archive, not a determination that the
experiments are obsolete or scientifically invalid. Scripts were moved without
changing their contents. Associated configs are now in `workflow_configs/archive/`;
vial files remain in their original locations. Restore a script to `workflows/`
and its config to `workflow_configs/` after bringing its configuration into this
convention. Setup compatibility alone does not guarantee execution compatibility.

## No Matching Config

- `amine_protonation_workflow.py`
- `amine_protonation_workflow_SPmodified.py`
- `calibration_vials_short_mass_validation.py`
- `color_matching.py`
- `create_surfactant_substocks.py`
- `ilya_workflow.py`
- `microgel_workflow.py`
- `mof_synthesis_workflow_concurrent.py`
- `mof_synthesis_workflow_multiple.py`
- `nanoparticle_workflow.py`
- `peroxide_serena_v2.py` (uses the differently named `peroxide_serena_v3` config)
- `polymer_phospholipid_turbidity_assay.py`
- `run_4d_algorithm_comparison.py`
- `safe_position_pipetting_test.py`
- `sample_workflow_v2.py`
- `sands_workflow_SP.py`
- `zif8_bsa_workflow.py`

## Vial-File Key Missing From Matching Config

- `color_mixing.py`
- `enhanced_SP_arm_position_program_v1.1_xy_coordinate_system.py`
- `glycerol_dispense_baseline.py`
- `ilya_workflow_v2.py`
- `mof_synthesis_workflow.py`
- `polymer_dye_kinetics_workflow.py`
- `surfactant_grid_replay.py`

## Retained Exceptions

`workflows/vial_risk_analyzer.py` remains in place because
`surfactant_multidimensional_workflow.py` imports it. It is an analysis helper,
not a standalone queued experiment. `workflows/workflow_template.py` remains
available for creating new workflows; its placeholder inputs are not runnable.

The scheduler currently discovers only top-level workflow scripts. Archived
scripts therefore disappear from its list when the scheduler is reopened.
The scheduler now uses a generic matching-YAML/INPUT_VIAL_STATUS_FILE Setup
handler. Restored scripts need to follow that convention; this does not by
itself certify their execution compatibility.