# Config lists

`fluorescence_six_pairs.yaml` is the local twelve-job fluorescence queue:
ethyl acetate preparations, DMSO preparations, then the six measurements.
Its full configs and run notes are in
`workflow_configs/fluorescence_six_pairs/` (locally Git-ignored).

In the Scheduler GUI, select **Import Queue** and choose a YAML file from this
folder. The jobs are appended in file order. The initial empty row is removed.
Each imported config becomes a private copy that can be reviewed in **Setup**.
Importing does not run experiments or change the original config files.

A list contains workflow names (without `.py`) and config file paths:

```yaml
- workflow: fluorescence_calibration_workflow
  config_file: fluorescence/nilered_ethylacetate_prepare.yaml
- workflow: fluorescence_calibration_workflow
  config_file: fluorescence/nilered_ethylacetate_execute.yaml
```

Config file paths are relative to the list file, or can be absolute. Paths
*inside* each workflow config, including `INPUT_VIAL_STATUS_FILE`, keep the
usual convention of being relative to the repository root.

For a twelve-job fluorescence queue, list six preparation configs and six
execution configs in the desired order. Preparation configs use
`PREPARE_ONLY: true`; execution configs use `PREPARE_ONLY: false`. Both members
of each pair must use matching recipes and the same vial inventory so the
execution job can reuse prepared substocks. Every file must be a complete
workflow config; the list does not supply defaults or overrides.

All listed files and workflow names are checked before rows are added.
Review the imported rows and simulate the full queue to check liquid use and
state carried between jobs before running hardware.
