# Degradation workflow (Serena)

Run from the repository root:

```powershell
python -m workflows.Degradation_serena
```

Dispenses an acid (from `ACID_LIBRARY`: HCl, TFA, p-TSA, citric acid, H2SO4,
or H3PO4) into polymer solution vials and reads UV-Vis spectra over time on a
schedule, to track degradation kinetics. Launch happens through
`execute(config=None, show_gui=True)`; importing the module does not start
the experiment.

Key config: `INPUT_VIAL_STATUS_FILE` is the starting vial layout,
`SCHEDULE_FILE` is the per-vial UV-Vis measurement time schedule, and
`CYTATION_PROTOCOL_FILE` points to the Cytation 5 sweep protocol
(`300_900_sweep.prt` by default). `REPLICATES` sets how many wells are read per
timepoint. `SIMULATE` controls hardware vs. simulated execution.
`VALIDATE_LIQUIDS=True` runs a compact pipetting validation (toluene volumes)
before the main run; `PREP_SOLUTIONS` controls whether acid/solution prep runs
as part of this call.

The post-experiment spectral analysis step imports an external analyzer that
is not part of this repository; that dependency only needs to be present on
the machine performing the real (non-simulated) analysis, not at import time
or during simulated runs.
