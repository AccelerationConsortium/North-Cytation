# Amine Protonation Workflow — Summary

**Purpose**
Automates the preparation of protonated-amine / metal-salt mixtures in ethanol for deposition onto slides. Each call to the workflow function processes one reactor vial from start to finish.

**What the robot does, in order**

| Step | Action |
|---|---|
| 1 | Dispense amine stock into the reactor vial (already contains 5 mL ethanol) |
| 2 | Move reactor vial into the photoreactor; start stir motor |
| 3 | Add 5 uL of 6 M HCl while stirring — protonates the amine |
| 4 | Stop stirring; return reactor vial to its rack position |
| 5 | Dispense metal salt stock into the reactor vial |
| 6 | Vortex the vial for 10 s to mix |
| 7 | *(stub)* Deposit droplet onto slide — to be implemented once slide holder positions are defined |

Slack messages are sent at start and end of each experiment (hardware only).

**Files**
- Workflow: `workflows/amine_protonation_workflow.py`
- Vial layout: `status/amine_protonation_vials.csv`

**Fixed parameters** (set at the top of the workflow file, not per-experiment):

| Parameter | Value | Notes |
|---|---|---|
| HCl volume | 5 uL | 6 M HCl |
| Stir RPM | 600 | Photoreactor fan |
| Vortex time | 10 s | After metal salt addition |
| Slide heater | 60 °C | Drives off ethanol from deposited droplets |

## Running a single experiment

The slide heater is turned on once before all experiments and off once after — it is not part of the per-experiment function. Robot homing also happens once at the start.

```python
lash_e = Lash_E("status/amine_protonation_vials.csv", simulate=False, ...)
lash_e.nr_robot.home_robot_components()
lash_e.temp_controller.set_temp(60)  # slide heater on

try:
    amine_protonation_workflow(
        lash_e,
        reactor_vial         = "reactor_vial_1",
        amine_vial           = "amine_1",
        amine_volume_mL      = 0.500,
        metal_salt_vial      = "metal_salt_1",
        metal_salt_volume_mL = 0.500,
    )
finally:
    lash_e.temp_controller.turn_off_heating()  # slide heater off
    lash_e.nr_robot.move_home()
```

## Combinatorial use — running a library

The `Lash_E` object is initialized once; `amine_protonation_workflow()` is called once per condition. You can vary the amine identity, metal salt identity, and their volumes across calls. Everything that is fixed (HCl dose, stir speed, vortex time, slide heater) stays constant automatically.

```python
experiments = [
    # (reactor_vial,      amine_vial,  vol_mL,  metal_vial,     vol_mL)
    ("reactor_vial_1",   "amine_1",   0.300,   "metal_salt_1", 0.700),
    ("reactor_vial_2",   "amine_1",   0.500,   "metal_salt_1", 0.500),
    ("reactor_vial_3",   "amine_2",   0.500,   "metal_salt_1", 0.500),
    ("reactor_vial_4",   "amine_2",   0.500,   "metal_salt_2", 0.500),
]

lash_e.nr_robot.home_robot_components()
lash_e.temp_controller.set_temp(60)  # slide heater on — stays on for full batch
try:
    for rv, av, avol, mv, mvol in experiments:
        amine_protonation_workflow(lash_e, rv, av, avol, mv, mvol)
finally:
    lash_e.temp_controller.turn_off_heating()  # slide heater off after all experiments
    lash_e.nr_robot.move_home()
```

Each new amine or metal salt just needs to be added as a row in `amine_protonation_vials.csv`. Reactor vials must be pre-loaded with 5 mL ethanol and have `cap_type=open`.
