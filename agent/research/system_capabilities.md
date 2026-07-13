# System Capabilities

System: North Robotics SDL (`Lash_E` coordinator + peripheral instruments)

---

## Instruments

| Instrument | Class | Notes |
|---|---|---|
| North Robot (liquid handler) | `North_Robot` | Robotic arm on a linear platform. Moves vials between rack positions, clamp, and photoreactor. Caps/decaps vials. Pipettes (aspirate, dispense, mix). Vortexes. Dispenses from reservoirs. |
| North Track | `North_Track` | Automated wellplate transport between pipetting area and Cytation |
| Cytation 5 plate reader | `biotek_new` | UV-Vis absorbance + fluorescence; protocol-driven |
| Photoreactor | `photoreactor_controller` | RPi Pico-controlled; light intensity + stir speed + duration |
| Heater + stirrer | `North_Temp` | Single channel, up to 100°C; integrated magnetic stirring; block temperature feedback (not in-vial) |
| Powder dispenser | `North_Powder` | Closed-loop gravimetric dispensing to mg-level targets; functional |

---

## Measurement

### Cytation 5 (UV-Vis / Fluorescence)
- Wellplate-based — samples must be transferred to a wellplate (polystyrene or quartz)
- Supported plate formats: 96-well and 48-well; other formats can be configured in principle
- Measurements are protocol-driven (`.prt` files configured in Cytation software)
- Supports multiple protocols per plate read (e.g. fluorescence + absorbance in one call)
- Supports replicate reads
- Returns data as a pandas DataFrame with MultiIndex columns `(rep_protocol, wavelength)`
- **Speed:** single-wavelength absorbance reads are fastest; spectral scans are slower; fluorescence reads are slower than absorbance reads

### Gravimetric (scale)
- Integrated scale on the robot; real-time continuous reading
- Resolution: ~0.2 mg
- Used during powder dispensing and for liquid volume verification by mass
- In-vial measurement — no plate transfer required

### Measurement design notes
- Optical measurements can serve as proxies for other properties — e.g. absorbance at 600 nm as a proxy for turbidity/precipitation, absorbance at other wavelengths for coloured products or reactants
- Fluorescent probes can be used if the experiment is designed around them (e.g. solvatochromic dyes, fluorescent indicators); probe must be compatible with the sample chemistry and not interfere with other measurements

### What is not available
- NMR, MS, GC, HPLC, FTIR — no chromatographic or spectroscopic instruments beyond UV-Vis/fluorescence
- Real-time in-vial optical monitoring — optical measurement requires transfer to a wellplate
- Kinetic plate reads are possible (multiple reads over time) but the wellplate must remain in the Cytation

---

## Process Operations

### Liquid handling
- Aspirate/dispense from vials into vials or wellplate wells
- Dispense from reservoirs (for high-volume reagents used across many wells)
- Mix via aspirate/dispense cycles in vials or wells
- Vortex vials in-place for mixing
- Substock preparation (serial dilution from stock to working concentrations)
- Minimum reliable pipette volume: ~5 uL (tip-dependent)
- Maximum reliable viscosity: ~1500 cP (glycerol-range); pipetting parameters are calibrated per liquid using AI-guided (Bayesian) calibration

### Photoreactor
- Single vial, single position
- Light colours available: white, green, blue, violet — selected at experiment start; cannot switch mid-experiment
- Controllable: light intensity, stir speed (RPM), duration
- Vial is moved to the reactor position and returned home automatically

### Heating / stirring
- Set temperature on a single heater channel (up to 100°C)
- Independent magnetic stirrer speed control
- Temperature feedback is from a sensor in the heating block, not inside the vial — actual vial temperature lags the setpoint; account for equilibration time

### Powder dispensing
- Closed-loop gravimetric dispensing using an integrated scale
- Targets in mg; multi-channel cartridge

### Wellplate management
- Automated plate pickup from source stack
- Automated plate discard to waste stack
- Track moves plate between pipetting area and Cytation
- Lids are handled separately from plates; a plate can be moved with a lid on, and lids can be removed/replaced independently

---

## Workflow Pattern

See [workflow_construction.md](workflow_construction.md) for the full step-by-step pattern with method references. 
