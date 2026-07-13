# Safety & Chemical Compatibility

System: North Robotics SDL (liquid handling robot + track + heater + photoreactor)

---

## Quick-Reference Constraints

| Category | Limit |
|---|---|
| Max temperature | 100°C (heater block) |
| Min temperature | Ambient (no cooling) |
| Pressure | Atmospheric only — no pressurized reactions |
| Ventilation | Snorkel vent only — **not a fume hood** |
| Wetted materials | PTFE, stainless steel, borosilicate glass, silicone/PTFE septa |
| Default wellplate | Polystyrene 96-well — solvent-incompatible chemistry requires quartz plate |

---

## Solvents

### Approved — no special restrictions
Aqueous buffers, water, glycerol, ethylene glycol, propylene glycol, DMSO, propylene carbonate, dimethyl carbonate, ethyl lactate.

### Approved — volatile (evaporation precautions apply)
Cap vials when not in use. Measure promptly after dispensing. Avoid prolonged open exposure.
Use **PTFE-lined caps** for all solvents in this category — silicone-backed caps degrade with repeated organic solvent contact.

- Ethanol, methanol, isopropanol (IPA), n-propanol, n-butanol, tert-butanol
- Acetone, methyl ethyl ketone (MEK / 2-butanone)
- Ethyl acetate, ethyl lactate (if used as solvent rather than co-solvent), diethyl carbonate
- Acetonitrile (moderate toxicity — keep volumes reasonable; acceptable under snorkel vent)
- Tetrahydrofuran (THF) — also check polystyrene compatibility before use
- 2-Methyltetrahydrofuran (2-MeTHF) — same compatibility caveats as THF; greener alternative
- Glymes: 1,2-dimethoxyethane (DME/monoglyme), diglyme — flammable, check polystyrene compatibility
- Acetic acid (glacial) — volatile and corrosive; use dilute solutions where possible

### Restricted — non-chlorinated organic solvents
Toluene, xylene, cyclohexane, hexane, heptane, and similar.
- Keep vials **capped when not in use** — same evaporation precautions as volatile category above
- Use **PTFE-lined caps** — silicone septa degrade rapidly with aromatic and aliphatic hydrocarbons
- **Not compatible with polystyrene wellplates** — use quartz plate
- Permitted under snorkel vent in small volumes
- Flammable solvents must not contact the heater block above their flash point

### Not supported
- Chlorinated solvents (DCM, chloroform, CHCl₃, CCl₄, etc.) — not safe under snorkel vent only
- Diethyl ether — flash point too low for safe use near heater
- **DMF (N,N-dimethylformamide) and NMP (N-methylpyrrolidone)** — commonly requested; not supported due to reproductive toxicity and vapour hazard under snorkel vent only
- 1,4-Dioxane — possible carcinogen, not safe under snorkel vent only
- Pyridine — acute toxicity and strong odour; not safe under snorkel vent only
- Benzene — carcinogen; not supported under any circumstance
- Any solvent with significant acute inhalation toxicity at room-temperature vapour pressures

---

## Temperature & Pressure

- **Maximum heater temperature: 100°C.** Do not exceed.
- No cooling capability — samples equilibrate to ambient after heating.
- Reactions that generate gas or significantly increase headspace pressure are **not supported**. Pressurized vials are a safety hazard with the robotic gripper and pipetting system.

### Solvent heating limits
The boiling point is the hard ceiling for heating any solvent:
- **Sealed vial at or above boiling point** → pressure buildup → safety hazard. Do not heat sealed vials within ~10°C of the solvent's boiling point.
- **Open or loosely capped vial near boiling point** → rapid evaporation, incorrect concentrations, possible dry vial.
- Volatile solvents heated above ~40°C should use sealed vials and PTFE-lined caps; account for any evaporation in volume calculations.

Common solvent boiling points for reference:

| Solvent | Boiling point (°C) | Practical max at 100°C heater |
|---|---|---|
| Acetone | 56 | Not suitable for heated experiments |
| Methanol | 65 | Not suitable for heated experiments |
| THF / 2-MeTHF | 66 / 80 | Not suitable for heated experiments |
| Ethanol | 78 | Marginal — stay below 65°C sample temp |
| Acetonitrile | 82 | Stay below 70°C sample temp |
| IPA | 82 | Stay below 70°C sample temp |
| Ethyl acetate | 77 | Not suitable for heated experiments |
| Toluene | 111 | Compatible with heater range |
| DME (glyme) | 85 | Stay below 70°C sample temp |
| DMSO | 189 | Fully compatible with heater range |
| Water / buffers | 100 | Compatible — degas if gas evolution expected |

---

## Chemical Compatibility

### Acids
Dilute to moderately concentrated acids (HCl, H₂SO₄, H₃PO₄, acetic acid, citric acid, etc.) are generally compatible.
- **Avoid HF** — etches glass vials and degrades PTFE components at concentration.
- **Avoid concentrated oxidizing acids** (concentrated HNO₃, aqua regia) — can damage stainless steel wetted parts and generate fumes.

### Bases
Dilute bases and biological buffers (NaOH, KOH, amines, carbonates) are acceptable.
- **Avoid strong bases at very high pH (>13)** — can degrade silicone septa and attack glass vials over time.
- Concentrated NaOH/KOH solutions should be used with care; etching of glass vials is possible at high concentration and elevated temperature.

### Oxidizers
Mild oxidizers (H₂O₂ at low concentration, NaIO₄, etc.) are acceptable with caution.
- Strong oxidizers (concentrated H₂O₂, KMnO₄, persulfates at high concentration) — risk of degrading septa, caps, and PTFE components; assess case-by-case.

### Toxic and volatile compounds
**Do not use** compounds that are both acutely toxic and volatile without proper fume hood containment. The snorkel vent reduces but does not eliminate exposure risk.

### Biological and radioactive materials
Not supported. No BSL-2+ pathogens, no radioactive materials.

---

## Labware Compatibility

### Vials
Borosilicate glass 8 mL vials with plastic screw caps. Caps have a PTFE-lined silicone septa.
- PTFE lining provides reasonable chemical resistance, but silicone backing can degrade with concentrated organics over long contact times.
- Use PTFE-only caps for aggressive organic solvents if available.
- Anything that etches glass (HF, hot strong base) degrades the vial itself.

### Wellplates
- **Polystyrene (default):** incompatible with most organic solvents (toluene, THF, hexane, etc.). Compatible with aqueous, DMSO (<5%), short-contact ethanol/methanol.
- **Quartz (available, not preferred):** chemically inert, use when polystyrene is incompatible. Reserve for cases where polystyrene is clearly unsuitable — quartz plates are expensive and limited in quantity.

---         

