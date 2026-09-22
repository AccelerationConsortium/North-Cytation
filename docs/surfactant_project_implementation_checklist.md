# Surfactant Project Implementation Checklist

This checklist is based on the proposal and limited to the workflow needed for the surfactant project. It intentionally excludes the broader polymer/lipid extraction work and the separate multi-workflow comparison work.

## Priority 1: Must implement for the proposal

### 1. Replicate-aware experiment execution
- Add support for running the same recipe set multiple times in one experiment.
- Track replicate number in the output table.
- Save replicate-specific metadata so results can be grouped and compared.
- Default target: 3 repeats for each condition or plate set.

### 2. Dye selection / assay parameterization
- Make the dye configurable instead of hard-coded to pyrene.
- Support at minimum:
  - pyrene
  - coumarin-6
  - Nile-red
- Ensure the measurement protocol and readout columns match the selected dye.
- Keep the current pyrene path as the default for compatibility.

### 3. Plate format flexibility
- Add support for 96-well, 48-well, and 24-well layouts.
- Parameterize well volume assumptions, plate geometry, and plate-specific recipe scaling.
- Do not assume the current 96-well recipe layout for all experiments.

### 4. Solvent / dye addition order controls
- Add configurable dye solvent choice.
- Allow testing of DMSO vs alternative solvent pathways.
- Add a mode for dye addition before surfactant mixing or after surfactant mixing.
- Include metadata for solvent and addition order in the saved results.

### 5. Randomized well placement option
- Add a mode that assigns recipes to physical wells in a randomized pattern rather than a fixed linear sequence.
- Preserve the original fixed-order mode for comparison.
- Store the randomized mapping in the results output.

### 6. Experimental uncertainty / repeatability reporting
- Compute per-condition variability across replicates.
- Add summary statistics for:
  - mean
  - standard deviation
  - coefficient of variation
  - replicate count
- Make the uncertainty output directly usable for comparing real changes vs assay noise.

## Not needed for this phase / explicitly excluded

### 7. Polymer surfactant project branch
- Do not treat this as part of the current surfactant CMC/CMB workflow.
- Polymer/lipid solubilization work should remain a separate assay class or distinct workflow branch.
- This is not a small extension of the current script; it is a different application.

## Implementation notes

### Scope that matches the current workflow
The current workflow already handles the following and should remain the base assay engine:
- 2D surfactant concentration grid
- pyrene-based fluorescence and turbidity readout
- recipe replay from saved CSV
- refill logic during execution
- plate-by-plate processing
- result file writing and analysis heatmaps

### Scope that needs change
The following are the actual additions required to fit the proposal:
- replicate mode
- dye configuration
- plate type configuration
- solvent and addition-order parameterization
- randomized well assignment
- replicate-based uncertainty output

## Suggested order of execution
1. Add replicate-aware result metadata and repeat execution.
2. Add dye selection and measurement protocol switching.
3. Add plate-format configuration.
4. Add solvent and addition-order controls.
5. Add randomization option.
6. Add uncertainty summary generation.
7. Leave polymer surfactant workflow separate.

## Practical recommendation
The best near-term approach is to extend the existing replay workflow rather than create a brand-new general assay framework. The replay workflow is already close to the proposal’s basic CMC/CMB assay and can be adapted to support the first six items without redesigning the whole system.
