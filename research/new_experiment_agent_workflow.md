# New Experiment Agent Workflow

A repeatable process for using an AI agent to go from a scientific topic to a running SDL workflow on Lash-E.

---

## Overview

The agent handles the full pipeline: literature research → experiment proposal → user approval → code generation → chemical sourcing. Each stage produces a concrete artifact, and the agent pauses for human review before writing any code or placing any orders.

---

## Stage 1 — Literature Research and Feasibility Screening

**Input:** A topic or research goal from the user (e.g. "synthesis of MOFs", "surfactant CMC screening", "photocatalytic dye degradation").

**Agent actions:**

1. Search for recent literature on the topic. Focus on:
   - What experimental parameters matter (temperature, concentration, solvent, pH, reagent ratios, light dose, etc.)
   - What outputs are measured and how (UV-Vis, fluorescence, yield, turbidity, etc.)
   - Typical scale and timescale of experiments

2. Cross-reference findings against `research/system_capabilities.md`:
   - Are the required measurements achievable with the Cytation 5 (UV-Vis / fluorescence)?
   - Are required operations supported (liquid handling, heating, photoreaction, powder dispensing)?
   - Are volumes and concentrations within reliable pipetting range (>5 uL, typically aqueous to moderate viscosity)?

3. Cross-reference against `research/safety_chemical_compatibility.md`:
   - Are all chemicals compatible with wetted materials (PTFE, glass, stainless steel)?
   - Are solvents permissible under snorkel ventilation?
   - Are temperatures within range (ambient to 100°C, atmospheric pressure only)?
   - Flag any chemicals that require fume hood or special containment — these cannot be used.

**Output:** A brief summary of findings: what can be adapted for Lash-E and what cannot.

---

## Stage 2 — Identify Inputs and Outputs

From the feasibility-screened literature, define the active learning campaign space:

**Inputs (parameters the robot will vary):**
- List each variable with its type (`continuous`, `categorical`, `integer`), range or set of values, and units
- Example: `{"name": "linker_concentration_mM", "type": "continuous", "bounds": [1.0, 50.0]}`

**Outputs (what gets measured):**
- List each measured quantity, the instrument used, and what it represents physically
- Note which output is the primary optimization target and whether it should be minimized or maximized
- Example: `absorbance_700nm` (Cytation 5, proxy for MOF particle yield / turbidity)

**Constraints:**
- Any linear or physical constraints linking inputs (e.g. total volume must equal 200 uL)
- Any chemistry constraints (e.g. metal:linker ratio must stay above 1:2)

---

## Stage 3 — Write Experiment Proposal

The agent writes a markdown file (saved to `research/proposals/<topic>_proposal.md`) containing:

1. **Background** — 2–4 sentences on the scientific motivation and what makes this suitable for active learning
2. **Experiment design** — what a single trial looks like (what gets dispensed, what gets measured, how long it takes)
3. **Lash-E integration** — which instruments are used and in what order; confirm all operations are within system capabilities
4. **Safety summary** — confirm all chemicals pass the safety screen; list any precautions
5. **Inputs table** — name, type, range/values, units
6. **Outputs table** — name, instrument, physical meaning, optimization direction
7. **Expected number of trials** — rough estimate of Sobol initialization trials + Bayesian trials needed to see useful optimization signal

**The agent stops here and sends the proposal to the user for review.**

---

## Stage 4 — User Approval

The user reviews the proposal and either:

- **Approves** → agent proceeds to Stage 5
- **Requests changes** → agent revises the proposal and returns to this stage
- **Rejects** → agent discards and can restart with a different topic

---

## Stage 5 — Code Generation

On approval, the agent generates three artifacts. Refer to the relevant guide for conventions before writing each one:

| Artifact | Guide | Output path |
|---|---|---|
| Workflow | `research/workflow_construction.md` | `workflows/<topic>_workflow.py` |
| Recommender | `research/recommender_guide.md` | `recommenders/<topic>_recommender.py` |
| Analysis + plotting | `research/analysis_and_data.md` | `analysis/<topic>_analysis.py` |

**Important conventions to follow during generation:**
- Workflow must support `simulate=True` from the start; synthetic data function must be included in the analysis module (see `analysis_and_data.md` — Synthetic Data for Simulation)
- No Unicode characters in log messages (`uL` not `μL`, `->` not `→`)
- Save `well_recipes_df` to CSV after every measurement and analysis step
- Default recommender is Ax; only use BayBe if there is a specific reason
- No silent defaults — missing values must raise, not substitute

---

## Stage 6 — Chemical Sourcing

The agent compiles a **chemicals list** for the experiment:

1. For each reagent identified in the proposal:
   - Chemical name and CAS number
   - Required purity (default: reagent grade unless the experiment requires higher)
   - Estimated quantity needed for a full optimization campaign (N trials × volume per trial × safety factor of 2×)
   - Supplier link (Sigma-Aldrich / MilliporeSigma, Fisher Scientific, or equivalent)
   - Price per unit at time of search
   - Recommended purchase quantity (minimize cost per unit while not over-buying)

2. The list is saved to `research/proposals/<topic>_chemicals.md`.

3. **The agent sends a message to the user containing:**
   - The chemicals list with links and prices
   - A note that the workflow, recommender, and analysis files have been created and are ready for review
   - A reminder to run in `simulate=True` mode first before using real reagents

---

## Agent Prompt Template

Use this prompt to invoke the full pipeline:

```
Research the topic: "<TOPIC>"

Follow the new_experiment_agent_workflow.md in research/:
1. Search for literature on this topic and screen against Lash-E's capabilities (system_capabilities.md) and safety constraints (safety_chemical_compatibility.md).
2. Identify the input parameters and output measurements for an active learning campaign.
3. Write a proposal markdown to research/proposals/<topic>_proposal.md and present it for approval before writing any code.
4. On approval, generate the workflow (workflows/), recommender (recommenders/), and analysis module (analysis/) following the conventions in the respective guide files in research/.
5. Compile a chemicals list with supplier links and prices to research/proposals/<topic>_chemicals.md, then message the user with the list and confirm that all code files are ready for review.
```

---

## Notes

- The agent must not write workflow code before receiving explicit user approval of the proposal (Stage 4).
- If a chemical fails the safety screen, it must be excluded from the proposal — do not suggest workarounds that require equipment not present on Lash-E.
- The synthetic data function in the analysis module should capture the qualitative shape of the expected response based on literature findings — it should not be a trivial constant or pure noise.
- Chemicals list should prioritize vendors that supply to University of Toronto / Canadian academic institutions to avoid import delays.
