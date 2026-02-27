# Complete Anionic/Cationic Surfactant Screening Plan
**Objective:** Screen ALL anionic/cationic combinations efficiently  
**Strategy:** 2 pairings per night using 3 surfactants total per night  
**Total:** 25 pairs across 13 nights

## Surfactant Inventory
### Anionic (5):
- **SDS** (50 mM), **NaC** (50 mM), **SDBS** (50 mM), **NaLS** (50 mM)
- **DSS** (25 mM)

### Cationic (5): 
- **TTAB** (50 mM), **DTAB** (50 mM), **BDDAC** (50 mM), **BZT** (50 mM)
- **CTAB** (5 mM)

## Efficient 13-Night Schedule
**Format:** Each night uses 1 anionic + 2 cationic = 3 surfactants total

### Schedule
**Night 1:** SDS + TTAB, SDS + DTAB (3)
**Night 2:** SDS + CTAB, SDS + BDDAC (3)
**Night 3:** SDS + BZT, NaC + TTAB (4)
**Night 4:** NaC + DTAB, NaC + CTAB (3)
**Night 5:** NaC + BDDAC, NaC + BZT (3)
**Night 6:** SDBS + TTAB, SDBS + DTAB (3)
**Night 7:** SDBS + CTAB, SDBS + BDDAC (3)
**Night 8:** SDBS + BZT, NaLS + TTAB (4)
**Night 9:** NaLS + DTAB, NaLS + CTAB (3)
**Night 10:** NaLS + BDDAC, NaLS + BZT (3)
**Night 11:** DSS + TTAB, DSS + DTAB (3)
**Night 12:** DSS + CTAB, DSS + BDDAC (3)
**Night 13:** DSS + BZT (2) *Single pairing night*

## Per-Night Configuration Examples

### Night 1: SDS + TTAB, SDS + DTAB
```python
# Pairing 1 (Run first)
SURFACTANT_A = "SDS"
SURFACTANT_B = "TTAB" 
EXPERIMENT_TAG = "sds_ttab"

# Pairing 2 (Run second, same night)
SURFACTANT_A = "SDS" 
SURFACTANT_B = "DTAB"
EXPERIMENT_TAG = "sds_dtab"
```

### Night 13: DSS + BZT (Single pairing)
```python
# Only one pairing this night
SURFACTANT_A = "DSS"
SURFACTANT_B = "BZT"
EXPERIMENT_TAG = "dss_bzt"
```

## Workflow Settings
```python
WORKFLOW_TYPE = 'iterative'
ITERATIVE_MEASUREMENT_TOTAL = 192  # Per pairing
ADD_BUFFER = False  # Or True with SELECTED_BUFFER = "MES"
SIMULATE = False    # Real hardware
```

## Efficiency Benefits
- **Minimal setup:** Only 2-3 surfactants per night
- **Stock reuse:** Same anionic used for both pairs most nights  
- **Complete coverage:** All 25 combinations in 13 nights
- **Balanced workload:** ~2-3 weeks per surfactant focus
- **~370 wells/night:** Most nights = 384 wells (2×192), Night 13 = 192 wells

## Total Deliverables
- **25 complete interaction maps** (192 wells each)
- **4,800 total wells** across 13 nights
- **Comprehensive anionic/cationic interaction database**
- **Chain length, structure, and concentration effects**