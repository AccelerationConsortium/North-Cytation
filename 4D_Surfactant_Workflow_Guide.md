# 4D Surfactant Workflow Implementation Guide

## Overview
Scaling from 2D (SDS + TTAB) to 4D surfactant experiments using existing 96-well plates with orbital shaking.

## Volume Constraints for 4D

**Standard 96-Well Plates (shaking-safe):**
- Total working volume: 250µL (max safe for orbital shaking)
- Per surfactant volume: ~55µL maximum (vs 90µL in 2D)
- Water/buffer: ~20µL
- Pyrene: ~5µL  
- Shaking headroom: ~100µL

**Result:** Tighter concentration ranges but still workable for 4D exploration.

## Required Code Changes

### 1. Global Configuration
```python
# Replace these globals:
SURFACTANT_A = "SDS"
SURFACTANT_B = "TTAB"

# With:
SURFACTANTS = ["SDS", "TTAB", "CTAB", "DTAB"]  # 4D example
N_DIMENSIONS = 4
WELL_VOLUME_UL = 250  # Max safe shaking volume
```

### 2. Initial Grid Strategy
- **Current 2D**: 9×9 = 81 wells
- **New 4D**: 4×4×4×4 = 256 wells (too many!)
- **Solution**: 3×3×3×3 = 81 wells (same as current 2D)
- **Rely on Bayesian iteration** for dense exploration

### 3. Well Recipe Generation
```python
def create_4d_well_recipe(surf_concentrations, well_volume_ul=250):
    """Create recipe for 4 surfactants"""
    surfactant_volume_each = (well_volume_ul - PYRENE_VOLUME_UL - BUFFER_VOLUME_UL) / 4
    # ~55µL per surfactant maximum
    
    recipe = {}
    for surf_name, target_conc in surf_concentrations.items():
        stock_conc = SURFACTANT_LIBRARY[surf_name]['stock_conc']
        vol_needed = (target_conc * well_volume_ul) / stock_conc
        water_vol = surfactant_volume_each - vol_needed
        
        recipe[f"{surf_name}_volume_ul"] = vol_needed
        recipe[f"{surf_name}_water_ul"] = water_vol
    
    return recipe
```

### 4. DataFrame Columns
```python
# Dynamic column generation
CONCENTRATION_COLUMNS = [f"{surf}_conc_mm" for surf in SURFACTANTS]
VOLUME_COLUMNS = [f"{surf}_volume_ul" for surf in SURFACTANTS]

# Input columns for Bayesian recommender
INPUT_COLUMNS = CONCENTRATION_COLUMNS  # ['SDS_conc_mm', 'TTAB_conc_mm', 'CTAB_conc_mm', 'DTAB_conc_mm']
```

### 5. Bayesian Recommender (Already Supports 4D!)
```python
recommender = BayesianTransitionRecommender(
    input_columns=INPUT_COLUMNS,  # 4D concentration space
    output_columns=['ratio', 'turbidity_600'],
    log_transform_inputs=True,    # Same as before
    # All other parameters identical
)
```

### 6. Dispensing Loop
Replace the dual dispense loops with:
```python
def dispense_4d_surfactants(lash_e, well_recipes_df):
    for surf_name in SURFACTANTS:
        vol_column = f"{surf_name}_volume_ul"
        water_column = f"{surf_name}_water_ul"
        
        print(f"\n=== Dispensing {surf_name} ===")
        for idx, row in well_recipes_df.iterrows():
            surf_vol = row[vol_column]
            water_vol = row[water_column]
            well_idx = row['wellplate_index']
            
            # Same dispensing logic as current workflow
            dispense_surfactant_and_water(lash_e, surf_name, surf_vol, water_vol, well_idx)
```

### 7. Substock Management
Extend existing substock system to handle 4 surfactants:
```python
def calculate_4d_dilution_plans(lash_e, target_concentrations_4d):
    all_plans = {}
    for i, surf_name in enumerate(SURFACTANTS):
        # Extract concentrations for this surfactant from 4D tuples
        target_concs = [conc_tuple[i] for conc_tuple in target_concentrations_4d]
        plan, tracker = calculate_smart_dilution_plan(lash_e, surf_name, target_concs)
        all_plans[surf_name] = plan
    return all_plans
```

## Implementation Strategy

1. **Copy existing workflow** → `surfactant_grid_4d_adaptive_concentrations.py`
2. **Update globals** (5 mins)
3. **Modify column definitions** (10 mins) 
4. **Update dispensing loops** (20 mins)
5. **Extend substock management** (30 mins)
6. **Test with simulation first** (always!)

## Expected Performance

- **Initial grid**: 81 wells (3⁴ = manageable)
- **Bayesian iterations**: 100-150 additional wells  
- **Total experiment**: ~200-250 wells across multiple plates
- **Concentration ranges**: Reduced but sufficient for boundary detection
- **Algorithm advantage**: Bayesian excels in high-dimensional spaces

## Key Benefits

- **Existing hardware**: No new plate types needed
- **Same automation**: All current pipetting/shaking protocols work
- **Proven algorithms**: Bayesian recommender already supports N-D
- **Manageable complexity**: Mostly loop extensions, not algorithmic changes

**Estimated Implementation Time: 2-3 hours**