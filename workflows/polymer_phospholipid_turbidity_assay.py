"""
Polymer-Phospholipid Turbidity Assay Workflow
Intelligent substock management for minimum volume constraints.
Creates samples with specified phospholipid and polymer concentrations.
"""
import sys
sys.path.append("../utoronto_demo")
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from master_usdl_coordinator import Lash_E

# WORKFLOW CONSTANTS
WELL_VOLUME_UL = 200  # μL per well
MIN_PIPETTE_VOLUME_UL = 10  # Minimum pipetting volume
WATER_REFILL_THRESHOLD_ML = 4.0  # Refill water when below this volume
SUBSTOCK_VOLUME_ML = 6.0  # Default substock preparation volume
N_REPLICATES = 3  # Default number of replicates per condition

# Stock concentrations - adjust as needed
STOCK_CONCENTRATIONS = {
    'phospholipid_a': 10.0,  # mM
    'phospholipid_b': 15.0,  # mM  
    'polymer_x': 5.0,        # mg/mL
    'polymer_y': 8.0         # mg/mL
}

# File paths
INPUT_VIAL_STATUS_FILE = "../utoronto_demo/status/polymer_phospholipid_vials.csv"
MEASUREMENT_PROTOCOL_FILE = r"C:\Protocols\Quick_Measurement.prt"

class SubstockTracker:
    """Track substock vial contents and concentrations."""
    
    def __init__(self):
        self.substocks = {}  # vial_name: {'component_type': 'phospholipid'/'polymer', 'component_name': str, 'concentration': float, 'volume': float}
        self.next_available_substock = 1
        

    
    def find_best_solution_for_component(self, component_type, component_name, target_concentration, well_volume_ml):
        """
        Find best available solution (stock or substock) for a specific component.
        Returns solution dict or None if no suitable solution exists.
        """
        options = []
        
        # Check stock solution first
        stock_conc = STOCK_CONCENTRATIONS.get(component_name, 0)
        if stock_conc > 0:
            vol_needed_ml = (target_concentration * well_volume_ml) / stock_conc
            vol_needed_ul = vol_needed_ml * 1000
            
            if vol_needed_ul >= MIN_PIPETTE_VOLUME_UL and vol_needed_ml <= well_volume_ml:
                options.append({
                    'vial_name': component_name,
                    'concentration': stock_conc,
                    'volume_needed_ml': vol_needed_ml,
                    'is_stock': True
                })
        
        # Check existing substocks
        for vial_name, contents in self.substocks.items():
            if (contents['component_type'] == component_type and 
                contents['component_name'] == component_name and
                contents['volume'] > 0):
                
                vol_needed_ml = (target_concentration * well_volume_ml) / contents['concentration']
                vol_needed_ul = vol_needed_ml * 1000
                
                if (vol_needed_ul >= MIN_PIPETTE_VOLUME_UL and 
                    vol_needed_ml <= well_volume_ml and
                    contents['volume'] >= vol_needed_ml):
                    
                    options.append({
                        'vial_name': vial_name,
                        'concentration': contents['concentration'],
                        'volume_needed_ml': vol_needed_ml,
                        'is_stock': False
                    })
        
        if not options:
            return None
            
        # Return most concentrated option (best option)
        options.sort(key=lambda x: x['concentration'], reverse=True)
        return options[0]
    
    def calculate_nice_dilution_concentration(self, component_name, target_concentration, well_volume_ml):
        """
        Calculate a nice round concentration for diluting a component.
        Ensures >= MIN_PIPETTE_VOLUME and produces clean decimal concentrations.
        """
        # Target 15-20 μL volumes for more reusable, dilute solutions
        # Calculate concentration for target volume (not minimum volume)
        target_volume_ul = 17.5  # Sweet spot between 15-20 μL
        target_volume_ml = target_volume_ul / 1000
        target_concentration_for_reusability = (target_concentration * well_volume_ml) / target_volume_ml
        
        # But ensure we don't go below minimum pipette volume as absolute constraint
        min_volume_ml = MIN_PIPETTE_VOLUME_UL / 1000
        max_concentration_absolute = (target_concentration * well_volume_ml) / min_volume_ml
        
        # Use the lower concentration (more dilute) for better reusability
        max_concentration = min(target_concentration_for_reusability, max_concentration_absolute)
        logging.debug(f"calculate_nice_dilution_concentration: {component_name}, target={target_concentration}")
        logging.debug(f"  targeting {target_volume_ul} μL -> max_conc={target_concentration_for_reusability:.3f}")
        logging.debug(f"  absolute min 10 μL -> max_conc={max_concentration_absolute:.3f}")
        logging.debug(f"  using more dilute: {max_concentration:.3f}")
        
        # Round DOWN to nice numbers with proper significant figures  
        def round_to_nice_concentration(value):
            if value <= 0:
                return 0
            
            # Get the order of magnitude
            log_val = np.log10(value)
            magnitude = 10 ** np.floor(log_val)
            
            # Get the leading digit(s)
            normalized = value / magnitude
            
            # Round DOWN to nice values: 1, 2, 3, 5, 10
            if normalized <= 1.0:
                nice_normalized = 1.0
            elif normalized <= 2.0:
                nice_normalized = 2.0
            elif normalized <= 3.0:
                nice_normalized = 3.0
            elif normalized <= 5.0:
                nice_normalized = 3.0  # Round DOWN from 5.0 to 3.0
            elif normalized <= 10.0:
                nice_normalized = 5.0  # Round DOWN from 10.0 to 5.0
            else:
                nice_normalized = 5.0  # Cap at 5x in this magnitude
            
            return nice_normalized * magnitude
        
        # Use 80% of max concentration for safety margin
        safe_concentration = max_concentration * 0.8
        nice_concentration = round_to_nice_concentration(safe_concentration)
        logging.debug(f"  safe_concentration={safe_concentration}, nice_concentration={nice_concentration}")
        
        return nice_concentration
    
    def calculate_volumes_for_solutions(self, phospholipid_solution, target_conc_phospholipid,
                                      polymer_solution, target_conc_polymer, well_volume_ml):
        """
        Calculate volumes needed when using specific solutions for both components.
        """
        phospholipid_vol_ml = phospholipid_solution['volume_needed_ml']
        polymer_vol_ml = polymer_solution['volume_needed_ml']
        water_vol_ml = well_volume_ml - phospholipid_vol_ml - polymer_vol_ml
        
        if water_vol_ml < 0:
            raise ValueError(f"Cannot fit components in well: need {phospholipid_vol_ml*1000:.1f} + {polymer_vol_ml*1000:.1f} = {(phospholipid_vol_ml + polymer_vol_ml)*1000:.1f} μL in {well_volume_ml*1000:.0f} μL well")
        
        water_vol_ul = water_vol_ml * 1000
        if water_vol_ul > 0 and water_vol_ul < MIN_PIPETTE_VOLUME_UL:
            raise ValueError(f"Water volume too small: {water_vol_ul:.1f} μL (need ≥ {MIN_PIPETTE_VOLUME_UL} μL or 0)")
        
        return {
            'phospholipid_volume_ml': phospholipid_vol_ml,
            'polymer_volume_ml': polymer_vol_ml,
            'water_volume_ml': water_vol_ml,
            'achieved_conc_phospholipid': target_conc_phospholipid,
            'achieved_conc_polymer': target_conc_polymer,
            'phospholipid_vial': phospholipid_solution['vial_name'],
            'polymer_vial': polymer_solution['vial_name'],
            'phospholipid_is_stock': phospholipid_solution['is_stock'],
            'polymer_is_stock': polymer_solution['is_stock']
        }
    
    def get_next_substock_vial(self):
        """Get next available substock vial name."""
        if self.next_available_substock > 10:
            raise ValueError("No more substock vials available!")
        
        vial_name = f"substock_{self.next_available_substock}"
        self.next_available_substock += 1
        return vial_name
    
    def register_substock(self, vial_name, component_type, component_name, concentration, volume, 
                         stock_used=None, water_added_ml=None):
        """Register a new substock with its contents and preparation details."""
        self.substocks[vial_name] = {
            'component_type': component_type,  # 'phospholipid' or 'polymer'
            'component_name': component_name,  # e.g. 'phospholipid_a', 'polymer_x'
            'concentration': concentration,
            'volume': volume,
            'stock_used': stock_used,
            'water_added_ml': water_added_ml
        }
        
    def update_volume(self, vial_name, volume_used_ml):
        """Update substock volume after use."""
        if vial_name in self.substocks:
            self.substocks[vial_name]['volume'] -= volume_used_ml
            if self.substocks[vial_name]['volume'] < 0:
                self.substocks[vial_name]['volume'] = 0

def create_output_folder(simulate=True):
    """Create timestamped output folder for results."""
    if simulate:
        print("Simulation mode: skipping output folder creation")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workflow_name = "polymer_phospholipid_turbidity_assay"
    folder_name = f"{workflow_name}_{timestamp}"
    
    output_dir = os.path.join("output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output folder: {output_dir}")
    return output_dir

def check_water_level(lash_e, water_vial="water_supply", max_volume=8.0):
    """Check water level and refill from reservoir if needed."""
    print(f"Checking water level in {water_vial}")
    
    # Get current water volume
    current_volume = lash_e.nr_robot.get_vial_info(water_vial, 'vial_volume')
    print(f"  Current water volume: {current_volume:.2f} mL")
    
    # If water is low, refill from reservoir
    if current_volume < WATER_REFILL_THRESHOLD_ML:
        refill_volume = max_volume - current_volume
        print(f"  Water low ({current_volume:.1f} mL), refilling {refill_volume:.1f} mL from reservoir")
        lash_e.nr_robot.dispense_into_vial_from_reservoir(
            reservoir_index=0, 
            vial_index=water_vial, 
            volume=refill_volume
        )
        print(f"  Water refilled to {max_volume:.1f} mL")
    else:
        print(f"  Water level OK ({current_volume:.1f} mL)")


def save_experiment_results(results_df, substock_tracker, experiment_name="polymer_phospholipid_experiment"):
    """
    Save experiment results and substock details to output folder.
    """
    import os
    from datetime import datetime
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/polymer_phospholipid_assay/{experiment_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    results_file = f"{output_dir}/sample_results.csv"
    results_df.to_csv(results_file, index=False)
    logging.info(f"Results saved to: {results_file}")
    
    # Save substock details
    if substock_tracker.substocks:
        substock_data = []
        for vial_name, contents in substock_tracker.substocks.items():
            substock_data.append({
                'vial_name': vial_name,
                'component_type': contents['component_type'],
                'component_name': contents['component_name'], 
                'concentration': contents['concentration'],
                'final_volume_ml': contents['volume'],
                'stock_used': contents.get('stock_used', 'N/A'),
                'water_added_ml': contents.get('water_added_ml', 'N/A')
            })
        
        substock_df = pd.DataFrame(substock_data)
        substock_file = f"{output_dir}/substock_details.csv"
        substock_df.to_csv(substock_file, index=False)
        logging.info(f"Substock details saved to: {substock_file}")
        
        # Also save a summary
        summary_file = f"{output_dir}/experiment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Polymer-Phospholipid Turbidity Assay Results\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total samples: {len(results_df)}\n")
            f.write(f"Substocks created: {len(substock_tracker.substocks)}\n\n")
            
            f.write("Stock Solutions Used:\n")
            stocks_used = set()
            for _, row in results_df.iterrows():
                if row['phospholipid_is_stock']:
                    stocks_used.add(f"{row['phospholipid_type']} (stock)")
                if row['polymer_is_stock']:
                    stocks_used.add(f"{row['polymer_type']} (stock)")
            
            for stock in sorted(stocks_used):
                f.write(f"  - {stock}\n")
                
            f.write(f"\nSubstocks Created:\n")
            for vial_name, contents in substock_tracker.substocks.items():
                f.write(f"  - {vial_name}: {contents['concentration']} {contents['component_name']} ({contents['volume']:.1f} mL)\n")
        
        logging.info(f"Summary saved to: {summary_file}")
    
    return output_dir


def prepare_substock_recursive(lash_e, substock_tracker, vial_name, component_type, component_name, target_concentration):
    """
    Prepare substock using recursive dilution to ensure ALL pipetting volumes ≥10 μL.
    
    This unified approach treats substock creation like sample creation:
    1. Find best available solution (stock or existing substock)
    2. If volume needed < 10 μL, create intermediate substock first
    3. Repeat recursively until all volumes ≥ 10 μL
    """
    logging.info(f"Preparing {component_type} substock {vial_name}: {component_name} {target_concentration:.3e}")
    
    # Find best available source for this substock
    source_solution = substock_tracker.find_best_solution_for_component(
        component_type, component_name, target_concentration, SUBSTOCK_VOLUME_ML  # Function expects mL
    )
    
    if source_solution is None:
        # Need to create intermediate substock first
        logging.info(f"  No suitable source found, creating intermediate substock...")
        
        # Calculate what concentration would give us 15-20 μL from stock
        intermediate_conc = substock_tracker.calculate_nice_dilution_concentration(
            component_name, target_concentration, SUBSTOCK_VOLUME_ML
        )
        
        # Get next vial for intermediate substock
        intermediate_vial = substock_tracker.get_next_substock_vial()
        
        # Recursively create intermediate substock
        prepare_substock_recursive(lash_e, substock_tracker, intermediate_vial, 
                                 component_type, component_name, intermediate_conc)
        
        # Now try again with the intermediate substock
        source_solution = substock_tracker.find_best_solution_for_component(
            component_type, component_name, target_concentration, SUBSTOCK_VOLUME_ML
        )
    
    if source_solution is None:
        raise ValueError(f"Cannot create substock for {component_name} at {target_concentration}")
    
    # Calculate dilution from the source solution
    source_conc = source_solution['concentration']
    source_name = source_solution['vial_name']
    is_stock = source_solution.get('is_stock', False)
    
    # Calculate volumes for 6 mL total substock
    source_volume_ml = (target_concentration * SUBSTOCK_VOLUME_ML) / source_conc
    water_volume_ml = SUBSTOCK_VOLUME_ML - source_volume_ml
    
    logging.info(f"  Using source: {source_name} ({source_conc} {component_name})")
    logging.info(f"  Recipe: {source_volume_ml:.3f} mL {source_name} + {water_volume_ml:.3f} mL water")
    
    # Validate minimum pipetting volume
    if source_volume_ml < MIN_PIPETTE_VOLUME_UL / 1000:
        raise ValueError(f"Source volume too small: {source_volume_ml*1000:.1f} μL < {MIN_PIPETTE_VOLUME_UL} μL")
    
    # Add water first if needed (from reservoir for large volumes)
    if water_volume_ml > 0:
        if water_volume_ml >= 1.0:  # Use reservoir for volumes ≥ 1 mL
            logging.info(f"  Adding {water_volume_ml:.3f} mL water from reservoir")
            lash_e.nr_robot.dispense_into_vial_from_reservoir(
                reservoir_index=0,  # Water reservoir
                vial_index=vial_name,
                volume=water_volume_ml
            )
        else:
            # Use water vial for small amounts
            check_water_level(lash_e, "water_supply")
            lash_e.nr_robot.dispense_from_vial_into_vial(
                source_vial_name="water_supply",
                dest_vial_name=vial_name,
                volume=water_volume_ml,
                liquid='water'
            )
    
    # Add source solution
    if source_volume_ml > 0:
        lash_e.nr_robot.dispense_from_vial_into_vial(
            source_vial_name=source_name,
            dest_vial_name=vial_name,
            volume=source_volume_ml,
            liquid='water'
        )
        
        # Update source volume if it's a substock
        if not is_stock:
            substock_tracker.substocks[source_name]['volume'] -= source_volume_ml
    
    # Vortex to mix
    lash_e.nr_robot.vortex_vial(vial_name=vial_name, vortex_time=5)
    
    # Register in tracker with preparation details
    source_description = f"{source_volume_ml:.3f} mL {source_name}"
    if water_volume_ml >= 1.0:
        water_description = f"{water_volume_ml:.3f} mL from reservoir"
    else:
        water_description = water_volume_ml
        
    substock_tracker.register_substock(
        vial_name=vial_name,
        component_type=component_type,
        component_name=component_name,
        concentration=target_concentration,
        volume=SUBSTOCK_VOLUME_ML,
        stock_used=source_description,
        water_added_ml=water_description
    )
    
    logging.info(f"  ✅ Created {vial_name}: {target_concentration} {component_name} (6.0 mL)")
    
    return {
        'concentration': target_concentration,
        'volume_ml': SUBSTOCK_VOLUME_ML
    }

# Legacy function name for compatibility
def prepare_substock(lash_e, substock_tracker, vial_name, component_type, component_name, target_concentration):
    """Legacy wrapper - use prepare_substock_recursive for new code."""
    return prepare_substock_recursive(lash_e, substock_tracker, vial_name, component_type, component_name, target_concentration)

def create_sample(lash_e, substock_tracker, phospholipid_type, target_conc_phospholipid, 
                 polymer_type, target_conc_polymer, well_index, replicates=1):
    """
    Create sample wells with specified phospholipid and polymer concentrations.
    Uses separate substocks for phospholipid and polymer components.
    
    Args:
        lash_e: Lash_E controller  
        substock_tracker: SubstockTracker instance
        phospholipid_type: Type of phospholipid
        target_conc_phospholipid: Target phospholipid concentration
        polymer_type: Type of polymer
        target_conc_polymer: Target polymer concentration
        well_index: Starting well index
        replicates: Number of replicates to create
        
    Returns:
        tuple: (next_well_index, sample_info_list)
    """
    well_volume_ml = WELL_VOLUME_UL / 1000.0
    sample_info = []
    
    logging.info(f"Creating sample: {phospholipid_type} {target_conc_phospholipid:.3e} + {polymer_type} {target_conc_polymer:.3e}")
    
    # Step 1: Check existing solutions for both components
    phospholipid_solution = substock_tracker.find_best_solution_for_component(
        'phospholipid', phospholipid_type, target_conc_phospholipid, well_volume_ml
    )
    logging.info(f"Phospholipid solution found: {phospholipid_solution}")
    
    polymer_solution = substock_tracker.find_best_solution_for_component(
        'polymer', polymer_type, target_conc_polymer, well_volume_ml
    )
    logging.info(f"Polymer solution found: {polymer_solution}")
    
    # Step 2: Create substocks for components that need them
    if phospholipid_solution is None:
        logging.info("Need to create phospholipid substock...")
        optimal_conc = substock_tracker.calculate_nice_dilution_concentration(
            phospholipid_type, target_conc_phospholipid, well_volume_ml
        )
        logging.info(f"Creating {optimal_conc:.3e} {phospholipid_type} substock")
        new_vial = substock_tracker.get_next_substock_vial()
        prepare_substock(lash_e, substock_tracker, new_vial, 'phospholipid', phospholipid_type, optimal_conc)
        
        # Get the solution we just created
        phospholipid_solution = substock_tracker.find_best_solution_for_component(
            'phospholipid', phospholipid_type, target_conc_phospholipid, well_volume_ml
        )
        logging.info(f"After creating substock, phospholipid solution: {phospholipid_solution}")
    
    # Keep trying to create polymer substocks until we get one that works
    polymer_attempts = 0
    max_attempts = 5  # Don't create too many substocks
    
    while polymer_solution is None and polymer_attempts < max_attempts:
        polymer_attempts += 1
        logging.info(f"Need to create polymer substock (attempt {polymer_attempts})...")
        
        optimal_conc = substock_tracker.calculate_nice_dilution_concentration(
            polymer_type, target_conc_polymer, well_volume_ml
        )
        logging.info(f"Creating {optimal_conc:.3e} {polymer_type} substock")
        
        try:
            new_vial = substock_tracker.get_next_substock_vial()
            prepare_substock(lash_e, substock_tracker, new_vial, 'polymer', polymer_type, optimal_conc)
            
            # Get the solution we just created
            polymer_solution = substock_tracker.find_best_solution_for_component(
                'polymer', polymer_type, target_conc_polymer, well_volume_ml
            )
            logging.info(f"After creating substock, polymer solution: {polymer_solution}")
            
            if polymer_solution is None:
                logging.warning(f"Substock attempt {polymer_attempts} still needs < 10 μL, trying more dilute...")
        except ValueError as e:
            logging.error(f"No more vials available for polymer substocks: {e}")
            break
    
    # Step 3: Calculate final volumes
    if phospholipid_solution is None or polymer_solution is None:
        logging.error(f"Debug: phospholipid_solution = {phospholipid_solution}")
        logging.error(f"Debug: polymer_solution = {polymer_solution}")
        logging.error(f"Debug: current substocks = {substock_tracker.substocks}")
        raise RuntimeError("Failed to create suitable substocks!")
    
    volumes = substock_tracker.calculate_volumes_for_solutions(
        phospholipid_solution, target_conc_phospholipid,
        polymer_solution, target_conc_polymer, 
        well_volume_ml
    )
    
    # Step 4: Check water level before proceeding
    check_water_level(lash_e)
    
    # Step 5: Create replicates
    phospholipid_vial = volumes['phospholipid_vial']
    polymer_vial = volumes['polymer_vial']
    
    logging.info(f"Using solutions: {volumes['phospholipid_volume_ml']*1000:.1f} μL {phospholipid_vial} + "
          f"{volumes['polymer_volume_ml']*1000:.1f} μL {polymer_vial} + "
          f"{volumes['water_volume_ml']*1000:.1f} μL water")
    
    for rep in range(replicates):
        current_well = well_index + rep
        
        # Pipette phospholipid solution
        lash_e.nr_robot.aspirate_from_vial(
            phospholipid_vial,
            volumes['phospholipid_volume_ml'],
            liquid='water'
        )
        
        # Pipette polymer solution
        lash_e.nr_robot.aspirate_from_vial(
            polymer_vial,
            volumes['polymer_volume_ml'],
            liquid='water'
        )
        
        # Pipette water if needed
        if volumes['water_volume_ml'] > 0:
            # Check water level BEFORE aspiration to avoid negative volumes
            check_water_level(lash_e, "water_supply")
            
            lash_e.nr_robot.aspirate_from_vial(
                "water_supply",
                volumes['water_volume_ml'],
                liquid='water'
            )
        
        # Dispense into well
        lash_e.nr_robot.dispense_into_wellplate(
            dest_wp_num_array=[current_well],
            amount_mL_array=[well_volume_ml],
            liquid='water'
        )
        
        # Update substock volumes (only if not using stocks)
        if not volumes['phospholipid_is_stock']:
            substock_tracker.update_volume(phospholipid_vial, volumes['phospholipid_volume_ml'])
        if not volumes['polymer_is_stock']:
            substock_tracker.update_volume(polymer_vial, volumes['polymer_volume_ml'])
        
        # Record sample information
        sample_info.append({
            'well': current_well,
            'phospholipid_type': phospholipid_type,
            'phospholipid_conc': volumes['achieved_conc_phospholipid'],
            'polymer_type': polymer_type, 
            'polymer_conc': volumes['achieved_conc_polymer'],
            'target_phospholipid_conc': target_conc_phospholipid,
            'target_polymer_conc': target_conc_polymer,
            'replicate': rep + 1,
            'phospholipid_vial': phospholipid_vial,
            'polymer_vial': polymer_vial,
            'phospholipid_volume_ul': volumes['phospholipid_volume_ml'] * 1000,
            'polymer_volume_ul': volumes['polymer_volume_ml'] * 1000,
            'water_volume_ul': volumes['water_volume_ml'] * 1000,
            'phospholipid_is_stock': volumes['phospholipid_is_stock'],
            'polymer_is_stock': volumes['polymer_is_stock']
        })
    
    next_well_index = well_index + replicates
    return next_well_index, sample_info

def run_turbidity_assay(conditions_list, simulate=True):
    """
    Main workflow function to run polymer-phospholipid turbidity assay.
    
    Args:
        conditions_list: List of dicts with keys: 'phospholipid_type', 'phospholipid_conc', 
                        'polymer_type', 'polymer_conc', 'replicates'
        simulate: Run in simulation mode
        
    Returns:
        tuple: (results_df, measurement_data)
    """
    print("=== Polymer-Phospholipid Turbidity Assay ===")
    
    # Initialize system
    lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=simulate)
    substock_tracker = SubstockTracker()
    
    # Check files and get wellplate
    lash_e.nr_robot.check_input_file()
    lash_e.nr_track.check_input_file()
    lash_e.grab_new_wellplate()
    
    # Create storage folder
    output_folder = create_output_folder(simulate)
    
    # Process all conditions
    well_index = 0
    all_sample_info = []
    
    for condition in conditions_list:
        next_well, sample_info = create_sample(
            lash_e=lash_e,
            substock_tracker=substock_tracker,
            phospholipid_type=condition['phospholipid_type'],
            target_conc_phospholipid=condition['phospholipid_conc'],
            polymer_type=condition['polymer_type'],
            target_conc_polymer=condition['polymer_conc'],
            well_index=well_index,
            replicates=condition.get('replicates', N_REPLICATES)
        )
        
        all_sample_info.extend(sample_info)
        well_index = next_well
    
    # Remove pipette
    lash_e.nr_robot.remove_pipet()
    
    # Measure turbidity
    well_indices = [info['well'] for info in all_sample_info]
    print(f"Measuring turbidity in wells {min(well_indices)} to {max(well_indices)}")
    measurement_data = lash_e.measure_wellplate(MEASUREMENT_PROTOCOL_FILE, well_indices)
    
    # Combine results
    results_df = pd.DataFrame(all_sample_info)
    
    # Add turbidity measurements (mock data in simulation)
    if simulate or measurement_data is None:
        results_df['turbidity'] = np.random.random(len(results_df)) * 0.5 + 0.2
    else:
        results_df['turbidity'] = measurement_data.get('turbidity', [0.5] * len(results_df))
    
    # Save results
    if output_folder:
        results_file = os.path.join(output_folder, "turbidity_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Save substock tracking info
        substock_file = os.path.join(output_folder, "substock_tracker.json")
        with open(substock_file, 'w') as f:
            json.dump(substock_tracker.substocks, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        print(f"Substock info saved to: {substock_file}")
    
    # Cleanup
    lash_e.discard_used_wellplate() 
    lash_e.nr_robot.move_home()
    
    print("=== Workflow Complete ===")
    return results_df, measurement_data, substock_tracker

# Example usage
def example_experiment():
    """Example experiment with various phospholipid-polymer combinations."""
    conditions = [
        {
            'phospholipid_type': 'phospholipid_a',
            'phospholipid_conc': 1e-3,  # 1 mM
            'polymer_type': 'polymer_x', 
            'polymer_conc': 0.5,        # 0.5 mg/mL
            'replicates': 3
        },
        {
            'phospholipid_type': 'phospholipid_a',
            'phospholipid_conc': 1e-4,  # 0.1 mM  
            'polymer_type': 'polymer_x',
            'polymer_conc': 0.1,        # 0.1 mg/mL
            'replicates': 3
        },
        {
            'phospholipid_type': 'phospholipid_b',
            'phospholipid_conc': 5e-4,  # 0.5 mM
            'polymer_type': 'polymer_y',
            'polymer_conc': 0.2,        # 0.2 mg/mL  
            'replicates': 3
        }
    ]
    
    return run_turbidity_assay(conditions, simulate=True)

def demonstrate_recursive_dilution():
    """
    Demonstrate the recursive substock creation algorithm without hardware.
    Shows how the system handles extreme dilutions that require multiple steps.
    """
    print("=== Recursive Substock Creation Demo ===")
    print("Testing extreme case: phospholipid_a at 0.001 mM")
    print("Stock: 10.0 mM -> direct would need 0.6 uL (below 10 uL minimum)")
    
    substock_tracker = SubstockTracker()
    
    # Simulate the planning logic (without actual robot calls)
    target_conc = 0.001  # mM
    component_name = 'phospholipid_a'
    
    print(f"\nStep 1: Check if stock works directly...")
    solution = substock_tracker.find_best_solution_for_component(
        'phospholipid', component_name, target_conc, 6.0  # 6 mL substock
    )
    
    if solution is None:
        print("  No - stock needs < 10 uL")
        print("  -> Need intermediate substock")
        
        # Calculate intermediate concentration
        intermediate_conc = substock_tracker.calculate_nice_dilution_concentration(
            component_name, target_conc, 6.0
        )
        print(f"\nStep 2: Create intermediate at {intermediate_conc:.3f} mM")
        
        # Simulate creating the intermediate
        substock_tracker.substocks['substock_1'] = {
            'component_type': 'phospholipid',
            'component_name': component_name,
            'concentration': intermediate_conc,
            'volume': 6.0
        }
        
        # Check if intermediate works for final target
        final_solution = substock_tracker.find_best_solution_for_component(
            'phospholipid', component_name, target_conc, 6.0
        )
        
        if final_solution:
            vol_ul = final_solution['volume_needed_ml'] * 1000
            print(f"Step 3: Use substock_1 -> need {vol_ul:.1f} uL (>= 10? {vol_ul >= 10})")
            print(f"\nSUCCESS: Recursive dilution solved the tiny volume problem!")
            return True
        else:
            print("Step 3: Still need more dilution steps...")
            return False
    else:
        print("Direct stock works - no recursion needed")
        return True

if __name__ == "__main__":
    """
    Run example experiment or demonstrate recursive dilution.
    Set simulate=False when ready for hardware.
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Run recursive dilution demonstration
        demonstrate_recursive_dilution()
    else:
        # Run normal experiment
        results, measurements, substock_tracker = example_experiment()
        print(f"Experiment complete: {len(results)} samples measured")
        print("\nSample results:")
        print(results)
        
        # Save results with substock details to timestamped output folder
        output_dir = save_experiment_results(results, substock_tracker, 'example_polymer_phospholipid_assay')
        print(f"\n📁 All results saved to: {output_dir}")
        
        print(f"\n💡 To see recursive dilution demo: python {__file__} demo")