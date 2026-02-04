"""Reservoir calibration protocol for North Robot system.

Unified minimal interface:
    initialize(cfg) -> state (dict)
    measure(state, volume_mL, params, replicates) -> list[dict]
    wrapup(state) -> None

Per-replicate result keys:
    replicate, volume (mL), elapsed_s
Optional: start_time, end_time, echoed params.
"""
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from calibration_protocol_base import CalibrationProtocolBase

# Add parent directory to path for North Robot imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import ReservoirParameters
import slack_agent

# Line priming volume (mL) - prime the reservoir line before calibration
LINE_PRIMING_VOLUME = 0.5

# Liquid densities for mass-to-volume conversion (g/mL)
LIQUIDS = {
    "water": {"density": 1.00},
    "ethanol": {"density": 0.789},
    "toluene": {"density": 0.867},
    "2MeTHF": {"density": 0.852},
    "isopropanol": {"density": 0.789},
    "DMSO": {"density": 1.1},
    "acetone": {"density": 0.79},
    "glycerol": {"density": 1.26},
    "PEG_Water": {"density": 1.05},
    "4%_hyaluronic_acid_water": {"density": 1.01},
    "agar_water": {"density": 1.01},
}

class ReservoirCalibrationProtocol(CalibrationProtocolBase):
    """Reservoir calibration protocol implementing the abstract interface."""
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize reservoir protocol with North Robot's internal simulation."""
               
        # Get liquid from experiment config - FAIL if missing
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        
        if 'liquid' not in cfg['experiment']:
            raise ValueError("Missing required 'liquid' in experiment configuration")
        liquid = cfg['experiment']['liquid']
        
        # Get reservoir index from config (default to 0)
        reservoir_index = 1
        
        print(f"Initializing North Robot reservoir protocol for {liquid} (reservoir {reservoir_index})")

        # Use hardware simulation mode (different from our simulation protocol)
        simulate = False  # This enables North Robot's internal simulation
        
        # Enable continuous monitoring for reservoir operations
        continuous_monitoring = True
        
        # Quality control threshold for mass measurement stability (in grams)
        # Default: 0.002g (2mg) - slightly more lenient than pipetting since volumes are larger
        # For stricter control: 0.001g (1mg) 
        # For lenient control (quick tests): 0.005g (5mg)
        self.quality_std_threshold = 0.004  # <<< CHANGE THIS VALUE FOR DIFFERENT QUALITY LEVELS

        # Initialize hardware
        try:
            # Initialize vial file path
            vial_file = "status/reservoir_calibration_vials.csv"
            
            # Initialize Lash_E coordinator
            lash_e = Lash_E(vial_file, simulate=simulate, initialize_biotek=False)
            
            # Validate hardware files
            lash_e.nr_robot.check_input_file()
            
            # Simple vial management: Set up measurement vials
            measurement_vial = "measurement_vial_0" 
            measurement_vial_index = 0  # Track current vial index for swapping
            
            # Move measurement vial to clamp position for dispensing operations
            lash_e.nr_robot.move_vial_to_location(measurement_vial, "clamp", 0)

            # Line priming: Prime the reservoir line to remove air bubbles
            print(f"Priming reservoir {reservoir_index} line with {LINE_PRIMING_VOLUME:.1f} mL into overflow vial...")
            lash_e.nr_robot.prime_reservoir_line(reservoir_index, measurement_vial, LINE_PRIMING_VOLUME)
                       
            print("READY: Reservoir hardware initialized successfully")
            
            return {
                'initialized_at': datetime.now(),
                'liquid': liquid,
                'reservoir_index': reservoir_index,
                'lash_e': lash_e,
                'measurement_vial': measurement_vial,
                'measurement_vial_index': measurement_vial_index,
                'measurement_count': 0,
                'continuous_mass_monitoring': continuous_monitoring
            }
            
        except Exception as e:
            raise RuntimeError(f"Reservoir hardware initialization failed: {e}") from e

    def _check_and_swap_vials(self, state: Dict[str, Any], target_volume_mL: float = 0.0) -> None:
        """Check measurement vial volume and swap if needed with predictive volume checking."""
        try:
            lash_e = state['lash_e']
            current_vial = state['measurement_vial']
            current_index = state['measurement_vial_index']
            
            # Check current volume in measurement vial
            current_volume = lash_e.nr_robot.get_vial_info(current_vial, 'vial_volume')
            max_capacity = 7.0  # mL - maximum safe capacity for vials
            
            if current_volume is None:
                current_volume = 0.0
                
            # DEBUG: Print current volume
            print(f"STATUS: {current_vial} ({current_volume:.2f}mL), adding {target_volume_mL:.3f}mL")
            
            # Check if current volume + target volume would exceed capacity
            predicted_volume = current_volume + target_volume_mL
            if predicted_volume > max_capacity:
                # Need to swap to next numbered vial
                new_index = current_index + 1
                new_vial = f"measurement_vial_{new_index}"
                
                print(f"SWAPPING: {current_vial} ({current_volume:.2f}mL) + {target_volume_mL:.3f}mL = {predicted_volume:.2f}mL > {max_capacity}mL")
                print(f"          Moving to {new_vial}")
                
                # Return current vial to home position first
                lash_e.nr_robot.return_vial_home(current_vial)
                
                # Then move new vial to clamp position
                lash_e.nr_robot.move_vial_to_location(new_vial, "clamp", 0)
                
                # Update state with new vial information
                state['measurement_vial'] = new_vial
                state['measurement_vial_index'] = new_index
                
                print(f"SUCCESS: Now using {new_vial} for measurements")
            
        except Exception as e:
            print(f"WARNING: Vial volume check/swap failed: {e}")

    def _evaluate_measurement(self, stability_info: Dict[str, Any], std_threshold: float = 0.002) -> bool:
        """Evaluate if a reservoir measurement is acceptable based on stability criteria.
        
        A baseline is considered stable if:
        - Stable readings percentage > 50%, OR
        - Standard deviation < threshold
        
        Both baselines must be stable for measurement to be trustworthy.
        
        Args:
            stability_info: Dictionary with stability metrics from dispense_into_vial_from_reservoir
            std_threshold: Maximum acceptable standard deviation in grams (default: 0.002g = 2.0mg)
            
        Returns:
            bool: True if measurement is acceptable, False if should be retried
        """
        # Check pre-baseline stability
        pre_stable_pct = (stability_info['pre_stable_count'] / max(stability_info['pre_total_count'], 1)) * 100
        pre_stable = (pre_stable_pct > 50.0) or (stability_info['pre_baseline_std'] < std_threshold)
        
        # Check post-baseline stability  
        post_stable_pct = (stability_info['post_stable_count'] / max(stability_info['post_total_count'], 1)) * 100
        post_stable = (post_stable_pct > 50.0) or (stability_info['post_baseline_std'] < std_threshold)
        
        # Both baselines must be stable
        is_acceptable = pre_stable and post_stable
        
        print(f"    Quality check: pre={pre_stable_pct:.1f}% stable (std={stability_info['pre_baseline_std']:.6f}g), "
              f"post={post_stable_pct:.1f}% stable (std={stability_info['post_baseline_std']:.6f}g)")
        print(f"    Result: {'ACCEPTABLE' if is_acceptable else 'RETRY NEEDED'} (threshold: {std_threshold:.6f}g)")
        
        return is_acceptable

    def measure(self, state: Dict[str, Any], volume_mL: float, params: Dict[str, Any], replicates: int = 1) -> List[Dict[str, Any]]:
        """Perform reservoir measurement with given parameters."""
        
        results = []
        lash_e = state['lash_e']

        if state.get('continuous_mass_monitoring', False):
            continuous_mass_monitoring = True
            save_mass_data = True
        else:
            continuous_mass_monitoring = False
            save_mass_data = False
        
        for rep in range(replicates):
            start_time = datetime.now()
                       
            # Extract parameters from nested structure
            hw_params = params.get('parameters', {}) if isinstance(params.get('parameters'), dict) else params
            
            # Extract reservoir parameters with safe fallbacks
            try:
                reservoir_params = ReservoirParameters(
                    aspirate_speed=hw_params.get('aspirate_speed', 11),
                    dispense_speed=hw_params.get('dispense_speed', 11),
                    aspirate_wait_time=hw_params.get('aspirate_wait_time', 0.0),
                    dispense_wait_time=hw_params.get('dispense_wait_time', 0.0),
                    valve_switch_delay=hw_params.get('valve_switch_delay', 0.1),
                    overaspirate_vol=params.get('overaspirate_vol', 0.0)  # This is at top level
                )
            except KeyError as e:
                raise ValueError(f"Missing required parameter structure - params dict malformed: {e}") from e
            
            print(f"  Rep {rep+1}/{replicates}: {volume_mL:.3f}mL with reservoir params")
            print(f"    aspirate_speed={reservoir_params.aspirate_speed}, dispense_speed={reservoir_params.dispense_speed}")
            print(f"    wait_times: asp={reservoir_params.aspirate_wait_time:.1f}s, disp={reservoir_params.dispense_wait_time:.1f}s")
            print(f"    overaspirate_vol={reservoir_params.overaspirate_vol:.3f}mL")
            
            # Perform reservoir operation: dispense from reservoir into measurement vial
            reservoir_index = state['reservoir_index']
            measurement_vial = state['measurement_vial']  # Get current vial after potential swapping
            
            # Check if we're in simulation mode
            simulate = state['lash_e'].nr_robot.simulate  # Get from actual robot state
            
            if not simulate:
                # Real hardware measurements with quality-controlled retry loop
                max_retries = 3
                retry_count = 0
                measurement_acceptable = False
                
                while not measurement_acceptable and retry_count <= max_retries:
                    if retry_count > 0:
                        print(f"    Retry attempt {retry_count}/{max_retries}")
                    
                    self._check_and_swap_vials(state, volume_mL)
                    # Start timing just before the successful measurement attempt
                    rep_start = time.perf_counter()
                    
                    # Get current measurement vial after potential swapping
                    current_measurement_vial = state['measurement_vial']
                    
                    # Dispense from reservoir into measurement vial and measure weight
                    dispense_result = lash_e.nr_robot.dispense_into_vial_from_reservoir(
                        reservoir_index, current_measurement_vial, volume_mL, 
                        reservoir_params=reservoir_params, measure_weight=True, 
                        continuous_mass_monitoring=continuous_mass_monitoring, save_mass_data=save_mass_data, return_home=False
                    )
                    
                    # Handle return format (mass, stability_info) or just mass
                    if isinstance(dispense_result, tuple):
                        measured_mass_g, stability_info = dispense_result
                        # Evaluate measurement quality if continuous monitoring was used
                        if continuous_mass_monitoring:
                            measurement_acceptable = self._evaluate_measurement(stability_info, self.quality_std_threshold)
                        else:
                            measurement_acceptable = True  # Accept single-point measurements
                    else:
                        # Backwards compatibility - old format returns just mass
                        measured_mass_g = dispense_result
                        stability_info = None
                        measurement_acceptable = True
                    
                    retry_count += 1
                        
                    if not measurement_acceptable and retry_count <= max_retries:
                        print(f"    WARNING! Measurement quality insufficient, retrying...")
                
                # Convert mass to volume using liquid density
                liquid = state['liquid']
                if liquid not in LIQUIDS:
                    raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
                
                density_g_mL = LIQUIDS[liquid]['density']
                measured_volume_mL = measured_mass_g / density_g_mL  # mass (g) / density (g/mL) = volume (mL)
                measured_mass_mg = measured_mass_g * 1000  # Convert g to mg for display
                
                print(f"    Mass: {measured_mass_mg:.2f}mg -> Volume: {measured_volume_mL:.3f}mL (density: {density_g_mL:.3f}g/mL)")
                
            else:
                # Simple simulation: target volume - 5% + overaspirate + noise
                import random
                
                # Start timing for simulation
                rep_start = time.perf_counter()
                
                # Still call the robot method for vial tracking, but ignore measurement result
                lash_e.nr_robot.dispense_into_vial_from_reservoir(
                    reservoir_index, measurement_vial, volume_mL, 
                    reservoir_params=reservoir_params, measure_weight=True
                )
                
                # Basic simulation logic (reservoirs are more accurate than pipetting)
                base_efficiency = 0.95  # Start at 95% efficiency (target - 5%)
                overaspirate_effect = reservoir_params.overaspirate_vol  # Already in mL
                noise = random.uniform(-0.01, 0.01) * volume_mL  # ±1% noise
                
                # Simple formula: base efficiency + overaspirate helps + noise
                simulated_volume_mL = (volume_mL * base_efficiency) + overaspirate_effect + noise
                simulated_volume_mL = max(simulated_volume_mL, 0)  # Can't be negative
                
                measured_volume_mL = simulated_volume_mL
            
            rep_end = time.perf_counter()
            end_time = datetime.now()
            elapsed_s = rep_end - rep_start
            
            # Increment measurement counter
            state['measurement_count'] += 1
            
            result = {
                'replicate': rep + 1,
                'volume': measured_volume_mL,
                'elapsed_s': elapsed_s,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'target_volume_mL': volume_mL,
                'measured_volume_mL': measured_volume_mL,
                'reservoir_index': reservoir_index,
                # Echo back parameters as dict for compatibility
                'aspirate_speed': reservoir_params.aspirate_speed,
                'dispense_speed': reservoir_params.dispense_speed,
                'aspirate_wait_time': reservoir_params.aspirate_wait_time,
                'dispense_wait_time': reservoir_params.dispense_wait_time,
                'valve_switch_delay': reservoir_params.valve_switch_delay,
                'overaspirate_vol': reservoir_params.overaspirate_vol
            }
            
            # Add stability information if available
            if stability_info is not None:
                result['stability_info'] = stability_info
                result['retry_count'] = retry_count - 1  # Actual number of retries performed
            
            results.append(result)
            print(f"    Measured: {measured_volume_mL:.3f}mL (target: {volume_mL:.3f}mL) in {elapsed_s:.2f}s")
            print()  # Add visual separation between measurement cycles
        
        return results

    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        
        lash_e = state.get('lash_e')
        if lash_e:
            try:
                # Move robot to safe position
                lash_e.nr_robot.return_vial_home(state['measurement_vial'])
                lash_e.nr_robot.move_home()
                
                print(f"COMPLETE: Reservoir hardware cleanup completed. Total measurements: {state.get('measurement_count', 0)}")
                
                # Send slack notification
                try:
                    slack_agent.send_slack_message("🧪 Reservoir calibration finished! All measurements completed.")
                except Exception as e:
                    print(f"WARNING: Slack notification failed: {e}")
                
                # Send slack notification
                try:
                    slack_agent.send_slack_message("🧪 Reservoir calibration finished! All measurements completed.")
                except Exception as e:
                    print(f"WARNING: Slack notification failed: {e}")
                
            except Exception as e:
                print(f"WARNING: Hardware cleanup warning: {e}")

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for reservoir system."""
        constraints = []
        
        # Reservoir syringe volume constraint (typically 2.5 mL max per syringe)
        max_syringe_volume_ml = 2.5
        
        available_volume_ml = max_syringe_volume_ml - target_volume_ml
        constraint = f"overaspirate_vol <= {available_volume_ml:.3f}"
        
        constraints.append(constraint)
        
        return constraints


# Export the protocol instance for clean importing
protocol_instance = ReservoirCalibrationProtocol()