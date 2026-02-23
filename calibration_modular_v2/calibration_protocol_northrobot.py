"""Hardware calibration protocol for North Robot system.

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
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from calibration_protocol_base import CalibrationProtocolBase

# Add parent directory to path for North Robot imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from master_usdl_coordinator import Lash_E
from pipetting_data.pipetting_parameters import PipettingParameters
import slack_agent

# Tip conditioning volume will be calculated dynamically based on target volumes

# Liquid densities for mass-to-volume conversion (g/mL)
LIQUIDS = {
    "water": {"density": 1.00, "refill_pipets": False},
    "ethanol": {"density": 0.789, "refill_pipets": False},
    "toluene": {"density": 0.867, "refill_pipets": False},
    "2MeTHF": {"density": 0.852, "refill_pipets": False},
    "isopropanol": {"density": 0.789, "refill_pipets": False},
    "DMSO": {"density": 1.1, "refill_pipets": False},
    "acetone": {"density": 0.79, "refill_pipets": False},
    "glycerol": {"density": 1.26, "refill_pipets": True},
    "PEG_Water": {"density": 1.05, "refill_pipets": True},
    "4%_hyaluronic_acid_water": {"density": 1.01, "refill_pipets": True},
    "agar_water": {"density": 1.01, "refill_pipets": False},
    "agar_water_refill": {"density": 1.01, "refill_pipets": True},
    "TFA": {"density": 1.49, "refill_pipets": False},
    "6M_HCl": {"density": 1.10, "refill_pipets": False},
}


from pipetting_data.pipetting_parameters import PipettingParameters

class HardwareCalibrationProtocol(CalibrationProtocolBase):
    """Hardware calibration protocol implementing the abstract interface."""
    
    def get_tip_conditioning_volume(self, target_volume_ml: float) -> float:
        """Calculate appropriate conditioning volume based on target pipetting volume."""
        if target_volume_ml <= 0.20:  # Small tip (200 uL)
            return min(0.200, target_volume_ml * 1.2)
        else:  # Large tip (1000 uL) 
            return min(1.000, target_volume_ml * 1.2)  
    
    def initialize(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize hardware protocol with North Robot's internal simulation."""
               
        # Get liquid from experiment config - FAIL if missing
        if not cfg or 'experiment' not in cfg:
            raise ValueError("Missing required 'experiment' configuration section")
        
        if 'liquid' not in cfg['experiment']:
            raise ValueError("Missing required 'liquid' in experiment configuration")
        liquid = cfg['experiment']['liquid']
        
        print(f"Initializing North Robot hardware protocol for {liquid}")
        
        # Read full config directly from file to get volume targets
        try:
            config_path = Path(__file__).parent / "experiment_config.yaml"
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            volume_targets = full_config.get('experiment', {}).get('volume_targets_ml', [])
            print(f"Read volume targets from config: {volume_targets}")
        except Exception as e:
            print(f"Warning: Could not read config file ({e}), using fallback")
            volume_targets = []
        
        # DEBUG: Show what's in the passed config vs full config
        print(f"DEBUG: cfg keys = {list(cfg.keys()) if cfg else 'cfg is None'}")
        print(f"DEBUG: cfg['experiment'] keys = {list(cfg['experiment'].keys()) if cfg and 'experiment' in cfg else 'no experiment'}")
        print(f"DEBUG: volume_targets_ml from file = {volume_targets}")
        
        # Extract volume targets to determine tip conditioning
        if volume_targets:
            max_volume = max(volume_targets)
            # Store for use in tip conditioning
            self.conditioning_volume = self.get_tip_conditioning_volume(max_volume)
            print(f"Set conditioning volume to {self.conditioning_volume:.3f}mL based on max target {max_volume:.3f}mL")
        else:
            # Fallback to conservative default
            self.conditioning_volume = 0.05
            print("No volume targets found, using default conditioning volume 0.05mL")

        # Use hardware simulation mode (different from our simulation protocol)
        simulate = False  # This enables North Robot's internal simulation
        
        # Vial management mode - swap roles when measurement vial gets too full
        SWAP = False  # If True, enables vial swapping when needed

        SINGLE_VIAL = True #True if you just want to use one vial, eg for a capped vial use "liquid_source_0"

        continuous_monitoring = True
               
        # Quality control threshold for mass measurement stability (in grams)
        # Default: 0.001g (1mg) - good for most pipetting
        # For stricter control (low volume): 0.0005g (0.5mg) 
        # For lenient control (quick tests): 0.002g (2mg)
        self.quality_std_threshold = 0.0005  # <<< CHANGE THIS VALUE FOR DIFFERENT QUALITY LEVELS

        if not simulate:
            slack_agent.send_slack_message("🤖 North Robot calibration/validation started!")

        
        # Initialize hardware
        try:
            # Initialize vial file path
            vial_file = "status/calibration_vials_short.csv"
            
            # Initialize Lash_E coordinator
            lash_e = Lash_E(vial_file, simulate=simulate, initialize_biotek=False)
            
            # Validate hardware files
            #lash_e.nr_robot.check_input_file()
            #lash_e.nr_track.check_input_file()
            lash_e.nr_robot.home_robot_components()
            
            # Simple vial management: Set up source and measurement vials
            if SINGLE_VIAL:
                source_vial = "liquid_source_0"
                measurement_vial = "liquid_source_0" 
            else:
                source_vial = "liquid_source_0"
                measurement_vial = "measurement_vial_0"
            
            # Move measurement vial to clamp position for pipetting operations
            lash_e.nr_robot.move_vial_to_location(measurement_vial, "clamp", 0)

            #Tip Conditioning: Pre-wet tips with 4 aspirate/dispense cycles
            if liquid in LIQUIDS and LIQUIDS[liquid]['refill_pipets'] == False:
                conditioning_params = PipettingParameters(
                    aspirate_speed=15,
                    dispense_speed=5,
                    dispense_wait_time=0.0,
                    blowout_vol=0.5
                )
                lash_e.nr_robot.aspirate_from_vial(source_vial, self.conditioning_volume, move_up=False, parameters=conditioning_params)
                lash_e.nr_robot.dispense_into_vial(source_vial, self.conditioning_volume, initial_move=False, parameters=conditioning_params)
                for i in range(3):
                    lash_e.nr_robot.aspirate_from_vial(source_vial, self.conditioning_volume,  move_to_aspirate=False, parameters=conditioning_params)
                    lash_e.nr_robot.dispense_into_vial(source_vial, self.conditioning_volume, initial_move=False, parameters=conditioning_params)
                lash_e.nr_robot.move_home()
            
            print("READY: Hardware initialized successfully")
            
            return {
                'initialized_at': datetime.now(),
                'liquid': liquid,
                'lash_e': lash_e,
                'source_vial': source_vial,
                'measurement_vial': measurement_vial,
                'swap_enabled': SWAP,
                'measurement_count': 0,
                'continuous_mass_monitoring': continuous_monitoring,
                'simulate': simulate
            }
            
        except Exception as e:
            raise RuntimeError(f"Hardware initialization failed: {e}") from e

    def _check_and_swap_vials(self, state: Dict[str, Any], swap_enabled: bool = True) -> None:
        """Check measurement vial volume and swap if needed."""
        if not swap_enabled:
            return
            
        try:
            lash_e = state['lash_e']
            measurement_vial = state['measurement_vial']
            source_vial = state['source_vial']
            
            # Check current volume in both vials
            measurement_volume = lash_e.nr_robot.get_vial_info(measurement_vial, 'vial_volume')
            source_volume = lash_e.nr_robot.get_vial_info(source_vial, 'vial_volume')
            min_source_volume = 3.0  # mL - threshold for swapping when source gets low
            
            # DEBUG: Always print current volumes
            print(f"STATUS: measurement_vial={measurement_vial} ({measurement_volume:.2f}mL), source_vial={source_vial} ({source_volume:.2f}mL)")
            
            # Swap when source vial gets too low (< 2 mL)
            # AND measurement vial has accumulated enough liquid to become the new source (> 1 mL)
            should_swap = (source_volume is not None and 
                          source_volume < min_source_volume and
                          measurement_volume is not None and 
                          measurement_volume > 1.0)
            
            if should_swap:
                print(f"🔄 SWAP: Source vial ({source_vial}) low at {source_volume:.2f}mL (< {min_source_volume}mL), measurement vial ({measurement_vial}) has {measurement_volume:.2f}mL")
                
                # First: Return the old measurement vial (at clamp) to its home position
                lash_e.nr_robot.remove_pipet()
                lash_e.nr_robot.return_vial_home(measurement_vial)
                
                # Swap the vial roles in state
                state['source_vial'] = measurement_vial  # Old measurement becomes new source
                state['measurement_vial'] = source_vial   # Old source becomes new measurement
                
                # Finally: Move the new measurement vial to clamp position
                lash_e.nr_robot.move_vial_to_location(state['measurement_vial'], "clamp", 0)

                #Tip Conditioning: Pre-wet tips with 4 aspirate/dispense cycles
                liquid = state['liquid']  # Get liquid from state
                if liquid in LIQUIDS and LIQUIDS[liquid]['refill_pipets'] == False:
                    conditioning_params = PipettingParameters(
                        aspirate_speed=15,
                        dispense_speed=5,
                        dispense_wait_time=0.0,
                        blowout_vol=0.5
                    )
                    lash_e.nr_robot.aspirate_from_vial(state['source_vial'], self.conditioning_volume, move_up=False, parameters=conditioning_params)
                    lash_e.nr_robot.dispense_into_vial(state['source_vial'], self.conditioning_volume, initial_move=False, parameters=conditioning_params)
                    for i in range(3):
                        lash_e.nr_robot.aspirate_from_vial(state['source_vial'], self.conditioning_volume, move_to_aspirate=False, parameters=conditioning_params)
                        lash_e.nr_robot.dispense_into_vial(state['source_vial'], self.conditioning_volume, initial_move=False, parameters=conditioning_params)
                    lash_e.nr_robot.move_home()
                
                print(f"SWAP complete: source={state['source_vial']}, measurement={state['measurement_vial']}")
            else:
                print(f"✋ NO SWAP: source_vial has {source_volume:.2f}mL (>= {min_source_volume}mL threshold) or measurement_vial has {measurement_volume:.2f}mL (<= 1.0mL)")
                
        except Exception as e:
            print(f"WARNING: Vial swap check failed: {e}")

    def _evaluate_measurement(self, stability_info: Dict[str, Any], std_threshold: float = 0.001) -> bool:
        """Evaluate if a measurement is acceptable based on stability criteria.
        
        A baseline is considered stable if:
        - Stable readings percentage > 50%, OR
        - Standard deviation < threshold
        
        Both baselines must be stable for measurement to be trustworthy.
        
        Args:
            stability_info: Dictionary with stability metrics from dispense_into_vial
            std_threshold: Maximum acceptable standard deviation in grams (default: 0.001g = 1.0mg)
            
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
        """Perform hardware measurement with given parameters."""
        
        # Check if we need to swap vials before pipetting
        self._check_and_swap_vials(state, swap_enabled=state.get('swap_enabled', False))
        
        results = []
        lash_e = state['lash_e']

        if state.get('continuous_mass_monitoring', True):
            continuous_mass_monitoring=True
            save_mass_data=True
        else:
            continuous_mass_monitoring=False
            save_mass_data=False
        
        for rep in range(replicates):
            start_time = datetime.now()
            
            # Convert volume to microliters
            volume_uL = volume_mL * 1000
            
            # Extract parameters from nested structure
            hw_params = params.get('parameters', {}) if isinstance(params.get('parameters'), dict) else params
            
            # Extract pipetting parameters with safe fallbacks
            try:
                pipet_params = {
                    'aspirate_speed': hw_params.get('aspirate_speed', 10),
                    'dispense_speed': hw_params.get('dispense_speed', 10), 
                    'aspirate_wait_time': hw_params.get('aspirate_wait_time', 0.0),
                    'dispense_wait_time': hw_params.get('dispense_wait_time', 1.5),
                    'pre_asp_air_vol': hw_params.get('pre_asp_air_vol', 0.0),
                    'retract_speed': hw_params.get('retract_speed', 5.0),
                    'blowout_vol': hw_params.get('blowout_vol', 0.0),
                    'post_asp_air_vol': hw_params.get('post_asp_air_vol', 0.0),
                    'post_retract_wait_time': hw_params.get('post_retract_wait_time', 0.0),
                    'overaspirate_vol': params.get('overaspirate_vol', 0.0)  # This is at top level
                }
            except KeyError as e:
                raise ValueError(f"Missing required parameter structure - params dict malformed: {e}") from e
            
            print(f"  Rep {rep+1}/{replicates}: {volume_uL:.1f}uL with params {pipet_params}")
            
            # Perform pipetting operation: aspirate from source, dispense into measurement vial
            source_vial = state['source_vial']
            measurement_vial = state['measurement_vial']
            simulate = state['simulate']
           
            if not simulate:
                # Real hardware measurements with quality-controlled retry loop
                max_retries = 3
                retry_count = 0
                measurement_acceptable = False
                
                while not measurement_acceptable and retry_count <= max_retries:
                    if retry_count > 0:
                        print(f"    Retry attempt {retry_count}/{max_retries}")
                    
                    # Start timing just before the successful measurement attempt
                    rep_start = time.perf_counter()
                    
                    # Aspirate from source vial
                    lash_e.nr_robot.aspirate_from_vial(source_vial, volume_mL, parameters=pipet_params)
                    
                    # Dispense into measurement vial and measure weight
                    dispense_result = lash_e.nr_robot.dispense_into_vial(
                        measurement_vial, volume_mL, parameters=pipet_params, measure_weight=True, 
                        continuous_mass_monitoring=continuous_mass_monitoring, save_mass_data=save_mass_data
                    )
                    
                    # Handle new return format (mass, stability_info)
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
                    # Check if pipet removal is needed for this liquid (viscous liquids)
                    if LIQUIDS[state['liquid']]['refill_pipets']:
                        lash_e.nr_robot.remove_pipet()
                        print(f"    Removed pipet (refill_pipets=True for {state['liquid']})")
                        
                    if not measurement_acceptable and retry_count <= max_retries:
                        print(f"    WARNING! Measurement quality insufficient, retrying...")
                
                # Convert mass to volume using liquid density
                liquid = state['liquid']
                if liquid not in LIQUIDS:
                    raise ValueError(f"Unknown liquid '{liquid}' - must be one of: {list(LIQUIDS.keys())}")
                
                density_g_mL = LIQUIDS[liquid]['density']
                if not simulate: 
                    measured_volume_mL = measured_mass_g / density_g_mL  # mass (g) / density (g/mL) = volume (mL)
                    measured_mass_mg = measured_mass_g * 1000  # Convert g to mg for display
                else:
                    measured_mass_mg = 0.0
                    measured_volume_mL = 0.0
                
                print(f"    Mass: {measured_mass_mg:.2f}mg -> Volume: {measured_volume_mL*1000:.2f}uL (density: {density_g_mL:.3f}g/mL)")
                
                
                
            else:
                # Simple simulation: target volume - 20% + overaspirate + noise
                import random
                
                # Start timing for simulation
                rep_start = time.perf_counter()
                
                # Still call the robot methods for vial tracking, but ignore measurement result
                lash_e.nr_robot.aspirate_from_vial(source_vial, volume_mL, parameters=pipet_params)
                lash_e.nr_robot.dispense_into_vial(
                    measurement_vial, volume_mL, parameters=pipet_params, measure_weight=True
                )
                
                # Basic simulation logic
                base_efficiency = 0.8  # Start at 80% efficiency (target - 20%)
                overaspirate_effect = pipet_params.get('overaspirate_vol', 0.004) * 1000  # Convert to uL
                noise = random.uniform(-0.02, 0.02) * volume_mL  # ±2% noise
                
                # Simple formula: base efficiency + overaspirate helps + noise
                simulated_volume_mL = (volume_mL * base_efficiency) + (overaspirate_effect / 1000) + noise
                simulated_volume_mL = max(simulated_volume_mL, 0)  # Can't be negative
                
                measured_volume_mL = simulated_volume_mL
            
            # Convert to microliters for display
            measured_volume_uL = measured_volume_mL * 1000
            
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
                'measured_volume_uL': measured_volume_uL,
                'target_volume_uL': volume_uL,
                **pipet_params  # Echo back parameters
            }
            
            results.append(result)
            print(f"    Measured: {measured_volume_uL:.1f}uL (target: {volume_uL:.1f}uL) in {elapsed_s:.2f}s")
            print()  # Add visual separation between measurement cycles
        
        return results

    def wrapup(self, state: Dict[str, Any]) -> None:
        """Clean up hardware resources."""
        
        lash_e = state.get('lash_e')
        if lash_e:
            try:
                # Move robot to safe position
                lash_e.nr_robot.remove_pipet()
                lash_e.nr_robot.return_vial_home(state['measurement_vial'])
                lash_e.nr_robot.move_home()
                
                print(f"COMPLETE: Hardware cleanup completed. Total measurements: {state.get('measurement_count', 0)}")
                               
                # Send slack notification
                try:
                    if not state.get('simulate', True):
                        slack_agent.send_slack_message("🤖 North Robot calibration finished! All measurements completed.")
                except Exception as e:
                    print(f"WARNING: Slack notification failed: {e}")
                
            except Exception as e:
                print(f"WARNING: Hardware cleanup warning: {e}")

    def get_parameter_constraints(self, target_volume_ml: float) -> List[str]:
        """Get hardware-specific parameter constraints for North Robot system."""
        constraints = []
        
        # North Robot tip volume constraint
        # Use 0.2 mL tips for volumes <= 150 uL, otherwise 1.0 mL tips
        if target_volume_ml <= 0.20:  # 150 uL or less
            tip_volume_ml = 0.2
        else:
            tip_volume_ml = 1.0
            

        max_volume = 1.0    
        
        # Check if post_asp_air_vol parameter exists in hardware_parameters
        hardware_params = self.config.get_hardware_parameters()
        has_post_asp_air = 'post_asp_air_vol' in hardware_params.keys()
        
        # Add tip volume constraint - only include post_asp_air_vol if it exists
        if has_post_asp_air:
            constraints.append(f"post_asp_air_vol + overaspirate_vol <= {tip_volume_ml-target_volume_ml:.6f}")
        else:
            constraints.append(f"overaspirate_vol <= {tip_volume_ml-target_volume_ml:.6f}")

        if has_post_asp_air:
            constraints.append(f"pre_asp_air_vol + post_asp_air_vol + overaspirate_vol <= {max_volume-target_volume_ml:.6f}")
        else:
            constraints.append(f"pre_asp_air_vol + overaspirate_vol <= {max_volume-target_volume_ml:.6f}")
        
        return constraints


# Export the protocol instance for clean importing
protocol_instance = HardwareCalibrationProtocol()