#The purpose of this file is to combine multiple pieces of equipment into intuitive tools
import sys
sys.path.append("../utoronto_demo")
from North_Safe import North_Robot
from North_Safe import North_Track
from North_Safe import North_Temp
from North_Safe import North_Powder
from North_Safe import North_Spin
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging
import threading
from experiment_tracking import experiment_run_logger

# Import ConfigManager for workflow config handling
try:
    from workflow_config_manager import ConfigManager
except ImportError:
    ConfigManager = None

# ================================================================================
# CYTATION DATA PROCESSING UTILITIES
# ================================================================================

def flatten_cytation_data(data, measurement_type="unknown"):
    """
    Reliably flatten Cytation MultiIndex DataFrame to simple columns.
    
    This utility handles the common Cytation data structure issues:
    - MultiIndex columns like [('rep1_CMC_Absorbance_96', '600')] -> ['600']
    - Well positions as index -> 'well_position' column
    - Proper column naming for different measurement types
    
    Args:
        data: DataFrame from lash_e.measure_wellplate()
        measurement_type: 'turbidity' or 'fluorescence' for column naming
        
    Returns:
        DataFrame with simple columns: ['well_position', 'turbidity_600'] or ['well_position', '334_373', '334_384']
        
    Usage:
        turbidity_data = flatten_cytation_data(turbidity_data, 'turbidity')
        fluorescence_data = flatten_cytation_data(fluorescence_data, 'fluorescence')
        
    Note:
        Ratio calculation for pyrene fluorescence: 334_373 / 334_384 (I₃/I₁ ratio)
        This is the standard pyrene ratio for CMC determination.
    """
    if data is None:
        return None
        
    # Handle MultiIndex columns - flatten to single level
    if isinstance(data.columns, pd.MultiIndex):
        # Take the second level (wavelength/measurement) as column names
        data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]
    
    # Rename columns to standard names based on measurement type
    if measurement_type == 'turbidity':
        column_mapping = {}
        for col in data.columns:
            if '600' in str(col) or 'Absorbance' in str(col):
                column_mapping[col] = 'turbidity_600'
        if column_mapping:
            data = data.rename(columns=column_mapping)
    
    # Reset index to convert well positions (A1, A2, etc.) to a column
    data = data.reset_index()
    if 'index' in data.columns:
        data = data.rename(columns={'index': 'well_position'})
    elif data.columns[0] == 0 or str(data.columns[0]).startswith('Unnamed'):
        # Handle cases where reset_index creates unnamed column
        data.columns = ['well_position'] + list(data.columns[1:])
    
    return data

# ================================================================================
# ASYNC MEASUREMENT HANDLE
# ================================================================================

class MeasurementHandle:
    """
    Returned by Lash_E.measure_wellplate_async(). Holds the in-progress cytation
    read in a background thread. Call handle.complete() when you are ready to collect
    the result and return the wellplate to the pipetting area.

    Timeline:
        handle = lash_e.measure_wellplate_async(...)   # plate already at cytation, read running
        lash_e.nr_robot.dispense_stuff(...)            # arm works freely while cytation reads
        data = handle.complete()                       # parks arm, waits for read, returns plate
    """

    def __init__(self, lash_e, protocol_file_path, wells_to_measure, wellplate_index,
                 plate_type, cytation_plate_type, repeats, use_lid, plate_id=None):
        self._lash_e = lash_e
        self._wellplate_index = wellplate_index
        self.plate_id = plate_id
        self._returned = False
        self._plate_type = plate_type
        self._use_lid = use_lid
        self._result = None
        self._error = None
        self._done_event = threading.Event()

        # Capture everything the background thread needs
        self._protocol_file_path = protocol_file_path
        self._wells_to_measure = wells_to_measure
        self._cytation_plate_type = cytation_plate_type
        self._repeats = repeats

        self._thread = threading.Thread(target=self._run_cytation, daemon=True, name="cytation_read")
        self._thread.start()
        lash_e.logger.info("MeasurementHandle: cytation read started in background thread")

    def _run_cytation(self):
        """Background thread: runs the cytation protocol and stores the result."""
        try:
            lash_e = self._lash_e
            all_data = []

            if not lash_e.simulate and self._protocol_file_path is not None:
                protocol_paths = (self._protocol_file_path
                                  if isinstance(self._protocol_file_path, list)
                                  else [self._protocol_file_path])

                for i in range(self._repeats):
                    for protocol_path in protocol_paths:
                        lash_e.logger.info(f"MeasurementHandle: running protocol {protocol_path} (rep {i+1})")
                        try:
                            data = lash_e.cytation.run_protocol(
                                protocol_path, self._wells_to_measure,
                                plate_type=self._cytation_plate_type
                            )
                            if data is not None:
                                label = f"rep{i+1}_{os.path.splitext(os.path.basename(protocol_path))[0]}"
                                data.columns = pd.MultiIndex.from_tuples([(label, col) for col in data.columns])
                                all_data.append(data)
                        except RuntimeError as e:
                            lash_e.logger.error(f"MeasurementHandle: plate read failed: {e}")
                            self._error = e
                            return

            self._result = pd.concat(all_data, axis=1) if all_data else None
            lash_e.logger.info("MeasurementHandle: cytation read complete")

        except Exception as e:
            self._lash_e.logger.error(f"MeasurementHandle: unexpected error in background thread: {e}")
            self._error = e
        finally:
            self._done_event.set()

    def complete(self, destination="pipetting_area"):
        """
        Finish the measurement cycle:
          1. Parks the robot arm (unconditionally, before track moves)
          2. Waits for cytation read to finish (may already be done)
          3. Returns plate from cytation to pipetting area
          4. Returns the measurement DataFrame (or None in simulate mode)

        Raises any exception that occurred during the cytation read.
        """
        if self._returned:
            return self._result
        lash_e = self._lash_e
        lash_e.logger.info("MeasurementHandle.complete(): parking arm before returning plate")

        # Step 1: arm home — unconditional, must happen before track moves
        lash_e.nr_robot.move_home()

        # Step 2: wait for cytation read (may already be done)
        lash_e.logger.info("MeasurementHandle.complete(): waiting for cytation read to finish")
        self._done_event.wait()

        if self._error is not None:
            lash_e.nr_robot.pause_after_error(f"Cytation read failed in background thread: {self._error}")
            raise self._error

        # Step 3: return plate
        lash_e.logger.info("MeasurementHandle.complete(): returning plate from cytation")
        lash_e.move_wellplate_back_from_cytation(
            self._wellplate_index, plate_type=self._plate_type, use_lid=self._use_lid,
            plate_id=self.plate_id, destination=destination
        )
        lash_e.nr_track.origin()

        self._returned = True
        if self._result is not None:
            self._result.attrs.update(plate_id=self.plate_id, plate_type=self._plate_type)
        return self._result


# ================================================================================

# The most recently created coordinator, so the scheduler can tidy up after a workflow finishes
last_coordinator = None


class Lash_E:
    nr_robot = None
    nr_track = None
    cytation = None
    spinner = None
    photoreactor = None
    temp_controller = None
    powder_dispenser = None
    simulate = None

    def __init__(self, vial_file=None, initialize_robot=True,initialize_track=True,initialize_biotek=True,initialize_t8=False,initialize_p2=False,simulate=False,logging_folder="../utoronto_demo/logs", workflow_globals=None, workflow_name=None, show_gui=True, config=None, dispense_recording="off"):
        """
        Initialize Lash_E coordinator.
        
        Args:
            show_gui (bool): If True (default), shows the vial manager GUI for status review.
                           If False, skips GUI and proceeds directly to hardware initialization.
                           Use show_gui=False for batch operations or automated workflows.
            config (mapping): Complete automated workflow settings. Requires
                              show_gui=False; saved YAML is not read or written.
                              Provide workflow_globals/name to identify required keys.
            dispense_recording (str): "auto" records known-liquid pipette additions at
                                      clamp 0; "off" disables routine recording.
                                      Simulation never saves measurements.
            workflow_config (attribute): Confirmed settings after review. Build
                                         experiment plans from this snapshot.
        """
        
        if dispense_recording not in ("off", "auto"):
            raise ValueError("dispense_recording must be 'off' or 'auto'")
        self.dispense_recording = dispense_recording

        # Handle workflow config loading before any other initialization
        self.workflow_config = None
        launch_config = None
        if config is not None or (workflow_globals is not None and workflow_name is not None):
            if ConfigManager is None:
                raise RuntimeError("ConfigManager is required for workflow configuration.")
            launch_config = ConfigManager.resolve_workflow_config(
                workflow_name, workflow_globals, config=config, show_gui=show_gui
            )
            simulate = launch_config['SIMULATE']
            vial_file = launch_config['INPUT_VIAL_STATUS_FILE']
            self.workflow_name = workflow_name
            self.config_loaded = config is None
            self.config_keys_loaded = list(launch_config)
        else:
            self.workflow_name = None
            self.config_loaded = False
            self.config_keys_loaded = []
        
        self.simulate = simulate
        self.vial_file = vial_file  # Store vial file for GUI access
        self.show_gui = show_gui  # Control GUI display
        
        # Flag to control workflow continuation
        self._workflow_should_continue = True
        global last_coordinator
        last_coordinator = self

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self.log_filename = None
        self._run_tracker = None

        # Check input status before hardware initialization (if GUI enabled)
        if self.show_gui:
            self.check_input_status()
            
            # Exit early if workflow was aborted
            if not self._workflow_should_continue:
                self.logger.info("Workflow aborted by user - skipping hardware initialization")
                return
        else:
            self.logger.info("GUI disabled - skipping status review, proceeding directly to hardware initialization")
            
        # Reload config from file after GUI may have updated YAML values
        if launch_config is not None:
            self.workflow_config = ConfigManager.confirm_workflow_config(
                workflow_name, workflow_globals, launch_config,
                supplied=config is not None, logger=self.logger,
            )
            vial_file = self.workflow_config['INPUT_VIAL_STATUS_FILE']
            self.vial_file = vial_file
            updated_simulate = self.workflow_config['SIMULATE']
            if updated_simulate != self.simulate:
                self.update_simulate_flag(updated_simulate)

        if os.environ.get("NORTH_SCHEDULER_SIMULATION_STATE"):
            from scheduler.simulation_state import validate_launch
            validate_launch(self.simulate, vial_file)

        suffix = "_simulate" if self.simulate else ""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"experiment_log{timestamp}{suffix}.log"
        os.makedirs(logging_folder, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logging_folder, self.log_filename))
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        if os.environ.get("NORTH_SCHEDULER_SIMULATION_STATE"):
            from scheduler.simulation_state import attach_diagnostics
            attach_diagnostics(self.logger)

        if self.config_loaded:
            self.logger.info(f"Workflow config confirmed: {self.workflow_name}")
            self.logger.info(f"Config keys loaded: {len(self.config_keys_loaded)} ({', '.join(self.config_keys_loaded)})")
        elif config is not None:
            self.logger.info(f"Supplied workflow config: {self.workflow_name}")
        elif workflow_globals is not None or workflow_name is not None:
            self.logger.warning("Incomplete config info provided - config loading skipped")
        self.logger.info(f"Confirmed SIMULATE mode: {self.simulate}")

        self._run_tracker = experiment_run_logger.start_run(
            self.workflow_name, self.simulate, self.log_filename, self.logger, logging_folder,
            self.workflow_config
        )
        if self._run_tracker is not None:
            self._run_tracker.vial_file = vial_file

        # Create hardware objects AFTER config reload to use correct simulate flag
        if not self.simulate:
            from north import NorthC9
            c9 = NorthC9("A", network_serial="AU06CNCF")
            c8 = NorthC9('D', network=c9.network)
        else:
            from unittest.mock import MagicMock
            from scheduler.fault_injection import simulated_c9
            c9 = simulated_c9()  # a plain stand-in, unless a test asks for a fault (NORTH_SIM_FAULT)
            c8 = MagicMock()

        if initialize_robot and not self.simulate and vial_file:
            # Live run start: every vial gets a container ID, and a vial a person emptied or
            # replaced (0 mL, but with history in the transfer log) gets a new one. Never raises.
            from experiment_tracking.sample_tracking import prepare_vial_file, snapshot
            prepare_vial_file(vial_file, retire_emptied=True, logger=self.logger)
            # How things started: the vial file and the stored contents, saved with the run's data.
            tracker = getattr(self, "_run_tracker", None)
            if tracker is not None:
                snapshot(tracker.run_id, "start", vial_file, logger=self.logger)

        if initialize_robot and not self.simulate:
            # Reservoirs estimated to be running low: a warning only, never a stop.
            from experiment_tracking.sample_tracking import reservoir_warnings
            for warning in reservoir_warnings():
                self.logger.warning(warning)

        if initialize_robot:
            self.nr_robot = North_Robot(c9, c8, vial_file,simulate=self.simulate, logger=self.logger,
                                        dispense_recording=self.dispense_recording)
            # Pass log filename to robot for organized mass measurement file storage
            self.nr_robot.log_filename = self.log_filename
            tracker = getattr(self, "_run_tracker", None)  # run ID stamped on each sample-tracking transfer row
            self.nr_robot.run_id = tracker.run_id if tracker is not None else None
            self.spinner = North_Spin(c9, c8, simulate=self.simulate, logger=self.logger)

        if initialize_biotek:
            from biotek_new import Biotek_Wrapper
            self.cytation = Biotek_Wrapper(simulate=self.simulate, logger=self.logger)
        
        if initialize_track:
            self.nr_track = North_Track(c9, simulate=self.simulate, logger=self.logger)
            self.nr_track.check_reader_motion = self._check_reader_motion
            if initialize_robot:
                self.nr_robot.plate_inventory = self.nr_track.inventory
        

        # Photoreactor initialization
        if not self.simulate:
            from photoreactor_controller import Photoreactor_Controller
            self.photoreactor = Photoreactor_Controller() #ADd logger

            if initialize_p2:
                self.powder_dispenser = North_Powder(c9, simulate=self.simulate, logger=self.logger)
            if initialize_t8:
                self.temp_controller = North_Temp(c9, c8, simulate=self.simulate, logger=self.logger)

        else:
            from unittest.mock import MagicMock
            self.photoreactor = MagicMock()
            # The real powder code on the simulated controller, so simulation checks the channel,
            # the carousel move and the capped-vial check. Not started unless asked, as in a live run.
            self.powder_dispenser = (North_Powder(c9, simulate=True, logger=self.logger)
                                     if initialize_p2 else None)
            self.temp_controller = MagicMock()
            #self.nr_track = MagicMock(c9)

            if os.environ.get("NORTH_SCHEDULER_SIMULATION_STATE"):
                from scheduler.simulation_state import register_coordinator
                register_coordinator(self)

    @property
    def run_id(self):
        """Unique ID of this run (also in experiment_tracking/experiment_runs.csv), or None if not started."""
        return self._run_tracker.run_id if self._run_tracker is not None else None

    def save_run_info(self, output_folder):
        """Save this run's config and ID into a workflow's output folder, and record the folder.

        Writes config.yaml (the confirmed workflow config, if any) and run_info.yaml (run_id,
        workflow name, simulate flag, start time). The folder is added to experiment_runs.csv.
        Call once, after the workflow creates its output folder.
        """
        import yaml
        os.makedirs(output_folder, exist_ok=True)
        tracker = self._run_tracker
        if self.workflow_config is not None:
            with open(os.path.join(output_folder, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(dict(self.workflow_config), f, sort_keys=False)
        run_info = {"run_id": self.run_id, "workflow_name": self.workflow_name, "simulate": self.simulate,
                    "datetime_started": tracker.datetime_started if tracker is not None else None}
        with open(os.path.join(output_folder, "run_info.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(run_info, f, sort_keys=False)
        if tracker is not None:
            experiment_run_logger.record_output_folder(tracker, output_folder)

    # ---- Module state readout (see docs/plans/module_state_readout.md) ----

    MODULE_STATE_SETTINGS_FILE = "robot_state/robot_hardware.yaml"
    # Resources whose state is not saved anywhere yet; reported as unknown placeholders
    _UNTRACKED_RESOURCES = ("spin_coater", "carousel", "powder_dispenser", "reservoir_pumps", "camera")

    @staticmethod
    def _resource(status, reasons=(), source="saved", **fields):
        """One resource entry: common fields first, then the resource's own fields."""
        return {"status": status, "reasons": list(reasons), "in_use_by": None, "source": source, **fields}

    def _vial_name(self, vial_df, vial_index):
        if vial_index is None:
            return None
        if vial_df is not None and vial_index in vial_df.index:
            return vial_df.loc[vial_index, "vial_name"]
        return vial_index

    @staticmethod
    def _vials_at(vial_df, location):
        if vial_df is None:
            return []
        return list(vial_df.loc[vial_df["location"] == location, "vial_name"])

    @staticmethod
    def _vials_not_home(vial_df):
        if vial_df is None:
            return []
        def same(a, b):
            try:
                return int(a) == int(b)
            except (TypeError, ValueError):
                return str(a) == str(b)
        return [row.vial_name for row in vial_df.itertuples()
                if row.location != row.home_location or not same(row.location_index, row.home_location_index)]

    def get_module_state(self):
        """Report the saved state of each Lash-E resource. Read-only: never moves or queries hardware.

        Returns {resource_name: {status, reasons, in_use_by, source, ...resource fields}}.
        Movers and stations are ready / not_ready; instruments are idle / has_plate / unknown.
        Resources whose state is not saved yet are reported with status and source unknown /
        not_tracked. in_use_by is reserved for claiming resources in asynchronous operation.
        """
        import yaml
        with open(self.MODULE_STATE_SETTINGS_FILE, encoding="utf-8") as f:
            settings = (yaml.safe_load(f) or {})["module_state"]
        robot = getattr(self, "nr_robot", None)
        track = getattr(self, "nr_track", None)
        vial_df = getattr(robot, "VIAL_DF", None) if robot is not None else None
        state = {}

        # Robot arm (mover)
        if robot is None:
            state["robot_arm"] = self._resource("unknown", source="not_initialized")
        else:
            holding = robot.GRIPPER_STATUS
            item = self._vial_name(vial_df, robot.GRIPPER_VIAL_INDEX)
            not_home = self._vials_not_home(vial_df)
            reasons = []
            if holding is not None:
                reasons.append(f"gripper holding {holding} ({item})")
            if robot.HELD_PIPET_TYPE is not None:
                reasons.append(f"tip attached ({robot.HELD_PIPET_TYPE})")
            if not_home:
                reasons.append(f"vials not home: {', '.join(map(str, not_home))}")
            state["robot_arm"] = self._resource(
                "not_ready" if reasons else "ready", reasons,
                gripper_holding=holding, gripper_item_id=item,
                tip_type=robot.HELD_PIPET_TYPE, tip_liquid_mL=robot.PIPET_FLUID_VOLUME,
                tip_source_vial=self._vial_name(vial_df, robot.PIPET_FLUID_VIAL_INDEX),
                vials_not_home=not_home)

        # Stations that hold vials
        for name in ("clamp", "clamp_2"):
            vials = self._vials_at(vial_df, settings["station_locations"][name])
            state[name] = self._resource("not_ready" if vials else "ready",
                                         [f"occupied by {', '.join(map(str, vials))}"] if vials else [],
                                         source="saved" if vial_df is not None else "not_tracked",
                                         vial=vials[0] if vials else None)

        # Track (mover) and the Cytation's plate
        if track is None:
            state["track"] = self._resource("unknown", source="not_initialized")
            state["cytation"] = self._resource("unknown", source="not_initialized")
        else:
            inventory = track.inventory
            held = inventory.contents("gripper") if "gripper" in inventory.positions else []
            pending = inventory.data.get("pending_transfer")
            location = track.CURRENT_GRIPPER_LOCATION
            reasons = []
            if held:
                reasons.append(f"holding {held[0]}")
            if pending:
                reasons.append(f"unfinished move of {pending.get('object_id')}")
            if location != settings["track_parked_location"]:
                reasons.append(f"not parked (at {location})")
            state["track"] = self._resource(
                "not_ready" if reasons else "ready", reasons,
                location=location, gripper_plate=held[0] if held else None, pending_transfer=pending,
                pipetting_area=inventory.contents("pipetting_area"))
            plate = inventory.contents("cytation_tray")
            state["cytation"] = self._resource("has_plate" if plate else "idle",
                                               [f"plate {plate[0]} in tray"] if plate else [],
                                               plate=plate[0] if plate else None)

        # Instruments that hold vials: contents are saved, running state is not
        for name in ("heater", "photoreactor"):
            state[name] = self._resource("unknown", source="partly_saved",
                                         vials=self._vials_at(vial_df, settings["station_locations"][name]))

        for name in self._UNTRACKED_RESOURCES:
            state[name] = self._resource("unknown", source="not_tracked")
        return state

    # ---- Safe stop: the one cleanup used after any error or interrupt ----

    # Instruments safe_stop can't switch off yet (no stop method); reported so an operator checks them
    _SAFE_STOP_NOT_HANDLED = ("reservoir pumps",)

    def tidy_after_workflow(self):
        """After a workflow finishes normally: return any vial it left away from home.

        A flow should end with every vial home, so anything found here means the workflow
        should be fixed: each case is logged as a warning naming the workflow (the scheduler
        shows warnings without blocking). Nothing found: nothing logged. Never raises.
        Held tips, gripper contents and plates are left to the scheduler's own end-of-job
        check, which already blocks the next job.
        """
        robot = getattr(self, "nr_robot", None)
        if robot is None:
            return [], []
        name = getattr(self, "workflow_name", None) or "workflow"
        try:
            if not robot._vials_available():
                return [], []
            returned, left = robot.return_all_vials_home()
        except Exception as error:
            self.logger.warning(f"{name} ended; could not check vials are home: {type(error).__name__}: {error}")
            return [], []
        if returned:
            self.logger.warning(f"{name} ended with vials away from home: {', '.join(map(str, returned))}. "
                                "Returned them home; the workflow should do this itself.")
        for vial, why in left:
            self.logger.warning(f"{name} ended with vial {vial} away from home ({why}); left in place.")
        return returned, left

    def safe_stop(self, reason=""):
        """Bring Lash-E to a safe state after an error or interrupt, and report what was left.

        Steps, in order (each runs even if an earlier one fails):
          1. Instruments off: heater and its stirring, photoreactor LEDs and fans, spin coater spin,
             powder cartridge valves closed (slots from powder_channels.yaml).
             Heater channels and reactor numbers come from safe_stop in robot_hardware.yaml.
          2. Track: back to origin, unless it is holding a plate or stopped mid-move. Then it is
             left alone (its saved state may be wrong and moving could crash it) and reported.
          3. Robot: return liquid in the tip, drop the tip, recap or return a held vial, arm home.
          4. Vials: every movable vial that is not home is returned, one by one.
          5. Report: log and (live runs) Slack what was done and what needs an operator.

        Returns a report dict: done, failed, left_for_operator, vials_returned, not_handled.
        """
        import yaml
        try:
            with open(self.MODULE_STATE_SETTINGS_FILE, encoding="utf-8") as f:
                settings = (yaml.safe_load(f) or {}).get("safe_stop", {})
        except Exception:
            self.logger.exception("safe_stop could not read its settings; using none")
            settings = {}
        report = {"reason": reason, "done": [], "failed": [], "left_for_operator": [],
                  "vials_returned": [], "not_handled": list(self._SAFE_STOP_NOT_HANDLED)}
        # The error being handled when safe_stop was called (if any): its code decides the
        # "what to do" line and whether Slack is told (not for Ctrl+C or an E-stop).
        cause = sys.exc_info()[1]
        error = None
        if cause is not None:
            try:
                from experiment_tracking.error_records import classify
                error = classify(cause)
                report["error_code"] = error["code"]
            except Exception:
                self.logger.exception("safe_stop could not classify the error")
        tracker = vars(self.logger).get("_experiment_run_tracker")
        if tracker is not None:
            tracker.safe_stop_report = report

        def step(label, action):
            try:
                action()
                report["done"].append(label)
            except (Exception, KeyboardInterrupt):
                self.logger.exception("Safe stop step failed: %s", label)
                report["failed"].append(label)

        self.logger.warning(f"Safe stop started{': ' + reason if reason else ''}")

        # 1. Instruments off
        temp = getattr(self, "temp_controller", None)
        if temp is not None:
            for channel in settings.get("heater_channels", []):
                step(f"heater channel {channel} off", lambda c=channel: temp.turn_off_heating(c))
            step("heater stirring off", temp.turn_off_stirring)
        photoreactor = getattr(self, "photoreactor", None)
        if photoreactor is not None:
            for reactor in settings.get("photoreactor_reactors", []):
                step(f"photoreactor {reactor} LED off", lambda r=reactor: photoreactor.turn_off_reactor_led(r))
                step(f"photoreactor {reactor} fan off", lambda r=reactor: photoreactor.turn_off_reactor_fan(r))
        spinner = getattr(self, "spinner", None)
        if spinner is not None:
            step("spin coater stopped", spinner.stop_spin)
        powder = getattr(self, "powder_dispenser", None)
        if powder is not None:
            step("powder valves closed", powder.close_all_valves)

        # 2. Track
        track = getattr(self, "nr_track", None)
        if track is not None:
            try:
                inventory = track.inventory
                held = inventory.positions.get("gripper")
                pending = inventory.data.get("pending_transfer")
            except Exception:
                held, pending = None, None
            if held or pending:
                report["left_for_operator"].append(
                    f"track may be holding a plate ({held}) or stopped mid-move ({pending}); check it by hand")
            else:
                step("track to origin", track.reset_after_initialization)

        # 3 and 4. Robot and vials
        robot = getattr(self, "nr_robot", None)
        if robot is not None:
            step("robot cleanup (tip, gripper, arm home)", robot._perform_physical_cleanup)

            def vials_home():
                returned, left = robot.return_all_vials_home()
                report["vials_returned"] = returned
                report["left_for_operator"].extend(f"vial {name} not home ({why})" for name, why in left)
            step("vials home", vials_home)
            if robot.GRIPPER_STATUS is not None:
                report["left_for_operator"].append(f"robot gripper still holding {robot.GRIPPER_STATUS}")
            if robot.HELD_PIPET_TYPE is not None:
                report["left_for_operator"].append(f"robot still has a tip ({robot.HELD_PIPET_TYPE})")

        # 5. Report
        summary = (f"Safe stop{' (' + reason + ')' if reason else ''}: "
                   f"{len(report['done'])} steps done, {len(report['failed'])} failed"
                   + (f" ({', '.join(report['failed'])})" if report["failed"] else "")
                   + (f". Vials returned: {', '.join(map(str, report['vials_returned']))}" if report["vials_returned"] else "")
                   + (f". Needs an operator: {'; '.join(report['left_for_operator'])}" if report["left_for_operator"] else "")
                   + f". Not handled automatically: {', '.join(report['not_handled'])}."
                   + (f" Error {error['code']}: {error['what_to_do']}" if error and error["code"] != "UNKNOWN" else ""))
        self.logger.warning(summary)
        if not self.simulate and (error is None or error["slack"]):
            try:
                import slack_agent
                slack_agent.send_slack_message(summary)
            except Exception:
                self.logger.exception("safe_stop could not send its Slack summary")
        return report

    def update_simulate_flag(self, new_simulate_value):
        """
        Update the simulation flag after initialization (e.g., after ConfigManager loads updated values).
        
        This fixes the issue where Lash_E is initialized with one simulate value, but ConfigManager
        later updates the global SIMULATE variable. This method synchronizes the instance with
        the updated global value.
        
        Args:
            new_simulate_value: Updated simulation flag value (True/False)
        """
        if self.simulate == new_simulate_value:
            return  # No change needed
            
        old_simulate = self.simulate
        self.simulate = new_simulate_value
        
        self.logger.info(f"SIMULATE flag updated: {old_simulate} -> {new_simulate_value}")
        
        # Update simulation flag in all initialized components
        components_updated = []
        
        if hasattr(self, 'nr_robot') and self.nr_robot:
            self.nr_robot.simulate = new_simulate_value
            components_updated.append('nr_robot')
            
        if hasattr(self, 'nr_track') and self.nr_track:
            self.nr_track.simulate = new_simulate_value
            components_updated.append('nr_track')
            
        if hasattr(self, 'cytation') and self.cytation:
            self.cytation.simulate = new_simulate_value
            components_updated.append('cytation')
            
        if hasattr(self, 'spinner') and self.spinner:
            self.spinner.simulate = new_simulate_value
            components_updated.append('spinner')
            
        if hasattr(self, 'powder_dispenser') and self.powder_dispenser:
            self.powder_dispenser.simulate = new_simulate_value
            components_updated.append('powder_dispenser')
            
        if hasattr(self, 'temp_controller') and self.temp_controller:
            self.temp_controller.simulate = new_simulate_value
            components_updated.append('temp_controller')
        
        if components_updated:
            self.logger.debug(f"Updated simulate flag in components: {', '.join(components_updated)}")
        else:
            self.logger.debug("No components needed simulate flag update")

    def check_input_status(self):
        """Launch GUI for reviewing and finalizing robot states before hardware initialization."""

        self.logger.info("Launching status review GUI...")
        
        try:
            # Import and launch the GUI
            from vial_manager_gui import VialManagerMainWindow
            from PySide6.QtWidgets import QApplication
            import sys
            
            # Create Qt application if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Create and configure the GUI window
            gui = VialManagerMainWindow()
            gui.setWindowTitle("Robot Status Review - Workflow Initialization")
            
            # Detect workflow name from main module
            workflow_name = self.workflow_name
            try:
                import __main__
                if workflow_name is None and hasattr(__main__, '__file__') and __main__.__file__:
                    script_name = os.path.basename(__main__.__file__)
                    workflow_name = os.path.splitext(script_name)[0]  # Remove .py extension
                    self.logger.info(f"Detected workflow name: {workflow_name}")
            except Exception as e:
                self.logger.warning(f"Could not detect workflow name: {e}")
            
            # Configure GUI for workflow mode
            gui._setup_workflow_mode(self.vial_file, self, workflow_name)
            
            # Show GUI and wait for user decision
            gui.show()
            app.exec()  # Block until GUI closes
            
            # Check if workflow should continue
            self._workflow_should_continue = getattr(gui, '_workflow_continue', True)
            
            if self._workflow_should_continue:
                self.logger.info("Status review completed - proceeding with hardware initialization")
            else:
                self.logger.info("Workflow aborted by user")
                
        except Exception as e:
            self.logger.error(f"Error launching status GUI: {e}")
            # Continue with workflow if GUI fails
            self._workflow_should_continue = True

    def mass_dispense_into_vial(self,vial,mass_mg,channel=0, return_home=True):
        """Flow: dispense powder into a vial at the clamp. Returns the mass dispensed (mg).

        Like dispense_into_vial_from_reservoir: the vial goes to the clamp, a closed cap is taken
        off (powder goes through an open cap), and it is recapped after.
        """
        if self.powder_dispenser is None:
            raise RuntimeError("Powder dispenser isn't started: create Lash_E with initialize_p2=True")
        robot = self.nr_robot
        self.logger.info(f"Dispensing into vial: {vial} with mass: {mass_mg:.3f} mg")
        if robot.get_vial_info(vial, 'location') != 'clamp':
            robot.move_vial_to_location(vial,'clamp',0)
        if not robot.is_vial_pipetable(vial):  # closed cap; powder goes through an open cap
            robot.uncap_clamp_vial()
        robot.move_home()
        dispensed_mg = self.powder_dispenser.dispense_powder_mg(mass_mg=mass_mg,channel=channel) #Dispense 50 mg of solid into source_vial_a
        robot.track_powder_dispense(channel, vial, dispensed_mg)  # log row and the vial's stored contents
        if robot.GRIPPER_STATUS == "Cap":  # this flow took the cap off: put it back
            robot.recap_clamp_vial()
        if return_home:
            robot.return_vial_home(vial)
        return dispensed_mg

    def _resolve_plate(self, location, plate_type=None, use_lid=None, plate_id=None):
        inv = self.nr_track.inventory
        plate_id = inv.plate_at(location, plate_type, plate_id, use_lid)
        actual_type = inv.plate_type(plate_id)
        reader_type = inv.definitions[actual_type].get("name_in_cytation") or actual_type
        return plate_id, actual_type, plate_id in inv.attachments, reader_type

    def _check_reader_motion(self):
        handle = getattr(self, "_active_measurement", None)
        if getattr(self, "_cytation_reading", False) or (handle is not None and not handle._done_event.is_set()):
            raise ValueError("Cytation read is active; its carrier cannot move yet")

    def move_wellplate_to_cytation(self, wellplate_index=0, plate_type=None, use_lid=None, safe_movement=True, plate_id=None):
        self._check_reader_motion()
        plate_id, plate_type, use_lid, reader_type = self._resolve_plate("pipetting_area", plate_type, use_lid, plate_id)
        inv = self.nr_track.inventory
        inv.check_destination(plate_id, "cytation_tray")
        inv.stack_z("pipetting_area", "grab")
        inv.stack_z("cytation_tray", "release")
        self.nr_robot.move_home()
        if safe_movement:
            self.nr_track.move_through_path(['cytation_safe_area'])
        self.nr_track.grab_wellplate_from_location('pipetting_area', plate_type)
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            self.cytation.CarrierOut()
        self.nr_track.release_wellplate_in_location('cytation_tray', plate_type)
        if not self.simulate:
            self.cytation.CarrierIn(plate_type=reader_type, use_lid=use_lid)
        self.logger.info("Plate %s (%s) loaded at Cytation", plate_id, plate_type)
        return plate_id

    def move_wellplate_back_from_cytation(self, wellplate_index=0, plate_type=None, use_lid=None, plate_id=None, destination="pipetting_area"):
        self._check_reader_motion()
        plate_id, plate_type, use_lid, reader_type = self._resolve_plate("cytation_tray", plate_type, use_lid, plate_id)
        inv = self.nr_track.inventory
        # Check before touching the carrier or picking up, so completion is retryable.
        inv.check_destination(plate_id, destination)
        inv.stack_z(destination, "release")
        self.nr_robot.move_home()
        if not self.simulate:
            self.cytation.CarrierOut()
        self.nr_track.grab_wellplate_from_location('cytation_tray', plate_type)
        self.nr_track.move_through_path(['cytation_safe_area'])
        if not self.simulate:
            self.cytation.CarrierIn(plate_type=reader_type, use_lid=use_lid)
        self.nr_track.release_wellplate_in_location(destination, plate_type)
        return plate_id

    #Note from OAM: The data formatting from this can be annoying. Need to think about how to handle it. 
    def measure_wellplate(self, protocol_file_path=None, wells_to_measure=None, wellplate_index=0, plate_type=None, repeats=1, use_lid=None, safe_movement=True, preparation_protocol=None, plate_id=None, plate_setup=None):
        """
        Measure a wellplate on the Cytation reader. Supports multiple protocols and replicate measurements.
        An optional preparation_protocol runs once after loading, before all reads,
        using a single well selection so a whole-plate shake is not repeated by row.
        Each replicate includes all protocols, e.g.:
            rep1: fluorescence, absorbance
            rep2: fluorescence, absorbance
            ...
        
        Args:
            plate_type (str): Robot's wellplate type (e.g., "96 WELL PLATE", "quartz") - used for robot movements
            use_lid (bool): Whether the wellplate has a lid (default: False)
            plate_setup (callable, optional): Procedure inspection/modification hook
                passed to the reader for measurement protocols only.
        """
        self.logger.info("Measuring wellplate %d with protocols: %s", wellplate_index, protocol_file_path)
        
        plate_id, plate_type, use_lid, cytation_plate_type = self._resolve_plate(
            "pipetting_area", plate_type, use_lid, plate_id)
        if wells_to_measure is not None:
            count = self.nr_track.inventory.definitions[plate_type]["num_wells"]
            wells_to_measure = list(wells_to_measure)
            if any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) or not 0 <= w < count for w in wells_to_measure):
                raise ValueError(f"Invalid wells for plate {plate_id} ({plate_type})")
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index, plate_type=plate_type, use_lid=use_lid, safe_movement=safe_movement, plate_id=plate_id)

        # A shake acts on the whole plate. Selecting one well prevents the reader
        # wrapper from repeating it for every row/group in wells_to_measure.
        if preparation_protocol is not None and not self.simulate:
            self._run_plate_protocol(preparation_protocol, [0], plate_type=cytation_plate_type)

        all_data = []

        if not self.simulate and protocol_file_path is not None:  #Note from OAM: The data formatting from this can be annoying. Need to think about how to handle it. 
            # Ensure it's a list for consistent handling
            protocol_paths = protocol_file_path if isinstance(protocol_file_path, list) else [protocol_file_path]

            for i in range(repeats):
                for protocol_path in protocol_paths:
                    print(f"Running protocol {protocol_path} (rep {i+1})")
                    try:
                        # Use cytation_plate_type for the biotek system, not the robot's plate_type
                        setup_options = {} if plate_setup is None else {"plate_setup": plate_setup}
                        data = self._run_plate_protocol(protocol_path, wells_to_measure, plate_type=cytation_plate_type, **setup_options)
                        if data is not None:
                            # Add a MultiIndex: (replicate + protocol name, wavelength/column)
                            label = f"rep{i+1}_{os.path.splitext(os.path.basename(protocol_path))[0]}"
                            data.columns = pd.MultiIndex.from_tuples([(label, col) for col in data.columns])
                            all_data.append(data)
                    except RuntimeError as e:
                        self.logger.error(f"Plate read failed: {e}")
                        raise

            combined_data = pd.concat(all_data, axis=1) if all_data else None
        else:
            combined_data = None

        self.move_wellplate_back_from_cytation(wellplate_index, plate_type=plate_type, use_lid=use_lid, plate_id=plate_id)
        self.nr_track.origin()
        if combined_data is not None:
            combined_data.attrs.update(plate_id=plate_id, plate_type=plate_type)
        return combined_data


    def measure_wellplate_async(self, protocol_file_path=None, wells_to_measure=None, wellplate_index=0, plate_type=None, repeats=1, use_lid=None, safe_movement=True, plate_id=None):
        """
        Non-blocking version of measure_wellplate(). Moves the wellplate to the Cytation
        and starts the read in a background thread, then returns a MeasurementHandle
        immediately so the robot arm can continue dispensing.

        Usage:
            handle = lash_e.measure_wellplate_async(protocol_file, wells_list)
            # ... arm does next round of dispensing ...
            data = handle.complete()  # parks arm, waits for cytation, returns plate, returns data

        Args: same as measure_wellplate()
        Returns: MeasurementHandle
        """
        self.logger.info("measure_wellplate_async: moving plate to cytation then starting background read")

        plate_id, plate_type, use_lid, cytation_plate_type = self._resolve_plate(
            "pipetting_area", plate_type, use_lid, plate_id)
        if wells_to_measure is not None:
            count = self.nr_track.inventory.definitions[plate_type]["num_wells"]
            wells_to_measure = list(wells_to_measure)
            if any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) or not 0 <= w < count for w in wells_to_measure):
                raise ValueError(f"Invalid wells for plate {plate_id} ({plate_type})")
        # Step 1: park arm and move plate to cytation (synchronous — track needs c9)
        self.nr_robot.move_home()
        self.move_wellplate_to_cytation(wellplate_index, plate_type=plate_type, use_lid=use_lid, safe_movement=safe_movement, plate_id=plate_id)

        # Step 2: start cytation read in background (c9 is now idle)
        handle = MeasurementHandle(
            lash_e=self,
            protocol_file_path=protocol_file_path,
            wells_to_measure=wells_to_measure,
            wellplate_index=wellplate_index,
            plate_type=plate_type,
            cytation_plate_type=cytation_plate_type,
            repeats=repeats,
            use_lid=use_lid,
            plate_id=plate_id,
        )

        self._active_measurement = handle
        return handle


    def run_photoreactor(self,vial_index,target_rpm,intensity,duration,reactor_num):
        self.logger.info("Running photoreactor for vial %d with target RPM: %d, intensity: %d, duration: %d, reactor number: %d", vial_index, target_rpm, intensity, duration, reactor_num)
        self.nr_robot.move_vial_to_location(vial_index,'photoreactor_array',reactor_num)
        self.photoreactor.run_photoreactor(target_rpm,duration,intensity,reactor_num)
        self.nr_robot.return_vial_home(vial_index)

    def _run_plate_protocol(self, *args, **kwargs):
        self._cytation_reading = True
        try:
            return self.cytation.run_protocol(*args, **kwargs)
        finally:
            self._cytation_reading = False

    def move_wellplate(self, source, destination, expected_type=None):
        if source == "cytation_tray":
            return self.move_wellplate_back_from_cytation(plate_type=expected_type, destination=destination)
        if destination == "cytation_tray":
            if source != "pipetting_area":
                raise ValueError("Load Cytation from the configured pipetting area")
            return self.move_wellplate_to_cytation(plate_type=expected_type)
        self.nr_robot.move_home()
        return self.nr_track.move_wellplate(source, destination, expected_type)

    def grab_new_wellplate(self):
        self.nr_robot.move_home()
        return self.nr_track.get_new_wellplate(move_home_afterwards=True)
    
    def discard_used_wellplate(self):
        self.nr_robot.move_home()
        return self.nr_track.discard_wellplate(move_home_afterwards=True)
