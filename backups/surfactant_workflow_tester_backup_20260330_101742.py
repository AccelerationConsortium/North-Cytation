# -*- coding: utf-8 -*-
"""
Surfactant Workflow Flow Path Tester GUI
Test specific workflow flow paths with individual operation buttons.

This GUI provides step-by-step workflow testing with buttons for:
- Source operations (get materials from vials)
- Transfer operations (move wellplates between positions)
- Liquid handling (aspirate, dispense, mixing)
- Analysis operations (Cytation measurements)
- Disposal operations (waste handling, cleanup)

Each button performs a single workflow step, allowing custom sequences.
Example flow: Source -> Pipetting Area -> Mix -> Cytation -> Waste
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import traceback
from datetime import datetime

# Add parent directory to path to find master_usdl_coordinator
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from master_usdl_coordinator import Lash_E

class WorkflowTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Surfactant Workflow Flow Path Tester")
        self.root.geometry("900x800")
        
        # Initialize variables
        self.lash_e = None
        self.test_running = False
        self.stop_requested = False
        
        # Workflow state tracking
        self.current_wellplate_pos = "unknown"
        self.current_vial = None
        self.current_volume = 0.0
        self.pipet_attached = False
        
        # Statistics
        self.operation_count = 0
        self.success_count = 0
        self.error_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the GUI interface."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Surfactant Workflow Flow Path Tester", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Connection Section
        conn_frame = ttk.LabelFrame(main_frame, text="System Connection", padding="10")
        conn_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(conn_frame, text="Initialize System (Simulate)", 
                  command=self.init_system_simulate).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(conn_frame, text="Initialize System (Real Hardware)", 
                  command=self.init_system_real).grid(row=0, column=1, padx=(0, 10))
        
        self.status_label = ttk.Label(conn_frame, text="Status: Not Connected", 
                                     foreground="red")
        self.status_label.grid(row=0, column=2, padx=(20, 0))
        
        # Flow Path Operations - organized in columns
        self.create_source_operations(main_frame, row=2, col=0)
        self.create_transfer_operations(main_frame, row=2, col=1)
        self.create_liquid_operations(main_frame, row=2, col=2)
        self.create_analysis_operations(main_frame, row=2, col=3)
        self.create_disposal_operations(main_frame, row=3, col=0)
        
        # Workflow Status Panel
        status_frame = ttk.LabelFrame(main_frame, text="Current Workflow State", padding="10")
        status_frame.grid(row=3, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.wellplate_status = ttk.Label(status_frame, text="Wellplate: Unknown Position", foreground="orange")
        self.wellplate_status.grid(row=0, column=0, sticky=tk.W)
        
        self.vial_status = ttk.Label(status_frame, text="Current Vial: None", foreground="gray")
        self.vial_status.grid(row=0, column=1, padx=(20, 0), sticky=tk.W)
        
        self.pipet_status = ttk.Label(status_frame, text="Pipet: Not Attached", foreground="gray")
        self.pipet_status.grid(row=0, column=2, padx=(20, 0), sticky=tk.W)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Operation Control", padding="10")
        control_frame.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(control_frame, text="Reset Workflow State", 
                  command=self.reset_workflow_state).grid(row=0, column=0, padx=(0, 20))
        ttk.Button(control_frame, text="Emergency Stop", 
                  command=self.emergency_stop, style='Accent.TButton').grid(row=0, column=1, padx=(0, 20))
        ttk.Button(control_frame, text="Clear Log", 
                  command=self.clear_log).grid(row=0, column=2)
        
        # Results Section
        result_frame = ttk.LabelFrame(main_frame, text="Operation Results", padding="10")
        result_frame.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Statistics
        stat_frame = ttk.Frame(result_frame)
        stat_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.operation_count_var = tk.StringVar(value="Operations: 0")
        self.success_count_var = tk.StringVar(value="Success: 0") 
        self.error_count_var = tk.StringVar(value="Errors: 0")
        
        ttk.Label(stat_frame, textvariable=self.operation_count_var).grid(row=0, column=0, padx=(0, 20))
        ttk.Label(stat_frame, textvariable=self.success_count_var, 
                 foreground="green").grid(row=0, column=1, padx=(0, 20))
        ttk.Label(stat_frame, textvariable=self.error_count_var, 
                 foreground="red").grid(row=0, column=2)
        
        # Log output
        ttk.Label(result_frame, text="Operation Log:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        self.log_text = scrolledtext.ScrolledText(result_frame, height=15, width=100)
        self.log_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(2, weight=1)
        
    def create_source_operations(self, parent, row, col):
        """Create source operation buttons."""
        frame = ttk.LabelFrame(parent, text="Source Operations", padding="10")
        frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(0, 10))
        
        ttk.Button(frame, text="Get Water Vial", 
                  command=lambda: self.run_operation(self.get_water_vial)).grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Get Surfactant A", 
                  command=lambda: self.run_operation(self.get_surfactant_a)).grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Get Surfactant B", 
                  command=lambda: self.run_operation(self.get_surfactant_b)).grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Get DMSO Vial", 
                  command=lambda: self.run_operation(self.get_dmso_vial)).grid(row=3, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Return Vial Home", 
                  command=lambda: self.run_operation(self.return_vial_home)).grid(row=4, column=0, sticky=(tk.W, tk.E))

    def create_transfer_operations(self, parent, row, col):
        """Create transfer operation buttons."""
        frame = ttk.LabelFrame(parent, text="Transfer Operations", padding="10")
        frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(0, 10))
        
        ttk.Button(frame, text="Move to Pipetting Area", 
                  command=lambda: self.run_operation(self.move_to_pipetting)).grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Move to Cytation", 
                  command=lambda: self.run_operation(self.move_to_cytation)).grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))  
        ttk.Button(frame, text="Return to Origin", 
                  command=lambda: self.run_operation(self.return_to_origin)).grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Check Position", 
                  command=lambda: self.run_operation(self.check_position)).grid(row=3, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Home Robot", 
                  command=lambda: self.run_operation(self.home_robot)).grid(row=4, column=0, sticky=(tk.W, tk.E))

    def create_liquid_operations(self, parent, row, col):
        """Create liquid handling operation buttons."""
        frame = ttk.LabelFrame(parent, text="Liquid Operations", padding="10")
        frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(0, 10))
        
        ttk.Button(frame, text="Attach Pipet", 
                  command=lambda: self.run_operation(self.attach_pipet)).grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Aspirate 10uL", 
                  command=lambda: self.run_operation(self.aspirate_small)).grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Aspirate 50uL", 
                  command=lambda: self.run_operation(self.aspirate_medium)).grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Dispense to Well", 
                  command=lambda: self.run_operation(self.dispense_to_well)).grid(row=3, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Vortex Current", 
                  command=lambda: self.run_operation(self.vortex_current)).grid(row=4, column=0, sticky=(tk.W, tk.E))
                  
    def create_analysis_operations(self, parent, row, col):
        """Create analysis operation buttons."""
        frame = ttk.LabelFrame(parent, text="Analysis Operations", padding="10")
        frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(0, 10))
        
        ttk.Button(frame, text="Carrier Out", 
                  command=lambda: self.run_operation(self.carrier_out)).grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Carrier In", 
                  command=lambda: self.run_operation(self.carrier_in)).grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Quick Read", 
                  command=lambda: self.run_operation(self.quick_read)).grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Shake Protocol", 
                  command=lambda: self.run_operation(self.shake_protocol)).grid(row=3, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Full Read Cycle", 
                  command=lambda: self.run_operation(self.full_read_cycle)).grid(row=4, column=0, sticky=(tk.W, tk.E))

    def create_disposal_operations(self, parent, row, col):
        """Create disposal operation buttons."""
        frame = ttk.LabelFrame(parent, text="Disposal & Cleanup", padding="10")
        frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N), padx=(0, 5), pady=(10, 0))
        
        ttk.Button(frame, text="Dispense to Waste", 
                  command=lambda: self.run_operation(self.dispense_to_waste)).grid(row=0, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Remove Pipet", 
                  command=lambda: self.run_operation(self.remove_pipet)).grid(row=1, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Clean Workspace", 
                  command=lambda: self.run_operation(self.clean_workspace)).grid(row=2, column=0, pady=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(frame, text="Reset All States", 
                  command=lambda: self.run_operation(self.reset_all_states)).grid(row=3, column=0, sticky=(tk.W, tk.E))
    
    def init_system_simulate(self):
        """Initialize system in simulation mode."""
        try:
            self.log("Initializing system in simulation mode...")
            vial_config_path = os.path.join(parent_dir, "status", "surfactant_grid_vials_expanded.csv")
            self.lash_e = Lash_E(vial_config_path, simulate=True)
            self.status_label.config(text="Status: Connected (Simulation)", foreground="blue")
            self.log("System initialized successfully in simulation mode")
            self.reset_counters()
            self.update_workflow_status()
        except Exception as e:
            self.log(f"ERROR: Failed to initialize system: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{str(e)}")
    
    def init_system_real(self):
        """Initialize system with real hardware."""
        try:
            self.log("Initializing system with real hardware...")
            vial_config_path = os.path.join(parent_dir, "status", "surfactant_grid_vials_expanded.csv")
            self.lash_e = Lash_E(vial_config_path, simulate=False)
            self.status_label.config(text="Status: Connected (Real Hardware)", foreground="green")
            self.log("System initialized successfully with real hardware")
            self.reset_counters()
            self.update_workflow_status()
        except Exception as e:
            self.log(f"ERROR: Failed to initialize system: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{str(e)}")
            
    def reset_counters(self):
        """Reset operation counters."""
        self.operation_count = 0
        self.success_count = 0
        self.error_count = 0
        self.update_status_display()
        
    def reset_workflow_state(self):
        """Reset workflow state tracking.""" 
        self.current_wellplate_pos = "unknown"
        self.current_vial = None
        self.current_volume = 0.0
        self.pipet_attached = False
        self.update_workflow_status()
        self.log("Workflow state reset")
        
    def update_workflow_status(self):
        """Update the workflow status display."""
        self.wellplate_status.config(text=f"Wellplate: {self.current_wellplate_pos}")
        self.vial_status.config(text=f"Current Vial: {self.current_vial or 'None'}")
        pipet_text = "Attached" if self.pipet_attached else "Not Attached"
        self.pipet_status.config(text=f"Pipet: {pipet_text}")
        
    def update_status_display(self):
        """Update the status display counters."""
        self.operation_count_var.set(f"Operations: {self.operation_count}")
        self.success_count_var.set(f"Success: {self.success_count}")
        self.error_count_var.set(f"Errors: {self.error_count}")
    
    def log(self, message):
        """Add message to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared")
        
    def emergency_stop(self):
        """Emergency stop - stop current operation."""
        self.stop_requested = True
        self.log("EMERGENCY STOP requested by user")
        messagebox.showwarning("Emergency Stop", "Emergency stop activated!")
        
    def check_system(self):
        """Check if system is initialized.""" 
        if self.lash_e is None:
            self.log("ERROR: System not initialized - please initialize first")
            messagebox.showerror("System Not Ready", "Please initialize the system first")
            return False
        return True
        
    def run_operation(self, operation_function):
        """Run a single workflow operation with error handling."""
        if not self.check_system():
            return
            
        if self.test_running:
            messagebox.showwarning("Operation In Progress", "Another operation is already running")
            return
        
        # Start operation in separate thread
        thread = threading.Thread(target=self._run_operation_thread, args=(operation_function,))
        thread.daemon = True
        thread.start()
        
    def _run_operation_thread(self, operation_function):
        """Run operation in separate thread to avoid blocking UI."""
        try:
            self.test_running = True
            self.operation_count += 1
            
            operation_name = operation_function.__name__.replace('_', ' ').title()
            self.log(f"Starting operation: {operation_name}")
            
            operation_function()
            
            self.success_count += 1
            self.log(f"Operation completed successfully: {operation_name}")
            
        except Exception as e:
            self.error_count += 1
            error_msg = str(e)
            self.log(f"Operation failed: {error_msg}")
            # Show detailed error in console for debugging
            tb_lines = traceback.format_exc().splitlines()
            for line in tb_lines[-3:]:  # Show last 3 lines of traceback
                self.log(f"  {line}")
                
        finally:
            self.test_running = False
            self.update_status_display()
            self.update_workflow_status()
            self.root.update_idletasks()
    
    # ========================================
    # SOURCE OPERATIONS
    # ========================================
    
    def get_water_vial(self):
        """Get water vial from source."""
        self.log("Getting water vial from source...")
        # Check if vial exists in system first
        try:
            self.lash_e.nr_robot.move_vial_to_location("water", 'clamp', 0)
            self.current_vial = "water"
            self.log("Water vial positioned at clamp 0")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Water vial operation completed (simulation mode)")
                self.current_vial = "water"
            else:
                raise
        
    def get_surfactant_a(self):
        """Get surfactant A vial from source."""
        self.log("Getting SDS vial (surfactant A) from source...")
        try:
            # Use actual vial name from the system
            self.lash_e.nr_robot.move_vial_to_location("SDS_stock", 'clamp', 0)
            self.current_vial = "SDS_stock"
            self.log("SDS stock vial positioned at clamp 0")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("SDS vial operation completed (simulation mode)")
                self.current_vial = "SDS_stock"
            else:
                raise
        
    def get_surfactant_b(self):
        """Get surfactant B vial from source."""
        self.log("Getting TTAB vial (surfactant B) from source...")
        try:
            # Use actual vial name from the system
            self.lash_e.nr_robot.move_vial_to_location("TTAB_stock", 'clamp', 0)
            self.current_vial = "TTAB_stock"
            self.log("TTAB stock vial positioned at clamp 0")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("TTAB vial operation completed (simulation mode)")
                self.current_vial = "TTAB_stock"
            else:
                raise
        
    def get_dmso_vial(self):
        """Get DMSO vial from source."""
        self.log("Getting pyrene_DMSO vial from source...")
        try:
            # Use actual vial name from the system
            self.lash_e.nr_robot.move_vial_to_location("pyrene_DMSO", 'clamp', 0)
            self.current_vial = "pyrene_DMSO"
            self.log("Pyrene_DMSO vial positioned at clamp 0")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Pyrene_DMSO vial operation completed (simulation mode)")
                self.current_vial = "pyrene_DMSO"
            else:
                raise
        
    def return_vial_home(self):
        """Return current vial to home position."""
        if self.current_vial:
            self.log(f"Returning {self.current_vial} to home position...")
            self.lash_e.nr_robot.return_vial_home(self.current_vial)
            self.current_vial = None
            time.sleep(1.0)
        else:
            self.log("No current vial to return home")
    
    # ========================================
    # TRANSFER OPERATIONS
    # ========================================
    
    def move_to_pipetting(self):
        """Move wellplate to pipetting area using smart positioning."""
        self.log("Moving wellplate to pipetting area...")
        
        # Smart positioning - only move if not already there
        self.lash_e.nr_track.get_track_status()
        current_pos = self.lash_e.nr_track.ACTIVE_WELLPLATE_POSITION
        
        if current_pos != 'pipetting_area':
            self.log(f"Current position: {current_pos}, moving to pipetting area")
            self.lash_e.nr_track.set_wellplate_position('pipetting_area')
            time.sleep(1.5)
        else:
            self.log("Already at pipetting area - no movement needed")
            
        self.current_wellplate_pos = 'pipetting_area'
        
    def move_to_cytation(self):
        """Move wellplate to cytation using smart positioning."""
        self.log("Moving wellplate to cytation...")
        
        # Smart positioning - only move if not already there
        self.lash_e.nr_track.get_track_status()
        current_pos = self.lash_e.nr_track.ACTIVE_WELLPLATE_POSITION
        
        if current_pos != 'cytation':
            self.log(f"Current position: {current_pos}, moving to cytation")
            self.lash_e.nr_track.set_wellplate_position('cytation')
            time.sleep(1.5)
        else:
            self.log("Already at cytation - no movement needed")
            
        self.current_wellplate_pos = 'cytation'
        
    def return_to_origin(self):
        """Return wellplate to origin position."""
        self.log("Returning wellplate to origin...")
        self.lash_e.nr_track.origin()
        self.current_wellplate_pos = 'origin'
        time.sleep(1.0)
        
    def check_position(self):
        """Check current positions of all components."""
        self.log("Checking current positions...")
        self.lash_e.nr_track.get_track_status()
        current_pos = self.lash_e.nr_track.ACTIVE_WELLPLATE_POSITION
        self.current_wellplate_pos = current_pos
        self.log(f"Current wellplate position: {current_pos}")
        
        # Also log robot position if available
        try:
            robot_positions = self.lash_e.nr_robot.get_robot_positions()
            self.log(f"Robot positions: {robot_positions}")
        except:
            self.log("Could not read robot positions")
        
    def home_robot(self):
        """Home the robot to safe position with proper error handling."""
        self.log("Homing robot to safe position...")
        try:
            # First remove any pipet for safety
            if self.pipet_attached:
                self.log("Removing pipet before homing...")
                self.lash_e.nr_robot.remove_pipet()
                self.pipet_attached = False
                self.current_volume = 0.0
            
            # Home the robot
            self.lash_e.nr_robot.home_robot()
            self.log("Robot homed successfully")
            time.sleep(2.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Robot homing completed (simulation mode)")
            else:
                raise
    
    # ========================================
    # LIQUID OPERATIONS
    # ========================================
    
    def attach_pipet(self):
        """Attach pipet tip (conditioning)."""
        if not self.current_vial:
            raise ValueError("No vial selected - get a vial first")
            
        self.log(f"Conditioning pipet with {self.current_vial}...")
        
        # Determine liquid type for conditioning
        liquid_type = "water" if self.current_vial == "water" else "DMSO"
        conditioning_volume_ul = 150  # 150 uL for small tip conditioning
        
        try:
            # Condition tip (this is the "attach" equivalent in the surfactant workflow)
            volume_ml = conditioning_volume_ul / 1000
            self.lash_e.nr_robot.aspirate_from_vial(self.current_vial, volume_ml, liquid=liquid_type)
            # Dispense back to same vial to condition
            self.lash_e.nr_robot.dispense_into_vial(self.current_vial, volume_ml, liquid=liquid_type)
            self.pipet_attached = True
            self.log(f"Pipet conditioned with {conditioning_volume_ul} uL of {liquid_type}")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Pipet conditioning completed (simulation mode)")
                self.pipet_attached = True
            else:
                raise
        
    def aspirate_small(self):
        """Aspirate 10 microliters from current vial."""
        if not self.current_vial:
            raise ValueError("No vial selected - get a vial first")
        
        if not self.pipet_attached:
            self.log("No pipet attached - conditioning first...")
            self.attach_pipet()
        
        volume = 0.01  # 10 uL
        liquid_type = "water" if self.current_vial == "water" else "DMSO"
        self.log(f"Aspirating {volume*1000} uL from {self.current_vial} (liquid: {liquid_type})...")
        
        try:
            self.lash_e.nr_robot.aspirate_from_vial(self.current_vial, volume, liquid=liquid_type)
            self.current_volume += volume
            self.log(f"Aspiration successful - current volume: {self.current_volume*1000:.1f} uL")
            time.sleep(0.5)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Aspiration completed (simulation mode)")
                self.current_volume += volume
            else:
                raise
                
    def aspirate_medium(self):
        """Aspirate 50 microliters from current vial."""
        if not self.current_vial:
            raise ValueError("No vial selected - get a vial first")
            
        if not self.pipet_attached:
            self.log("No pipet attached - conditioning first...")
            self.attach_pipet()
        
        volume = 0.05  # 50 uL
        liquid_type = "water" if self.current_vial == "water" else "DMSO"
        self.log(f"Aspirating {volume*1000} uL from {self.current_vial} (liquid: {liquid_type})...")
        
        try:
            self.lash_e.nr_robot.aspirate_from_vial(self.current_vial, volume, liquid=liquid_type)
            self.current_volume += volume
            self.log(f"Aspiration successful - current volume: {self.current_volume*1000:.1f} uL")
            time.sleep(0.5)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Aspiration completed (simulation mode)")
                self.current_volume += volume
            else:
                raise
                
    def dispense_to_well(self):
        """Dispense current volume to wellplate well."""
        if self.current_volume <= 0:
            raise ValueError("No liquid to dispense - aspirate first")
            
        # Ensure wellplate is at pipetting area
        if self.current_wellplate_pos != 'pipetting_area':
            self.log("Moving to pipetting area first...")
            self.move_to_pipetting()
            
        well_index = 0  # A1 = well index 0
        liquid_type = "water" if self.current_vial == "water" else "DMSO"
        
        self.log(f"Dispensing {self.current_volume*1000:.1f} uL to well A1 (index {well_index})...")
        
        try:
            self.lash_e.nr_robot.dispense_into_wellplate(
                dest_wp_num_array=[well_index],
                amount_mL_array=[self.current_volume],
                liquid=liquid_type
            )
            self.log(f"Successfully dispensed {self.current_volume*1000:.1f} uL to well A1")
            self.current_volume = 0.0
            time.sleep(0.5)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Dispensing completed (simulation mode)")
                self.current_volume = 0.0
            else:
                raise
                
    def vortex_current(self):
        """Vortex the current vial using proper pattern."""
        if not self.current_vial:
            raise ValueError("No vial selected - get a vial first")
            
        self.log(f"Vortexing {self.current_vial} (3 seconds at 50% speed)...")
        
        try:
            # Standard vortex parameters from surfactant workflow
            vortex_time = 3
            vortex_speed = 50
            self.lash_e.nr_robot.vortex_vial(self.current_vial, vortex_time=vortex_time, vortex_speed=vortex_speed)
            self.log(f"Vortex completed: {vortex_time}s at {vortex_speed}% speed")
            time.sleep(3.5)  # Wait for vortex to complete plus settling time
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Vortex operation completed (simulation mode)")
                time.sleep(3.5)
            else:
                raise
    
    # ========================================
    # ANALYSIS OPERATIONS
    # ========================================
    
    def carrier_out(self):
        """Move Cytation carrier out."""
        self.log("Moving Cytation carrier out...")
        try:
            self.lash_e.cytation.CarrierOut()
            self.log("Carrier successfully moved out")
            time.sleep(2.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Carrier out completed (simulation mode)")
            else:
                raise
        
    def carrier_in(self):
        """Move Cytation carrier in."""
        self.log("Moving Cytation carrier in...")
        try:
            self.lash_e.cytation.CarrierIn(plate_type="96 WELL PLATE", use_lid=False)
            self.log("Carrier successfully moved in")
            time.sleep(2.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Carrier in completed (simulation mode)")
            else:
                raise
        
    def quick_read(self):
        """Perform quick turbidity reading using atomic operations pattern."""
        self.log("Performing quick turbidity reading...")
        
        # Smart positioning - only move if needed
        if self.current_wellplate_pos != 'cytation':
            self.log("Smart positioning: moving to cytation")
            self.move_to_cytation()
        else:
            self.log("Smart positioning: already at cytation")
        
        # Test wells A1-A3 (indices 0-2)
        test_wells = [0, 1, 2]
        
        try:
            if not self.lash_e.simulate:
                # Use actual protocol file if available
                protocol_file = "protocols/turbidity_600nm.LHC"  # Example protocol
                self.log(f"Running turbidity protocol: {protocol_file}")
                turbidity_data = self.lash_e.cytation.run_protocol(protocol_file, test_wells)
                self.log(f"Quick read completed - measured {len(test_wells)} wells")
            else:
                self.log("Running quick measurement protocol (simulated)...")
                time.sleep(2.0)
                self.log("Quick reading completed (simulation mode)")
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Quick reading completed (simulation mode)")
            else:
                raise
        
    def shake_protocol(self):
        """Run shake protocol on Cytation."""
        self.log("Running shake protocol...")
        
        # Ensure plate is at cytation
        if self.current_wellplate_pos != 'cytation':
            self.move_to_cytation()
        
        try:
            if not self.lash_e.simulate:
                # Use actual shake protocol file if available
                shake_protocol = "protocols/shake_wait.LHC"  # Example protocol
                self.log(f"Running shake protocol: {shake_protocol}")
                self.lash_e.cytation.run_protocol(shake_protocol, [0])
            else:
                self.log("Shake protocol (simulated)")
                time.sleep(3.0)
            self.log("Shake protocol completed")
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Shake protocol completed (simulation mode)")
            else:
                raise
        
    def full_read_cycle(self):
        """Perform full read cycle with carrier operations (atomic pattern)."""
        self.log("Starting full read cycle with atomic operations...")
        
        # Step 1: Smart positioning
        if self.current_wellplate_pos != 'cytation':
            self.move_to_cytation()
            
        # Step 2: Carrier operations and measurement
        try:
            # Carrier out
            self.log("Step 2a: Carrier out")
            self.carrier_out()
            
            # Carrier in  
            self.log("Step 2b: Carrier in")
            self.carrier_in()
            
            # Measurement
            self.log("Step 2c: Full measurement")
            test_wells = [0, 1, 2, 3, 4]  # Test more wells
            if not self.lash_e.simulate:
                protocol_file = "protocols/turbidity_600nm.LHC"
                self.lash_e.cytation.run_protocol(protocol_file, test_wells)
            else:
                time.sleep(2.0)
                
            # Carrier out again for plate access
            self.log("Step 2d: Final carrier out")
            self.carrier_out()
            
            self.log("Full read cycle completed successfully")
            
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Full read cycle completed (simulation mode)")
            else:
                # Ensure carrier is out on error for safety
                try:
                    self.lash_e.cytation.CarrierOut()
                except:
                    pass
                raise
    
    # ========================================
    # DISPOSAL & CLEANUP OPERATIONS
    # ========================================
    
    def dispense_to_waste(self):
        """Dispense current volume to waste using proper waste handling pattern."""
        if self.current_volume <= 0:
            self.log("No liquid to dispose")
            return
            
        liquid_type = "water" if self.current_vial == "water" else "DMSO"
        self.log(f"Dispensing {self.current_volume*1000:.1f} uL to waste (liquid: {liquid_type})...")
        
        try:
            # Use waste vial - common pattern in the system
            waste_vial = "waste"
            self.lash_e.nr_robot.dispense_into_vial(waste_vial, self.current_volume, liquid=liquid_type)
            self.log(f"Successfully dispensed {self.current_volume*1000:.1f} uL to waste")
            self.current_volume = 0.0
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log(f"Waste disposal completed (simulation mode)")
                self.current_volume = 0.0
            else:
                # Fallback - just clear the volume if waste disposal fails
                self.log(f"Waste disposal failed: {e}, clearing volume anyway")
                self.current_volume = 0.0
        
    def remove_pipet(self):
        """Remove current pipet tip."""
        self.log("Removing pipet tip...")
        try:
            self.lash_e.nr_robot.remove_pipet()
            self.pipet_attached = False
            self.current_volume = 0.0
            self.log("Pipet tip removed successfully")
            time.sleep(1.0)
        except Exception as e:
            if "simulation" in str(e).lower():
                self.log("Pipet removal completed (simulation mode)")
                self.pipet_attached = False
                self.current_volume = 0.0
            else:
                raise
        
    def clean_workspace(self):
        """Clean the workspace using proper cleanup sequence."""
        self.log("Cleaning workspace...")
        
        try:
            # Step 1: Remove pipet if attached
            if self.pipet_attached:
                self.log("Step 1: Removing pipet")
                self.remove_pipet()
            
            # Step 2: Return vial to home if any vial is active
            if self.current_vial:
                self.log("Step 2: Returning vial home")
                self.return_vial_home()
            
            # Step 3: Ensure carrier is out for safety
            try:
                self.log("Step 3: Safety check - ensuring carrier is out")
                self.lash_e.cytation.CarrierOut()
            except Exception as e:
                if not "simulation" in str(e).lower():
                    self.log(f"Carrier check warning: {e}")
            
            # Step 4: Return wellplate to safe position
            if self.current_wellplate_pos != 'pipetting_area':
                self.log("Step 4: Returning wellplate to pipetting area")
                self.move_to_pipetting()
                
            time.sleep(1.0)
            self.log("Workspace cleaned successfully")
            
        except Exception as e:
            self.log(f"Warning during workspace cleanup: {str(e)}")
            self.log("Continuing with state reset...")
        
    def reset_all_states(self):
        """Reset all workflow states to initial conditions with comprehensive error handling."""
        self.log("Resetting all workflow states...")
        
        try:
            # Step 1: Clean up current operations
            self.log("Step 1: Workspace cleanup")
            self.clean_workspace()
            
            # Step 2: Home robot for safety
            self.log("Step 2: Homing robot")
            try:
                self.lash_e.nr_robot.home_robot()
            except Exception as e:
                if not "simulation" in str(e).lower():
                    self.log(f"Robot homing warning: {e}")
            
            # Step 3: Reset state variables
            self.log("Step 3: Resetting state variables")
            self.reset_workflow_state()
            
            # Step 4: Reset counters
            self.log("Step 4: Resetting counters")
            self.reset_counters()
            
            self.log("All states reset to initial conditions successfully")
            
        except Exception as e:
            self.log(f"Error during comprehensive state reset: {str(e)}")
            # Always reset state variables even if operations fail
            self.reset_workflow_state()
            self.log("State variables reset despite errors")


def main():
    root = tk.Tk()
    app = WorkflowTester(root)
    root.mainloop()

if __name__ == "__main__":
    main()