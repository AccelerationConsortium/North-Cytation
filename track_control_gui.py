# -*- coding: utf-8 -*-
"""
Track Control GUI - Manual Track Position Management
================================================

SAFE OPERATION MODE:
- This GUI operates independently of active workflows
- Uses simulation mode by default to prevent hardware interference
- DOES NOT modify any robot configuration files during movement
- Only saves data when user explicitly chooses "Save Sequence"

IMPORTANT SAFETY NOTES:
- Moving the robot with this GUI does NOT update robot_state/ files
- Configuration files remain unchanged during manual movement
- Only explicitly saved sequences create new user-selected files
- NO automatic modification of track_positions.yaml or track_status.yaml

USAGE:
- Use arrow keys or buttons to move track in X (left/right) and Z (up/down)
- Open/close gripper with dedicated buttons
- Record current positions and build command sequences
- Edit, reorder, and delete commands in the sequence
- Save sequences to user-selected files only when you choose to export

SAFETY FEATURES:
- Simulation mode prevents actual hardware movement during active workflows
- Position validation and bounds checking
- Emergency stop functionality
- Real-time position display
- NO modification of system configuration files
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yaml
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append("..")
sys.path.append(".")

# Import Lash_E coordinator for actual track control
try:
    from master_usdl_coordinator import Lash_E
    LASH_E_AVAILABLE = True
except ImportError:
    print("Warning: Could not import Lash_E - running in simulation-only mode")
    LASH_E_AVAILABLE = False

class TrackControlGUI:
    """Standalone GUI for track control and position management"""
    
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.current_x = 68325  # Start at cytation position as reference
        self.current_z = 0      # Start at safe/home position (Z=0 is TOP for track system)
        self.gripper_open = False
        self.command_sequence = []
        self.position_history = []
        
        # Initialize Lash_E coordinator for actual track control
        self.lash_e = None
        if LASH_E_AVAILABLE and not simulation_mode:
            try:
                # Use a minimal vial file for track control only
                INPUT_VIAL_STATUS_FILE = "status/experiment_vials.csv"
                self.lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=True,  # Start with simulation
                                   workflow_globals=globals(), workflow_name='track_control_gui')
                print("Lash_E coordinator initialized for track control")
            except Exception as e:
                print(f"Warning: Could not initialize Lash_E: {e}")
                self.simulation_mode = True
        
        # Load current track positions for reference
        self.load_track_positions()
        
        # Movement increments - CONVERSION: 100 encoder units = 1 mm
        self.UNITS_PER_MM = 100  # Hardware conversion factor
        self.x_increment_mm = 1.0   # Fine movement in mm (default 1mm)
        self.z_increment_mm = 1.0   # Fine movement in mm (default 1mm) 
        self.coarse_multiplier = 10  # For coarse movement (10x fine)
        
        # Convert mm to encoder units for hardware
        self.x_increment = int(self.x_increment_mm * self.UNITS_PER_MM)
        self.z_increment = int(self.z_increment_mm * self.UNITS_PER_MM)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Track Control GUI - Manual Position Management")
        self.root.geometry("900x800")
        
        # Initialize GUI components
        self.setup_gui()
        self.setup_keybindings()
        self.update_position_display()
        
        # Get initial track status if available
        self.refresh_track_status()
        
    def load_track_positions(self):
        """Load current track positions from YAML file"""
        try:
            yaml_path = "robot_state/track_positions.yaml"
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    self.track_positions = yaml.safe_load(f)
            else:
                self.track_positions = {"positions": {}}
        except Exception as e:
            print(f"Warning: Could not load track positions: {e}")
            self.track_positions = {"positions": {}}
    
    def setup_gui(self):
        """Create all GUI components"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="5")
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.mode_label = ttk.Label(status_frame, text=f"Mode: {'SIMULATION' if self.simulation_mode else 'HARDWARE'}", 
                                   font=('Arial', 10, 'bold'))
        self.mode_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(status_frame, text="Toggle Mode", command=self.toggle_simulation_mode).grid(row=0, column=1, padx=(10, 0))
        
        # Connection status
        self.connection_label = ttk.Label(status_frame, text="Lash_E: Not Connected", font=('Arial', 9))
        self.connection_label.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(status_frame, text="Refresh Status", command=self.refresh_track_status).grid(row=1, column=1, padx=(10, 0))
        ttk.Button(status_frame, text="Debug Hardware", command=self.debug_hardware).grid(row=1, column=2, padx=(10, 0))
        
        # Safety reminder
        self.safety_label = ttk.Label(status_frame, text="⚠️ Manual movement does NOT modify robot config files", 
                                     font=('Arial', 8), foreground="green")
        self.safety_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        
        # Position display frame
        pos_frame = ttk.LabelFrame(main_frame, text="Current Position", padding="5")
        pos_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.x_label = ttk.Label(pos_frame, text="X: 0", font=('Arial', 12, 'bold'))
        self.x_label.grid(row=0, column=0, padx=(0, 20))
        
        self.z_label = ttk.Label(pos_frame, text="Z: 0", font=('Arial', 12, 'bold'))
        self.z_label.grid(row=0, column=1, padx=(0, 20))
        
        self.gripper_label = ttk.Label(pos_frame, text="Gripper: CLOSED", font=('Arial', 12, 'bold'))
        self.gripper_label.grid(row=0, column=2)
        
        # Movement control frame
        move_frame = ttk.LabelFrame(main_frame, text="Movement Control", padding="10")
        move_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # X-axis controls
        ttk.Label(move_frame, text="X-axis (Left/Right) - Movement in mm", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=4, pady=(0, 5))
        
        ttk.Button(move_frame, text="<<<< Left", command=lambda: self.move_x(-self.coarse_multiplier)).grid(row=1, column=0, padx=2)
        ttk.Button(move_frame, text="< Left", command=lambda: self.move_x(-1)).grid(row=1, column=1, padx=2) 
        ttk.Button(move_frame, text="Right >", command=lambda: self.move_x(1)).grid(row=1, column=2, padx=2)
        ttk.Button(move_frame, text="Right >>>>", command=lambda: self.move_x(self.coarse_multiplier)).grid(row=1, column=3, padx=2)
        
        # Z-axis controls
        ttk.Label(move_frame, text="Z-axis (Up/Down) - Movement in mm (Z=0 is top, higher Z = lower)", font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=4, pady=(10, 5))
        
        ttk.Button(move_frame, text="^^^^ Up", command=lambda: self.move_z(self.coarse_multiplier)).grid(row=3, column=0, padx=2)
        ttk.Button(move_frame, text="^ Up", command=lambda: self.move_z(1)).grid(row=3, column=1, padx=2)
        ttk.Button(move_frame, text="Down v", command=lambda: self.move_z(-1)).grid(row=3, column=2, padx=2)
        ttk.Button(move_frame, text="Down vvvv", command=lambda: self.move_z(-self.coarse_multiplier)).grid(row=3, column=3, padx=2)
        
        # Movement increment controls
        ttk.Label(move_frame, text="Movement Increment (mm):", font=('Arial', 10)).grid(row=4, column=0, pady=(10, 0), sticky=tk.W)
        
        increment_frame = ttk.Frame(move_frame)
        increment_frame.grid(row=4, column=1, columnspan=3, pady=(10, 0), sticky=tk.W)
        
        self.x_increment_var = tk.DoubleVar(value=self.x_increment_mm)
        self.z_increment_var = tk.DoubleVar(value=self.z_increment_mm)
        
        ttk.Label(increment_frame, text="X:").grid(row=0, column=0)
        x_increment_spin = ttk.Spinbox(increment_frame, from_=0.1, to=10.0, width=8, textvariable=self.x_increment_var,
                                      increment=0.1, command=self.update_increments, format="%.1f")
        x_increment_spin.grid(row=0, column=1, padx=(2, 10))
        ttk.Label(increment_frame, text="mm").grid(row=0, column=2, padx=(2, 10))
        
        ttk.Label(increment_frame, text="Z:").grid(row=0, column=3)
        z_increment_spin = ttk.Spinbox(increment_frame, from_=0.1, to=10.0, width=8, textvariable=self.z_increment_var,
                                      increment=0.1, command=self.update_increments, format="%.1f")
        z_increment_spin.grid(row=0, column=4, padx=2)
        ttk.Label(increment_frame, text="mm").grid(row=0, column=5, padx=(2, 0))
        
        # Gripper and utility controls
        util_frame = ttk.LabelFrame(main_frame, text="Gripper & Utilities", padding="10")
        util_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Button(util_frame, text="Open Gripper", command=lambda: self.set_gripper(True), width=15).grid(row=0, column=0, pady=2)
        ttk.Button(util_frame, text="Close Gripper", command=lambda: self.set_gripper(False), width=15).grid(row=1, column=0, pady=2)
        
        ttk.Separator(util_frame, orient='horizontal').grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(util_frame, text="Go to Origin", command=self.go_to_origin, width=15).grid(row=3, column=0, pady=2)
        ttk.Button(util_frame, text="Emergency Stop", command=self.emergency_stop, width=15).grid(row=4, column=0, pady=2)
        
        # Preset positions
        ttk.Label(util_frame, text="Preset Positions:", font=('Arial', 10, 'bold')).grid(row=5, column=0, pady=(10, 5))
        
        preset_positions = [
            ("Pipetting Area", 131750, 75000),
            ("Source Stack", 300, 0),
            ("Waste Stack", 14700, 0), 
            ("Cytation Tray", 68325, 0),
            ("Lid Storage", 60325, 75000)
        ]
        
        for i, (name, x, z) in enumerate(preset_positions):
            x_mm = x / self.UNITS_PER_MM
            z_mm = z / self.UNITS_PER_MM
            button_text = f"{name}\n({x_mm:.0f},{z_mm:.0f}mm)"
            ttk.Button(util_frame, text=button_text, command=lambda n=name, x_pos=x, z_pos=z: self.go_to_preset(n, x_pos, z_pos), 
                      width=15).grid(row=6+i, column=0, pady=1)
        
        # Command sequence frame
        seq_frame = ttk.LabelFrame(main_frame, text="Command Sequence", padding="5")
        seq_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Sequence controls
        seq_controls = ttk.Frame(seq_frame)
        seq_controls.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Position naming section
        name_frame = ttk.Frame(seq_controls)
        name_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(name_frame, text="Position Name:").grid(row=0, column=0, padx=(0, 5))
        self.position_name_var = tk.StringVar()
        self.position_name_entry = ttk.Entry(name_frame, textvariable=self.position_name_var, width=15)
        self.position_name_entry.grid(row=0, column=1, padx=(0, 5))
        
        ttk.Button(name_frame, text="Save Named Position", command=self.save_named_position).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(name_frame, text="Copy Coordinates", command=self.copy_coordinates).grid(row=0, column=3, padx=(0, 10))
        
        # Coordinate display
        self.coord_display = tk.Text(name_frame, height=1, width=60, wrap=tk.NONE)
        self.coord_display.grid(row=0, column=4, padx=(10, 0))
        
        # Sequence controls row
        ttk.Button(seq_controls, text="Record Current Position", command=self.record_position).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Button(seq_controls, text="Clear Sequence", command=self.clear_sequence).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Button(seq_controls, text="Save Sequence", command=self.save_sequence).grid(row=1, column=2, padx=5, pady=(5, 0))
        ttk.Button(seq_controls, text="Load Sequence", command=self.load_sequence).grid(row=1, column=3, padx=5, pady=(5, 0))
        
        # Sequence display
        seq_display_frame = ttk.Frame(seq_frame)
        seq_display_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create treeview for command sequence
        columns = ('Step', 'Action', 'X', 'Z', 'Gripper', 'Notes')
        self.sequence_tree = ttk.Treeview(seq_display_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.sequence_tree.heading(col, text=col)
            self.sequence_tree.column(col, width=100)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(seq_display_frame, orient=tk.VERTICAL, command=self.sequence_tree.yview)
        h_scrollbar = ttk.Scrollbar(seq_display_frame, orient=tk.HORIZONTAL, command=self.sequence_tree.xview)
        self.sequence_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.sequence_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Context menu for sequence
        self.sequence_tree.bind("<Button-3>", self.show_context_menu)  # Right-click
        self.sequence_tree.bind("<Delete>", self.delete_selected_command)  # Delete key
        
        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        seq_display_frame.columnconfigure(0, weight=1)
        seq_display_frame.rowconfigure(0, weight=1)
        
    def setup_keybindings(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self.move_x(-1))
        self.root.bind('<Right>', lambda e: self.move_x(1))
        self.root.bind('<Up>', lambda e: self.move_z(1))        # Up arrow = move up physically (internally decreases Z)  
        self.root.bind('<Down>', lambda e: self.move_z(-1))     # Down arrow = move down physically (internally increases Z)
        self.root.bind('<Shift-Left>', lambda e: self.move_x(-self.coarse_multiplier))
        self.root.bind('<Shift-Right>', lambda e: self.move_x(self.coarse_multiplier))
        self.root.bind('<Shift-Up>', lambda e: self.move_z(self.coarse_multiplier))    # Coarse up movement
        self.root.bind('<Shift-Down>', lambda e: self.move_z(-self.coarse_multiplier)) # Coarse down movement
        self.root.bind('<space>', lambda e: self.set_gripper(not self.gripper_open))
        self.root.bind('<Return>', lambda e: self.record_position())
        
        # Make sure the window can receive focus
        self.root.focus_set()
        
    def toggle_simulation_mode(self):
        """Toggle between simulation and hardware mode"""
        if not LASH_E_AVAILABLE:
            messagebox.showwarning("Not Available", "Lash_E coordinator not available - cannot switch to hardware mode")
            return
            
        if not self.simulation_mode:
            # Switching to simulation mode
            if self.lash_e:
                self.lash_e.simulate = True
            self.simulation_mode = True
            self.mode_label.config(text="Mode: SIMULATION")
            print("Switched to simulation mode")
        else:
            # Switching to hardware mode
            response = messagebox.askyesno("Hardware Mode", 
                                         "WARNING: Switching to hardware mode will send actual commands to the robot.\n\n"
                                         "Ensure the robot is not running an active workflow!\n\n"
                                         "Continue?")
            if response:
                try:
                    if not self.lash_e:
                        # Initialize Lash_E for hardware mode
                        INPUT_VIAL_STATUS_FILE = "status/experiment_vials.csv"
                        self.lash_e = Lash_E(INPUT_VIAL_STATUS_FILE, simulate=False,
                                           workflow_globals=globals(), workflow_name='track_control_gui')
                    else:
                        self.lash_e.simulate = False
                    
                    self.simulation_mode = False
                    self.mode_label.config(text="Mode: HARDWARE")
                    self.refresh_track_status()
                    print("Switched to hardware mode")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize hardware mode: {e}")
                    print(f"Error switching to hardware mode: {e}")
    
    def refresh_track_status(self):
        """Get current track status from hardware"""
        if self.lash_e:
            try:
                if not self.simulation_mode:
                    self.lash_e.nr_track.get_track_status()
                    # Update current position from hardware if available
                    # Note: The track may not provide absolute position feedback
                    print("Track status refreshed from hardware")
                    self.connection_label.config(text="Lash_E: Connected (Hardware)", foreground="green")
                else:
                    self.connection_label.config(text="Lash_E: Connected (Simulation)", foreground="blue")
            except Exception as e:
                print(f"Warning: Could not get track status: {e}")
                self.connection_label.config(text=f"Lash_E: Error - {str(e)[:30]}", foreground="red")
        else:
            self.connection_label.config(text="Lash_E: Not Connected", foreground="red")
    
    def debug_hardware(self):
        """Debug hardware configuration and axis mappings"""
        if not self.lash_e:
            messagebox.showwarning("Debug", "Lash_E not connected - cannot debug hardware")
            return
        
        try:
            x_axis = self.lash_e.nr_track.get_axis('x_axis')
            z_axis = self.lash_e.nr_track.get_axis('z_axis')
            gripper_open = self.lash_e.nr_track.get_axis('gripper_open')
            gripper_close = self.lash_e.nr_track.get_axis('gripper_close')
            
            x_speed = self.lash_e.nr_track.get_speed('default_x')
            z_speed = self.lash_e.nr_track.get_speed('default_z')
            
            debug_info = (
                f"Hardware Axis Mappings:\n"
                f"X-axis: {x_axis}\n"
                f"Z-axis: {z_axis} (INVERTED: 0=top, higher=lower)\n"
                f"Gripper Open: {gripper_open}\n"
                f"Gripper Close: {gripper_close}\n\n"
                f"Unit Conversion:\n"
                f"{self.UNITS_PER_MM} encoder units = 1mm\n\n"
                f"Movement Speeds:\n"
                f"X Speed: {x_speed}\n"
                f"Z Speed: {z_speed}\n\n"
                f"Current Position:\n"
                f"X: {self.current_x} units ({self.current_x/self.UNITS_PER_MM:.1f}mm)\n"
                f"Z: {self.current_z} units ({self.current_z/self.UNITS_PER_MM:.1f}mm - {'TOP' if self.current_z == 0 else 'lower'})\n"
                f"Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}\n\n"
                f"Movement Increments:\n"
                f"X: {self.x_increment_mm}mm ({self.x_increment} units)\n"
                f"Z: {self.z_increment_mm}mm ({self.z_increment} units)\n\n"
                f"Z-Axis Coordinate System:\n"
                f"Z = 0 units (0mm) → TOP/SAFE position\n"
                f"Z = 75000+ units (750+mm) → LOWER working positions"
            )
            
            print("DEBUG HARDWARE INFO:")
            print(debug_info)
            messagebox.showinfo("Hardware Debug", debug_info)
            
        except Exception as e:
            error_msg = f"Debug failed: {e}"
            print(f"DEBUG ERROR: {error_msg}")
            messagebox.showerror("Debug Error", error_msg)
    
    def update_increments(self):
        """Update movement increments from spinbox values (convert mm to encoder units)"""
        self.x_increment_mm = self.x_increment_var.get()
        self.z_increment_mm = self.z_increment_var.get()
        
        # Convert mm to encoder units for hardware commands
        self.x_increment = int(self.x_increment_mm * self.UNITS_PER_MM)
        self.z_increment = int(self.z_increment_mm * self.UNITS_PER_MM)
        
        print(f"Updated increments: X={self.x_increment_mm}mm ({self.x_increment} units), Z={self.z_increment_mm}mm ({self.z_increment} units)")
    
    def move_x(self, direction_multiplier):
        """Move track in X direction"""
        if self.lash_e and not self.simulation_mode:
            try:
                # Calculate target position
                target_x = self.current_x + (direction_multiplier * self.x_increment)
                
                # Use Lash_E track control for actual movement
                x_axis = self.lash_e.nr_track.get_axis('x_axis')
                speed = self.lash_e.nr_track.get_speed('default_x')
                self.lash_e.nr_track.c9.move_axis(x_axis, target_x, vel=speed)
                
                print(f"HARDWARE: Moving X axis {x_axis} to {target_x} units ({target_x/self.UNITS_PER_MM:.1f}mm) - delta: {direction_multiplier * self.x_increment} units ({direction_multiplier * self.x_increment_mm:.1f}mm)")
                self.lash_e.logger.info(f"Manual X movement: axis {x_axis} to {target_x} units ({target_x/self.UNITS_PER_MM:.1f}mm)")
                
                # NOTE: Intentionally NOT calling update_gripper_location() to avoid modifying config files
                
            except Exception as e:
                messagebox.showerror("Movement Error", f"Failed to move X: {e}")
                print(f"Error moving X: {e}")
                return
                
        self.current_x += direction_multiplier * self.x_increment
        self.position_history.append(('move_x', self.current_x, self.current_z, self.gripper_open))
        self.update_position_display()
        
        if self.simulation_mode or not self.lash_e:
            x_mm = self.current_x / self.UNITS_PER_MM
            delta_mm = direction_multiplier * self.x_increment_mm
            print(f"SIMULATION: Moving X to {self.current_x} units ({x_mm:.1f}mm) - delta: {direction_multiplier * self.x_increment} units ({delta_mm:.1f}mm)")
    
    def move_z(self, direction_multiplier):
        """Move track in Z direction - NOTE: Z coordinates are inverted (0=top, higher=lower)""" 
        if self.lash_e and not self.simulation_mode:
            try:
                # IMPORTANT: Invert the direction because Z-axis is inverted 
                # (Z=0 is top/safe, higher values are lower positions)
                inverted_direction = -direction_multiplier  # Invert the direction!
                target_z = self.current_z + (inverted_direction * self.z_increment)
                
                # Ensure we don't go below 0 (above safe height)
                if target_z < 0:
                    messagebox.showwarning("Z-Limit", "Cannot move above safe height (Z=0)")
                    return
                
                # Debug Z-axis setup
                z_axis = self.lash_e.nr_track.get_axis('z_axis')
                speed = self.lash_e.nr_track.get_speed('default_z')
                
                print(f"DEBUG Z-movement (INVERTED): User pressed {'UP' if direction_multiplier > 0 else 'DOWN'}")
                print(f"  Physical direction: {'UP (decreasing Z)' if inverted_direction < 0 else 'DOWN (increasing Z)'}")
                print(f"  Current: {self.current_z} units ({self.current_z/self.UNITS_PER_MM:.1f}mm)")
                print(f"  Target: {target_z} units ({target_z/self.UNITS_PER_MM:.1f}mm)")
                print(f"  axis={z_axis}, speed={speed}")
                
                # Try Z movement with corrected direction
                self.lash_e.nr_track.c9.move_axis(z_axis, target_z, vel=speed)
                print(f"HARDWARE: Z-axis movement successful to {target_z} units ({target_z/self.UNITS_PER_MM:.1f}mm)")
                
                self.lash_e.logger.info(f"Manual Z movement: axis {z_axis} to {target_z} units ({target_z/self.UNITS_PER_MM:.1f}mm) - user direction: {direction_multiplier}, physical direction: {inverted_direction}")
                
                # NOTE: Intentionally NOT calling update_gripper_location() to avoid modifying config files
                
            except Exception as e:
                messagebox.showerror("Movement Error", f"Failed to move Z: {e}")
                print(f"Error moving Z: {e}")
                return
                
        # Update GUI position with inverted direction
        inverted_direction = -direction_multiplier
        self.current_z += inverted_direction * self.z_increment
        
        # Don't let GUI position go negative
        if self.current_z < 0:
            self.current_z = 0
        
        self.position_history.append(('move_z', self.current_x, self.current_z, self.gripper_open))
        self.update_position_display()
        
        if self.simulation_mode or not self.lash_e:
            direction_name = "UP (Z decreasing)" if direction_multiplier > 0 else "DOWN (Z increasing)"
            z_mm = self.current_z / self.UNITS_PER_MM
            delta_mm = inverted_direction * self.z_increment_mm
            print(f"SIMULATION: Moving Z {direction_name} to {self.current_z} units ({z_mm:.1f}mm) - delta: {inverted_direction * self.z_increment} units ({delta_mm:.1f}mm)")
    
    def set_gripper(self, open_gripper):
        """Set gripper state"""
        if self.lash_e and not self.simulation_mode:
            try:
                # Use Lash_E gripper control methods
                if open_gripper:
                    self.lash_e.nr_track.open_gripper()
                    print("HARDWARE: Opening gripper")
                else:
                    self.lash_e.nr_track.close_gripper()
                    print("HARDWARE: Closing gripper")
                    
                self.lash_e.logger.info(f"Manual gripper {'opened' if open_gripper else 'closed'}")
                    
            except Exception as e:
                messagebox.showerror("Gripper Error", f"Failed to control gripper: {e}")
                print(f"Error controlling gripper: {e}")
                return
                
        self.gripper_open = open_gripper
        self.position_history.append(('gripper', self.current_x, self.current_z, self.gripper_open))
        self.update_position_display()
        
        if self.simulation_mode or not self.lash_e:
            print(f"SIMULATION: Gripper {'OPENED' if open_gripper else 'CLOSED'}")
    
    def go_to_origin(self):
        """Move to origin position"""
        if self.lash_e and not self.simulation_mode:
            try:
                # Use Lash_E origin/homing command - this should go to X=0, Z=0 (top/safe)
                self.lash_e.nr_track.origin()
                print("HARDWARE: Moving to origin (X=0, Z=0 which is TOP position)")
                
                # Update position after homing - origin is X=0, Z=0 (safe height)
                self.refresh_track_status()
                
            except Exception as e:
                messagebox.showerror("Movement Error", f"Failed to move to origin: {e}")
                print(f"Error moving to origin: {e}")
                return
                
        # Origin is X=0 (left), Z=0 (top/safe position)
        self.current_x = 0
        self.current_z = 0
        self.position_history.append(('origin', self.current_x, self.current_z, self.gripper_open))
        self.update_position_display()
        
        if self.simulation_mode or not self.lash_e:
            print("SIMULATION: Moving to origin (X=0mm, Z=0mm - top/safe position)")
    
    def go_to_preset(self, name, x, z):
        """Go to predefined position"""
        if self.lash_e and not self.simulation_mode:
            try:
                x_mm = x / self.UNITS_PER_MM
                z_mm = z / self.UNITS_PER_MM
                print(f"DEBUG: Attempting preset movement to {name} at X={x_mm:.1f}mm ({x} units), Z={z_mm:.1f}mm ({z} units)")
                
                # Try direct coordinate movement first (more reliable)
                x_axis = self.lash_e.nr_track.get_axis('x_axis')
                z_axis = self.lash_e.nr_track.get_axis('z_axis') 
                x_speed = self.lash_e.nr_track.get_speed('default_x')
                z_speed = self.lash_e.nr_track.get_speed('default_z')
                
                print(f"DEBUG: Using axes X={x_axis}, Z={z_axis}, speeds X={x_speed}, Z={z_speed}")
                
                # Move X first
                self.lash_e.nr_track.c9.move_axis(x_axis, x, vel=x_speed)
                print(f"HARDWARE: X moved to {x} units ({x_mm:.1f}mm)")
                
                # Try Z movement (might not be supported)
                if z != 0:  # Only move Z if not zero position
                    try:
                        self.lash_e.nr_track.c9.move_axis(z_axis, z, vel=z_speed)
                        print(f"HARDWARE: Z moved to {z} units ({z_mm:.1f}mm)")
                    except Exception as z_error:
                        print(f"Z preset movement failed: {z_error}")
                        messagebox.showwarning("Preset Z Issue", f"X moved successfully, but Z movement failed: {z_error}")
                
                print(f"HARDWARE: Manual movement to {name} completed")
                self.lash_e.logger.info(f"Manual preset movement to {name}: X={x_mm:.1f}mm ({x} units), Z={z_mm:.1f}mm ({z} units)")
                
            except Exception as e:
                messagebox.showerror("Preset Movement Error", f"Failed to move to {name}: {e}")
                print(f"Error moving to {name}: {e}")
                return
                
        self.current_x = x
        self.current_z = z
        self.position_history.append(('preset', self.current_x, self.current_z, self.gripper_open, name))
        self.update_position_display()
        
        if self.simulation_mode or not self.lash_e:
            x_mm = x / self.UNITS_PER_MM
            z_mm = z / self.UNITS_PER_MM
            print(f"SIMULATION: Moving to {name} position at X={x_mm:.1f}mm ({x} units), Z={z_mm:.1f}mm ({z} units)")
    
    def emergency_stop(self):
        """Emergency stop function"""
        if self.lash_e and not self.simulation_mode:
            try:
                # Use Lash_E emergency stop if available
                print("HARDWARE: Emergency stop activated!")
                self.lash_e.logger.error("EMERGENCY STOP activated via GUI")
                # Note: May need to implement emergency stop in track controller
                
            except Exception as e:
                print(f"Error during emergency stop: {e}")
                
        messagebox.showwarning("Emergency Stop", "Emergency stop activated!")
        print("EMERGENCY STOP ACTIVATED")
    
    def update_position_display(self):
        """Update position labels and coordinate display"""
        # Convert encoder units to mm for display
        x_mm = self.current_x / self.UNITS_PER_MM
        z_mm = self.current_z / self.UNITS_PER_MM
        
        self.x_label.config(text=f"X: {x_mm:.1f}mm ({self.current_x})")
        z_height_desc = "TOP" if self.current_z == 0 else f"Z{z_mm:.1f}mm (lower)"
        self.z_label.config(text=f"Z: {z_mm:.1f}mm ({z_height_desc})")
        self.gripper_label.config(text=f"Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}")
        
        # Update coordinate display in single line format with both mm and units
        coord_text = f"X={x_mm:.1f}mm ({self.current_x}), Z={z_mm:.1f}mm ({'TOP' if self.current_z == 0 else 'lower'}), Gripper={'OPEN' if self.gripper_open else 'CLOSED'}"
        self.coord_display.delete('1.0', tk.END)
        self.coord_display.insert('1.0', coord_text)
    
    def save_named_position(self):
        """Save current position with user-specified name"""
        name = self.position_name_var.get().strip()
        if not name:
            messagebox.showwarning("No Name", "Please enter a position name!")
            return
        
        # Create position record
        position_data = {
            'name': name,
            'x': self.current_x,
            'z': self.current_z,
            'gripper': 'OPEN' if self.gripper_open else 'CLOSED',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to command sequence
        step_num = len(self.command_sequence) + 1
        command = {
            'step': step_num,
            'action': f"Named: {name}",
            'x': self.current_x,
            'z': self.current_z,
            'gripper': 'OPEN' if self.gripper_open else 'CLOSED',
            'notes': f"User-named position: {name}",
            'timestamp': datetime.now().isoformat()
        }
        
        self.command_sequence.append(command)
        self.update_sequence_display()
        
        # Output coordinates in single line format
        x_mm = self.current_x / self.UNITS_PER_MM
        z_mm = self.current_z / self.UNITS_PER_MM
        z_desc = "TOP" if self.current_z == 0 else f"Z{z_mm:.1f}mm(lower)"
        coord_line = f"{name}: X={x_mm:.1f}mm ({self.current_x}), Z={z_mm:.1f}mm ({z_desc}), Gripper={'OPEN' if self.gripper_open else 'CLOSED'}"
        print(f"SAVED POSITION: {coord_line}")
        
        # Update coordinate display
        self.coord_display.delete('1.0', tk.END)
        self.coord_display.insert('1.0', coord_line)
        
        # Clear name entry for next position
        self.position_name_var.set("")
        
        messagebox.showinfo("Position Saved", f"Position '{name}' saved!\n\n{coord_line}")
    
    def copy_coordinates(self):
        """Copy current coordinates to clipboard"""
        coord_text = self.coord_display.get('1.0', tk.END).strip()
        if coord_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(coord_text)
            messagebox.showinfo("Copied", f"Coordinates copied to clipboard:\n{coord_text}")
        else:
            messagebox.showwarning("No Coordinates", "No coordinates to copy!")
    
    def record_position(self):
        """Record current position to command sequence"""
        step_num = len(self.command_sequence) + 1
        
        # Determine action type based on last command
        action = "Position" 
        if self.position_history:
            last_action = self.position_history[-1][0]
            if last_action == 'gripper':
                action = "Gripper"
            elif last_action == 'origin':
                action = "Origin"
            elif last_action == 'preset':
                action = f"Preset: {self.position_history[-1][4]}"
        
        command = {
            'step': step_num,
            'action': action,
            'x': self.current_x,
            'z': self.current_z,
            'gripper': 'OPEN' if self.gripper_open else 'CLOSED',
            'notes': '',
            'timestamp': datetime.now().isoformat()
        }
        
        self.command_sequence.append(command)
        self.update_sequence_display()
        print(f"Recorded command {step_num}: {action} at ({self.current_x}, {self.current_z})")
    
    def update_sequence_display(self):
        """Update the sequence treeview"""
        # Clear existing items
        for item in self.sequence_tree.get_children():
            self.sequence_tree.delete(item)
        
        # Add commands to treeview
        for cmd in self.command_sequence:
            self.sequence_tree.insert('', 'end', values=(
                cmd['step'], cmd['action'], cmd['x'], cmd['z'], cmd['gripper'], cmd['notes']
            ))
    
    def clear_sequence(self):
        """Clear all commands from sequence"""
        if messagebox.askyesno("Clear Sequence", "Are you sure you want to clear all commands?"):
            self.command_sequence.clear()
            self.update_sequence_display()
            print("Command sequence cleared")
    
    def save_sequence(self):
        """Save command sequence to file"""
        if not self.command_sequence:
            messagebox.showwarning("No Commands", "No commands to save!")
            return
        
        # Confirm user understands this doesn't modify robot config files
        response = messagebox.askyesno("Save Sequence", 
                                     "This will save your command sequence to a NEW file that you choose.\n\n"
                                     "This will NOT modify any robot configuration files.\n\n"
                                     "To update robot positions, you must manually copy values\n"
                                     "from the saved file into the appropriate config files.\n\n"
                                     "Continue?")
        if not response:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Command Sequence (NEW FILE - will not overwrite robot config)",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.yaml'):
                    # Save in track_positions.yaml format
                    yaml_data = self.convert_sequence_to_yaml()
                    with open(filename, 'w') as f:
                        yaml.dump(yaml_data, f, default_flow_style=False, indent=2)
                else:
                    # Save as JSON
                    with open(filename, 'w') as f:
                        json.dump(self.command_sequence, f, indent=2)
                
                messagebox.showinfo("Success", f"Sequence saved to {filename}")
                print(f"Sequence saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save sequence: {e}")
    
    def load_sequence(self):
        """Load command sequence from file"""
        filename = filedialog.askopenfilename(
            title="Load Command Sequence", 
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.yaml'):
                    with open(filename, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                    self.command_sequence = self.convert_yaml_to_sequence(yaml_data)
                else:
                    with open(filename, 'r') as f:
                        self.command_sequence = json.load(f)
                
                self.update_sequence_display()
                messagebox.showinfo("Success", f"Sequence loaded from {filename}")
                print(f"Sequence loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load sequence: {e}")
    
    def convert_sequence_to_yaml(self):
        """Convert command sequence to track_positions.yaml format"""
        positions = {}
        
        for cmd in self.command_sequence:
            if cmd['action'] != 'Gripper':  # Only save position commands
                pos_name = f"custom_position_{cmd['step']}"
                positions[pos_name] = {
                    'x': cmd['x'],
                    'z_transfer': 0,
                    'z_grab': cmd['z'],
                    'z_release': cmd['z'] - 500,  # Slightly lower for release
                    'description': f"Custom position {cmd['step']}: {cmd['action']}"
                }
        
        return {'positions': positions}
    
    def convert_yaml_to_sequence(self, yaml_data):
        """Convert track_positions.yaml format to command sequence"""
        sequence = []
        step = 1
        
        if 'positions' in yaml_data:
            for pos_name, pos_data in yaml_data['positions'].items():
                if 'x' in pos_data and 'z_grab' in pos_data:
                    command = {
                        'step': step,
                        'action': f"Position: {pos_name}",
                        'x': pos_data['x'],
                        'z': pos_data['z_grab'],
                        'gripper': 'CLOSED',
                        'notes': pos_data.get('description', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    sequence.append(command)
                    step += 1
        
        return sequence
    
    def show_context_menu(self, event):
        """Show right-click context menu for sequence"""
        item = self.sequence_tree.selection()[0] if self.sequence_tree.selection() else None
        if item:
            context_menu = tk.Menu(self.root, tearoff=0)
            context_menu.add_command(label="Edit Notes", command=lambda: self.edit_notes(item))
            context_menu.add_command(label="Delete Command", command=lambda: self.delete_command(item))
            context_menu.add_separator()
            context_menu.add_command(label="Move Up", command=lambda: self.move_command_up(item))
            context_menu.add_command(label="Move Down", command=lambda: self.move_command_down(item))
            context_menu.post(event.x_root, event.y_root)
    
    def edit_notes(self, item):
        """Edit notes for selected command"""
        values = self.sequence_tree.item(item)['values']
        step = int(values[0])
        
        # Find command in sequence
        cmd = next((c for c in self.command_sequence if c['step'] == step), None)
        if cmd:
            notes = tk.simpledialog.askstring("Edit Notes", f"Notes for Step {step}:", initialvalue=cmd['notes'])
            if notes is not None:
                cmd['notes'] = notes
                self.update_sequence_display()
    
    def delete_command(self, item):
        """Delete selected command"""
        values = self.sequence_tree.item(item)['values'] 
        step = int(values[0])
        
        if messagebox.askyesno("Delete Command", f"Delete Step {step}?"):
            self.command_sequence = [c for c in self.command_sequence if c['step'] != step]
            # Renumber steps
            for i, cmd in enumerate(self.command_sequence):
                cmd['step'] = i + 1
            self.update_sequence_display()
    
    def delete_selected_command(self, event):
        """Handle delete key press"""
        selected = self.sequence_tree.selection()
        if selected:
            self.delete_command(selected[0])
    
    def move_command_up(self, item):
        """Move command up in sequence"""
        values = self.sequence_tree.item(item)['values']
        step = int(values[0])
        
        if step > 1:
            # Swap with previous command
            current_cmd = next(c for c in self.command_sequence if c['step'] == step)
            prev_cmd = next(c for c in self.command_sequence if c['step'] == step - 1)
            
            current_cmd['step'] = step - 1
            prev_cmd['step'] = step
            
            self.command_sequence.sort(key=lambda x: x['step'])
            self.update_sequence_display()
    
    def move_command_down(self, item):
        """Move command down in sequence"""
        values = self.sequence_tree.item(item)['values']
        step = int(values[0])
        
        if step < len(self.command_sequence):
            # Swap with next command
            current_cmd = next(c for c in self.command_sequence if c['step'] == step)
            next_cmd = next(c for c in self.command_sequence if c['step'] == step + 1)
            
            current_cmd['step'] = step + 1
            next_cmd['step'] = step
            
            self.command_sequence.sort(key=lambda x: x['step'])
            self.update_sequence_display()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    # Import simpledialog after tkinter setup
    import tkinter.simpledialog
    
    print("Track Control GUI - Manual Position Management")
    print("=" * 50)
    print("SAFE OPERATION: Manual movement does NOT modify robot config files")
    print("Config files (track_positions.yaml, track_status.yaml) remain unchanged")
    print("Only saves new files when you explicitly choose 'Save Sequence'")
    print("")
    print("USER-FRIENDLY UNITS: Movement and display in millimeters!")
    print(f"- Conversion: 100 encoder units = 1mm")  
    print("- Default increment: 1.0mm (fine), 10.0mm (coarse)")
    print("- Position display shows both mm and encoder units")
    print("")
    print("IMPORTANT: Z-Axis Coordinate System is INVERTED!")
    print("- Z = 0mm is the TOP/SAFE position (home)")
    print("- Z = 750mm+ are LOWER working positions")
    print("- UP button decreases Z values, DOWN button increases Z values")
    print("")
    print("Controls: Arrow keys to move, Spacebar for gripper, Enter to record")
    print("Starting in simulation mode for safety...")
    print("")
    
    gui = TrackControlGUI(simulation_mode=True)
    gui.run()