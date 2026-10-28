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
from tkinter import ttk, messagebox, filedialog, simpledialog
import yaml
import json
import os
import sys
import queue
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append("..")
sys.path.append(".")

from robot_state.manual_track import ManualTrack, OperationRunner, ROOT

class TrackControlGUI:
    """Standalone GUI for track control and position management"""
    
    def __init__(self, simulation_mode=True):
        self.simulation_mode = True
        self.current_x = 0  # Simulation coordinates; live coordinates are read on connection.
        self.current_z = 0      # Start at safe/home position (Z=0 is TOP for track system)
        self.gripper_open = None
        self.command_sequence = []
        self.position_history = []
        
        self.hardware = None
        self.runner = OperationRunner()
        self.stop_runner = OperationRunner()
        self._operation_done = None
        self._closed = False
        self._position_known = True
        self._stage = "Ready — simulation"
        
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
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        
        # Initialize GUI components
        self.setup_gui()
        self.setup_keybindings()
        self.update_position_display()
        
        # Get initial track status if available
        self.refresh_track_status()
        self.root.after(50, self._poll_operations)
        if not simulation_mode:
            self.root.after(0, lambda: self._enable_hardware(confirm=False))
        
    def load_track_positions(self):
        """Load current track positions from YAML file"""
        try:
            yaml_path = ROOT / "robot_state/track_positions.yaml"
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
        self.connection_label = ttk.Label(status_frame, text="Track: Not Connected", font=('Arial', 9))
        self.connection_label.grid(row=1, column=0, sticky=tk.W)
        
        ttk.Button(status_frame, text="Refresh Status", command=self.refresh_track_status).grid(row=1, column=1, padx=(10, 0))
        ttk.Button(status_frame, text="Debug Hardware", command=self.debug_hardware).grid(row=1, column=2, padx=(10, 0))
        
        # Safety reminder
        self.safety_label = ttk.Label(status_frame, text="⚠️ Manual movement does NOT modify robot config files", 
                                     font=('Arial', 8), foreground="green")
        self.safety_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        self.operation_label = ttk.Label(status_frame, text="Ready — simulation", wraplength=650)
        self.operation_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
        
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
        ttk.Button(util_frame, text="SDK Quick Stop", command=self.emergency_stop, width=15).grid(row=4, column=0, pady=2)
        
        # Preset positions
        ttk.Label(util_frame, text="Preset Positions:", font=('Arial', 10, 'bold')).grid(row=5, column=0, pady=(10, 5))
        
        preset_positions = self.preset_positions()
        
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

    def preset_positions(self):
        """Use configured transfer coordinates, rather than old calibration values."""
        labels = {"pipetting_area": "Pipetting Area", "source_stack": "Source Stack",
                  "waste_stack": "Waste Stack", "cytation_tray": "Cytation Tray",
                  "lid_storage": "Lid Storage", "lid_storage_96": "96-well Lid Storage"}
        positions = self.track_positions.get("positions", {})
        return [(label, positions[name]["x"], positions[name]["z_transfer"])
                for name, label in labels.items() if name in positions]

    def _submit(self, label, function, completed=None):
        if self.runner.busy or self.stop_runner.busy:
            self.operation_label.config(text="An operation is still running; additional commands were not queued.")
            return False
        self._operation_done = completed
        self._operation_started = time.monotonic()
        self._stage = label
        self.operation_label.config(text=label)
        return self.runner.start(label, function)

    def _poll_operations(self):
        if self._closed:
            return
        for runner in (self.runner, self.stop_runner):
            while True:
                try:
                    kind, payload = runner.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "progress":
                    self._stage = payload
                else:
                    runner.busy = False
                    label, value = payload
                    if kind == "error":
                        self._stage = f"{label} failed: {value}"
                        self.connection_label.config(text="Track: operation failed; inspect controller", foreground="red")
                        self.gripper_open = None
                        self._position_known = False
                        self.update_position_display()
                        messagebox.showerror(label, value)
                    else:
                        if runner is self.runner and self._operation_done:
                            self._operation_done(value)
                        self._stage = f"{label} complete"
                    if runner is self.runner:
                        self._operation_done = None
        suffix = f" ({time.monotonic() - self._operation_started:.1f}s)" if self.runner.busy else ""
        self.operation_label.config(text=self._stage + suffix)
        self.root.after(50, self._poll_operations)

    def _set_position(self, position):
        self.current_x, self.current_z = position
        self._position_known = True
        self.update_position_display()

    def _enable_hardware(self, confirm=True):
        if self.runner.busy or self.stop_runner.busy:
            return
        if confirm and not messagebox.askyesno("Hardware Mode",
                "Connect to the real track? Ensure no workflow is using it. Connecting will not move it."):
            return
        existing = self.hardware

        def connect(progress):
            track = existing or ManualTrack.connect(progress)
            return track, track.read_position(progress)

        def connected(result):
            self.hardware, position = result
            self.simulation_mode = False
            self.gripper_open = None
            self.mode_label.config(text="Mode: HARDWARE")
            self.connection_label.config(text="Track: Connected (Hardware)", foreground="green")
            self._set_position(position)

        self._submit("Connect track", connect, connected)

    def toggle_simulation_mode(self):
        if self.runner.busy or self.stop_runner.busy:
            self.operation_label.config(text="Wait for the current request before changing modes.")
            return
        if self.simulation_mode:
            self._enable_hardware()
        else:
            self.simulation_mode = True
            self.mode_label.config(text="Mode: SIMULATION")
            self.connection_label.config(text="Track: Simulation (hardware parked)", foreground="blue")
            self._position_known = True

    def refresh_track_status(self):
        if self.simulation_mode:
            self.connection_label.config(text="Track: Simulation", foreground="blue")
        elif self.hardware:
            self._submit("Refresh track encoders", self.hardware.read_position, self._set_position)

    def debug_hardware(self):
        positions = self.track_positions.get("positions", {})
        messagebox.showinfo("Track diagnostics", f"Mode: {'simulation' if self.simulation_mode else 'hardware'}\n"
                            f"Operation: {self._stage}\n"
                            f"Configured holders: {', '.join(positions)}\n"
                            "Connection uses controller A / USB AU06CNCF; axes come from robot_hardware.yaml.\n"
                            "No workflow, inventory recovery or photoreactor is initialized.")

    def update_increments(self):
        try:
            x, z = self.x_increment_var.get(), self.z_increment_var.get()
            if not all(0.1 <= value <= 10 for value in (x, z)):
                raise ValueError("Movement increments must be between 0.1 and 10 mm")
            self.x_increment_mm, self.z_increment_mm = x, z
            self.x_increment, self.z_increment = round(x * self.UNITS_PER_MM), round(z * self.UNITS_PER_MM)
            return True
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Movement increment", str(error))
            return False

    def _motion(self, label, function, action, name=None, gripper=None):
        if self.runner.busy or self.stop_runner.busy:
            self.operation_label.config(text="An operation is still running; additional commands were not queued.")
            return
        track = self.hardware
        track.cancel.clear()

        def completed(position):
            if gripper is not None:
                self.gripper_open = gripper
            self._set_position(position)
            entry = (action, self.current_x, self.current_z, self.gripper_open)
            self.position_history.append(entry + (name,) if name else entry)

        self._submit(label, function, completed)

    def _jog(self, axis, direction):
        if not self.update_increments():
            return
        delta = direction * (self.x_increment if axis == "x" else -self.z_increment)
        if self.simulation_mode:
            if self.runner.busy or self.stop_runner.busy:
                return
            self.current_x += delta if axis == "x" else 0
            self.current_z = max(0, self.current_z + (delta if axis == "z" else 0))
            self.position_history.append(("move_" + axis, self.current_x, self.current_z, self.gripper_open))
            self.update_position_display()
        elif self.hardware:
            self._motion(f"Jog {axis.upper()}", lambda progress: self.hardware.jog(axis, delta, progress), "move_" + axis)

    def move_x(self, direction_multiplier):
        self._jog("x", direction_multiplier)

    def move_z(self, direction_multiplier):
        self._jog("z", direction_multiplier)

    def set_gripper(self, open_gripper):
        if self.simulation_mode:
            self.gripper_open = open_gripper
            self.position_history.append(('gripper', self.current_x, self.current_z, self.gripper_open))
            self.update_position_display()
        elif self.hardware:
            self._motion("Open gripper" if open_gripper else "Close gripper",
                         lambda progress: self.hardware.set_gripper(open_gripper, progress), "gripper", gripper=open_gripper)

    def go_to_origin(self):
        # A coordinate move, not homing/recalibration; never updates track_status.
        self.go_to_preset("Origin", 0, 0, action="origin")

    def go_to_preset(self, name, x, z, action="preset"):
        if self.simulation_mode:
            self.current_x, self.current_z = x, z
            self.position_history.append((action, x, z, self.gripper_open, name))
            self.update_position_display()
        elif self.hardware:
            self._motion(f"Move to {name}", lambda progress: self.hardware.move_to(x, z, progress), action, name=name)

    def emergency_stop(self):
        if self.simulation_mode or not self.hardware:
            self.operation_label.config(text="Simulation: no stop sent to hardware")
            return
        if self.stop_runner.busy:
            return
        self.hardware.cancel.set()
        self.stop_runner.start("SDK quick stop", self.hardware.stop)
        self.operation_label.config(text="Requesting SDK quick stop; use the physical E-stop if needed.")

    def _close(self):
        self._closed = True
        # Never wait on an SDK thread on the UI thread, or silently command motion.
        self.root.destroy()

    def update_position_display(self):
        """Update position labels and coordinate display"""
        # Convert encoder units to mm for display
        x_mm = self.current_x / self.UNITS_PER_MM
        z_mm = self.current_z / self.UNITS_PER_MM
        
        self.x_label.config(text=f"X: {x_mm:.1f}mm ({self.current_x})")
        z_height_desc = "TOP" if self.current_z == 0 else f"Z{z_mm:.1f}mm (lower)"
        self.z_label.config(text=f"Z: {z_mm:.1f}mm ({z_height_desc})")
        gripper = 'UNKNOWN' if self.gripper_open is None else 'OPEN' if self.gripper_open else 'CLOSED'
        self.gripper_label.config(text=f"Gripper: {gripper}")
        if not self._position_known:
            self.x_label.config(text=f"X: unconfirmed (last {self.current_x})")
            self.z_label.config(text=f"Z: unconfirmed (last {self.current_z})")
        
        # Update coordinate display in single line format with both mm and units
        coord_text = f"X={x_mm:.1f}mm ({self.current_x}), Z={z_mm:.1f}mm ({'TOP' if self.current_z == 0 else 'lower'}), Gripper={gripper}"
        if not self._position_known:
            coord_text = "UNCONFIRMED last reading: " + coord_text
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
            notes = simpledialog.askstring("Edit Notes", f"Notes for Step {step}:", initialvalue=cmd['notes'])
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
