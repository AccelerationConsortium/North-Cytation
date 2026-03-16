# NorthC9 Teach & Save (Gameboy-style)  
# - Jog with buttons (±deg joints, ±mm Z)
# - Name & Save current position (joint counts + xyz/theta)
# - Go To Selected (safe) for re-teach flow
# - Update Selected after fine-tuning
# - Import/Export JSON
# - Enhanced: Predefined workflow positions and timestamped saves
#
# Safety: start with low vel/accel, verify clearances, keep physical E-Stop within reach.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, math, os
from datetime import datetime

# -------- Defaults --------
DEFAULT_ADDR   = "A"     # Controller address (e.g., 'A')
DEFAULT_SERIAL = None    # FTDI network serial (or None to auto-pick first)
DEFAULT_VEL    = 2000    # counts/s (conservative for teaching/jogging)
DEFAULT_ACC    = 20000   # counts/s^2
STEP_DEG       = 1.0     # rotational jog step (deg)
STEP_MM        = 0.5     # Z jog step (mm)

# Fast movement speeds for homing and position changes
FAST_VEL       = 12000   # counts/s (fast for major movements)
FAST_ACC       = 60000   # counts/s^2 (fast acceleration)
FAST_HOME_VEL  = 15000   # counts/s (very fast for homing)
FAST_ACC       = 40000   # counts/s^2 (fast acceleration)
FAST_HOME_VEL  = 10000   # counts/s (very fast for homing)

# Predefined workflow positions from the main surfactant screening workflow
WORKFLOW_POSITIONS = {
    "Clamp Position": {"location": "clamp", "index": 0},
    "Water Vial Position": {"location": "main_8mL_rack", "index": 44},
    "Water 2 Vial Position": {"location": "main_8mL_rack", "index": 45}, 
    "Pyrene DMSO Position": {"location": "main_8mL_rack", "index": 47},
    "Buffer Position": {"location": "main_8mL_rack", "index": 47},
    "Safe Position 46": {"location": "main_8mL_rack", "index": 46},
    "Safe Position 43": {"location": "main_8mL_rack", "index": 43},
    "Safe Position 36": {"location": "main_8mL_rack", "index": 36},
    "Cytation Position": {"location": "cytation", "index": None},
    "Pipetting Area": {"location": "pipetting_area", "index": None},
    "Home Position": {"location": "home", "index": None}
}

try:
    from north.north_c9 import NorthC9
    # Add parent directory to path to import coordinator from root
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from master_usdl_coordinator import Lash_E
except Exception as e:
    raise SystemExit("Could not import north.north_c9 or Lash_E. Install the North Robotics SDK.\n" + str(e))

# Axis IDs from API
GRIPPER  = NorthC9.GRIPPER
ELBOW    = NorthC9.ELBOW
SHOULDER = NorthC9.SHOULDER
ZAXIS    = NorthC9.Z_AXIS
FREE     = NorthC9.FREE

class TeachSaveApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NorthC9 Teach & Save (Gameboy)")
        self.geometry("980x640")
        self.c9 = None
        self.lash_e = None
        self.poll_ms = 150
        self.locations = []
        self.safe_z_counts = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.auto_reconnect = tk.BooleanVar(value=True)
        self.connecting = False  # Track connection state
        
        # In-memory position tracking (cleared on restart)
        self.workflow_positions_taught = {}  # Store learned workflow positions
        self.home_position = None  # Store home position as reference
        self.session_positions = {}  # Store any positions learned this session

        # Session state
        self.addr_var = tk.StringVar(value=DEFAULT_ADDR)
        self.ser_var  = tk.StringVar(value=DEFAULT_SERIAL if DEFAULT_SERIAL else "")
        self.vel_var  = tk.IntVar(value=DEFAULT_VEL)
        self.acc_var  = tk.IntVar(value=DEFAULT_ACC)
        self.step_deg = tk.DoubleVar(value=STEP_DEG)
        self.step_mm  = tk.DoubleVar(value=STEP_MM)
        self.tool_len_mm  = tk.DoubleVar(value=0.0)
        self.pipette_tip  = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Disconnected")

        self._build_ui()
        self._set_controls("disabled")

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Addr").grid(row=0, column=0)
        ttk.Entry(top, textvariable=self.addr_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(top, text="Network Serial").grid(row=0, column=2)
        ttk.Entry(top, textvariable=self.ser_var, width=24).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Connect", command=self.connect).grid(row=0, column=4, padx=6)
        ttk.Button(top, text="Disconnect", command=self.disconnect).grid(row=0, column=5, padx=6)
        ttk.Checkbutton(top, text="Auto-reconnect", variable=self.auto_reconnect).grid(row=0, column=6, padx=6)
        ttk.Button(top, text="Home", command=self.home).grid(row=0, column=7, padx=6)
        ttk.Button(top, text="Quick Stop", command=self.quick_stop).grid(row=0, column=8, padx=6)
        
        # Connection progress bar
        progress_frame = ttk.Frame(self, padding=4)
        progress_frame.pack(fill="x", padx=8)
        ttk.Label(progress_frame, text="Connection Progress:").pack(side="left")
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(side="left", padx=6, fill="x", expand=True)

        cfg = ttk.LabelFrame(self, text="Move Config (for Jogging/Teaching Only)", padding=8); cfg.pack(fill="x", padx=8, pady=6)
        ttk.Label(cfg, text="Vel (cts/s)", foreground="blue").grid(row=0, column=0, sticky="e")
        ttk.Entry(cfg, textvariable=self.vel_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(cfg, text="Acc (cts/s²)", foreground="blue").grid(row=0, column=2, sticky="e")
        ttk.Entry(cfg, textvariable=self.acc_var, width=8).grid(row=0, column=3, padx=6)
        ttk.Label(cfg, text="Step (deg)").grid(row=0, column=4, sticky="e")
        ttk.Entry(cfg, textvariable=self.step_deg, width=8).grid(row=0, column=5, padx=6)
        ttk.Label(cfg, text="Z Step (mm)").grid(row=0, column=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.step_mm, width=8).grid(row=0, column=7, padx=6)
        ttk.Label(cfg, text="Tool length (mm)").grid(row=0, column=8, sticky="e")
        ttk.Entry(cfg, textvariable=self.tool_len_mm, width=8).grid(row=0, column=9, padx=6)
        ttk.Checkbutton(cfg, text="Pipette tip offset", variable=self.pipette_tip).grid(row=0, column=10, padx=8)
        ttk.Button(cfg, text="Set Safe-Z = Current", command=self.set_safe_z_current).grid(row=0, column=11, padx=8)
        
        # Add speed info label
        speed_info = ttk.Label(cfg, text="Note: Homing & position movements use fast speeds automatically", 
                              font=("Arial", 8), foreground="green")
        speed_info.grid(row=1, column=0, columnspan=12, pady=3, sticky="w")

        # Predefined Workflow Positions
        workflow_frame = ttk.LabelFrame(self, text="Workflow Positions", padding=8)
        workflow_frame.pack(fill="x", padx=8, pady=6)
        
        ttk.Label(workflow_frame, text="Position:").grid(row=0, column=0, sticky="w", padx=6)
        self.position_var = tk.StringVar()
        position_combo = ttk.Combobox(workflow_frame, textvariable=self.position_var, 
                                     values=list(WORKFLOW_POSITIONS.keys()), 
                                     state="readonly", width=25)
        position_combo.grid(row=0, column=1, padx=6, pady=3)
        position_combo.set("Clamp Position")  # Default selection
        
        ttk.Button(workflow_frame, text="Go to Position", 
                  command=self.go_to_workflow_position).grid(row=0, column=2, padx=6)
        ttk.Button(workflow_frame, text="Save New Position", 
                  command=self.save_new_position_timestamped).grid(row=0, column=3, padx=6)
        
        # Add session info
        session_info = ttk.Label(workflow_frame, 
                               text="Note: Positions taught this session, auto-home on connect", 
                               font=("Arial", 8), foreground="green")
        session_info.grid(row=1, column=0, columnspan=4, pady=3, sticky="w")
        
        # Add session info
        session_info = ttk.Label(workflow_frame, 
                               text="Note: Positions taught this session, auto-home on connect", 
                               font=("Arial", 8), foreground="green")
        session_info.grid(row=1, column=0, columnspan=4, pady=3, sticky="w")

        # Jogging
        jog = ttk.LabelFrame(self, text="Jog (Gameboy style)", padding=8); jog.pack(fill="x", padx=8, pady=6)
        self._mk_jog_row(jog, "Shoulder (±deg)", SHOULDER, row=0)
        self._mk_jog_row(jog, "Elbow (±deg)", ELBOW, row=1)
        self._mk_jog_row(jog, "Gripper (±deg)", GRIPPER, row=2)
        ttk.Label(jog, text="Z (±mm)").grid(row=3, column=0, sticky="w")
        ttk.Button(jog, text="▲  Z Up",   command=lambda: self.jog_z(-1)).grid(row=3, column=1, padx=6, pady=3, sticky="ew")
        ttk.Button(jog, text="▼  Z Down", command=lambda: self.jog_z(+1)).grid(row=3, column=2, padx=6, pady=3, sticky="ew")

        # Live pose
        pose = ttk.LabelFrame(self, text="Live Pose", padding=8); pose.pack(fill="x", padx=8, pady=6)
        self.pose_txt = tk.Text(pose, height=8, font=("Consolas", 10)); self.pose_txt.pack(fill="x")

        # Naming + Save
        teach = ttk.LabelFrame(self, text="Teach / Save", padding=8); teach.pack(fill="x", padx=8, pady=6)
        self.name_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        ttk.Label(teach, text="Name").grid(row=0, column=0, sticky="e")
        ttk.Entry(teach, textvariable=self.name_var, width=24).grid(row=0, column=1, padx=6)
        ttk.Label(teach, text="Note").grid(row=0, column=2, sticky="e")
        ttk.Entry(teach, textvariable=self.note_var, width=40).grid(row=0, column=3, padx=6)
        ttk.Button(teach, text="Add @ Current", command=self.add_current).grid(row=0, column=4, padx=6)
        ttk.Button(teach, text="Update Selected", command=self.update_selected).grid(row=0, column=5, padx=6)

        # List + GoTo + Export
        listf = ttk.LabelFrame(self, text="Locations", padding=8); listf.pack(fill="both", expand=True, padx=8, pady=6)
        cols = ("name","x","y","z","theta","g_cts","e_cts","s_cts","z_cts","note")
        self.tree = ttk.Treeview(listf, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90 if c not in ("name","note") else 140)
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set); sb.pack(side="right", fill="y")

        btns = ttk.Frame(self, padding=8); btns.pack(fill="x")
        ttk.Button(btns, text="Go To Selected (safe)", command=self.goto_selected).pack(side="left", padx=6)
        ttk.Button(btns, text="Import JSON", command=self.import_json).pack(side="left", padx=6)
        ttk.Button(btns, text="Export JSON", command=self.export_json).pack(side="left", padx=6)

        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def _mk_jog_row(self, parent, label, axis, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Button(parent, text="⟵", command=lambda: self.jog_axis(axis, -1)).grid(row=row, column=1, padx=6, pady=3, sticky="ew")
        ttk.Button(parent, text="⟶", command=lambda: self.jog_axis(axis, +1)).grid(row=row, column=2, padx=6, pady=3, sticky="ew")

    # ---------- Connection & Poll ----------
    def connect(self):
        """Connect to robot with fallback strategies and auto-reconnection"""
        if self.connecting:
            self.status.set("Connection already in progress...")
            return
            
        self.connecting = True
        self.connection_attempts += 1
        
        # Start progress bar and disable connect button
        self.progress_bar.start(10)  # Animation speed
        self._set_connect_button_state("disabled")
        
        # Update UI and process events
        self.status.set(f"🔄 Connecting... (attempt {self.connection_attempts}/3)")
        self.update_idletasks()
        
        try:
            # Strategy 1: Try simple direct NorthC9 connection first (fastest)
            self.status.set(f"🔄 Step 1/2: Testing direct NorthC9 connection...")
            self.update_idletasks()
            
            self.c9 = NorthC9(
                addr=self.addr_var.get().strip(),
                network_serial=(self.ser_var.get().strip() or None),
                verbose=False, 
                project=True
            )
            
            # Test connection by getting status
            self.status.set(f"🔄 Step 2/2: Verifying robot communication...")
            self.update_idletasks()
            
            status = self.c9.get_robot_status()
            
            # Success!
            self._connection_success(f"✅ Connected Successfully! (Direct NorthC9) - Robot Status: {status}")
            return
            
        except Exception as direct_error:
            self.status.set(f"❌ Direct connection failed: {str(direct_error)[:40]}...")
            self.update_idletasks()
            
            # Strategy 2: Try with minimal Lash_E setup if direct fails
            try:
                self.status.set(f"🔄 Trying fallback method (Lash_E coordinator)...")
                self.update_idletasks()
                
                # Create a minimal vial file if it doesn't exist
                self._ensure_minimal_vial_file()
                
                self.status.set(f"🔄 Initializing Lash_E coordinator...")
                self.update_idletasks()
                
                self.lash_e = Lash_E(
                    vial_file="status/minimal_vials.csv", 
                    initialize_robot=True,
                    initialize_track=False,
                    initialize_biotek=False,
                    simulate=False,
                    show_gui=False
                )
                self.c9 = self.lash_e.nr_robot.c9
                
                # Success with fallback!
                self._connection_success("✅ Connected Successfully! (Lash_E Fallback)")
                return
                
            except Exception as lash_e_error:
                self._connection_failure(direct_error, lash_e_error)
                
    def _connection_success(self, message):
        """Handle successful connection"""
        self.progress_bar.stop()
        self.status.set("✅ Connected! Auto-homing for position reference...")
        self.connection_attempts = 0
        self.connecting = False
        
        # Force enable controls immediately
        self._set_controls("normal")
        self._set_connect_button_state("normal")
        
        # Auto-home to establish position reference
        self.after(500, self._auto_home_and_setup)
        
    def _auto_home_and_setup(self):
        """Auto-home after connection and set up position tracking"""
        try:
            self.status.set("🏠 Homing robot to establish position reference...")
            self.update_idletasks()
            
            # Home with fast speed - force maximum speed
            try:
                # Try multiple methods to ensure fast homing
                self.c9.home_robot(wait=True, vel=FAST_HOME_VEL, accel=FAST_ACC)
            except:
                try:
                    self.c9.home_robot(wait=True, vel_cts_per_sec=FAST_HOME_VEL)
                except:
                    # Final fallback
                    self.c9.home_robot(wait=True)
            
            # Store home position as reference
            self.home_position = self.c9.get_robot_positions()
            
            # Clear any previous session data
            self.workflow_positions_taught.clear()
            self.session_positions.clear()
            
            self.status.set("✅ Ready! Robot homed and position tracking initialized")
            
            # Start polling
            self.after(self.poll_ms, self._poll)
            
            # Show success message
            messagebox.showinfo("Setup Complete", 
                              "Robot connected and homed successfully!\n\n" + 
                              "• All workflow positions are now available\n" +
                              "• Position tracking active for this session\n" +
                              "• Robot will re-home when program restarts")
                              
        except Exception as e:
            messagebox.showerror("Homing Failed", f"Could not home robot:\n{str(e)}\n\nPlease home manually using the Home button.")
            self.status.set("⚠️ Connected but homing failed - home manually")
            self.after(self.poll_ms, self._poll)
    
    def _connection_failure(self, direct_error, lash_e_error):
        """Handle connection failure"""
        self.progress_bar.stop()
        self.status.set(f"❌ All connection methods failed")
        
        # Auto-retry logic
        if self.auto_reconnect.get() and self.connection_attempts < self.max_connection_attempts:
            self.status.set(f"🔄 Auto-retry in 2 seconds... ({self.connection_attempts}/{self.max_connection_attempts})")
            self.after(2000, self._retry_connection)  # Retry after 2 seconds
            return
        else:
            # Final failure
            self.connecting = False
            self._set_connect_button_state("normal")
            
            error_msg = (f"Connection failed after {self.connection_attempts} attempts:\n\n"
                        f"Method 1 (Direct): {str(direct_error)[:100]}...\n\n"
                        f"Method 2 (Lash_E): {str(lash_e_error)[:100]}...\n\n"
                        f"Please check:\n"
                        f"• Robot power and network connection\n"
                        f"• Network serial number (if specified)\n"
                        f"• No other programs using the robot")
            
            messagebox.showerror("Connection Failed", error_msg)
            self.status.set("❌ Connection failed - check robot status")
            self.connection_attempts = 0
            self._set_controls("disabled")
    
    def _retry_connection(self):
        """Retry connection (called by timer)"""
        self.connecting = False  # Reset flag for retry
        self.connect()
        
    def _set_connect_button_state(self, state):
        """Enable/disable just the connect button"""
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Button) and widget['text'] == 'Connect':
                        widget.configure(state=state)
                    
    def _ensure_minimal_vial_file(self):
        """Create minimal vial file if it doesn't exist"""
        import os
        vial_path = "status/minimal_vials.csv"
        if not os.path.exists(vial_path):
            os.makedirs("status", exist_ok=True)
            with open(vial_path, "w") as f:
                f.write("vial_name,location,location_index,contents,volume_mL\n")
                f.write("dummy_vial,clamp,0,empty,0.0\n")
    
    def disconnect(self):
        """Clean disconnect"""
        if self.connecting:
            self.progress_bar.stop()
            self.connecting = False
            
        if self.c9:
            self.c9 = None
        if self.lash_e:
            self.lash_e = None
        self._set_controls("disabled")
        self._set_connect_button_state("normal")
        self.status.set("Disconnected")

    def _poll(self):
        if not self.c9: 
            if self.auto_reconnect.get() and not self.connecting:
                self.status.set("🔄 No connection - auto-reconnecting...")
                self.connect()
            return
            
        try:
            self._refresh_pose()
            robot_status = self.c9.get_robot_status()
            
            if robot_status == FREE:
                # Ensure controls stay enabled when robot is free
                self._set_controls("normal")
                # Only update status if it's not already a success message
                if not ("✅" in self.status.get() or "Connected Successfully" in self.status.get()):
                    self.status.set(f"✅ Connected and ready - Robot status: {robot_status}")
            else:
                # Keep controls enabled but show robot is busy
                # Don't disable controls - user should still be able to use UI
                self.status.set(f"⚠️ Robot busy - Status: {robot_status}")
                
        except Exception as e:
            self.status.set(f"❌ Communication error: {str(e)[:40]}...")
            
            # Auto-reconnect on communication errors
            if self.auto_reconnect.get() and not self.connecting:
                self.c9 = None  # Clear bad connection
                self.after(1000, self.connect)  # Try to reconnect in 1 second
                return
                
        finally:
            if not self.connecting:
                self.after(self.poll_ms, self._poll)

    def _refresh_pose(self):
        g,e,s,z = self.c9.get_robot_positions()  # counts
        g_rad = NorthC9.counts_to_rad(GRIPPER, g)
        e_rad = NorthC9.counts_to_rad(ELBOW,   e)
        s_rad = NorthC9.counts_to_rad(SHOULDER,s)
        z_mm  = NorthC9.counts_to_mm(ZAXIS,    z)
        x_mm, y_mm, theta_mm = self.c9.n9_fk(
            g, e, s,
            tool_length=self.tool_len_mm.get(),
            pipette_tip_offset=self.pipette_tip.get()
        )
        self.pose_txt.delete("1.0","end")
        self.pose_txt.insert("end",
            f"Counts [G,E,S,Z]: {g:7d} {e:7d} {s:7d} {z:7d}\n"
            f"Joints [deg]:     {math.degrees(g_rad):8.3f} {math.degrees(e_rad):8.3f} {math.degrees(s_rad):8.3f}\n"
            f"Z [mm]:           {z_mm:8.3f}\n"
            f"Task [x,y,θ]:     {x_mm:8.3f} {y_mm:8.3f} {theta_mm:8.3f}\n"
        )

    def _set_controls(self, state):
        """Enable/disable interactive widgets except connection row"""
        # Handle all frames (both LabelFrame and regular Frame)
        for frame in self.winfo_children():
            if isinstance(frame, (ttk.LabelFrame, ttk.Frame)):
                # Skip the connection frame (first frame with Connect button)
                is_connection_frame = False
                for widget in frame.winfo_children():
                    if isinstance(widget, ttk.Button) and widget.cget('text') == 'Connect':
                        is_connection_frame = True
                        break
                
                if is_connection_frame:
                    continue  # Skip connection controls
                
                # Enable/disable all widgets in this frame
                self._set_frame_controls(frame, state)
    
    def _set_frame_controls(self, frame, state):
        """Recursively enable/disable controls in a frame"""
        for widget in frame.winfo_children():
            try:
                if isinstance(widget, ttk.Combobox):
                    # Special handling for comboboxes - use 'readonly' or 'disabled'
                    if state == "normal":
                        widget.configure(state="readonly")  # Keep readonly but functional
                    else:
                        widget.configure(state="disabled")
                elif isinstance(widget, (ttk.Button, ttk.Entry, ttk.Checkbutton)):
                    widget.configure(state=state)
                elif isinstance(widget, tk.Text):
                    # Text widgets use different state values
                    widget.configure(state="normal" if state == "normal" else "disabled")
                elif hasattr(widget, 'configure') and hasattr(widget, 'cget'):
                    # Try to configure any other widget that supports state
                    try:
                        widget.configure(state=state)
                    except tk.TclError:
                        pass  # Widget doesn't support state attribute
                
                # Recursively handle nested frames
                if isinstance(widget, (ttk.Frame, ttk.LabelFrame)):
                    self._set_frame_controls(widget, state)
                    
            except (tk.TclError, AttributeError):
                pass  # Skip widgets that don't support state configuration

    # ---------- Safety / Utilities ----------
    def home(self):
        if not self.c9: return
        try:
            # Use new fast homing speeds with multiple fallback methods
            self.status.set(f"Homing (fast speed - {FAST_HOME_VEL} cts/s)...")
            try:
                self.c9.home_robot(wait=False, vel=FAST_HOME_VEL, accel=FAST_ACC)
            except:
                try:
                    self.c9.home_robot(wait=False, vel_cts_per_sec=FAST_HOME_VEL)
                except:
                    self.c9.home_robot(wait=False)
                    self.status.set("Homing (default speed - API limitation)...")
        except Exception as e:
            messagebox.showerror("Home failed", str(e))

    def quick_stop(self):
        if not self.c9: return
        try:
            self.c9.quick_stop()
            self.status.set("Quick stop sent")
        except Exception as e:
            messagebox.showerror("Quick stop error", str(e))

    def set_safe_z_current(self):
        if not self.c9: return
        z = self.c9.get_axis_position(ZAXIS)
        self.safe_z_counts = z
        mm = NorthC9.counts_to_mm(ZAXIS, z)
        self.status.set(f"Safe-Z set to {mm:.2f} mm")

    # ---------- Jogging ----------
    def jog_axis(self, axis, sign):
        if not self.c9: return
        step_deg = self.step_deg.get() * sign
        curr = self.c9.get_axis_position(axis)
        delta = NorthC9.rad_to_counts(axis, math.radians(step_deg))
        try:
            self.c9.move_axis(axis=axis, cts=curr + delta,
                              vel=self.vel_var.get(), accel=self.acc_var.get(), wait=False)
            self.status.set("Moving...")
        except Exception as e:
            messagebox.showerror("Jog error", str(e))

    def jog_z(self, sign):
        if not self.c9: return
        step_mm = self.step_mm.get() * sign
        curr = self.c9.get_axis_position(ZAXIS)
        delta = NorthC9.mm_to_counts(ZAXIS, step_mm)
        try:
            self.c9.move_axis(axis=ZAXIS, cts=curr + delta,
                              vel=self.vel_var.get(), accel=self.acc_var.get(), wait=False)
            self.status.set("Moving...")
        except Exception as e:
            messagebox.showerror("Jog Z error", str(e))

    # ---------- Teach / Save ----------
    def _current_record(self):
        g,e,s,z = self.c9.get_robot_positions()
        g_deg = math.degrees(NorthC9.counts_to_rad(GRIPPER,  g))
        e_deg = math.degrees(NorthC9.counts_to_rad(ELBOW,    e))
        s_deg = math.degrees(NorthC9.counts_to_rad(SHOULDER, s))
        z_mm  = NorthC9.counts_to_mm(ZAXIS, z)
        x_mm, y_mm, theta_mm = self.c9.n9_fk(
            g, e, s,
            tool_length=self.tool_len_mm.get(),
            pipette_tip_offset=self.pipette_tip.get()
        )
        return {
            "name": self.name_var.get().strip() or f"Loc_{len(self.locations)+1}",
            "note": self.note_var.get().strip(),
            # Robot-native joint counts -> used for replay
            "g_cts": g, "e_cts": e, "s_cts": s, "z_cts": z,
            # Human-readable references
            "x": x_mm, "y": y_mm, "z": z_mm, "theta": theta_mm,
            "g_deg": g_deg, "e_deg": e_deg, "s_deg": s_deg,
            # session parameters for context
            "tool_len_mm": self.tool_len_mm.get(),
            "pipette_tip": bool(self.pipette_tip.get()),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }

    def add_current(self):
        if not self.c9: return
        rec = self._current_record()
        self.locations.append(rec)
        self._refresh_table()
        self.status.set(f"Added: {rec['name']}")

    def update_selected(self):
        if not self.c9: return
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Update", "Select a location in the list.")
            return
        rec = self._current_record()
        self.locations[idx] = rec
        self._refresh_table()
        self.status.set(f"Updated: {rec['name']}")

    def _refresh_table(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for i, r in enumerate(self.locations):
            vals = (r["name"], f"{r['x']:.3f}", f"{r['y']:.3f}", f"{r['z']:.3f}", f"{r['theta']:.3f}",
                    r["g_cts"], r["e_cts"], r["s_cts"], r["z_cts"], r.get("note",""))
            self.tree.insert("", "end", text=str(i), values=vals)

    def _selected_index(self):
        sel = self.tree.selection()
        if not sel: return None
        return int(self.tree.item(sel[0], "text"))

    # ---------- Go To (Safe) ----------
    def goto_selected(self):
        if not self.c9: return
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Go To", "Select a location.")
            return
        r = self.locations[idx]
        loc = [r["g_cts"], r["e_cts"], r["s_cts"], r["z_cts"]]  # [gripper, elbow, shoulder, z]
        safe = self.safe_z_counts  # None -> controller's default max safe height
        try:
            # 1) Safe XY to target using joint coords from the record - use fast speeds
            self.c9.goto_xy_safe(loc, safe_height=safe,
                                 vel=FAST_VEL, accel=FAST_ACC)
            # 2) Then Z down to recorded Z - fast speed
            self.c9.goto_z(loc, vel=FAST_VEL, accel=FAST_ACC, wait=False)
            self.status.set(f"Moving to {r['name']} (fast speed)...")
        except Exception as e:
            messagebox.showerror("Go To error", str(e))

    # ---------- Workflow Position Management ----------
    def go_to_workflow_position(self):
        """Move robot to selected predefined workflow position using in-memory tracking"""
        if not self.c9: 
            messagebox.showwarning("Connection", "Robot not connected!")
            return
            
        if not self.home_position:
            messagebox.showwarning("Reference Missing", "No home reference position! Please home the robot first.")
            return
            
        position_name = self.position_var.get()
        if not position_name:
            messagebox.showwarning("Selection", "Please select a position first!")
            return
            
        # Handle special case: Home position
        if position_name == "Home Position":
            self._execute_home_movement()
            return
            
        # Check if we've already taught this position
        if position_name in self.workflow_positions_taught:
            self._go_to_taught_position(position_name)
            return
            
        # For new positions, offer to teach current position
        result = messagebox.askyesno("Teach Position", 
                                   f"Position '{position_name}' not yet taught.\n\n"
                                   f"Current robot position will be saved as '{position_name}'.\n\n"
                                   f"Make sure robot is at desired position, then click YES to save.\n"
                                   f"Click NO to jog to position first.")
        
        if result:
            self._teach_current_as_workflow_position(position_name)
        else:
            messagebox.showinfo("Manual Positioning", 
                              f"To teach '{position_name}':\n\n"
                              f"1. Use jog controls below to move robot to desired position\n"
                              f"2. Select '{position_name}' from dropdown again\n"
                              f"3. Click 'Go to Position' \n"
                              f"4. Choose 'YES' to save that position")
    
    def _go_to_taught_position(self, position_name):
        """Move to a previously taught workflow position with FAST speed"""
        try:
            taught_pos = self.workflow_positions_taught[position_name]
            loc = [taught_pos["g_cts"], taught_pos["e_cts"], taught_pos["s_cts"], taught_pos["z_cts"]]
            
            self.status.set(f"Moving to {position_name} (fast speed - {FAST_VEL} cts/s)...")
            
            # Use safe movement with FAST speeds
            if self.safe_z_counts is not None:
                # Move to safe Z first, then XY, then final Z
                self.c9.move_axis(ZAXIS, self.safe_z_counts, vel=FAST_VEL, accel=FAST_ACC, wait=True)
                
            # Move to XY position with fast speed 
            self.c9.move_robot_delta([loc[0], loc[1], loc[2], loc[3]], 
                                   vel=FAST_VEL, accel=FAST_ACC, wait=False)
            
        except Exception as e:
            messagebox.showerror("Movement Error", f"Failed to move to {position_name}:\n{str(e)}")
    
    def _teach_current_as_workflow_position(self, position_name):
        """Save current position as a workflow position"""
        try:
            current_pos = self.c9.get_robot_positions()
            g, e, s, z = current_pos
            
            # Store in workflow position tracking
            self.workflow_positions_taught[position_name] = {
                "g_cts": g, "e_cts": e, "s_cts": s, "z_cts": z,
                "timestamp": datetime.now().isoformat(),
                "taught_this_session": True
            }
            
            # Also add to regular locations list for reference
            record = self._current_record()
            record["name"] = f"{position_name}_session"
            record["note"] = f"Workflow position: {position_name}"
            self.locations.append(record)
            self._refresh_table()
            
            self.status.set(f"✅ Position '{position_name}' saved! Ready for fast movement.")
            
            # Ask if they want to test the movement
            test_move = messagebox.askyesno("Test Position", 
                                          f"Position '{position_name}' saved successfully!\n\n"
                                          f"Would you like to test the fast movement?\n\n"
                                          f"Robot will move away and then return to this position.")
            
            if test_move:
                # Move slightly away then back to test
                self.status.set("Testing movement - moving slightly away...")
                self.after(1000, lambda: self._test_position_movement(position_name))
            
        except Exception as e:
            messagebox.showerror("Teaching Error", f"Could not save position:\n{str(e)}")
    
    def _test_position_movement(self, position_name):
        """Test movement to a taught position"""
        try:
            # Move to a slightly different position first (small Z move)
            current_z = self.c9.get_axis_position(ZAXIS)
            test_z = current_z + NorthC9.mm_to_counts(ZAXIS, 5.0)  # Move 5mm up
            
            self.c9.move_axis(ZAXIS, test_z, vel=FAST_VEL, accel=FAST_ACC, wait=True)
            self.status.set("Test movement - returning to taught position...")
            
            # Wait a moment then return to taught position
            self.after(1000, lambda: self._go_to_taught_position(position_name))
            
        except Exception as e:
            messagebox.showerror("Test Error", f"Test movement failed:\n{str(e)}")
    
    def _execute_home_movement(self):
        """Execute fast home movement"""
        try:
            self.status.set(f"Homing robot (fast speed - {FAST_HOME_VEL} cts/s)...")
            
            # Try multiple methods for fast homing
            try:
                self.c9.home_robot(wait=False, vel=FAST_HOME_VEL, accel=FAST_ACC)
            except:
                try:
                    self.c9.home_robot(wait=False, vel_cts_per_sec=FAST_HOME_VEL)
                except:
                    self.c9.home_robot(wait=False)  # Final fallback
                    self.status.set("Homing robot (default speed - API limitation)...")
                    
        except Exception as e:
            messagebox.showerror("Homing Error", f"Could not home robot:\n{str(e)}")
        
    def _execute_workflow_position(self, position_name, location, index):
        """Execute the actual workflow position movement"""
        try:
            if location == "clamp":
                # Move to clamp position 
                if self.lash_e and hasattr(self.lash_e, 'nr_robot'):
                    self.lash_e.nr_robot.move_vial_to_location("dummy_vial", "clamp", 0)
                    self.status.set(f"Moving to {position_name} (clamp via Lash_E - fast speed)")
                else:
                    self._show_manual_position_instructions(position_name, location, index)
                    
            elif location == "main_8mL_rack":
                # Move to specific rack position
                if self.lash_e and hasattr(self.lash_e, 'nr_robot'):
                    self.lash_e.nr_robot.move_vial_to_location("dummy_vial", "main_8mL_rack", index)
                    self.status.set(f"Moving to {position_name} (rack {index} via Lash_E - fast speed)")
                else:
                    self._show_manual_position_instructions(position_name, location, index)
                    
            elif location == "cytation":
                # Move wellplate to cytation (if applicable)
                if self.lash_e and hasattr(self.lash_e, 'move_wellplate_to_cytation'):
                    self.lash_e.move_wellplate_to_cytation()
                    self.status.set(f"Moving to {position_name}")
                else:
                    messagebox.showinfo("Info", "Cytation movement requires Lash_E with track initialization")
                    
            elif location == "pipetting_area":
                # Move wellplate to pipetting area (if applicable)
                if self.lash_e and hasattr(self.lash_e, 'move_wellplate_back_from_cytation'):
                    self.lash_e.move_wellplate_back_from_cytation()
                    self.status.set(f"Moving to {position_name}")
                else:
                    messagebox.showinfo("Info", "Pipetting area movement requires Lash_E with track initialization")
                    
            elif location == "home":
                # Home the robot (works with both connection types) - use fast homing
                try:
                    self.c9.home_robot(wait=False, vel_cts_per_sec=FAST_HOME_VEL)
                    self.status.set("Homing robot (fast speed)...")
                except:
                    # Fallback if speed parameter not supported
                    self.c9.home_robot(wait=False)
                    self.status.set("Homing robot (default speed)...")
                
            else:
                messagebox.showerror("Error", f"Unknown location type: {location}")
                
        except KeyError:
            messagebox.showerror("Error", f"Position '{position_name}' not found in workflow positions")
        except Exception as e:
            messagebox.showerror("Movement Error", f"Failed to move to {position_name}:\\n{str(e)}")
            messagebox.showerror("Movement Error", f"Failed to move to {position_name}:\n{str(e)}")

    def save_new_position_timestamped(self):
        """Save current position to a timestamped folder"""
        if not self.c9:
            messagebox.showwarning("Connection", "Robot not connected!")
            return
            
        try:
            # Create timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            position_name = self.position_var.get() or "custom_position"
            safe_name = position_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            folder_name = f"position_{safe_name}_{timestamp}"
            
            # Create positions folder if it doesn't exist
            positions_dir = "robot_positions"
            os.makedirs(positions_dir, exist_ok=True)
            
            # Create specific timestamped folder
            specific_folder = os.path.join(positions_dir, folder_name)
            os.makedirs(specific_folder, exist_ok=True)
            
            # Get current robot state
            record = self._current_record()
            record["workflow_position"] = position_name
            record["folder_created"] = timestamp
            
            # Save as JSON
            json_path = os.path.join(specific_folder, f"{safe_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                
            # Save as human-readable text
            txt_path = os.path.join(specific_folder, f"{safe_name}_readable.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Robot Position: {position_name}\n")
                f.write(f"Saved: {record['timestamp']}\n")
                f.write(f"Folder: {folder_name}\n\n")
                f.write("=== JOINT POSITIONS ===\n")
                f.write(f"Gripper:  {record['g_deg']:8.3f} deg ({record['g_cts']:7d} counts)\n")
                f.write(f"Elbow:    {record['e_deg']:8.3f} deg ({record['e_cts']:7d} counts)\n")
                f.write(f"Shoulder: {record['s_deg']:8.3f} deg ({record['s_cts']:7d} counts)\n")
                f.write(f"Z-Axis:   {record['z']:8.3f} mm  ({record['z_cts']:7d} counts)\n\n")
                f.write("=== CARTESIAN COORDINATES ===\n")
                f.write(f"X:     {record['x']:8.3f} mm\n")
                f.write(f"Y:     {record['y']:8.3f} mm\n")
                f.write(f"Z:     {record['z']:8.3f} mm\n")
                f.write(f"Theta: {record['theta']:8.3f} mm\n\n")
                f.write("=== TOOL SETTINGS ===\n")
                f.write(f"Tool Length: {record['tool_len_mm']:6.1f} mm\n")
                f.write(f"Pipette Tip Offset: {record['pipette_tip']}\n\n")
                f.write("=== NOTES ===\n")
                f.write(f"{record.get('note', 'No notes')}\n")
            
            # Update status
            self.status.set(f"Position saved to: {specific_folder}")
            
            # Show success message with folder location
            messagebox.showinfo("Position Saved", 
                              f"Position '{position_name}' saved successfully!\n\n"
                              f"Folder: {specific_folder}\n"
                              f"Files: {safe_name}.json, {safe_name}_readable.txt")
            
            # Optionally add to current session
            if messagebox.askyesno("Add to Session", 
                                 "Add this position to the current session list?"):
                self.name_var.set(f"{safe_name}_{timestamp}")
                self.add_current()
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save position:\n{str(e)}")

    # ---------- Import / Export ----------
    def import_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.locations = json.load(f)
            self._refresh_table()
            self.status.set(f"Imported {len(self.locations)} locations.")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def export_json(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.locations, f, indent=2)
            self.status.set(f"Exported {len(self.locations)} locations.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

if __name__ == "__main__":
    TeachSaveApp().mainloop()