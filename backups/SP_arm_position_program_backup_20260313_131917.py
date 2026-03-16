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
DEFAULT_VEL    = 2000    # counts/s (conservative for teaching)
DEFAULT_ACC    = 20000   # counts/s^2
STEP_DEG       = 1.0     # rotational jog step (deg)
STEP_MM        = 0.5     # Z jog step (mm)

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
except Exception as e:
    raise SystemExit("Could not import north.north_c9. Install the North Robotics SDK.\n" + str(e))

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
        self.poll_ms = 150
        self.locations = []
        self.safe_z_counts = None

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
        ttk.Button(top, text="Home", command=self.home).grid(row=0, column=5, padx=6)
        ttk.Button(top, text="Quick Stop", command=self.quick_stop).grid(row=0, column=6, padx=6)

        cfg = ttk.LabelFrame(self, text="Move Config", padding=8); cfg.pack(fill="x", padx=8, pady=6)
        ttk.Label(cfg, text="Vel (cts/s)").grid(row=0, column=0, sticky="e")
        ttk.Entry(cfg, textvariable=self.vel_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(cfg, text="Acc (cts/s²)").grid(row=0, column=2, sticky="e")
        ttk.Entry(cfg, textvariable=self.acc_var, width=8).grid(row=0, column=3, padx=6)
        ttk.Label(cfg, text="Step (deg)").grid(row=0, column=4, sticky="e")
        ttk.Entry(cfg, textvariable=self.step_deg, width=8).grid(row=0, column=5, padx=6)
        ttk.Label(cfg, text="Z Step (mm)").grid(row=0, column=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.step_mm, width=8).grid(row=0, column=7, padx=6)
        ttk.Label(cfg, text="Tool length (mm)").grid(row=0, column=8, sticky="e")
        ttk.Entry(cfg, textvariable=self.tool_len_mm, width=8).grid(row=0, column=9, padx=6)
        ttk.Checkbutton(cfg, text="Pipette tip offset", variable=self.pipette_tip).grid(row=0, column=10, padx=8)
        ttk.Button(cfg, text="Set Safe-Z = Current", command=self.set_safe_z_current).grid(row=0, column=11, padx=8)

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
        try:
            self.c9 = NorthC9(addr=self.addr_var.get().strip(),
                              network_serial=(self.ser_var.get().strip() or None),
                              verbose=False, project=True)
            self.status.set("Connected.")
            self._set_controls("normal")
            self.after(self.poll_ms, self._poll)
        except Exception as e:
            messagebox.showerror("Connect failed", str(e))
            self.status.set("Connect failed")
            self._set_controls("disabled")

    def _poll(self):
        if not self.c9: return
        try:
            self._refresh_pose()
            if self.c9.get_robot_status() == FREE:
                self._set_controls("normal")
            else:
                self._set_controls("disabled")
        except Exception as e:
            self.status.set(f"Poll error: {e}")
        finally:
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
        # enable/disable interactive widgets except connection row
        for lf in self.winfo_children():
            if isinstance(lf, ttk.LabelFrame):
                for w in lf.winfo_children():
                    try: w.configure(state=state)
                    except: pass

    # ---------- Safety / Utilities ----------
    def home(self):
        if not self.c9: return
        try:
            self.c9.home_robot(wait=False)  # non-blocking
            self.status.set("Homing...")
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
            # 1) Safe XY to target using joint coords from the record
            self.c9.goto_xy_safe(loc, safe_height=safe,
                                 vel=self.vel_var.get(), accel=self.acc_var.get())
            # 2) Then Z down to recorded Z
            self.c9.goto_z(loc, vel=self.vel_var.get(), accel=self.acc_var.get(), wait=False)
            self.status.set(f"Moving to {r['name']}...")
        except Exception as e:
            messagebox.showerror("Go To error", str(e))

    # ---------- Workflow Position Management ----------
    def go_to_workflow_position(self):
        """Move robot to selected predefined workflow position"""
        if not self.c9: 
            messagebox.showwarning("Connection", "Robot not connected!")
            return
            
        position_name = self.position_var.get()
        if not position_name:
            messagebox.showwarning("Selection", "Please select a position first!")
            return
            
        try:
            position_info = WORKFLOW_POSITIONS[position_name]
            location = position_info["location"]
            index = position_info["index"]
            
            vel = self.vel_var.get()
            acc = self.acc_var.get()
            
            if location == "clamp":
                # Move to clamp position
                self.c9.move_vial_to_location("dummy_vial", "clamp", 0, vel=vel, accel=acc)
                self.status.set(f"Moving to {position_name} (clamp)")
                
            elif location == "main_8mL_rack":
                # Move to specific rack position
                self.c9.move_vial_to_location("dummy_vial", "main_8mL_rack", index, vel=vel, accel=acc)
                self.status.set(f"Moving to {position_name} (rack {index})")
                
            elif location == "cytation":
                # Move wellplate to cytation (if applicable)
                try:
                    self.c9.move_wellplate_to_cytation()
                    self.status.set(f"Moving to {position_name}")
                except:
                    messagebox.showwarning("Position", "Cytation position requires wellplate on track")
                    
            elif location == "pipetting_area":
                # Move wellplate to pipetting area (if applicable)
                try:
                    self.c9.move_wellplate_back_from_cytation()
                    self.status.set(f"Moving to {position_name}")
                except:
                    messagebox.showwarning("Position", "Pipetting area requires wellplate to be returned from cytation")
                    
            elif location == "home":
                # Home the robot
                self.c9.home_robot(wait=False)
                self.status.set("Homing robot...")
                
            else:
                messagebox.showerror("Error", f"Unknown location type: {location}")
                
        except KeyError:
            messagebox.showerror("Error", f"Position '{position_name}' not found in workflow positions")
        except Exception as e:
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