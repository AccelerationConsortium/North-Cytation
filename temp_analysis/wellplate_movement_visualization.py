#!/usr/bin/env python3
"""
Visualization of wellplate movement during measure_wellplate() method
Shows the path a wellplate takes in (x, z) coordinate space

Coordinate system: Z=0 is highest, larger Z values are lower physical positions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Coordinates from track_positions.yaml
positions = {
    'pipetting_area': {'x': 131854, 'z': 88750},
    'cytation_tray': {'x': 68608, 'z_grab': 8500, 'z_release': 5500},
    'transfer_stack': {'x': 28000},
    'transfer_height': {'z': 75000},
    'max_safe_height': 0
}

# Create the visualization
fig, ax = plt.subplots(figsize=(14, 10))

# Define the movement sequence for measure_wellplate()
movement_sequence = [
    # 1. move_wellplate_to_cytation()
    # 1a. grab_wellplate_from_location('pipetting_area')
    {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '1a. Z up to safe height', 'color': 'blue'},
    {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z']), 'action': '1b. Z down, grab wellplate', 'color': 'green'},
    {'pos': (positions['pipetting_area']['x'], positions['transfer_height']['z']), 'action': '1c. Z up to transfer height', 'color': 'blue'},
    
    # 1b. release_wellplate_in_location('cytation_tray') using safe path
    # transfer_wellplate_via_path() sequence:
    {'pos': (positions['pipetting_area']['x'], positions['transfer_height']['z']), 'action': '1d. At transfer height', 'color': 'orange'},
    {'pos': (positions['transfer_stack']['x'], positions['transfer_height']['z']), 'action': '1e. X move to transfer area', 'color': 'orange'},
    {'pos': (positions['transfer_stack']['x'], positions['max_safe_height']), 'action': '1f. Z up to safe height', 'color': 'blue'},
    {'pos': (positions['cytation_tray']['x'], positions['max_safe_height']), 'action': '1g. X move to cytation area', 'color': 'blue'},
    {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_release']), 'action': '1h. Z down, release wellplate', 'color': 'red'},
    {'pos': (positions['cytation_tray']['x'], positions['max_safe_height']), 'action': '1i. Z up to safe height', 'color': 'blue'},
    
    # 2. MEASUREMENT HAPPENS HERE (wellplate stays at cytation)
    
    # 3. move_wellplate_back_from_cytation()
    # 3a. grab_wellplate_from_location('cytation_tray') 
    {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_grab']), 'action': '2a. Z down, grab from cytation', 'color': 'green'},
    {'pos': (positions['cytation_tray']['x'], positions['transfer_height']['z']), 'action': '2b. Z up to transfer height', 'color': 'blue'},
    
    # 3b. release_wellplate_in_location('pipetting_area') using safe path  
    # transfer_wellplate_via_path() sequence:
    {'pos': (positions['transfer_stack']['x'], positions['transfer_height']['z']), 'action': '2c. X move to transfer area', 'color': 'orange'},
    {'pos': (positions['transfer_stack']['x'], positions['max_safe_height']), 'action': '2d. Z up to safe height', 'color': 'blue'},
    {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '2e. X move to pipetting area', 'color': 'blue'},
    {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z']), 'action': '2f. Z down, release wellplate', 'color': 'red'},
    {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '2g. Z up to safe height', 'color': 'blue'},
]

# Plot the movement path
x_coords = [step['pos'][0] for step in movement_sequence]
z_coords = [step['pos'][1] for step in movement_sequence]

# Plot the movement path with proper step-by-step lines
for i in range(len(movement_sequence)-1):
    x1, z1 = movement_sequence[i]['pos']
    x2, z2 = movement_sequence[i+1]['pos']
    
    # Determine movement type and color
    if x1 == x2:  # Vertical movement (Z only)
        line_color = 'purple'
        line_style = '-'
        alpha = 0.8
    else:  # Horizontal movement (X only) 
        line_color = 'darkorange'
        line_style = '-'
        alpha = 0.8
    
    ax.plot([x1, x2], [z1, z2], color=line_color, linestyle=line_style, 
            linewidth=3, alpha=alpha)

# Add arrows to show direction
for i in range(len(movement_sequence)-1):
    x1, z1 = movement_sequence[i]['pos']
    x2, z2 = movement_sequence[i+1]['pos']
    
    # Add arrow in middle of each segment
    mid_x, mid_z = (x1 + x2) / 2, (z1 + z2) / 2
    dx, dz = x2 - x1, z2 - z1
    
    if abs(dx) > 1000 or abs(dz) > 1000:  # Only add arrows for significant movements
        ax.annotate('', xy=(x2, z2), xytext=(mid_x, mid_z),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Add numbered points for each step
for i, step in enumerate(movement_sequence):
    x, z = step['pos']
    color = step['color']
    
    # Add point
    ax.plot(x, z, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Add step number
    ax.annotate(f'{i+1}', (x, z), xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.7))

# Add location labels and boundaries
locations = {
    'Pipetting Area': (positions['pipetting_area']['x'], positions['pipetting_area']['z']),
    'Cytation Tray': (positions['cytation_tray']['x'], (positions['cytation_tray']['z_grab'] + positions['cytation_tray']['z_release'])/2),
    'Transfer Area': (positions['transfer_stack']['x'], positions['transfer_height']['z']),
    'Safe Height': (positions['pipetting_area']['x'], positions['max_safe_height'])
}

for name, (x, z) in locations.items():
    ax.annotate(name, (x, z), xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

# Add key areas as rectangles
# Pipetting area
pipetting_rect = patches.Rectangle((positions['pipetting_area']['x']-5000, positions['pipetting_area']['z']-2000), 
                                 10000, 4000, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
ax.add_patch(pipetting_rect)

# Cytation area  
cytation_rect = patches.Rectangle((positions['cytation_tray']['x']-5000, positions['cytation_tray']['z_release']-1000), 
                                10000, positions['cytation_tray']['z_grab']-positions['cytation_tray']['z_release']+2000, 
                                linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.3)
ax.add_patch(cytation_rect)

# Transfer area
transfer_rect = patches.Rectangle((positions['transfer_stack']['x']-5000, positions['transfer_height']['z']-2000), 
                                10000, 4000, linewidth=2, edgecolor='orange', facecolor='moccasin', alpha=0.3)
ax.add_patch(transfer_rect)

# Customize the plot
ax.set_xlabel('X Position (encoder counts)', fontsize=12)
ax.set_ylabel('Z Position (encoder counts)\n← Higher (Z=0) | Lower (larger Z) →', fontsize=12)
ax.set_title('Wellplate Movement During measure_wellplate() Method\n(x, z) Coordinate Path - Sequential Axis Movement', fontsize=14, fontweight='bold')

# Invert Y axis to show Z=0 at top (higher physical position)
ax.invert_yaxis()

# Add grid
ax.grid(True, alpha=0.3)

# Set axis limits with some padding
all_x = [pos['x'] for pos in positions.values() if isinstance(pos, dict) and 'x' in pos]
all_z = [positions['max_safe_height'], positions['pipetting_area']['z'], 
         positions['cytation_tray']['z_grab'], positions['cytation_tray']['z_release'],
         positions['transfer_height']['z']]

x_range = max(all_x) - min(all_x)
z_range = max(all_z) - min(all_z)

ax.set_xlim(min(all_x) - x_range*0.1, max(all_x) + x_range*0.1)
ax.set_ylim(max(all_z) + z_range*0.1, min(all_z) - z_range*0.1)  # Inverted for Z

# Add legend for movement types
movement_legend = [
    plt.Line2D([0], [0], color='purple', linewidth=3, label='Z movement (vertical)'),
    plt.Line2D([0], [0], color='darkorange', linewidth=3, label='X movement (horizontal)')
]
action_colors = {'blue': 'Movement', 'green': 'Grab', 'red': 'Release', 'orange': 'Transfer routing'}
action_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=8, label=action) for color, action in action_colors.items()]

all_legend = movement_legend + action_legend
ax.legend(handles=all_legend, loc='upper right')

# Add step-by-step description
step_text = "\n".join([f"{i+1}. {step['action']}" for i, step in enumerate(movement_sequence)])
ax.text(0.02, 0.98, "Movement Sequence:", transform=ax.transAxes, fontsize=10, 
        fontweight='bold', verticalalignment='top')
ax.text(0.02, 0.94, step_text, transform=ax.transAxes, fontsize=8, 
        verticalalignment='top', family='monospace')

plt.tight_layout()
plt.savefig('c:/Users/owenm/OneDrive/Desktop/North Robotics/utoronto_demo/utoronto_demo/temp_analysis/wellplate_movement_diagram.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Wellplate movement visualization created!")
print("Path: pipetting_area → cytation_tray → [MEASUREMENT] → cytation_tray → pipetting_area")
print("\nMovement Pattern - ALWAYS sequential:")
print("1. Z first (to safe height)")  
print("2. X second (horizontal movement)")
print("3. Z third (to target height)")
print("\nKey features:")
print("- Purple lines = Z movement (vertical only)")
print("- Orange lines = X movement (horizontal only)")  
print("- No diagonal movement - one axis at a time!")
print("- Uses safe routing via transfer area (x=28000) to avoid cytation collisions")
print("- Always moves to safe height (z=0) before horizontal movements") 
print("- Different grab/release heights for cytation (z_grab=8500, z_release=5500)")
print("- Standard height for pipetting area (z=88750 - very low for pipetting)")