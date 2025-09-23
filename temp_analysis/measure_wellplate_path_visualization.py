"""
Visualization of wellplate movement path during measure_wellplate operation in master_usdl_coordinator.

This shows the complete path the wellplate takes from pipetting_area to cytation and back.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load positions from track_positions.yaml (simplified version)
positions = {
    'pipetting_area': {'x': 131854, 'z_transfer': 75000, 'z_grab': 88750, 'z_release': 86350},
    'cytation_tray': {'x': 68608, 'z_transfer': 0, 'z_release': 5500, 'z_grab': 8500},
    'cytation_safe_area': {'x': 47747},
    'max_safe_height': 0,
}

def create_measure_wellplate_path_visualization():
    """Create visualization showing the complete measure_wellplate movement path"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define the movement sequence based on measure_wellplate workflow
    movement_sequence = [
        # measure_wellplate() calls move_wellplate_to_cytation()
        {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '1. Start at safe height over pipetting area', 'color': 'green', 'marker': 'o'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '2. Move to transfer height', 'color': 'blue', 'marker': 's'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_grab']), 'action': '3. Move down to grab wellplate', 'color': 'orange', 'marker': 'v'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '4. Lift wellplate to transfer height', 'color': 'blue', 'marker': '^'},
        
        # Navigate via cytation_safe_area waypoint
        {'pos': (positions['cytation_safe_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '5. Move to cytation safe area waypoint (X only)', 'color': 'purple', 'marker': 'D'},
        {'pos': (positions['cytation_safe_area']['x'], positions['cytation_tray']['z_transfer']), 'action': '6. Move to cytation transfer height (Z only)', 'color': 'purple', 'marker': 's'},
        {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_transfer']), 'action': '7. Move to cytation X position', 'color': 'red', 'marker': 's'},
        {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_release']), 'action': '8. Lower wellplate into cytation', 'color': 'red', 'marker': 'v'},
        
        # Measurement happens here (no movement)
        {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_release']), 'action': '9. *** MEASUREMENT IN CYTATION ***', 'color': 'gold', 'marker': '*'},
        
        # move_wellplate_back_from_cytation()
        {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_grab']), 'action': '10. Grab wellplate from cytation', 'color': 'red', 'marker': '^'},
        {'pos': (positions['cytation_tray']['x'], positions['cytation_tray']['z_transfer']), 'action': '11. Lift to transfer height', 'color': 'red', 'marker': 's'},
        
        # Navigate back via cytation_safe_area waypoint
        {'pos': (positions['cytation_safe_area']['x'], positions['cytation_tray']['z_transfer']), 'action': '12. Move to cytation safe area waypoint (X only)', 'color': 'purple', 'marker': 'D'},
        {'pos': (positions['cytation_safe_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '13. Move to pipetting transfer height (Z only)', 'color': 'purple', 'marker': 's'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '14. Return to pipetting area X position', 'color': 'blue', 'marker': 's'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_release']), 'action': '15. Lower wellplate to pipetting area', 'color': 'blue', 'marker': 'v'},
        {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '16. Return to safe height (complete)', 'color': 'green', 'marker': 'o'},
    ]
    
    # Plot the movement path
    x_coords = [step['pos'][0] for step in movement_sequence]
    z_coords = [step['pos'][1] for step in movement_sequence]
    
    # Draw path lines
    ax.plot(x_coords, z_coords, 'k--', alpha=0.5, linewidth=1, label='Movement Path')
    
    # Plot each movement step
    for i, step in enumerate(movement_sequence):
        x, z = step['pos']
        ax.scatter(x, z, c=step['color'], marker=step['marker'], s=100, 
                  edgecolors='black', linewidth=1, alpha=0.8, zorder=3)
        
        # Add step numbers
        ax.annotate(f"{i+1}", (x, z), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, fontweight='bold', color='black')
    
    # Add location labels and areas
    locations = {
        'Pipetting Area': (positions['pipetting_area']['x'], positions['pipetting_area']['z_grab']),
        'Cytation Tray': (positions['cytation_tray']['x'], positions['cytation_tray']['z_release']),
        'Cytation Safe Area': (positions['cytation_safe_area']['x'], 40000),
        'Safe Height Zone': (100000, positions['max_safe_height']),
    }
    
    for name, (x, z) in locations.items():
        ax.text(x, z, name, ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add area rectangles
    # Pipetting area
    pipetting_rect = patches.Rectangle(
        (positions['pipetting_area']['x']-10000, positions['pipetting_area']['z_release']-5000), 
        20000, 10000, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ax.add_patch(pipetting_rect)
    
    # Cytation area
    cytation_rect = patches.Rectangle(
        (positions['cytation_tray']['x']-10000, positions['cytation_tray']['z_release']-2000), 
        20000, 15000, linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.3)
    ax.add_patch(cytation_rect)
    
    # Safe area waypoint
    safe_area_rect = patches.Rectangle(
        (positions['cytation_safe_area']['x']-5000, 20000), 
        10000, 40000, linewidth=2, edgecolor='purple', facecolor='plum', alpha=0.3)
    ax.add_patch(safe_area_rect)
    
    # Formatting
    ax.set_xlim(40000, 140000)
    ax.set_ylim(-5000, 95000)
    ax.set_xlabel('X Position (encoder counts)', fontsize=12)
    ax.set_ylabel('Z Position (encoder counts)\n(0 = highest, larger values = lower)', fontsize=12)
    ax.set_title('Wellplate Movement Path During measure_wellplate() Operation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Z=0 is highest position
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start/End'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Transfer Height'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=10, label='Grab/Release'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='Waypoint'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Measurement'),
        plt.Line2D([0], [0], linestyle='--', color='black', alpha=0.5, label='Movement Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add operation phases
    ax.text(0.02, 0.98, 'Phase 1: Move to Cytation (steps 1-7)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    ax.text(0.02, 0.90, 'Phase 2: Measurement (step 8)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='gold'))
    ax.text(0.02, 0.82, 'Phase 3: Return to Pipetting (steps 9-14)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    plt.tight_layout()
    return fig, movement_sequence

def print_movement_summary(movement_sequence):
    """Print a text summary of the movement sequence"""
    print("\n" + "="*80)
    print("MEASURE_WELLPLATE MOVEMENT SEQUENCE SUMMARY")
    print("="*80)
    print(f"Total steps: {len(movement_sequence)}")
    print("\nDetailed sequence:")
    
    for i, step in enumerate(movement_sequence):
        x, z = step['pos']
        print(f"{i+1:2d}. {step['action']}")
        print(f"    Position: X={x:,}, Z={z:,}")
        if i < len(movement_sequence) - 1:
            next_x, next_z = movement_sequence[i+1]['pos']
            dx = abs(next_x - x)
            dz = abs(next_z - z)
            if dx > 0 or dz > 0:
                print(f"    Movement to next: ΔX={dx:,}, ΔZ={dz:,}")
        print()
    
    print("KEY OBSERVATIONS:")
    print("- Uses cytation_safe_area waypoint for obstacle avoidance")
    print("- Maintains transfer height during horizontal movements") 
    print("- Precise vertical movements for grab/release operations")
    print("- Complete round trip: pipetting → cytation → pipetting")

if __name__ == "__main__":
    # Create and show the visualization
    fig, movement_sequence = create_measure_wellplate_path_visualization()
    
    # Print movement summary
    print_movement_summary(movement_sequence)
    
    # Show the plot
    plt.show()
    
    # Save the plot
    plt.savefig('measure_wellplate_movement_path.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'measure_wellplate_movement_path.png'")