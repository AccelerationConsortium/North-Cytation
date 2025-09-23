"""
Visualization of wellplate movement path during grab_new_wellplate() and discard_used_wellplate() operations.

This shows the complete path the wellplate takes from source_stack → pipetting_area → waste_stack.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load positions from track_positions.yaml (simplified version)
positions = {
    'source_stack': {'x': 175, 'z_transfer': 0, 'z_grab': 83500, 'z_release': 83000},
    'pipetting_area': {'x': 131854, 'z_transfer': 75000, 'z_grab': 88750, 'z_release': 86350},
    'waste_stack': {'x': 14550, 'z_transfer': 0, 'z_grab': 83500, 'z_release': 83000},
    'transfer_stack': {'x': 28000},
    'max_safe_height': 0,
}

def create_grab_discard_path_visualization():
    """Create visualization showing the complete grab → discard movement path"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define the movement sequence based on grab_new_wellplate() + discard_used_wellplate()
    movement_sequence = [
        # === GRAB NEW WELLPLATE SEQUENCE ===
        {'pos': (positions['source_stack']['x'], positions['max_safe_height']), 'action': '1. Start at safe height over source stack', 'color': 'green', 'marker': 'o', 'phase': 'grab'},
        {'pos': (positions['transfer_stack']['x'], positions['max_safe_height']), 'action': '2. Move to transfer stack waypoint (X only)', 'color': 'purple', 'marker': 'D', 'phase': 'grab'},
        {'pos': (positions['source_stack']['x'], positions['source_stack']['z_transfer']), 'action': '3. Move to source stack transfer height (Z only)', 'color': 'purple', 'marker': 's', 'phase': 'grab'},
        {'pos': (positions['source_stack']['x'], positions['source_stack']['z_transfer']), 'action': '4. Move to source X position', 'color': 'cyan', 'marker': 's', 'phase': 'grab'},
        {'pos': (positions['source_stack']['x'], 83500), 'action': '5. Move down to grab wellplate from stack', 'color': 'cyan', 'marker': 'v', 'phase': 'grab'},
        {'pos': (positions['source_stack']['x'], positions['source_stack']['z_transfer']), 'action': '6. Lift wellplate to transfer height', 'color': 'cyan', 'marker': '^', 'phase': 'grab'},
        
        # Navigate to pipetting area via transfer_stack waypoint
        {'pos': (positions['transfer_stack']['x'], positions['source_stack']['z_transfer']), 'action': '7. Move to transfer stack waypoint (X only)', 'color': 'purple', 'marker': 'D', 'phase': 'grab'},
        {'pos': (positions['transfer_stack']['x'], positions['pipetting_area']['z_transfer']), 'action': '8. Move to pipetting transfer height (Z only)', 'color': 'purple', 'marker': 's', 'phase': 'grab'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '9. Move to pipetting area X position', 'color': 'blue', 'marker': 's', 'phase': 'grab'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_release']), 'action': '10. Lower wellplate to pipetting area', 'color': 'blue', 'marker': 'v', 'phase': 'grab'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '11. Return to transfer height', 'color': 'blue', 'marker': 's', 'phase': 'grab'},
        {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '12. Move to safe height (grab complete)', 'color': 'blue', 'marker': 'o', 'phase': 'grab'},
        
        # === PIPETTING/EXPERIMENTS HAPPEN HERE ===
        {'pos': (positions['pipetting_area']['x'], positions['max_safe_height']), 'action': '13. *** PIPETTING & EXPERIMENTS ***', 'color': 'gold', 'marker': '*', 'phase': 'work'},
        
        # === DISCARD WELLPLATE SEQUENCE ===
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '14. Move to pipetting transfer height', 'color': 'blue', 'marker': 's', 'phase': 'discard'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_grab']), 'action': '15. Move down to grab used wellplate', 'color': 'blue', 'marker': 'v', 'phase': 'discard'},
        {'pos': (positions['pipetting_area']['x'], positions['pipetting_area']['z_transfer']), 'action': '16. Lift used wellplate to transfer height', 'color': 'blue', 'marker': '^', 'phase': 'discard'},
        
        # Navigate to waste stack via transfer_stack waypoint
        {'pos': (positions['transfer_stack']['x'], positions['pipetting_area']['z_transfer']), 'action': '17. Move to transfer stack waypoint (X only)', 'color': 'purple', 'marker': 'D', 'phase': 'discard'},
        {'pos': (positions['transfer_stack']['x'], positions['waste_stack']['z_transfer']), 'action': '18. Move to waste transfer height (Z only)', 'color': 'purple', 'marker': 's', 'phase': 'discard'},
        {'pos': (positions['waste_stack']['x'], positions['waste_stack']['z_transfer']), 'action': '19. Move to waste stack X position', 'color': 'red', 'marker': 's', 'phase': 'discard'},
        {'pos': (positions['waste_stack']['x'], 83000), 'action': '20. Lower wellplate to waste stack', 'color': 'red', 'marker': 'v', 'phase': 'discard'},
        {'pos': (positions['waste_stack']['x'], positions['waste_stack']['z_transfer']), 'action': '21. Return to transfer height', 'color': 'red', 'marker': 's', 'phase': 'discard'},
        {'pos': (positions['waste_stack']['x'], positions['max_safe_height']), 'action': '22. Move to safe height', 'color': 'red', 'marker': 'o', 'phase': 'discard'},
        {'pos': (0, positions['max_safe_height']), 'action': '23. Return to home position (complete)', 'color': 'green', 'marker': 'H', 'phase': 'discard'},
    ]
    
    # Plot the movement path with different colors for each phase
    grab_coords = [(step['pos'][0], step['pos'][1]) for step in movement_sequence if step['phase'] == 'grab']
    work_coords = [(step['pos'][0], step['pos'][1]) for step in movement_sequence if step['phase'] == 'work']
    discard_coords = [(step['pos'][0], step['pos'][1]) for step in movement_sequence if step['phase'] == 'discard']
    
    # Draw path lines for each phase
    if grab_coords:
        grab_x, grab_z = zip(*grab_coords)
        ax.plot(grab_x, grab_z, 'b--', alpha=0.7, linewidth=2, label='Grab Phase Path')
    
    if discard_coords:
        discard_x, discard_z = zip(*discard_coords)
        ax.plot(discard_x, discard_z, 'r--', alpha=0.7, linewidth=2, label='Discard Phase Path')
    
    # Plot each movement step
    for i, step in enumerate(movement_sequence):
        x, z = step['pos']
        ax.scatter(x, z, c=step['color'], marker=step['marker'], s=120, 
                  edgecolors='black', linewidth=1, alpha=0.9, zorder=3)
        
        # Add step numbers
        ax.annotate(f"{i+1}", (x, z), xytext=(5, 5), textcoords='offset points', 
                   fontsize=7, fontweight='bold', color='black')
    
    # Add location labels and areas
    locations = {
        'Source Stack': (positions['source_stack']['x'], positions['source_stack']['z_grab']),
        'Pipetting Area': (positions['pipetting_area']['x'], positions['pipetting_area']['z_grab']),
        'Waste Stack': (positions['waste_stack']['x'], positions['waste_stack']['z_grab']),
        'Transfer Waypoint': (positions['transfer_stack']['x'], 50000),
        'Home Position': (0, 10000),
    }
    
    for name, (x, z) in locations.items():
        ax.text(x, z, name, ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add area rectangles
    # Source stack area
    source_rect = patches.Rectangle(
        (positions['source_stack']['x']-5000, positions['source_stack']['z_release']-5000), 
        10000, 15000, linewidth=2, edgecolor='cyan', facecolor='lightcyan', alpha=0.3)
    ax.add_patch(source_rect)
    
    # Pipetting area
    pipetting_rect = patches.Rectangle(
        (positions['pipetting_area']['x']-10000, positions['pipetting_area']['z_release']-5000), 
        20000, 15000, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ax.add_patch(pipetting_rect)
    
    # Waste stack area
    waste_rect = patches.Rectangle(
        (positions['waste_stack']['x']-5000, positions['waste_stack']['z_release']-5000), 
        10000, 15000, linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.3)
    ax.add_patch(waste_rect)
    
    # Transfer waypoint area
    transfer_rect = patches.Rectangle(
        (positions['transfer_stack']['x']-5000, 20000), 
        10000, 40000, linewidth=2, edgecolor='purple', facecolor='plum', alpha=0.3)
    ax.add_patch(transfer_rect)
    
    # Formatting
    ax.set_xlim(-5000, 140000)
    ax.set_ylim(-5000, 95000)
    ax.set_xlabel('X Position (encoder counts)', fontsize=12)
    ax.set_ylabel('Z Position (encoder counts)\n(0 = highest, larger values = lower)', fontsize=12)
    ax.set_title('Wellplate Movement Path: Grab New Wellplate → Use → Discard', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Z=0 is highest position
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start/End'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='Transfer Height'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='orange', markersize=10, label='Grab/Release'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='Waypoint'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Work Phase'),
        plt.Line2D([0], [0], marker='H', color='w', markerfacecolor='green', markersize=10, label='Home'),
        plt.Line2D([0], [0], linestyle='--', color='blue', alpha=0.7, linewidth=2, label='Grab Phase'),
        plt.Line2D([0], [0], linestyle='--', color='red', alpha=0.7, linewidth=2, label='Discard Phase'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add operation phases
    ax.text(0.02, 0.98, 'Phase 1: Grab New Wellplate (steps 1-12)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    ax.text(0.02, 0.90, 'Phase 2: Pipetting & Experiments (step 13)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='gold'))
    ax.text(0.02, 0.82, 'Phase 3: Discard Used Wellplate (steps 14-23)', transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    plt.tight_layout()
    return fig, movement_sequence

def print_grab_discard_summary(movement_sequence):
    """Print a text summary of the grab and discard movement sequence"""
    print("\n" + "="*80)
    print("GRAB NEW WELLPLATE → DISCARD USED WELLPLATE MOVEMENT SEQUENCE")
    print("="*80)
    print(f"Total steps: {len(movement_sequence)}")
    
    phases = {'grab': [], 'work': [], 'discard': []}
    for i, step in enumerate(movement_sequence):
        phases[step['phase']].append((i+1, step))
    
    for phase_name, steps in phases.items():
        print(f"\n{phase_name.upper()} PHASE ({len(steps)} steps):")
        print("-" * 40)
        
        for step_num, step in steps:
            x, z = step['pos']
            print(f"{step_num:2d}. {step['action']}")
            print(f"    Position: X={x:,}, Z={z:,}")
            
            # Calculate movement to next step (if not last)
            if step_num < len(movement_sequence):
                next_step = movement_sequence[step_num]  # step_num is 1-indexed, list is 0-indexed
                next_x, next_z = next_step['pos']
                dx = abs(next_x - x)
                dz = abs(next_z - z)
                if dx > 0 or dz > 0:
                    print(f"    Movement to next: ΔX={dx:,}, ΔZ={dz:,}")
            print()
    
    print("KEY OBSERVATIONS:")
    print("- Complete wellplate lifecycle: source → pipetting → waste")
    print("- Uses transfer_stack waypoint for safe routing in both directions")
    print("- Maintains consistent transfer heights during horizontal movements")
    print("- Stack height calculations for dynamic Z positioning")
    print("- Returns to home position after discard operation")

if __name__ == "__main__":
    # Create and show the visualization
    fig, movement_sequence = create_grab_discard_path_visualization()
    
    # Print movement summary
    print_grab_discard_summary(movement_sequence)
    
    # Show the plot
    plt.show()
    
    # Save the plot
    plt.savefig('grab_discard_wellplate_movement_path.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'grab_discard_wellplate_movement_path.png'")