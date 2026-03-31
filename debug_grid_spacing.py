#!/usr/bin/env python3
"""
Debug Grid Spacing Analysis

Systematically investigates differences between generated grid and actual rack_pip positions
to determine what's wrong with the grid generation algorithm.
"""

import sys
import math
sys.path.append('.')

from master_usdl_coordinator import Lash_E

# Generated grid (wrong)
generated_grid = [
    [2196, 6824, 35310, 18300], [2097, 7788, 34740, 18300], [2000, 8931, 34548, 18300], 
    [1902, 10278, 34735, 18300], [1800, 11883, 35332, 18300], [1687, 13884, 36453, 18300], 
    [2198, 5557, 32869, 18300], [2096, 6556, 32275, 18300], [1998, 7715, 32086, 18300], 
    [1901, 9050, 32278, 18300], [1802, 10595, 32849, 18300], [1698, 12436, 33855, 18300], 
    [2184, 4428, 30281, 18300], [2079, 5480, 29719, 18300], [1981, 6677, 29594, 18300], 
    [1885, 8024, 29851, 18300], [1790, 9550, 30468, 18300], [1691, 11315, 31467, 18300], 
    [2150, 3441, 27473, 18300], [2045, 4562, 27026, 18300], [1948, 5808, 27038, 18300], 
    [1855, 7184, 27426, 18300], [1764, 8712, 28149, 18300], [1670, 10443, 29212, 18300], 
    [2093, 2615, 24379, 18300], [1991, 3814, 24172, 18300], [1898, 5115, 24413, 18300], 
    [1811, 6525, 24998, 18300], [1725, 8067, 25882, 18300], [1637, 9785, 27063, 18300], 
    [2008, 1985, 20979, 18300], [1915, 3258, 21173, 18300], [1832, 4610, 21742, 18300], 
    [1752, 6053, 22591, 18300], [1673, 7610, 23683, 18300], [1592, 9326, 25021, 18300], 
    [1895, 1593, 17352, 18300], [1820, 2921, 18107, 18300], [1750, 4309, 19089, 18300], 
    [1681, 5774, 20250, 18300], [1611, 7343, 21582, 18300], [581, 33440, 47248, 18300], 
    [209, 41022, 52877, 18300], [264, 39677, 51612, 18300], [321, 38278, 50273, 18300], 
    [380, 36806, 48849, 18300], [443, 35233, 47318, 18300], [511, 33516, 45638, 18300]
]

# Actual rack_pip (correct)
actual_rack_pip = [
    [2184, 6798, 35307, 16281], [2106, 7484, 34689, 16281], [2029, 8313, 34375, 16281], 
    [1951, 9292, 34359, 16281], [1873, 10437, 34648, 16281], [1792, 11787, 35268, 16281], 
    [2187, 5539, 32865, 16281], [2105, 6251, 32222, 16281], [2026, 7100, 31907, 16281], 
    [1949, 8084, 31902, 16281], [1872, 9213, 32194, 16281], [1794, 10512, 32789, 16281], 
    [2175, 4373, 30276, 16281], [2091, 5128, 29646, 16281], [2011, 6012, 29377, 16281], 
    [1934, 7020, 29430, 16281], [1859, 8156, 29776, 16281], [1783, 9438, 30407, 16281], 
    [2148, 3294, 27454, 16281], [2063, 4108, 26902, 16281], [1983, 5042, 26742, 16281], 
    [1908, 6088, 26910, 16281], [1835, 7247, 27363, 16281], [1762, 8531, 28079, 16281], 
    [2101, 2305, 24304, 16281], [2017, 3196, 23935, 16281], [1940, 4192, 23971, 16281], 
    [1868, 5286, 24324, 16281], [1799, 6477, 24939, 16281], [1731, 7776, 25788, 16281], 
    [2029, 1423, 20730, 16281], [1951, 2405, 20709, 16281], [1881, 3473, 21058, 16281], 
    [1816, 4620, 21677, 16281], [1752, 5847, 22511, 16281], [1689, 7167, 23537, 16281], 
    [1927, 678, 16673, 16281], [1864, 1762, 17234, 16281], [1806, 2903, 18032, 16281], 
    [1750, 4101, 18999, 16281], [1695, 5364, 20105, 16281], [1639, 6706, 21344, 16281], 
    [1794, 121, 12212, 16281], [1756, 1300, 13607, 16281], [1716, 2504, 14975, 16281], 
    [1673, 3744, 16353, 16281], [1628, 5036, 17767, 16281], [1580, 6396, 19242, 16281]
]

def analyze_grid_differences():
    """Main analysis function"""
    
    print("=" * 80)
    print("GRID SPACING DEBUG ANALYSIS")
    print("=" * 80)
    
    # Initialize robot for FK calculations
    try:
        print("Connecting to robot...")
        lash_e = Lash_E("status/experiment_vials.csv", simulate=False)
        robot = lash_e.nr_robot
        print("Robot connected successfully")
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        print("Running analysis without FK (joint differences only)")
        robot = None
    
    # 1. Joint space differences analysis
    print("\n" + "=" * 60)
    print("1. JOINT SPACE DIFFERENCES ANALYSIS")
    print("=" * 60)
    
    max_positions = min(len(generated_grid), len(actual_rack_pip))
    total_gripper_error = 0
    total_elbow_error = 0
    total_shoulder_error = 0
    
    print(f"{'Pos':<4} {'Gripper Δ':<12} {'Elbow Δ':<12} {'Shoulder Δ':<12} {'Generated':<30} {'Actual':<30}")
    print("-" * 110)
    
    for i in range(min(12, max_positions)):  # Show first 12 positions
        gen = generated_grid[i]
        act = actual_rack_pip[i]
        
        gripper_diff = gen[0] - act[0]
        elbow_diff = gen[1] - act[1]
        shoulder_diff = gen[2] - act[2]
        
        total_gripper_error += abs(gripper_diff)
        total_elbow_error += abs(elbow_diff)
        total_shoulder_error += abs(shoulder_diff)
        
        print(f"{i:<4} {gripper_diff:<+12} {elbow_diff:<+12} {shoulder_diff:<+12} "
              f"{gen[:3]!s:<30} {act[:3]!s:<30}")
    
    print(f"\nAverage absolute errors (first 12 positions):")
    print(f"  Gripper: {total_gripper_error/12:.1f} counts")
    print(f"  Elbow:   {total_elbow_error/12:.1f} counts") 
    print(f"  Shoulder: {total_shoulder_error/12:.1f} counts")
    
    # 2. Cartesian space analysis (if robot available)
    if robot and hasattr(robot, 'n9_fk'):
        print("\n" + "=" * 60)
        print("2. CARTESIAN SPACE ANALYSIS")
        print("=" * 60)
        
        print("\nConverting first 12 positions to X,Y coordinates...")
        print(f"{'Pos':<4} {'Generated X,Y':<20} {'Actual X,Y':<20} {'X Δ':<10} {'Y Δ':<10} {'Distance Δ':<12}")
        print("-" * 86)
        
        for i in range(min(12, max_positions)):
            try:
                # Generated position FK
                gen = generated_grid[i]
                gen_x, gen_y, _ = robot.n9_fk(gen[0], gen[1], gen[2])
                
                # Actual position FK
                act = actual_rack_pip[i]
                act_x, act_y, _ = robot.n9_fk(act[0], act[1], act[2])
                
                # Calculate differences
                dx = gen_x - act_x
                dy = gen_y - act_y
                distance_error = math.sqrt(dx*dx + dy*dy)
                
                print(f"{i:<4} ({gen_x:6.1f},{gen_y:6.1f}){'':<6} ({act_x:6.1f},{act_y:6.1f}){'':<6} "
                      f"{dx:+6.1f}{'':<4} {dy:+6.1f}{'':<4} {distance_error:8.1f}mm")
                
            except Exception as e:
                print(f"{i:<4} FK Error: {e}")
        
        # 3. Actual spacing analysis
        print("\n" + "=" * 60)
        print("3. ACTUAL RACK SPACING ANALYSIS")
        print("=" * 60)
        
        try:
            # Analyze first few positions to determine actual spacing
            positions_to_analyze = [0, 1, 6]  # origin, Y+1, X+1 
            coordinates = []
            
            for idx in positions_to_analyze:
                pos = actual_rack_pip[idx]
                x, y, _ = robot.n9_fk(pos[0], pos[1], pos[2])
                coordinates.append((x, y))
                print(f"Position {idx}: joints={pos[:3]} → X={x:.2f}mm, Y={y:.2f}mm")
            
            if len(coordinates) >= 3:
                x0, y0 = coordinates[0]  # Position 0 (origin)
                x1, y1 = coordinates[1]  # Position 1 (Y+1 step)
                x6, y6 = coordinates[2]  # Position 6 (X+1 step)
                
                # Calculate actual spacing vectors
                y_step_dx = x1 - x0
                y_step_dy = y1 - y0  
                y_spacing = math.sqrt(y_step_dx*y_step_dx + y_step_dy*y_step_dy)
                
                x_step_dx = x6 - x0
                x_step_dy = y6 - y0
                x_spacing = math.sqrt(x_step_dx*x_step_dx + x_step_dy*x_step_dy)
                
                print(f"\nACTUAL GRID SPACING:")
                print(f"  Y-step (0→1): Δx={y_step_dx:+6.2f}mm, Δy={y_step_dy:+6.2f}mm, spacing={y_spacing:.2f}mm")
                print(f"  X-step (0→6): Δx={x_step_dx:+6.2f}mm, Δy={x_step_dy:+6.2f}mm, spacing={x_spacing:.2f}mm")
                
                print(f"\nCOMPARISON TO ASSUMED 26mm:")
                print(f"  Y-spacing: {y_spacing:.2f}mm vs 26.0mm (error: {y_spacing-26:+.2f}mm)")
                print(f"  X-spacing: {x_spacing:.2f}mm vs 26.0mm (error: {x_spacing-26:+.2f}mm)")
                
                print(f"\nCORRECTED GRID PARAMETERS SHOULD BE:")
                print(f"  x_offset: {x_step_dx:.2f}mm (not 26.0mm)")
                print(f"  y_offset: {y_step_dy:.2f}mm (not 26.0mm)")
                
        except Exception as e:
            print(f"Spacing analysis failed: {e}")
    
    # 4. Pattern analysis
    print("\n" + "=" * 60)
    print("4. ERROR PATTERN ANALYSIS")
    print("=" * 60)
    
    print("Analyzing joint error patterns...")
    
    # Column-by-column analysis (positions 0-5, 6-11, 12-17, etc.)
    n_cols = 8
    n_rows = 6
    
    for col in range(min(3, n_cols)):  # Analyze first 3 columns
        print(f"\nColumn {col} (positions {col*n_rows} to {col*n_rows + n_rows-1}):")
        
        col_gripper_errors = []
        col_elbow_errors = []
        
        for row in range(n_rows):
            pos_idx = col * n_rows + row
            if pos_idx >= max_positions:
                break
                
            gen = generated_grid[pos_idx]
            act = actual_rack_pip[pos_idx]
            
            gripper_err = gen[0] - act[0]
            elbow_err = gen[1] - act[1]
            
            col_gripper_errors.append(gripper_err)
            col_elbow_errors.append(elbow_err)
            
            print(f"  Row {row}: gripper_err={gripper_err:+4d}, elbow_err={elbow_err:+4d}")
        
        if col_gripper_errors:
            avg_gripper = sum(col_gripper_errors) / len(col_gripper_errors)
            avg_elbow = sum(col_elbow_errors) / len(col_elbow_errors)
            print(f"  Column {col} averages: gripper={avg_gripper:+6.1f}, elbow={avg_elbow:+6.1f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if robot:
        print("Check the 'CORRECTED GRID PARAMETERS' section above for the right spacing values!")
    else:
        print("Reconnect robot for full Cartesian analysis including correct spacing calculation.")

if __name__ == "__main__":
    try:
        analyze_grid_differences()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()