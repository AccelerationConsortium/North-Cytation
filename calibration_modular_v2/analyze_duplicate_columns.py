#!/usr/bin/env python3
"""
Analyze duplicate columns in the master dataset.
"""

try:
    import pandas as pd
    import numpy as np
    from pathlib import Path
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure pandas and numpy are installed")
    exit(1)

def find_duplicate_columns(csv_file):
    """Analyze a CSV file to find columns that contain the same data."""
    print(f"Analyzing: {csv_file}")
    print("=" * 80)
    
    # Load the data
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    # Group columns by data patterns
    duplicate_groups = []
    processed_columns = set()
    
    for col1 in df.columns:
        if col1 in processed_columns:
            continue
            
        # Find columns with identical or very similar data
        similar_columns = [col1]
        
        for col2 in df.columns:
            if col2 == col1 or col2 in processed_columns:
                continue
                
            # Check if columns contain the same data
            if are_columns_duplicates(df[col1], df[col2]):
                similar_columns.append(col2)
        
        if len(similar_columns) > 1:
            duplicate_groups.append(similar_columns)
            processed_columns.update(similar_columns)
    
    # Report findings
    print(f"\nFound {len(duplicate_groups)} groups of duplicate columns:")
    print("-" * 60)
    
    total_duplicates = 0
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nGroup {i}: {len(group)} columns")
        for col in group:
            sample_values = df[col].dropna().head(3).tolist()
            print(f"  • {col}: {sample_values}")
        total_duplicates += len(group) - 1  # -1 because we keep one from each group
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Duplicate groups: {len(duplicate_groups)}")
    print(f"  Columns that can be removed: {total_duplicates}")
    print(f"  Columns after cleanup: {len(df.columns) - total_duplicates}")
    
    # Suggest which columns to keep vs remove
    print(f"\nRECOMMENDED ACTIONS:")
    print("-" * 40)
    
    keep_columns = []
    remove_columns = []
    
    for group in duplicate_groups:
        # Priority: prefer columns WITHOUT metadata_ prefix
        def column_priority(col):
            if col.startswith('metadata_'):
                return (1, col)  # Lower priority - drop these if alternatives exist
            else:
                return (0, col)  # Higher priority - keep these
        
        sorted_group = sorted(group, key=column_priority)
        keep = sorted_group[0]
        remove = sorted_group[1:]
        
        keep_columns.append(keep)
        remove_columns.extend(remove)
        
        print(f"\nKeep: {keep}")
        print(f"Remove: {', '.join(remove)}")
    
    return keep_columns, remove_columns, duplicate_groups

def are_columns_duplicates(col1, col2, tolerance=0.001):
    """Check if two columns contain the same data (accounting for minor differences)."""
    
    # Handle completely missing data
    if col1.isna().all() and col2.isna().all():
        return True
    
    # If one is all NaN and other has data, not duplicates
    if col1.isna().all() or col2.isna().all():
        return False
    
    # Get non-null values
    mask = col1.notna() & col2.notna()
    if mask.sum() == 0:
        return True  # Both all NaN in overlapping positions
    
    clean_col1 = col1[mask]
    clean_col2 = col2[mask]
    
    # If different number of non-null values, not duplicates
    if len(clean_col1) != len(clean_col2):
        return False
    
    # For numeric columns, check if values are nearly identical
    if pd.api.types.is_numeric_dtype(clean_col1) and pd.api.types.is_numeric_dtype(clean_col2):
        try:
            return np.allclose(clean_col1, clean_col2, rtol=tolerance, atol=tolerance, equal_nan=True)
        except:
            return False
    
    # For string/categorical columns, check exact matches
    try:
        return clean_col1.equals(clean_col2)
    except:
        return False

if __name__ == "__main__":
    # Find the most recent master dataset
    script_dir = Path(__file__).parent
    master_data_dir = script_dir / "master_data"
    
    # Also check if we're running from parent directory
    if not master_data_dir.exists():
        parent_master_data = script_dir.parent / "calibration_modular_v2" / "master_data"
        if parent_master_data.exists():
            master_data_dir = parent_master_data
    
    if not master_data_dir.exists():
        print(f"No master_data directory found!")
        print(f"Looked in: {script_dir / 'master_data'}")
        if 'parent_master_data' in locals():
            print(f"Also looked in: {parent_master_data}")
        print(f"Current directory: {Path.cwd()}")
        exit(1)
    
    # Find most recent measurements file
    measurement_files = list(master_data_dir.glob("master_measurements_*.csv"))
    if not measurement_files:
        print("No master measurements file found!")
        exit(1)
    
    latest_file = max(measurement_files, key=lambda p: p.stat().st_mtime)
    
    keep_columns, remove_columns, duplicate_groups = find_duplicate_columns(latest_file)
    
    print(f"\n" + "=" * 80)
    print("COLUMN CLEANUP SCRIPT:")
    print("=" * 80)
    print("# Remove these duplicate columns:")
    for col in remove_columns:
        print(f"# df.drop('{col}', axis=1, inplace=True)")
    
    print(f"\n# Total columns to remove: {len(remove_columns)}")
    print(f"# This will reduce from {len(keep_columns) + len(remove_columns)} to {len(keep_columns)} columns")