#!/usr/bin/env python3
"""
Parameter Corrections Analyzer

Analyzes how overaspirate parameters have changed over time for different
liquid and volume combinations. Shows trends for combinations with sufficient
data points (count >= 5).

Usage: python parameter_corrections_analyzer.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def load_and_prepare_data(csv_file):
    """Load CSV data and prepare for analysis"""
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    
    # Create a liquid-volume combination identifier
    df['liquid_volume'] = df['liquid_type'] + f'_' + df['target_volume_ml'].astype(str) + 'mL'
    
    # Check for environmental data columns
    env_columns = ['temp_c', 'humidity_pct', 'pressure_pa']
    has_env_data = any(col in df.columns for col in env_columns)
    
    if has_env_data:
        print("✓ Environmental data found in corrections log")
        # Show environmental data summary
        for col in env_columns:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                if valid_count > 0:
                    val_range = f"{df[col].min():.1f} - {df[col].max():.1f}"
                    units = {"temp_c": "°C", "humidity_pct": "%", "pressure_pa": "Pa"}
                    print(f"  {col}: {valid_count}/{len(df)} records, range: {val_range} {units.get(col, '')}")
    else:
        print("ℹ No environmental data in corrections log (older format)")
    
    return df

def get_frequent_combinations(df, min_count=5):
    """Get liquid-volume combinations with at least min_count data points"""
    combo_counts = df.groupby(['liquid_type', 'target_volume_ml']).size()
    frequent_combos = combo_counts[combo_counts >= min_count].index.tolist()
    
    print(f"\nLiquid-Volume combinations with >= {min_count} data points:")
    print("=" * 60)
    for liquid, volume in frequent_combos:
        count = combo_counts[(liquid, volume)]
        print(f"{liquid:>10} @ {volume:>6} mL: {count:>3} corrections")
    
    return frequent_combos

def plot_overaspirate_trends(df, frequent_combos):
    """Plot overaspirate trends over time for frequent combinations"""
    
    # Create subplots
    n_combos = len(frequent_combos)
    n_cols = 3
    n_rows = (n_combos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Overaspirate Parameter Evolution Over Time', fontsize=16, y=0.98)
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (liquid, volume) in enumerate(frequent_combos):
        # Filter data for this combination
        combo_data = df[(df['liquid_type'] == liquid) & (df['target_volume_ml'] == volume)].copy()
        combo_data = combo_data.sort_values('datetime')
        
        ax = axes[i]
        
        # Plot old and new overaspirate values
        ax.plot(combo_data['datetime'], combo_data['old_overaspirate_ml'] * 1000, 
               'o-', label='Old Overaspirate', alpha=0.7, color='red')
        ax.plot(combo_data['datetime'], combo_data['new_overaspirate_ml'] * 1000, 
               's-', label='New Overaspirate', alpha=0.8, color='blue')
        
        ax.set_title(f'{liquid} @ {volume} mL\n({len(combo_data)} corrections)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Overaspirate (uL)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_liquid_comprehensive(df, liquid_type):
    """Create comprehensive analysis for a specific liquid"""
    liquid_data = df[df['liquid_type'] == liquid_type].copy()
    
    if liquid_data.empty:
        print(f"No {liquid_type} data found!")
        return None
    
    liquid_volumes = liquid_data['target_volume_ml'].unique()
    print(f"\n{liquid_type.upper()} volumes found: {sorted(liquid_volumes)} mL")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Comprehensive {liquid_type.upper()} Parameter Analysis', fontsize=16)
    
    # 1. Overaspirate evolution over time (all volumes)
    for volume in sorted(liquid_volumes):
        vol_data = liquid_data[liquid_data['target_volume_ml'] == volume].sort_values('datetime')
        ax1.plot(vol_data['datetime'], vol_data['new_overaspirate_ml'] * 1000, 
               'o-', label=f'{volume} mL', alpha=0.8)
    
    ax1.set_title(f'{liquid_type.upper()}: New Overaspirate vs Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('New Overaspirate (uL)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Final overaspirate value distribution
    ax2.boxplot([liquid_data[liquid_data['target_volume_ml'] == vol]['new_overaspirate_ml'] * 1000
                for vol in sorted(liquid_volumes)], 
               labels=[f'{vol} mL' for vol in sorted(liquid_volumes)])
    ax2.set_title(f'{liquid_type.upper()}: Final Overaspirate Value Distribution')
    ax2.set_ylabel('Final Overaspirate (uL)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correction direction (positive vs negative)
    correction_summary = []
    for volume in sorted(liquid_volumes):
        vol_data = liquid_data[liquid_data['target_volume_ml'] == volume]
        pos_corrections = (vol_data['correction_ul'] > 0).sum()
        neg_corrections = (vol_data['correction_ul'] < 0).sum()
        zero_corrections = (vol_data['correction_ul'] == 0).sum()
        correction_summary.append([pos_corrections, neg_corrections, zero_corrections])
    
    correction_array = np.array(correction_summary)
    x_pos = np.arange(len(liquid_volumes))
    
    ax3.bar(x_pos, correction_array[:, 0], label='Increase (+)', alpha=0.8, color='green')
    ax3.bar(x_pos, correction_array[:, 1], bottom=correction_array[:, 0], 
           label='Decrease (-)', alpha=0.8, color='red')
    ax3.bar(x_pos, correction_array[:, 2], 
           bottom=correction_array[:, 0] + correction_array[:, 1],
           label='No Change (0)', alpha=0.8, color='gray')
    
    ax3.set_title(f'{liquid_type.upper()}: Correction Direction by Volume')
    ax3.set_xlabel('Volume (mL)')
    ax3.set_ylabel('Number of Corrections')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{vol}' for vol in sorted(liquid_volumes)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final overaspirate values (most recent for each volume)
    final_values = []
    final_counts = []
    for volume in sorted(liquid_volumes):
        vol_data = liquid_data[liquid_data['target_volume_ml'] == volume].sort_values('datetime')
        if not vol_data.empty:
            final_values.append(vol_data.iloc[-1]['new_overaspirate_ml'] * 1000)
            final_counts.append(len(vol_data))
        else:
            final_values.append(0)
            final_counts.append(0)
    
    bars = ax4.bar(range(len(liquid_volumes)), final_values, alpha=0.8)
    ax4.set_title(f'{liquid_type.upper()}: Current Overaspirate Values')
    ax4.set_xlabel('Volume (mL)')
    ax4.set_ylabel('Current Overaspirate (uL)')
    ax4.set_xticks(range(len(liquid_volumes)))
    ax4.set_xticklabels([f'{vol}' for vol in sorted(liquid_volumes)])
    ax4.grid(True, alpha=0.3)
    
    # Add count annotations on bars
    for i, (bar, count) in enumerate(zip(bars, final_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'n={count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_environmental_correlations(df):
    """Plot environmental correlations with parameter corrections if data is available"""
    env_columns = ['temp_c', 'humidity_pct', 'pressure_pa']
    available_env = [col for col in env_columns if col in df.columns and df[col].notna().sum() > 3]
    
    if not available_env:
        return None
    
    print(f"\nGenerating environmental correlation analysis...")
    
    # Filter to rows with environmental data
    env_df = df.dropna(subset=available_env, how='all')
    if len(env_df) < 3:
        print(f"Insufficient environmental data ({len(env_df)} rows)")
        return None
    
    n_envs = len(available_env)
    fig, axes = plt.subplots(2, n_envs, figsize=(5*n_envs, 10))
    if n_envs == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Environmental Factor vs Parameter Corrections', fontsize=16)
    
    labels = {'temp_c': 'Temperature (°C)', 'humidity_pct': 'Humidity (%)', 'pressure_pa': 'Pressure (Pa)'}
    
    for i, env_col in enumerate(available_env):
        # Top plot: correction magnitude vs environmental factor
        ax1 = axes[0, i]
        ax1.scatter(env_df[env_col], abs(env_df['correction_ul']), alpha=0.6, c='blue')
        ax1.set_xlabel(labels[env_col])
        ax1.set_ylabel('|Correction| (uL)')
        ax1.set_title(f'Correction Magnitude vs {labels[env_col]}')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(env_df) > 5:
            z = np.polyfit(env_df[env_col].dropna(), abs(env_df['correction_ul'].dropna()), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(env_df[env_col].min(), env_df[env_col].max(), 100)
            ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Bottom plot: correction direction vs environmental factor  
        ax2 = axes[1, i]
        increases = env_df[env_df['correction_ul'] > 0]
        decreases = env_df[env_df['correction_ul'] < 0]
        
        if len(increases) > 0:
            ax2.scatter(increases[env_col], increases['correction_ul'], 
                       alpha=0.6, c='green', label='Increases', s=50)
        if len(decreases) > 0:
            ax2.scatter(decreases[env_col], decreases['correction_ul'], 
                       alpha=0.6, c='red', label='Decreases', s=50)
        
        ax2.set_xlabel(labels[env_col])
        ax2.set_ylabel('Correction (uL)')
        ax2.set_title(f'Correction Direction vs {labels[env_col]}')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.legend()
    
    plt.tight_layout()
    return fig

def generate_summary_stats(df, frequent_combos):
    """Generate summary statistics for frequent combinations"""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for liquid, volume in frequent_combos:
        combo_data = df[(df['liquid_type'] == liquid) & (df['target_volume_ml'] == volume)]
        
        print(f"\n{liquid.upper()} @ {volume} mL ({len(combo_data)} corrections)")
        print("-" * 50)
        
        # Current vs initial overaspirate
        initial_overaspirate = combo_data.sort_values('datetime').iloc[0]['old_overaspirate_ml'] * 1000
        final_overaspirate = combo_data.sort_values('datetime').iloc[-1]['new_overaspirate_ml'] * 1000
        
        print(f"Initial overaspirate: {initial_overaspirate:6.2f} uL")
        print(f"Final overaspirate:   {final_overaspirate:6.2f} uL")
        print(f"Net change:           {final_overaspirate - initial_overaspirate:+6.2f} uL")
        
        # Correction statistics
        corrections = combo_data['correction_ul']
        print(f"Average correction:   {corrections.mean():+6.2f} uL")
        print(f"Correction std dev:   {corrections.std():6.2f} uL")
        print(f"Max correction:       {corrections.max():+6.2f} uL")
        print(f"Min correction:       {corrections.min():+6.2f} uL")
        
        # Direction analysis
        increases = (corrections > 0).sum()
        decreases = (corrections < 0).sum()
        no_change = (corrections == 0).sum()
        
        print(f"Increases/Decreases:  {increases}/{decreases}/{no_change} (+/-/0)")
        
        # Environmental data summary (if available)
        if any(col in combo_data.columns for col in ['temp_c', 'humidity_pct', 'pressure_pa']):
            env_summary = []
            if 'temp_c' in combo_data.columns and combo_data['temp_c'].notna().sum() > 0:
                temp_data = combo_data['temp_c'].dropna()
                env_summary.append(f"Temp: {temp_data.mean():.1f}°C ({temp_data.min():.1f}-{temp_data.max():.1f})")
            if 'humidity_pct' in combo_data.columns and combo_data['humidity_pct'].notna().sum() > 0:
                humid_data = combo_data['humidity_pct'].dropna()
                env_summary.append(f"Humidity: {humid_data.mean():.1f}% ({humid_data.min():.1f}-{humid_data.max():.1f})")
            if 'pressure_pa' in combo_data.columns and combo_data['pressure_pa'].notna().sum() > 0:
                press_data = combo_data['pressure_pa'].dropna()
                env_summary.append(f"Pressure: {press_data.mean():.0f} Pa")
            
            if env_summary:
                print(f"Environment range:    {' | '.join(env_summary)}")

def main():
    """Main analysis function"""
    csv_file = Path("pipetting_data/parameter_corrections.csv")
    
    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        print("Please ensure the CSV file is in the correct location.")
        return
    
    print("Loading parameter corrections data...")
    df = load_and_prepare_data(csv_file)
    
    print(f"Loaded {len(df)} parameter corrections from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Show data summary by liquid
    print(f"\nLIQUID DATA SUMMARY:")
    print("=" * 40)
    for liquid in sorted(df['liquid_type'].unique()):
        liquid_data = df[df['liquid_type'] == liquid]
        volumes = sorted(liquid_data['target_volume_ml'].unique())
        print(f"{liquid.upper():>10}: {len(liquid_data):>3} corrections across volumes {volumes}")
    
    print(f"\nTotal volumes: {sorted(df['target_volume_ml'].unique())} mL")
    
    # Get combinations with sufficient data
    frequent_combos = get_frequent_combinations(df, min_count=5)
    
    if not frequent_combos:
        print("No combinations found with >= 5 data points!")
        return
    
    # Generate summary statistics
    generate_summary_stats(df, frequent_combos)
    
    # Create plots
    print("\nGenerating trend plots...")
    
    # Plot trends for frequent combinations
    fig1 = plot_overaspirate_trends(df, frequent_combos)
    fig1.savefig('parameter_corrections_trends.png', dpi=300, bbox_inches='tight')
    print("Saved: parameter_corrections_trends.png")
    
    # Special comprehensive analysis for liquids with sufficient data
    liquid_counts = df['liquid_type'].value_counts()
    print(f"\nLiquid data counts: {dict(liquid_counts)}")
    
    for liquid in liquid_counts.index:
        if liquid_counts[liquid] >= 3:  # Need at least 3 points for meaningful analysis
            print(f"\nGenerating comprehensive analysis for {liquid.upper()}...")
            fig = plot_liquid_comprehensive(df, liquid)
            if fig:
                fig.savefig(f'{liquid}_parameter_analysis.png', dpi=300, bbox_inches='tight')
                print(f"Saved: {liquid}_parameter_analysis.png")
    
    # Environmental correlation analysis (if data available)
    env_fig = plot_environmental_correlations(df)
    if env_fig:
        env_fig.savefig('environmental_correlations.png', dpi=300, bbox_inches='tight')
        print("Saved: environmental_correlations.png")
    
    plt.show()

if __name__ == "__main__":
    main()