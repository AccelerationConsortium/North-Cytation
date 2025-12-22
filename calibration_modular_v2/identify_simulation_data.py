#!/usr/bin/env python3
"""
Identify Simulation vs Real Data in Existing Calibration Runs
============================================================

This script analyzes existing calibration output folders to help identify
which runs were simulation vs real hardware based on data patterns.

Usage: python identify_simulation_data.py

It will scan all output folders and provide educated guesses about which
runs are simulation vs real hardware.
"""

import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

def analyze_measurement_patterns(measurements_df):
    """Analyze measurement patterns to detect simulation characteristics."""
    indicators = {
        'simulation_score': 0.0,
        'reasons': []
    }
    
    if measurements_df.empty:
        return indicators
    
    # Check for overly consistent measurements (simulation trait)
    if 'measured_volume_ml' in measurements_df.columns:
        volumes = measurements_df['measured_volume_ml'].dropna()
        if len(volumes) > 5:
            cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 1.0
            
            # Real hardware typically has CV > 0.02, simulation often < 0.01
            if cv < 0.01:
                indicators['simulation_score'] += 0.4
                indicators['reasons'].append(f"Very low variability (CV={cv:.4f})")
            elif cv > 0.05:
                indicators['simulation_score'] -= 0.2
                indicators['reasons'].append(f"High variability suggests real hardware (CV={cv:.4f})")
    
    # Check for perfect parameter relationships (simulation trait)
    if 'overaspirate_vol' in measurements_df.columns and 'measured_volume_ml' in measurements_df.columns:
        try:
            # Look for overly linear relationships
            overasp = measurements_df['overaspirate_vol'].dropna()
            volumes = measurements_df['measured_volume_ml'].dropna()
            
            if len(overasp) > 3 and len(volumes) > 3:
                correlation = np.corrcoef(overasp[:min(len(overasp), len(volumes))], 
                                        volumes[:min(len(overasp), len(volumes))])[0,1]
                
                if abs(correlation) > 0.95:
                    indicators['simulation_score'] += 0.3
                    indicators['reasons'].append(f"Perfect parameter correlation (r={correlation:.3f})")
        except:
            pass
    
    # Check for unrealistic measurement precision
    if 'measured_volume_ml' in measurements_df.columns:
        volumes = measurements_df['measured_volume_ml'].dropna()
        if len(volumes) > 1:
            # Count decimal places (simulation often has excessive precision)
            decimal_places = []
            for vol in volumes:
                vol_str = f"{vol:.10f}".rstrip('0')
                if '.' in vol_str:
                    decimal_places.append(len(vol_str.split('.')[1]))
            
            avg_decimals = np.mean(decimal_places) if decimal_places else 0
            if avg_decimals > 6:
                indicators['simulation_score'] += 0.2
                indicators['reasons'].append(f"Excessive precision ({avg_decimals:.1f} decimal places)")
    
    # Check for suspiciously perfect target matching
    if 'target_volume_ml' in measurements_df.columns and 'measured_volume_ml' in measurements_df.columns:
        targets = measurements_df['target_volume_ml'].dropna()
        measured = measurements_df['measured_volume_ml'].dropna()
        
        if len(targets) > 0 and len(measured) > 0:
            errors = []
            for i in range(min(len(targets), len(measured))):
                error_pct = abs((measured.iloc[i] - targets.iloc[i]) / targets.iloc[i]) * 100
                errors.append(error_pct)
            
            if errors:
                avg_error = np.mean(errors)
                if avg_error < 2.0:  # Less than 2% average error is suspicious
                    indicators['simulation_score'] += 0.3
                    indicators['reasons'].append(f"Suspiciously low error ({avg_error:.1f}%)")
                elif avg_error > 15.0:  # High error suggests real hardware struggles
                    indicators['simulation_score'] -= 0.2
                    indicators['reasons'].append(f"High error suggests real hardware ({avg_error:.1f}%)")
    
    return indicators

def analyze_run_folder(run_path):
    """Analyze a single calibration run folder."""
    run_info = {
        'folder': run_path.name,
        'path': str(run_path),
        'timestamp': None,
        'has_summary': False,
        'has_measurements': False,
        'likely_simulation': None,
        'confidence': 0.0,
        'evidence': [],
        'total_measurements': 0
    }
    
    # Extract timestamp from folder name if possible
    try:
        if 'run_' in run_path.name:
            timestamp_str = run_path.name.split('run_')[1]
            run_info['timestamp'] = datetime.fromtimestamp(int(timestamp_str))
    except:
        pass
    
    # Check for summary file
    summary_path = run_path / "experiment_summary.json"
    if summary_path.exists():
        run_info['has_summary'] = True
        try:
            with open(summary_path) as f:
                summary = json.load(f)
                run_info['total_measurements'] = summary.get('total_measurements', 0)
                
                # Check if simulation flag exists (newer runs)
                if 'simulation_mode' in summary:
                    run_info['likely_simulation'] = summary['simulation_mode']
                    run_info['confidence'] = 1.0
                    run_info['evidence'].append("Has simulation_mode flag")
        except Exception as e:
            run_info['evidence'].append(f"Could not read summary: {e}")
    
    # If we don't have a definitive answer, analyze measurement patterns
    if run_info['likely_simulation'] is None:
        measurements_path = run_path / "raw_measurements.csv"
        if measurements_path.exists():
            run_info['has_measurements'] = True
            try:
                measurements_df = pd.read_csv(measurements_path)
                
                analysis = analyze_measurement_patterns(measurements_df)
                
                # Convert simulation score to boolean prediction
                if analysis['simulation_score'] > 0.5:
                    run_info['likely_simulation'] = True
                    run_info['confidence'] = min(analysis['simulation_score'], 1.0)
                elif analysis['simulation_score'] < -0.2:
                    run_info['likely_simulation'] = False
                    run_info['confidence'] = min(abs(analysis['simulation_score']), 1.0)
                else:
                    run_info['likely_simulation'] = None
                    run_info['confidence'] = 0.0
                    analysis['reasons'].append("Inconclusive - patterns unclear")
                
                run_info['evidence'].extend(analysis['reasons'])
                
            except Exception as e:
                run_info['evidence'].append(f"Could not analyze measurements: {e}")
    
    return run_info

def main():
    """Analyze all calibration run folders."""
    print("🔍 ANALYZING EXISTING CALIBRATION DATA")
    print("=" * 60)
    
    # Look in the calibration_modular_v2 output directory specifically
    output_dir = Path("output")  # This should be calibration_modular_v2/output
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir.absolute()}")
        return
    
    run_folders = [f for f in output_dir.iterdir() if f.is_dir() and f.name.startswith('run_')]
    
    if not run_folders:
        print("❌ No calibration run folders found")
        print(f"   Looked in: {output_dir.absolute()}")
        
        # Show what's actually in the directory
        contents = list(output_dir.iterdir())
        if contents:
            print(f"   Directory contains {len(contents)} items:")
            for item in contents[:10]:  # Show first 10 items
                print(f"     • {item.name}")
            if len(contents) > 10:
                print(f"     ... and {len(contents) - 10} more")
        else:
            print("   Directory is empty")
        return
    
    print(f"📁 Found {len(run_folders)} calibration runs to analyze\\n")
    
    # Analyze each run
    results = []
    for run_folder in sorted(run_folders):
        result = analyze_run_folder(run_folder)
        results.append(result)
    
    # Display results
    print("ANALYSIS RESULTS:")
    print("-" * 60)
    
    simulation_runs = []
    hardware_runs = []
    unclear_runs = []
    
    for result in results:
        status = "❓ UNCLEAR"
        confidence_bar = ""
        
        if result['likely_simulation'] is True:
            status = f"🖥️  SIMULATION"
            simulation_runs.append(result)
            confidence_bar = "█" * int(result['confidence'] * 10)
        elif result['likely_simulation'] is False:
            status = f"🔧 HARDWARE"
            hardware_runs.append(result)
            confidence_bar = "█" * int(result['confidence'] * 10)
        else:
            unclear_runs.append(result)
        
        timestamp_str = result['timestamp'].strftime("%Y-%m-%d %H:%M") if result['timestamp'] else "Unknown time"
        
        print(f"{status:<15} {result['folder']:<25} ({timestamp_str})")
        if confidence_bar:
            print(f"               Confidence: {confidence_bar:<10} ({result['confidence']:.1%})")
        
        if result['evidence']:
            for evidence in result['evidence'][:2]:  # Show top 2 pieces of evidence
                print(f"               • {evidence}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY:")
    print(f"🖥️  Likely SIMULATION runs: {len(simulation_runs)}")
    print(f"🔧 Likely HARDWARE runs: {len(hardware_runs)}")
    print(f"❓ UNCLEAR runs: {len(unclear_runs)}")
    
    if hardware_runs:
        print(f"\\n✅ RECOMMENDED: Focus on these {len(hardware_runs)} hardware runs for real data:")
        for run in hardware_runs:
            print(f"   • {run['folder']} (confidence: {run['confidence']:.1%})")
    
    if unclear_runs:
        print(f"\\n⚠️  These {len(unclear_runs)} runs need manual inspection:")
        for run in unclear_runs:
            print(f"   • {run['folder']} - check your lab notes for this time period")

if __name__ == "__main__":
    main()