#!/usr/bin/env python3
"""
Log Analysis Script for Counting Pipetting Actions

This script analyzes North Robotics experiment logs to count macro pipetting 
actions (complete liquid transfers) vs low-level operations. Each complete 
pipetting action involves multiple aspirations (pre-air, main liquid, post-air, 
mixing cycles) but typically one dispense. The dispense count represents the 
actual number of completed transfers.

Key Metrics:
- Macro Actions: Complete liquid transfers (= dispense count)
- Low-level Operations: Individual aspirate/dispense calls including air gaps

Author: GitHub Copilot
Date: April 16, 2026
"""

import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

class PippettingLogAnalyzer:
    """Analyzes North Robotics logs to count macro pipetting actions vs low-level operations"""
    
    def __init__(self):
        # Regex patterns for pipetting operations based on North_Safe.py logging
        self.aspiration_pattern = re.compile(r'INFO - Pipetting from vial (.+?), amount: ([\d.]+) mL')
        self.dispense_pattern = re.compile(r'INFO - Pipetting into vial (.+?), amount: ([\d.]+) mL')
        
        # Additional patterns for lower-level operations (optional)
        self.debug_aspirate_pattern = re.compile(r'DEBUG - Aspirating ([\d.]+) mL then waiting ([\d.]+) s')
        self.debug_dispense_pattern = re.compile(r'DEBUG - Dispensing ([\d.]+) mL then waiting ([\d.]+) s')
        
    def analyze_file(self, log_file_path):
        """
        Analyze a single log file for pipetting operations
        
        Args:
            log_file_path (str): Path to the log file
            
        Returns:
            dict: Analysis results with counts and details
        """
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Failed to read file {log_file_path}: {e}"}
            
        results = {
            "file": os.path.basename(log_file_path),
            "aspirations": [],
            "dispenses": [],
            "debug_aspirations": [],
            "debug_dispenses": [],
            "total_aspirations": 0,
            "total_dispenses": 0,
            "total_macro_actions": 0,
            "total_operations": 0,
            "total_volume_aspirated": 0.0,
            "total_volume_dispensed": 0.0
        }
        
        # Find high-level aspiration operations
        for match in self.aspiration_pattern.finditer(content):
            vial_name = match.group(1)
            volume = float(match.group(2))
            results["aspirations"].append({
                "vial": vial_name, 
                "volume_mL": volume,
                "line": match.group(0)
            })
            results["total_volume_aspirated"] += volume
            
        # Find high-level dispense operations  
        for match in self.dispense_pattern.finditer(content):
            vial_name = match.group(1)
            volume = float(match.group(2))
            results["dispenses"].append({
                "vial": vial_name, 
                "volume_mL": volume,
                "line": match.group(0)
            })
            results["total_volume_dispensed"] += volume
            
        # Find debug-level operations (for completeness)
        for match in self.debug_aspirate_pattern.finditer(content):
            volume = float(match.group(1))
            wait_time = float(match.group(2))
            results["debug_aspirations"].append({
                "volume_mL": volume, 
                "wait_time_s": wait_time,
                "line": match.group(0)
            })
            
        for match in self.debug_dispense_pattern.finditer(content):
            volume = float(match.group(1))
            wait_time = float(match.group(2))
            results["debug_dispenses"].append({
                "volume_mL": volume, 
                "wait_time_s": wait_time,
                "line": match.group(0)
            })
            
        # Calculate totals - dispenses represent complete macro pipetting actions
        results["total_aspirations"] = len(results["aspirations"])
        results["total_dispenses"] = len(results["dispenses"])
        results["total_macro_actions"] = results["total_dispenses"]  # Each dispense = 1 complete transfer
        results["total_operations"] = results["total_aspirations"] + results["total_dispenses"]
        
        return results
        
    def analyze_directory(self, log_directory, pattern="*.log"):
        """
        Analyze all log files in a directory
        
        Args:
            log_directory (str): Path to directory containing log files
            pattern (str): File pattern to match (default: "*.log")
            
        Returns:
            list: List of analysis results for each file
        """
        log_dir = Path(log_directory)
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {log_directory}")
            
        log_files = list(log_dir.glob(pattern))
        if not log_files:
            print(f"No log files found matching pattern '{pattern}' in {log_directory}")
            return []
            
        results = []
        for log_file in sorted(log_files):
            print(f"Analyzing: {log_file.name}")
            result = self.analyze_file(log_file)
            results.append(result)
            
        return results
        
    def generate_summary(self, results_list):
        """
        Generate a summary report from multiple log file analyses
        
        Args:
            results_list (list): List of analysis results
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            "files_analyzed": len(results_list),
            "total_aspirations": 0,
            "total_dispenses": 0,
            "total_macro_actions": 0, 
            "total_operations": 0,
            "total_volume_aspirated": 0.0,
            "total_volume_dispensed": 0.0,
            "files_with_operations": 0,
            "vial_usage": defaultdict(int),
            "operation_by_file": []
        }
        
        for result in results_list:
            if "error" in result:
                continue
                
            has_operations = result["total_macro_actions"] > 0
            if has_operations:
                summary["files_with_operations"] += 1
                
            summary["total_aspirations"] += result["total_aspirations"]
            summary["total_dispenses"] += result["total_dispenses"] 
            summary["total_macro_actions"] += result["total_macro_actions"]
            summary["total_operations"] += result["total_operations"]
            summary["total_volume_aspirated"] += result["total_volume_aspirated"]
            summary["total_volume_dispensed"] += result["total_volume_dispensed"]
            
            # Track vial usage
            for asp in result["aspirations"]:
                summary["vial_usage"][f"{asp['vial']} (source)"] += 1
            for disp in result["dispenses"]:
                summary["vial_usage"][f"{disp['vial']} (dest)"] += 1
                
            # File-by-file breakdown
            summary["operation_by_file"].append({
                "file": result["file"],
                "macro_actions": result["total_macro_actions"],
                "aspirations": result["total_aspirations"],
                "dispenses": result["total_dispenses"],
                "volume_transferred": result["total_volume_dispensed"],
                "volume_asp": result["total_volume_aspirated"]
            })
            
        return summary
        
    def print_report(self, results_list, detailed=False):
        """
        Print a formatted report of the analysis
        
        Args:
            results_list (list): List of analysis results
            detailed (bool): Whether to include detailed per-file information
        """
        summary = self.generate_summary(results_list)
        
        print("\n" + "="*60)
        print("PIPETTING OPERATIONS ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nFiles Analyzed: {summary['files_analyzed']}")
        print(f"Files with Operations: {summary['files_with_operations']}")
        
        print(f"\nMACRO PIPETTING ACTIONS (Complete Transfers):")
        print(f"  Total Actions: {summary['total_macro_actions']:,}")
        print(f"  Volume Transferred: {summary['total_volume_dispensed']:.3f} mL")
        
        print(f"\nLOW-LEVEL OPERATIONS (includes air gaps, cycles):")
        print(f"  Aspirations: {summary['total_aspirations']:,}")
        print(f"  Dispenses:   {summary['total_dispenses']:,}")
        print(f"  Ratio:       {summary['total_aspirations']/max(summary['total_dispenses'],1):.1f}:1")
        
        if summary['vial_usage']:
            print(f"\nMOST USED VIALS:")
            sorted_vials = sorted(summary['vial_usage'].items(), key=lambda x: x[1], reverse=True)
            for vial, count in sorted_vials[:10]:  # Top 10
                print(f"  {vial}: {count} operations")
        
        if detailed:
            print(f"\nPER-FILE BREAKDOWN:")
            df = pd.DataFrame(summary['operation_by_file'])
            if not df.empty:
                print(df.to_string(index=False))
                
        print("\n" + "="*60)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze North Robotics logs to count pipetting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python log_dispense_counter.py logs/experiment.log
  python log_dispense_counter.py logs/ --detailed
  python log_dispense_counter.py logs/ --pattern "experiment_*2026*.log"
  python log_dispense_counter.py logs/ --csv output.csv
        """
    )
    
    parser.add_argument(
        "path", 
        help="Path to log file or directory containing log files"
    )
    parser.add_argument(
        "--pattern", 
        default="*.log",
        help="File pattern for directory analysis (default: *.log)"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Show detailed per-file breakdown"
    )
    parser.add_argument(
        "--csv", 
        help="Save detailed results to CSV file"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    analyzer = PippettingLogAnalyzer()
    
    # Determine if path is file or directory
    target_path = Path(args.path)
    
    if target_path.is_file():
        # Single file analysis
        if not args.quiet:
            print(f"Analyzing single file: {target_path.name}")
        result = analyzer.analyze_file(target_path)
        results_list = [result]
    elif target_path.is_dir():
        # Directory analysis
        if not args.quiet:
            print(f"Analyzing directory: {target_path}")
            print(f"Pattern: {args.pattern}")
        results_list = analyzer.analyze_directory(target_path, args.pattern)
    else:
        print(f"Error: Path not found: {target_path}")
        return 1
        
    if not results_list:
        print("No files to analyze or no operations found.")
        return 0
        
    # Print report
    analyzer.print_report(results_list, detailed=args.detailed)
    
    # Save CSV if requested
    if args.csv:
        summary = analyzer.generate_summary(results_list)
        df = pd.DataFrame(summary['operation_by_file'])
        df.to_csv(args.csv, index=False)
        print(f"\nDetailed results saved to: {args.csv}")
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())