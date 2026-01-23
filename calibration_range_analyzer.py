#!/usr/bin/env python3
"""
Calibration Range Analyzer
Analyzes experiment logs to determine the range of volumes dispensed from reservoirs and vials.
This helps determine what calibration ranges are needed for the pipetting system.
"""

import re
import os
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

class CalibrationRangeAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.start_cutoff = datetime(2025, 8, 8, 13, 30, 29)  # First timestamped log
        self.dispensing_operations = []
        self.analysis_results = {}
        
        # Regex patterns for the actual dispensing operations in your logs
        self.dispensing_patterns = [
            # "Dispensing 1.900 mL from 2MeTHF to sample_2"
            r'Dispensing\s+([0-9.]+)\s+mL\s+from\s+([^\s]+)\s+to\s+([^\s]+)',
            # "Pipetting from vial 2MeTHF, amount: 0.950 mL"
            r'Pipetting\s+from\s+vial\s+([^,]+),\s+amount:\s+([0-9.]+)\s+mL',
            # "Pipetting into vial sample_2, amount: 0.950 mL"
            r'Pipetting\s+into\s+vial\s+([^,]+),\s+amount:\s+([0-9.]+)\s+mL',
            # "Adding 1.900 mL solvent to sample_2" (plain text, no timestamp)
            r'Adding\s+([0-9.]+)\s+mL\s+([^\s]+)\s+to\s+([^\s]+)'
        ]
        
        # Timestamp pattern
        self.timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d*'
        
        # Keywords that indicate waste/overvolume operations to ignore
        self.waste_keywords = ['waste', 'overvolume', 'priming', 'prime', 'flush', 'rinse', 'clean']
        
    def extract_timestamp_from_line(self, line):
        """Extract timestamp from a log line."""
        match = re.match(self.timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            try:
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return None
        return None
    
    def is_waste_operation(self, line, source=None, target=None):
        """Check if this is a waste/overvolume operation that should be ignored."""
        line_lower = line.lower()
        
        # Check line content for waste keywords
        if any(keyword in line_lower for keyword in self.waste_keywords):
            return True
            
        # Check source and target names
        if source and any(keyword in source.lower() for keyword in self.waste_keywords):
            return True
            
        if target and any(keyword in target.lower() for keyword in self.waste_keywords):
            return True
            
        return False
    
    def extract_dispensing_operations(self, line, timestamp):
        """Extract dispensing operations from a log line."""
        operations = []
        
        for pattern in self.dispensing_patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                groups = match.groups()
                
                # Initialize default values
                volume = None
                source = "unknown"
                target = "unknown" 
                operation_type = "unknown"
                
                # Match the actual patterns from your logs
                if 'Dispensing' in pattern:
                    # "Dispensing 1.900 mL from 2MeTHF to sample_2"
                    volume = float(groups[0])
                    source = groups[1]
                    target = groups[2]
                    operation_type = 'dispensing'
                    
                elif 'Pipetting' in pattern and 'from' in pattern:
                    # "Pipetting from vial 2MeTHF, amount: 0.950 mL"
                    source = groups[0]
                    volume = float(groups[1])
                    target = 'unknown'
                    operation_type = 'pipetting_from'
                    
                elif 'Pipetting' in pattern and 'into' in pattern:
                    # "Pipetting into vial sample_2, amount: 0.950 mL"
                    target = groups[0]
                    volume = float(groups[1])
                    source = 'unknown'
                    operation_type = 'pipetting_into'
                    
                elif 'Adding' in pattern:
                    # "Adding 1.900 mL solvent to sample_2"
                    volume = float(groups[0])
                    source = groups[1]
                    target = groups[2]
                    operation_type = 'adding'
                else:
                    continue
                
                # Skip if volume couldn't be extracted
                if volume is None:
                    continue
                    
                # Skip waste operations
                if self.is_waste_operation(line, source, target):
                    continue
                
                # Categorize source type
                source_type = 'vial'  # Default to vial
                if 'reservoir' in source.lower():
                    source_type = 'reservoir'
                    
                operation = {
                    'timestamp': timestamp,
                    'volume_mL': volume,
                    'source': source,
                    'target': target,
                    'source_type': source_type,
                    'operation_type': operation_type,
                    'log_line': line.strip()
                }
                
                operations.append(operation)
                
        return operations
    
    def analyze_log_file(self, log_path):
        """Analyze a single log file for dispensing operations."""
        print(f"Analyzing: {log_path.name}")
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                print(f"  - Empty file")
                return []
                
            operations = []
            for line_num, line in enumerate(lines, 1):
                timestamp = self.extract_timestamp_from_line(line)
                if timestamp:  # No cutoff filtering - analyze the whole file
                    line_operations = self.extract_dispensing_operations(line, timestamp)
                    for op in line_operations:
                        op['file'] = log_path.name
                        op['line_number'] = line_num
                    operations.extend(line_operations)
            
            print(f"  - Found {len(operations)} dispensing operations")
            return operations
            
        except Exception as e:
            print(f"  - Error reading file: {e}")
            return []
    
    def analyze_single_log(self, log_filename):
        """Analyze a specific log file for dispensing operations."""
        # Auto-add .log extension if not provided
        if not log_filename.endswith('.log'):
            log_filename = log_filename + '.log'
            
        print(f"Analyzing single log file: {log_filename}")
        
        log_path = self.logs_dir / log_filename
        if not log_path.exists():
            print(f"Error: Log file '{log_filename}' not found in {self.logs_dir}")
            # List available log files
            available_logs = list(self.logs_dir.glob("experiment_log*.log"))
            if available_logs:
                print("\nAvailable log files:")
                for log in sorted(available_logs)[-5:]:
                    print(f"  {log.name}")
                if len(available_logs) > 5:
                    print(f"  ... and {len(available_logs)-5} more")
            return []
            
        operations = self.analyze_log_file(log_path)
        self.dispensing_operations = operations
        
        print(f"\nTotal dispensing operations found: {len(self.dispensing_operations)}")
        return self.dispensing_operations
    
    def scan_all_logs(self):
        """Scan all experiment log files for dispensing operations."""
        print("Scanning log files for dispensing operations...")
        
        # Find all experiment log files with timestamps
        log_pattern = self.logs_dir / "experiment_log*.log"
        log_files = sorted(glob.glob(str(log_pattern)))
        
        print(f"Found {len(log_files)} experiment log files")
        
        self.dispensing_operations = []
        for log_file in log_files:
            operations = self.analyze_log_file(Path(log_file))
            self.dispensing_operations.extend(operations)
        
        print(f"\nTotal dispensing operations found: {len(self.dispensing_operations)}")
        return self.dispensing_operations
    
    def analyze_volume_ranges(self):
        """Analyze volume ranges by source type and operation type."""
        if not self.dispensing_operations:
            print("No dispensing operations to analyze")
            return
            
        df = pd.DataFrame(self.dispensing_operations)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Overall statistics
        total_ops = len(df)
        volume_range = (df['volume_mL'].min(), df['volume_mL'].max())
        volume_mean = df['volume_mL'].mean()
        volume_median = df['volume_mL'].median()
        
        # Analysis by source type
        source_type_stats = df.groupby('source_type')['volume_mL'].agg([
            'count', 'min', 'max', 'mean', 'median', 'std'
        ]).round(3)
        
        # Analysis by operation type
        operation_type_stats = df.groupby('operation_type')['volume_mL'].agg([
            'count', 'min', 'max', 'mean', 'median', 'std'
        ]).round(3)
        
        # Volume distribution bins for calibration planning
        volume_bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
        df['volume_bin'] = pd.cut(df['volume_mL'], bins=volume_bins, 
                                 labels=['<10µL', '10-50µL', '50-100µL', '100-500µL', 
                                        '0.5-1mL', '1-2mL', '2-5mL', '>5mL'])
        
        volume_distribution = df.groupby(['source_type', 'volume_bin']).size().reset_index(name='count')
        
        # Top sources by volume frequency
        source_frequency = df.groupby('source')['volume_mL'].agg(['count', 'min', 'max']).sort_values('count', ascending=False)
        
        self.analysis_results = {
            'total_operations': total_ops,
            'volume_range': volume_range,
            'volume_mean': volume_mean,
            'volume_median': volume_median,
            'source_type_stats': source_type_stats,
            'operation_type_stats': operation_type_stats,
            'volume_distribution': volume_distribution,
            'source_frequency': source_frequency,  # All sources, not just top 10
            'unique_sources': df['source'].nunique(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max())
        }
        
        return self.analysis_results
    
    def create_visualizations(self):
        """Create calibration range visualizations."""
        if not self.dispensing_operations or not self.analysis_results:
            print("No data to visualize")
            return
            
        df = pd.DataFrame(self.dispensing_operations)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pipetting Calibration Range Analysis', fontsize=16, fontweight='bold')
        
        # 1. Volume distribution by source type
        ax1 = axes[0, 0]
        source_volumes = [group['volume_mL'].values for name, group in df.groupby('source_type')]
        source_labels = list(df.groupby('source_type').groups.keys())
        
        ax1.boxplot(source_volumes, labels=source_labels)
        ax1.set_ylabel('Volume (mL)')
        ax1.set_title('Volume Distribution by Source Type')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volume histogram with calibration zones
        ax2 = axes[0, 1]
        volumes = df['volume_mL']
        
        # Create histogram
        n_bins = 50
        counts, bins, patches = ax2.hist(volumes, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add calibration zone markers
        calibration_zones = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
        
        for zone, color in zip(calibration_zones, colors):
            if zone <= volumes.max():
                ax2.axvline(zone, color=color, linestyle='--', alpha=0.8, 
                           label=f'{zone*1000:.0f}µL' if zone < 1 else f'{zone:.0f}mL')
        
        ax2.set_xlabel('Volume (mL)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Volume Distribution with Calibration Zones')
        ax2.set_xscale('log')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Operations over time
        ax3 = axes[1, 0]
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_ops = df.groupby(['date', 'source_type']).size().reset_index(name='count')
        
        # Pivot for plotting
        daily_pivot = daily_ops.pivot(index='date', columns='source_type', values='count').fillna(0)
        
        if len(daily_pivot) > 0:
            daily_pivot.plot(kind='bar', stacked=True, ax=ax3, alpha=0.8)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Number of Operations')
            ax3.set_title('Daily Operations by Source Type')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No time series data', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12)
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        results = self.analysis_results
        
        summary_text = f"""
CALIBRATION RANGE SUMMARY

OVERALL STATISTICS
Total Operations: {results['total_operations']}
Volume Range: {results['volume_range'][0]:.3f} - {results['volume_range'][1]:.3f} mL
Mean Volume: {results['volume_mean']:.3f} mL
Median Volume: {results['volume_median']:.3f} mL
Unique Sources: {results['unique_sources']}

RECOMMENDED CALIBRATION POINTS
Micro Range: 0.001 - 0.050 mL
Small Range: 0.050 - 0.500 mL  
Medium Range: 0.500 - 2.000 mL
Large Range: 2.000 - 5.000 mL

SOURCE TYPE BREAKDOWN
"""
        
        # Add source type statistics
        for source_type, stats in results['source_type_stats'].iterrows():
            summary_text += f"\n{source_type.upper()}:\n"
            summary_text += f"  Operations: {int(stats['count'])}\n"
            summary_text += f"  Range: {stats['min']:.3f} - {stats['max']:.3f} mL\n"
            summary_text += f"  Avg: {stats['mean']:.3f} mL\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'output/utilization/calibration_range_analysis.png'
        os.makedirs('output/utilization', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Calibration analysis saved to: {output_file}")
        
        # Close the figure
        plt.close()
        
    def generate_report(self):
        """Generate a simple volume range report focused on actual usage data."""
        if not self.analysis_results:
            print("No analysis results to report")
            return
            
        results = self.analysis_results
        df = pd.DataFrame(self.dispensing_operations)
        
        # Get detailed volume statistics for each source
        detailed_stats = df.groupby('source')['volume_mL'].agg([
            'count', 'min', 'max', 'mean', 'median'
        ]).round(4)
        detailed_stats = detailed_stats.sort_values('count', ascending=False)
        
        report = f"""
PIPETTING VOLUME ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {df['file'].iloc[0] if len(df) > 0 else 'N/A'}

=======================================================================
SOURCE VOLUME SUMMARY
=======================================================================
Total Operations: {results['total_operations']}
Overall Volume Range: {results['volume_range'][0]:.4f} - {results['volume_range'][1]:.4f} mL
Unique Sources: {results['unique_sources']}

=======================================================================
VOLUME RANGES BY SOURCE (sorted by frequency)
=======================================================================

"""
        
        # Add each source with its volume details
        for source, stats in detailed_stats.iterrows():
            source_data = df[df['source'] == source]
            volumes = sorted(source_data['volume_mL'].unique())
            source_type = source_data['source_type'].iloc[0]
            
            report += f"{source} ({source_type}):\n"
            report += f"  Operations: {int(stats['count'])}\n"
            report += f"  Range: {stats['min']:.4f} - {stats['max']:.4f} mL\n"
            report += f"  Average: {stats['mean']:.4f} mL\n"
            report += f"  Median: {stats['median']:.4f} mL\n"
            
            # Show all unique volumes used
            if len(volumes) <= 10:
                volume_str = ", ".join([f"{v:.4f}" for v in volumes])
                report += f"  Volumes used: {volume_str} mL\n"
            else:
                report += f"  Volumes used: {len(volumes)} different values\n"
                report += f"  Sample volumes: {volumes[0]:.4f}, {volumes[len(volumes)//2]:.4f}, {volumes[-1]:.4f} mL\n"
            report += "\n"
        
        # Add reservoir vs vial breakdown
        type_summary = df.groupby('source_type').agg({
            'volume_mL': ['count', 'min', 'max'],
            'source': 'nunique'
        }).round(4)
        
        report += "=======================================================================\n"
        report += "SUMMARY BY SOURCE TYPE\n"
        report += "=======================================================================\n\n"
        
        for source_type in type_summary.index:
            stats = type_summary.loc[source_type]
            report += f"{source_type.upper()}:\n"
            report += f"  Sources: {int(stats[('source', 'nunique')])}\n"
            report += f"  Operations: {int(stats[('volume_mL', 'count')])}\n"
            report += f"  Volume range: {stats[('volume_mL', 'min')]:.4f} - {stats[('volume_mL', 'max')]:.4f} mL\n\n"
        
        # Save report to file
        report_file = f"output/utilization/volume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs('output/utilization', exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(report)
        print(f"Volume analysis saved to: {report_file}")
        
        return report


def main():
    """Main analysis function."""
    import sys
    
    print("Pipetting Calibration Range Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CalibrationRangeAnalyzer("logs")
    
    # Check if specific log file is provided as command line argument
    if len(sys.argv) > 1:
        log_filename = sys.argv[1]
        print(f"Analyzing specific log: {log_filename}")
        analyzer.analyze_single_log(log_filename)
    else:
        # Ask user for log filename
        log_filename = input("\nEnter log filename (extension optional): ").strip()
        if not log_filename:
            print("No filename provided. Exiting.")
            return
        analyzer.analyze_single_log(log_filename)
    
    if not analyzer.dispensing_operations:
        print("No dispensing operations found. Check that log files contain volume dispensing information.")
        return
        
    # Analyze volume ranges
    analyzer.analyze_volume_ranges()
    
    # Generate visualizations
    analyzer.create_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    # Save dispensing operations data
    operations_data_file = "output/utilization/dispensing_operations.json"
    with open(operations_data_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        operations_json = []
        for op in analyzer.dispensing_operations:
            op_copy = op.copy()
            op_copy['timestamp'] = op_copy['timestamp'].isoformat()
            operations_json.append(op_copy)
        json.dump(operations_json, f, indent=2)
    print(f"Dispensing operations data saved to: {operations_data_file}")
    
    # Save summary statistics
    stats_file = "output/utilization/calibration_statistics.json"
    with open(stats_file, 'w') as f:
        # Convert pandas objects to JSON-serializable format
        stats_json = {
            'total_operations': analyzer.analysis_results['total_operations'],
            'volume_range': analyzer.analysis_results['volume_range'],
            'volume_mean': analyzer.analysis_results['volume_mean'],
            'volume_median': analyzer.analysis_results['volume_median'],
            'unique_sources': analyzer.analysis_results['unique_sources'],
            'date_range': [
                analyzer.analysis_results['date_range'][0].isoformat(),
                analyzer.analysis_results['date_range'][1].isoformat()
            ]
        }
        json.dump(stats_json, f, indent=2)
    print(f"Summary statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()