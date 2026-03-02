#!/usr/bin/env python3
"""
Laboratory System Utilization Analyzer
Analyzes log files to calculate equipment utilization metrics and create visualizations.
"""

import re
import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import json

class LabUtilizationAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.start_cutoff = datetime(2025, 8, 8, 13, 30, 29)  # First timestamped log
        self.sessions = []
        self.analysis_results = {}
        
        # Regex patterns for different timestamp formats
        self.timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+',  # 2025-08-08 13:30:29,855
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',      # 2025-08-08 13:30:29
        ]
        
    def extract_timestamp_from_line(self, line):
        """Extract timestamp from a log line."""
        for pattern in self.timestamp_patterns:
            match = re.match(pattern, line)
            if match:
                timestamp_str = match.group(1)
                try:
                    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
        return None
    
    def extract_session_info_from_filename(self, filename):
        """Extract session metadata from log filename."""
        # Pattern: experiment_log2025-08-08_13-30-29_simulate.log
        filename_pattern = r'experiment_log(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(_simulate)?\.log'
        match = re.match(filename_pattern, filename)
        
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            is_simulate = match.group(3) is not None
            
            # Convert to datetime
            datetime_str = f"{date_str} {time_str.replace('-', ':')}"
            try:
                file_start_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
                return {
                    'file_start_time': file_start_time,
                    'is_simulate': is_simulate,
                    'date': date_str
                }
            except ValueError:
                pass
        
        return None
    
    def analyze_log_file(self, log_path):
        """Analyze a single log file to extract session timing."""
        print(f"Analyzing: {log_path.name}")
        
        # Extract info from filename
        session_info = self.extract_session_info_from_filename(log_path.name)
        if not session_info:
            print(f"  - Could not parse filename format")
            return None
        
        if session_info['file_start_time'] < self.start_cutoff:
            print(f"  - Before cutoff date")
            return None
            
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                print(f"  - Empty file")
                return None
                
            # Get first and last timestamps from log content
            first_timestamp = None
            last_timestamp = None
            
            # Find first timestamp
            for line in lines[:50]:  # Check first 50 lines
                ts = self.extract_timestamp_from_line(line)
                if ts:
                    first_timestamp = ts
                    break
                    
            # Find last timestamp  
            for line in reversed(lines[-50:]):  # Check last 50 lines
                ts = self.extract_timestamp_from_line(line)
                if ts:
                    last_timestamp = ts
                    break
                    
            if not first_timestamp or not last_timestamp:
                print(f"  - No valid timestamps found in log content")
                return None
                
            # Calculate duration
            duration = last_timestamp - first_timestamp
            duration_hours = duration.total_seconds() / 3600
            
            session = {
                'filename': log_path.name,
                'file_start_time': session_info['file_start_time'],
                'log_start_time': first_timestamp,
                'log_end_time': last_timestamp,
                'duration_hours': duration_hours,
                'duration_minutes': duration.total_seconds() / 60,
                'is_simulate': session_info['is_simulate'],
                'date': session_info['date'],
                'total_lines': len(lines)
            }
            
            print(f"  - Duration: {duration_hours:.2f} hours ({duration.total_seconds()/60:.1f} minutes)")
            return session
            
        except Exception as e:
            print(f"  - Error reading file: {e}")
            return None
    
    def scan_all_logs(self):
        """Scan all experiment log files and extract session data."""
        print("Scanning log files...")
        
        # Find all experiment log files with timestamps
        log_pattern = self.logs_dir / "experiment_log*.log"
        log_files = sorted(glob.glob(str(log_pattern)))
        
        print(f"Found {len(log_files)} experiment log files")
        
        self.sessions = []
        for log_file in log_files:
            session = self.analyze_log_file(Path(log_file))
            if session:
                self.sessions.append(session)
        
        print(f"\nSuccessfully parsed {len(self.sessions)} sessions")
        return self.sessions
    
    def calculate_metrics(self):
        """Calculate utilization metrics from session data."""
        if not self.sessions:
            print("No sessions to analyze")
            return
            
        df = pd.DataFrame(self.sessions)
        df['log_start_time'] = pd.to_datetime(df['log_start_time'])
        df['log_end_time'] = pd.to_datetime(df['log_end_time'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Overall metrics
        total_runtime_hours = df['duration_hours'].sum()
        total_sessions = len(df)
        avg_session_hours = df['duration_hours'].mean()
        
        # Time period analysis
        start_date = df['log_start_time'].min()
        end_date = df['log_end_time'].max()
        total_period_days = (end_date - start_date).total_seconds() / (24 * 3600)
        total_period_hours = total_period_days * 24
        
        # Utilization calculation
        utilization_24h = (total_runtime_hours / total_period_hours) * 100 if total_period_hours > 0 else 0
        
        # Business hours utilization (8am-6pm, Mon-Fri)
        business_hours = self.calculate_business_hours(start_date, end_date)
        utilization_business = (total_runtime_hours / business_hours) * 100 if business_hours > 0 else 0
        
        # Weekly averaging metrics
        daily_usage = df.groupby(df['date'].dt.date)['duration_hours'].sum()
        if len(daily_usage) >= 7:
            weekly_avg_runtime = daily_usage.rolling(window=7, min_periods=1).mean().iloc[-1]
            peak_weekly_avg = daily_usage.rolling(window=7, min_periods=1).mean().max()
        else:
            weekly_avg_runtime = daily_usage.mean()
            peak_weekly_avg = daily_usage.max()
        
        # Simulation vs real experiments
        sim_sessions = df[df['is_simulate'] == True]
        real_sessions = df[df['is_simulate'] == False]
        
        self.analysis_results = {
            'total_runtime_hours': total_runtime_hours,
            'total_sessions': total_sessions,
            'avg_session_hours': avg_session_hours,
            'start_date': start_date,
            'end_date': end_date,
            'total_period_days': total_period_days,
            'utilization_24h': utilization_24h,
            'utilization_business': utilization_business,
            'weekly_avg_runtime': weekly_avg_runtime,
            'peak_weekly_avg': peak_weekly_avg,
            'simulation_sessions': len(sim_sessions),
            'real_sessions': len(real_sessions),
            'simulation_hours': sim_sessions['duration_hours'].sum() if len(sim_sessions) > 0 else 0,
            'real_hours': real_sessions['duration_hours'].sum() if len(real_sessions) > 0 else 0,
            'sessions_per_day': total_sessions / total_period_days if total_period_days > 0 else 0
        }
        
        return self.analysis_results
    
    def calculate_business_hours(self, start_date, end_date):
        """Calculate business hours (8am-6pm, Mon-Fri) between dates."""
        business_hours = 0
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_hours += 10  # 8am to 6pm = 10 hours
            current_date += timedelta(days=1)
            
        return business_hours
    
    def create_visualizations(self):
        """Create utilization visualizations."""
        if not self.sessions or not self.analysis_results:
            print("No data to visualize")
            return
            
        df = pd.DataFrame(self.sessions)
        df['log_start_time'] = pd.to_datetime(df['log_start_time'])
        df['log_end_time'] = pd.to_datetime(df['log_end_time'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Create figure with subplots - just weekly average and summary
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Laboratory System Utilization Analysis', fontsize=16, fontweight='bold')
        
        # 1. Weekly rolling average utilization
        ax1 = axes[0]
        daily_usage = df.groupby('date')['duration_hours'].sum().reset_index()
        daily_usage['date'] = pd.to_datetime(daily_usage['date'])
        
        # Create complete date range to handle missing days
        date_range = pd.date_range(start=daily_usage['date'].min(), 
                                 end=daily_usage['date'].max(), freq='D')
        complete_df = pd.DataFrame({'date': date_range})
        daily_usage = complete_df.merge(daily_usage, on='date', how='left')
        daily_usage['duration_hours'] = daily_usage['duration_hours'].fillna(0)
        
        # Calculate 7-day rolling average
        daily_usage['weekly_avg'] = daily_usage['duration_hours'].rolling(window=7, min_periods=1).mean()
        
        # Plot weekly rolling average
        ax1.plot(daily_usage['date'], daily_usage['weekly_avg'], 'o-', 
                color='green', linewidth=2, markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weekly Average Runtime (hours)')
        ax1.set_title('Weekly Rolling Average System Utilization (7-day window)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Format x-axis dates for weekly view
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # Show more frequent dates for weekly data
        
        # 2. Summary statistics
        ax2 = axes[1]
        ax2.axis('off')
        
        # Create summary text
        results = self.analysis_results
        summary_text = f"""
UTILIZATION SUMMARY
Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}
Duration: {results['total_period_days']:.0f} days

RUNTIME STATISTICS
Total Runtime: {results['total_runtime_hours']:.1f} hours
Total Sessions: {results['total_sessions']}
Avg Session: {results['avg_session_hours']:.2f} hours

WEEKLY TRENDS
Current Weekly Avg: {results['weekly_avg_runtime']:.1f} hours/day
Peak Weekly Avg: {results['peak_weekly_avg']:.1f} hours/day

UTILIZATION RATES
24/7 Utilization: {results['utilization_24h']:.1f}%
Business Hours: {results['utilization_business']:.1f}%
Sessions/Day: {results['sessions_per_day']:.1f}

SESSION BREAKDOWN
Simulation: {results['simulation_sessions']} sessions ({results['simulation_hours']:.1f}h)
Real Hardware: {results['real_sessions']} sessions ({results['real_hours']:.1f}h)
        """
        
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'output/utilization/utilization_analysis.png'
        os.makedirs('output/utilization', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        
        # Close the figure instead of showing it
        plt.close()
        
    def generate_report(self):
        """Generate a detailed text report."""
        if not self.analysis_results:
            print("No analysis results to report")
            return
            
        results = self.analysis_results
        
        report = f"""
LABORATORY SYSTEM UTILIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==================================================
ANALYSIS PERIOD
==================================================
Start Date: {results['start_date'].strftime('%Y-%m-%d %H:%M:%S')}
End Date: {results['end_date'].strftime('%Y-%m-%d %H:%M:%S')}
Total Period: {results['total_period_days']:.1f} days ({results['total_period_days']/7:.1f} weeks)

==================================================
RUNTIME SUMMARY  
==================================================
Total Experiment Runtime: {results['total_runtime_hours']:.2f} hours
Total Experiment Sessions: {results['total_sessions']}
Average Session Duration: {results['avg_session_hours']:.2f} hours
Sessions per Day: {results['sessions_per_day']:.1f}

==================================================
UTILIZATION METRICS
==================================================
24/7 System Utilization: {results['utilization_24h']:.1f}%
Business Hours Utilization: {results['utilization_business']:.1f}%
  (Business Hours = 8am-6pm, Monday-Friday)

==================================================
WEEKLY UTILIZATION TRENDS
==================================================
Current Weekly Average: {results['weekly_avg_runtime']:.2f} hours per day
Peak Weekly Average: {results['peak_weekly_avg']:.2f} hours per day
  (7-day rolling average)

==================================================
EXPERIMENT TYPE BREAKDOWN
==================================================
Simulation Experiments: {results['simulation_sessions']} sessions ({results['simulation_hours']:.1f} hours)
Real Hardware Experiments: {results['real_sessions']} sessions ({results['real_hours']:.1f} hours)

Simulation Percentage: {(results['simulation_hours']/results['total_runtime_hours']*100):.1f}% of total runtime

==================================================
INSIGHTS
==================================================
"""

        # Add insights based on the data
        if results['utilization_24h'] < 10:
            report += "- Low overall utilization (<10%) suggests significant capacity for additional experiments\n"
        elif results['utilization_24h'] < 25:
            report += "- Moderate utilization suggests room for optimization and increased throughput\n"
        else:
            report += "- High utilization suggests the system is being actively used\n"
            
        if results['utilization_business'] > results['utilization_24h'] * 3:
            report += "- Most experiments run during business hours - consider overnight/weekend runs for higher throughput\n"
            
        if results['simulation_hours'] > results['real_hours']:
            report += "- More simulation time than real hardware time - good for development/testing workflow\n"
        else:
            report += "- More real hardware time than simulation - system is being used for actual experiments\n"
            
        report += f"\n- Average experiment duration of {results['avg_session_hours']:.1f} hours suggests {'short development runs' if results['avg_session_hours'] < 1 else 'substantial experimental procedures'}\n"
        
        # Save report to file
        report_file = f"output/utilization/utilization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs('output/utilization', exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
            
        print(report)
        print(f"\nDetailed report saved to: {report_file}")
        
        return report


def main():
    """Main analysis function."""
    print("Laboratory System Utilization Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LabUtilizationAnalyzer("logs")
    
    # Scan and analyze logs
    analyzer.scan_all_logs()
    
    if not analyzer.sessions:
        print("No valid sessions found. Check that log files have proper timestamps.")
        return
        
    # Calculate metrics
    analyzer.calculate_metrics()
    
    # Generate visualizations
    analyzer.create_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    # Save session data
    session_data_file = "output/utilization/session_data.json"
    with open(session_data_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        sessions_json = []
        for session in analyzer.sessions:
            session_copy = session.copy()
            session_copy['file_start_time'] = session_copy['file_start_time'].isoformat()
            session_copy['log_start_time'] = session_copy['log_start_time'].isoformat()
            session_copy['log_end_time'] = session_copy['log_end_time'].isoformat()
            sessions_json.append(session_copy)
        json.dump(sessions_json, f, indent=2)
    print(f"Session data saved to: {session_data_file}")


if __name__ == "__main__":
    main()