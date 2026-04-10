#!/usr/bin/env python3
"""
Laboratory System Utilization Analyzer
Analyzes log files to calculate equipment utilization metrics and create visualizations.
"""

#=============================================================================
# CONFIGURATION - Edit these dates to analyze specific periods
#=============================================================================

# Set date range for analysis (set to None for all data)
# Examples:
# ANALYZE_START_DATE = "2026-03-01"    # Specific date
# ANALYZE_START_DATE = "March 2026"    # Whole month
# ANALYZE_START_DATE = None            # From beginning

ANALYZE_START_DATE = "March 2026"  # Edit this line to change start date
ANALYZE_END_DATE = None           # Edit this line to change end date (None = auto-detect end of month/period)

#=============================================================================

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
    def __init__(self, logs_dir="logs", start_date=None, end_date=None):
        self.logs_dir = Path(logs_dir)
        self.start_cutoff = datetime(2025, 8, 8, 13, 30, 29)  # First timestamped log
        self.sessions = []
        self.analysis_results = {}
        
        # Date range filtering
        self.filter_start_date = start_date
        self.filter_end_date = end_date
        
        if start_date:
            print(f"Filtering from: {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            print(f"Filtering to: {end_date.strftime('%Y-%m-%d')}")
        
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
                
        # Check date range filters
        if self.filter_start_date and session_info['file_start_time'].date() < self.filter_start_date:
            print(f"  - Before filter start date")
            return None
        if self.filter_end_date and session_info['file_start_time'].date() > self.filter_end_date:
            print(f"  - After filter end date")
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
            
            # Override: sessions > 1 hour tagged as simulate are almost certainly real hardware runs
            is_simulate = session_info['is_simulate']
            if is_simulate and duration_hours > 1.0:
                is_simulate = False
                print(f"  - Tagged simulate but duration {duration_hours:.2f}h > 1h - treating as hardware run")
            
            session = {
                'filename': log_path.name,
                'file_start_time': session_info['file_start_time'],
                'log_start_time': first_timestamp,
                'log_end_time': last_timestamp,
                'duration_hours': duration_hours,
                'duration_minutes': duration.total_seconds() / 60,
                'is_simulate': is_simulate,
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
        if self.filter_start_date or self.filter_end_date:
            print(f"Note: Weekly metrics will exclude incomplete weeks at period boundaries")
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
        
        # Weekly total metrics - only include complete weeks within the date range
        df['week'] = df['date'].dt.to_period('W')
        weekly_usage = df.groupby('week')['duration_hours'].sum()
        
        # Filter out incomplete weeks if we have date filters
        if self.filter_start_date or self.filter_end_date:
            filtered_weeks = []
            for week_period, hours in weekly_usage.items():
                week_start = week_period.start_time.date()
                week_end = week_start + timedelta(days=6)
                
                # Check if this week is mostly within our date range
                week_in_range = True
                if self.filter_start_date:
                    # Week must start on or after filter start date
                    if week_start < self.filter_start_date:
                        week_in_range = False
                if self.filter_end_date:
                    # Week must end on or before filter end date
                    if week_end > self.filter_end_date:
                        week_in_range = False
                
                if week_in_range:
                    filtered_weeks.append((week_period, hours))
            
            # Rebuild weekly_usage with only complete weeks
            if filtered_weeks:
                weekly_usage = pd.Series([hours for _, hours in filtered_weeks], 
                                       index=[week for week, _ in filtered_weeks])
            else:
                weekly_usage = pd.Series(dtype=float)
        
        avg_weekly_runtime = weekly_usage.mean() if len(weekly_usage) > 0 else 0
        peak_weekly_runtime = weekly_usage.max() if len(weekly_usage) > 0 else 0
        current_weekly_runtime = weekly_usage.iloc[-1] if len(weekly_usage) > 0 else 0
        
        # Daily total metrics  
        daily_usage = df.groupby('date')['duration_hours'].sum()
        avg_daily_runtime = daily_usage.mean()
        peak_daily_runtime = daily_usage.max()
        days_with_experiments = len(daily_usage)
        total_calendar_days = (end_date.date() - start_date.date()).days + 1
        
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
            'avg_weekly_runtime': avg_weekly_runtime,
            'peak_weekly_runtime': peak_weekly_runtime,
            'current_weekly_runtime': current_weekly_runtime,
            'simulation_sessions': len(sim_sessions),
            'real_sessions': len(real_sessions),
            'simulation_hours': sim_sessions['duration_hours'].sum() if len(sim_sessions) > 0 else 0,
            'real_hours': real_sessions['duration_hours'].sum() if len(real_sessions) > 0 else 0,
            'sessions_per_day': total_sessions / total_period_days if total_period_days > 0 else 0,
            # Daily metrics
            'avg_daily_runtime': avg_daily_runtime,
            'peak_daily_runtime': peak_daily_runtime,
            'days_with_experiments': days_with_experiments,
            'total_calendar_days': total_calendar_days,
            'experiment_days_ratio': days_with_experiments / total_calendar_days if total_calendar_days > 0 else 0
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
        
        # Create figure with subplots - daily, weekly, and summary
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Laboratory System Utilization Analysis', fontsize=16, fontweight='bold')
        
        # 1. Daily hours utilization
        ax1 = axes[0]
        
        # Group by day and sum hours
        daily_usage = df.groupby('date')['duration_hours'].sum().reset_index()
        daily_usage['date'] = pd.to_datetime(daily_usage['date'])
        
        # Plot daily data (no rolling average for speed)
        ax1.plot(daily_usage['date'], daily_usage['duration_hours'], 
                 'o-', color='lightblue', linewidth=1.5, markersize=4, alpha=0.8, label='Daily Usage')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Runtime (hours)')
        ax1.set_title('Daily System Utilization')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # Format x-axis dates for daily view
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_usage)//10)))
        
        # 2. Weekly hours utilization
        ax2 = axes[1]
        
        # Group by week and sum hours - filter incomplete weeks
        df['week'] = df['log_start_time'].dt.to_period('W')
        weekly_usage = df.groupby('week')['duration_hours'].sum().reset_index()
        
        # Filter out incomplete weeks if we have date filters
        if self.filter_start_date or self.filter_end_date:
            filtered_weekly = []
            for _, row in weekly_usage.iterrows():
                week_start = row['week'].start_time.date()
                week_end = week_start + timedelta(days=6)
                
                # Check if this week is completely within our date range
                week_in_range = True
                if self.filter_start_date:
                    if week_start < self.filter_start_date:
                        week_in_range = False
                if self.filter_end_date:
                    if week_end > self.filter_end_date:
                        week_in_range = False
                
                if week_in_range:
                    filtered_weekly.append(row)
            
            # Rebuild dataframe with only complete weeks
            if filtered_weekly:
                weekly_usage = pd.DataFrame(filtered_weekly)
            else:
                weekly_usage = pd.DataFrame(columns=['week', 'duration_hours'])
        
        if len(weekly_usage) > 0:
            weekly_usage['week_start'] = weekly_usage['week'].dt.start_time
            
            # Plot raw weekly data only (no rolling average for speed)
            ax2.plot(weekly_usage['week_start'], weekly_usage['duration_hours'], 
                     'o-', color='darkgreen', linewidth=2, markersize=6, alpha=0.8, label='Weekly Usage')
            
            # Format x-axis dates for weekly view
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
        else:
            ax2.text(0.5, 0.5, 'No complete weeks in date range', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12)
        
        ax2.set_xlabel('Week Starting')
        ax2.set_ylabel('Total Runtime (hours)')
        ax2.set_title('Weekly System Utilization (Complete Weeks Only)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Summary statistics
        ax3 = axes[2]
        ax3.axis('off')
        
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

WEEKLY STATISTICS
Current Week: {results['current_weekly_runtime']:.1f} hours
Average Week: {results['avg_weekly_runtime']:.1f} hours
Peak Week: {results['peak_weekly_runtime']:.1f} hours

UTILIZATION RATES
24/7 Utilization: {results['utilization_24h']:.1f}%
Business Hours: {results['utilization_business']:.1f}%
Sessions/Day: {results['sessions_per_day']:.1f}

DAILY STATISTICS
Average Day: {results['avg_daily_runtime']:.1f} hours
Peak Day: {results['peak_daily_runtime']:.1f} hours
Active Days: {results['days_with_experiments']}/{results['total_calendar_days']} ({results['experiment_days_ratio']:.1%})

SESSION BREAKDOWN
Simulation: {results['simulation_sessions']} sessions ({results['simulation_hours']:.1f}h)
Real Hardware: {results['real_sessions']} sessions ({results['real_hours']:.1f}h)
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=9, 
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
WEEKLY UTILIZATION STATISTICS
==================================================
Current Week Runtime: {results['current_weekly_runtime']:.2f} hours
Average Weekly Runtime: {results['avg_weekly_runtime']:.2f} hours
Peak Weekly Runtime: {results['peak_weekly_runtime']:.2f} hours

==================================================
DAILY UTILIZATION STATISTICS
==================================================
Average Daily Runtime: {results['avg_daily_runtime']:.2f} hours
Peak Daily Runtime: {results['peak_daily_runtime']:.2f} hours
Days with Experiments: {results['days_with_experiments']} out of {results['total_calendar_days']} ({results['experiment_days_ratio']:.1%})

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


def main(start_date=None, end_date=None):
    """Main analysis function."""
    print("Laboratory System Utilization Analyzer")
    
    # Use configuration if no parameters provided
    if start_date is None and end_date is None:
        if ANALYZE_START_DATE:
            start_date = parse_date_argument(ANALYZE_START_DATE)
            print(f"Using configured start date: {ANALYZE_START_DATE}")
        if ANALYZE_END_DATE:
            end_date = parse_date_argument(ANALYZE_END_DATE)
            print(f"Using configured end date: {ANALYZE_END_DATE}")
        elif ANALYZE_START_DATE and not ANALYZE_END_DATE:
            # Auto-detect end date for month names
            date_str = ANALYZE_START_DATE.lower()
            if any(month in date_str for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                   'july', 'august', 'september', 'october', 'november', 'december']):
                # Get last day of the month
                year = start_date.year
                month = start_date.month
                if month == 12:
                    end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
                else:
                    end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
                print(f"Auto-detected end date: {end_date}")
    
    if start_date or end_date:
        date_range = f" ({start_date.strftime('%Y-%m-%d') if start_date else 'start'} to {end_date.strftime('%Y-%m-%d') if end_date else 'end'})"
        print(f"Analyzing period: {date_range}")
    print("=" * 50)
    
    # Initialize analyzer with date range
    analyzer = LabUtilizationAnalyzer("logs", start_date=start_date, end_date=end_date)
    
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


def parse_date_argument(date_str):
    """Parse various date formats: 'March', 'March 2026', '2026-03-01', etc."""
    if not date_str:
        return None
        
    date_str = date_str.strip()
    
    # Handle month names (e.g., "March", "March 2026")
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Try "March" or "March 2026" format
    parts = date_str.lower().split()
    if parts[0] in month_names:
        month = month_names[parts[0]]
        year = int(parts[1]) if len(parts) > 1 else datetime.now().year
        return datetime(year, month, 1).date()
    
    # Try standard date formats
    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m']:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.date()
        except ValueError:
            continue
            
    raise ValueError(f"Could not parse date: {date_str}")


if __name__ == "__main__":
    import sys
    
    # Simple command line argument parsing
    start_date = None
    end_date = None
    
    if len(sys.argv) > 1:
        # Usage examples:
        # python utilization_analyzer.py March
        # python utilization_analyzer.py "March 2026" 
        # python utilization_analyzer.py 2026-03-01 2026-03-31
        # python utilization_analyzer.py --start "March 1 2026" --end "March 31 2026"
        
        args = sys.argv[1:]
        
if __name__ == "__main__":
    import sys
    
    # If no command line arguments, use the configuration at the top of the file
    if len(sys.argv) == 1:
        try:
            main()  # Uses ANALYZE_START_DATE and ANALYZE_END_DATE from config
        except Exception as e:
            print(f"Error: {e}")
            print("\nTo change the analysis period, edit ANALYZE_START_DATE and ANALYZE_END_DATE at the top of this file.")
    else:
        # Original command line argument parsing for advanced users
        start_date = None
        end_date = None
        
        args = sys.argv[1:]
        
        if '--start' in args:
            start_idx = args.index('--start') + 1
            if start_idx < len(args):
                start_date = parse_date_argument(args[start_idx])
        
        if '--end' in args:
            end_idx = args.index('--end') + 1
            if end_idx < len(args):
                end_date = parse_date_argument(args[end_idx])
        
        # Handle simple cases: single month or two dates
        if '--start' not in args and '--end' not in args:
            if len(args) == 1:
                # Single month or date
                date_arg = args[0]
                start_date = parse_date_argument(date_arg)
                # If it's a month name, make it the full month
                if any(month in date_arg.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                               'july', 'august', 'september', 'october', 'november', 'december']):
                    # Get last day of the month
                    year = start_date.year
                    month = start_date.month
                    if month == 12:
                        end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
                    else:
                        end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
            elif len(args) == 2:
                # Start and end dates
                start_date = parse_date_argument(args[0])
                end_date = parse_date_argument(args[1])
        
        try:
            main(start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error: {e}")
            print("\nUsage examples:")
            print("  python utilization_analyzer.py                    # Use config at top of file")
            print("  python utilization_analyzer.py March              # Analyze March (current year)")
            print("  python utilization_analyzer.py \"March 2026\"       # Analyze March 2026")
            print("  python utilization_analyzer.py 2026-03-01 2026-03-31  # Analyze specific date range")
            print("  python utilization_analyzer.py --start 2026-03-01 --end 2026-03-31  # Named parameters")