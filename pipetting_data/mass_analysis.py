"""Mass analyzer for scale readings recorded during an aspirate or dispense.

The robot records readings (dicts with timestamp, time_relative, mass_g, phase, steady_status;
phase is 'baseline_pre', 'aspiration' or 'dispensing', or 'baseline_post'). This module turns
them into a mass and stability summary, and saves the readings and a plot. It does not touch
hardware. Used by North_Robot.aspirate_from_vial, dispense_into_vial and
dispense_into_vial_from_reservoir.
"""
import os

import pandas as pd

MIN_READINGS = 4  # fewer than this: no baselines, mass reported as 0.0


def empty_stability():
    """Stability summary before any readings (also the result when there are too few)."""
    return {
        'pre_stable_count': 0,
        'pre_total_count': 0,
        'post_stable_count': 0,
        'post_total_count': 0,
        'pre_baseline_std': 0.0,
        'post_baseline_std': 0.0,
    }


def _baseline(mass_df, phase):
    """(mean mass, method, readings in that phase, steady readings in that phase).
    Uses the steady readings if there are any, otherwise every reading in the phase."""
    in_phase = mass_df[mass_df['phase'] == phase]
    steady = in_phase[in_phase['steady_status'] == True]  # noqa: E712 (as recorded: True/False)
    if len(steady) > 0:
        return steady['mass_g'].mean(), "STABLE", in_phase, steady
    return in_phase['mass_g'].mean(), "TIME-BASED", in_phase, steady


def analyze_mass_readings(readings, operation):
    """Mass change from readings taken before, during and after one operation.

    Args:
        readings (list[dict]): scale readings as recorded by the robot
        operation (str): 'aspiration' (mass leaves the vial, reported positive) or
            'dispense' / 'reservoir' (mass added)

    Returns:
        dict with mass_g, pre_baseline_g, post_baseline_g, pre_method, post_method and
        stability (see empty_stability). With fewer than MIN_READINGS readings: mass_g 0.0,
        baselines None, stability all zero.
    """
    if len(readings) < MIN_READINGS:
        return {'mass_g': 0.0, 'pre_baseline_g': None, 'post_baseline_g': None,
                'pre_method': None, 'post_method': None, 'stability': empty_stability()}
    mass_df = pd.DataFrame(readings)
    pre, pre_method, pre_all, pre_steady = _baseline(mass_df, 'baseline_pre')
    post, post_method, post_all, post_steady = _baseline(mass_df, 'baseline_post')
    mass = pre - post if operation == 'aspiration' else post - pre
    stability = {
        'pre_stable_count': len(pre_steady),
        'pre_total_count': len(pre_all),
        'post_stable_count': len(post_steady),
        'post_total_count': len(post_all),
        'pre_baseline_std': pre_all['mass_g'].std() if len(pre_all) > 1 else 0.0,
        'post_baseline_std': post_all['mass_g'].std() if len(post_all) > 1 else 0.0,
    }
    return {'mass_g': mass, 'pre_baseline_g': pre, 'post_baseline_g': post,
            'pre_method': pre_method, 'post_method': post_method, 'stability': stability}


_PLOT_STYLE = {  # operation -> (phase, shading colour, label, title)
    'aspiration': ('aspiration', 'orange', 'Aspiration', 'Mass vs Time During Aspiration'),
    'dispense': ('dispensing', 'blue', 'Dispensing', 'Mass vs Time During Dispense'),
    'reservoir': ('dispensing', 'blue', 'Dispensing', 'Reservoir Mass vs Time During Dispense'),
}


def plot_mass_vs_time(readings, result, operation, vial_label, target_mL, plot_path):
    """Save a mass vs time plot: readings (steady green, unsteady red), phases shaded,
    baselines as dashed lines."""
    import matplotlib.pyplot as plt

    phase_name, colour, label, title = _PLOT_STYLE[operation]
    mass_df = pd.DataFrame(readings)
    plt.figure(figsize=(12, 6))
    plt.plot(mass_df['time_relative'], mass_df['mass_g'], 'k-', linewidth=2, label='Mass readings', zorder=3)
    stable = mass_df[mass_df['steady_status'] == True]  # noqa: E712
    unstable = mass_df[mass_df['steady_status'] == False]  # noqa: E712
    if len(stable) > 0:
        plt.scatter(stable['time_relative'], stable['mass_g'], c='green', s=20, alpha=0.7,
                    label='Stable readings', zorder=4)
    if len(unstable) > 0:
        plt.scatter(unstable['time_relative'], unstable['mass_g'], c='red', s=20, alpha=0.7,
                    label='Unstable readings', zorder=4)
    for phase, shade, shade_label in [('baseline_pre', 'green', 'Pre-baseline'), (phase_name, colour, label),
                                      ('baseline_post', 'red', 'Post-baseline')]:
        times = mass_df[mass_df['phase'] == phase]['time_relative']
        if len(times) > 0:
            plt.axvspan(times.min(), times.max(), alpha=0.2, color=shade, label=shade_label, zorder=1)
    pre, post = result['pre_baseline_g'], result['post_baseline_g']
    plt.axhline(y=pre, color='g', linestyle='--', alpha=0.7, label=f'Pre-avg: {pre:.3f}g')
    plt.axhline(y=post, color='r', linestyle='--', alpha=0.7, label=f'Post-avg: {post:.3f}g')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mass (g)')
    plt.title(f"{title}\\nVial: {vial_label}, Target: {target_mL:.3f} mL, Measured: {result['mass_g']:.4f} g")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_mass_readings(readings, result, operation, vial_label, target_mL, folder, csv_name, plot_name, logger):
    """Save the readings as CSV and, when there were enough for baselines, the plot."""
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, csv_name)
    pd.DataFrame(readings).to_csv(csv_path, index=False)
    logger.info(f"Saved {len(readings)} {operation} mass measurements to {csv_path}")
    if result['pre_baseline_g'] is None:
        logger.warning(f"Too few readings ({len(readings)}) for a {operation} mass plot; CSV saved only")
        return
    plot_path = os.path.join(folder, plot_name)
    plot_mass_vs_time(readings, result, operation, vial_label, target_mL, plot_path)
    logger.info(f"Saved {operation} mass vs time plot to {plot_path}")
