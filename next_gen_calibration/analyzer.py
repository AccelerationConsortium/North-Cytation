from __future__ import annotations
from typing import List, Dict, Any
import csv, os, math
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write_history_csv(self, records: List[Dict[str, Any]]):
        if not records:
            return
        path = os.path.join(self.out_dir, 'history.csv')
        fieldnames = sorted(records[0].keys())
        with open(path,'w',newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in records:
                w.writerow(r)
        return path

    def summary(self, records: List[Dict[str, Any]]):
        if not records:
            return {}
        devs = [r['avg_dev_ul'] for r in records if 'avg_dev_ul' in r]
        times = [r['avg_time_s'] for r in records if 'avg_time_s' in r]
        vars_ = [r['std_dev_ul'] for r in records if 'std_dev_ul' in r]
        summary = {
            'n': len(records),
            'dev_mean': sum(devs)/len(devs) if devs else math.nan,
            'time_mean': sum(times)/len(times) if times else math.nan,
            'var_mean': sum(vars_)/len(vars_) if vars_ else math.nan,
        }
        # Tolerance pass counts if present
        if any('pass_deviation' in r for r in records):
            summary.update({
                'pass_deviation_count': sum(1 for r in records if r.get('pass_deviation')),
                'pass_variation_count': sum(1 for r in records if r.get('pass_variation')),
                'pass_time_count': sum(1 for r in records if r.get('pass_time')),
                'pass_all_count': sum(1 for r in records if r.get('pass_all')),
            })
        return summary

    # --- Plotting ---
    def _save_fig(self, fig, name: str):
        path = os.path.join(self.out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    def plot_time_vs_deviation(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        fig, ax = plt.subplots(figsize=(5.2,4.2))
        xs = [r['avg_dev_ul'] for r in records]
        ys = [r['avg_time_s'] for r in records]
        vols = [r['volume_ul'] for r in records]
        scatter = ax.scatter(xs, ys, c=vols, cmap='viridis', alpha=0.8, edgecolors='none')
        ax.set_xlabel('Avg Deviation (uL)')
        ax.set_ylabel('Avg Time (s)')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Target Volume (uL)')
        ax.set_title('Time vs Deviation')
        return self._save_fig(fig, 'time_vs_deviation.png')

    def plot_deviation_over_iterations(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        fig, ax = plt.subplots(figsize=(5.4,3.8))
        devs = [r['avg_dev_ul'] for r in records]
        ax.plot(range(1,len(devs)+1), devs, marker='o', ms=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Deviation (uL)')
        ax.set_title('Deviation Over Iterations')
        return self._save_fig(fig, 'deviation_over_iterations.png')

    def plot_time_over_iterations(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        fig, ax = plt.subplots(figsize=(5.4,3.8))
        times = [r['avg_time_s'] for r in records]
        ax.plot(range(1,len(times)+1), times, marker='o', ms=3, color='tab:orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Time (s)')
        ax.set_title('Time Over Iterations')
        return self._save_fig(fig, 'time_over_iterations.png')

    def plot_volume_groups(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        # Plot deviation vs iteration per unique volume (small multiples stacked)
        vols = sorted(set(r['volume_ul'] for r in records))
        fig, axes = plt.subplots(len(vols), 1, figsize=(5.0, 2.2*len(vols)), sharex=True)
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        for ax, vol in zip(axes, vols):
            subset = [r for r in records if r['volume_ul']==vol]
            devs = [r['avg_dev_ul'] for r in subset]
            ax.plot(range(1,len(devs)+1), devs, marker='o', ms=3)
            ax.set_ylabel(f'{int(vol)}uL')
        axes[0].set_title('Deviation by Volume')
        axes[-1].set_xlabel('Iteration')
        return self._save_fig(fig, 'deviation_by_volume.png')

    def generate_all_plots(self, records: List[Dict[str, Any]]):
        return {
            'time_vs_deviation': self.plot_time_vs_deviation(records),
            'deviation_over_iterations': self.plot_deviation_over_iterations(records),
            'time_over_iterations': self.plot_time_over_iterations(records),
            'deviation_by_volume': self.plot_volume_groups(records)
        }
