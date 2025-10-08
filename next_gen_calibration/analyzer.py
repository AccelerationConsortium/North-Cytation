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
        # Union all keys across records to avoid ValueError when new fields appear later.
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        path = os.path.join(self.out_dir, 'history.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in records:
                w.writerow(r)
        return path

    def summary(self, records: List[Dict[str, Any]]):
        if not records:
            return {}
        # Backward compatibility helper: prefer new keys (avg_accuracy_ul, std_precision_ul)
        def _get(r, new_key, old_key):
            if new_key in r: return r[new_key]
            if old_key in r: return r[old_key]
            return None
        devs = [v for r in records if (v := _get(r,'avg_accuracy_ul','avg_dev_ul')) is not None]
        times = [r['avg_time_s'] for r in records if 'avg_time_s' in r]
        vars_ = [v for r in records if (v := _get(r,'std_precision_ul','std_dev_ul')) is not None]
        summary = {
            'n': len(records),
            # New canonical aggregate names (0.6.5+)
            'mean_accuracy_dev_ul': sum(devs)/len(devs) if devs else math.nan,
            'mean_time_s': sum(times)/len(times) if times else math.nan,
            'mean_precision_std_ul': sum(vars_)/len(vars_) if vars_ else math.nan,
            # Legacy keys preserved for downstream consumers
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
        # Precision test stats (next-gen schema)
        if any('precision_pass' in r for r in records):
            summary['precision_pass_total'] = sum(1 for r in records if r.get('precision_pass'))
            # Count distinct volumes that achieved at least one precision pass
            summary['precision_pass_volume_count'] = len({r['volume_ul'] for r in records if r.get('precision_pass')})
        return summary

    def per_volume_summary(self, records: List[Dict[str, Any]]):
        if not records:
            return {}
        by_vol: Dict[Any, list] = {}
        for r in records:
            vol = r.get('volume_ul')
            if vol is None: continue
            by_vol.setdefault(vol, []).append(r)
        out: Dict[Any, Dict[str, Any]] = {}
        def _get(r, new_key, old_key):
            if new_key in r: return r[new_key]
            if old_key in r: return r[old_key]
            return None
        for vol, recs in sorted(by_vol.items()):
            screen = [r for r in recs if r.get('phase')=='screening']
            opt = [r for r in recs if r.get('phase')=='optimization']
            prec = [r for r in recs if r.get('phase')=='precision_test']
            accs = [v for r in (screen+opt) if (v := _get(r,'avg_accuracy_ul','avg_dev_ul')) is not None]
            times = [r['avg_time_s'] for r in (screen+opt) if 'avg_time_s' in r]
            stds = [v for r in (screen+opt) if (v := _get(r,'std_precision_ul','std_dev_ul')) is not None]
            gate_hits = [r for r in (screen+opt) if r.get('gate_accuracy_met') and r.get('gate_time_met')]
            out[vol] = {
                'volume_ul': vol,
                'screening_trials': len(screen),
                'optimization_trials': len(opt),
                'precision_attempts': len(prec),
                'precision_pass': any(r.get('precision_pass') for r in prec),
                'best_accuracy_dev_ul': min(accs) if accs else math.nan,
                'best_time_s': min(times) if times else math.nan,
                'best_precision_std_ul': min(stds) if stds else math.nan,
                'achieved_gate': bool(gate_hits),
            }
        return out

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
        def _val(r, new_k, old_k):
            return r[new_k] if new_k in r else r.get(old_k, math.nan)
        xs = [_val(r,'avg_accuracy_ul','avg_dev_ul') for r in records]
        ys = [r.get('avg_time_s', math.nan) for r in records]
        vols = [r['volume_ul'] for r in records]
        scatter = ax.scatter(xs, ys, c=vols, cmap='viridis', alpha=0.8, edgecolors='none')
        ax.set_xlabel('Avg Accuracy Deviation (uL)')
        ax.set_ylabel('Avg Time (s)')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Target Volume (uL)')
        ax.set_title('Time vs Deviation')
        return self._save_fig(fig, 'time_vs_deviation.png')

    def plot_deviation_over_iterations(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        fig, ax = plt.subplots(figsize=(5.4,3.8))
        # Only include phases providing accuracy metrics (screening/optimization)
        metric_records = [r for r in records if ('avg_accuracy_ul' in r) or ('avg_dev_ul' in r)]
        devs = [r['avg_accuracy_ul'] if 'avg_accuracy_ul' in r else r.get('avg_dev_ul', math.nan) for r in metric_records]
        ax.plot(range(1,len(devs)+1), devs, marker='o', ms=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Accuracy Dev (uL)')
        ax.set_title('Accuracy Deviation Over Iterations')
        return self._save_fig(fig, 'deviation_over_iterations.png')

    def plot_time_over_iterations(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        fig, ax = plt.subplots(figsize=(5.4,3.8))
        time_records = [r for r in records if 'avg_time_s' in r]
        times = [r['avg_time_s'] for r in time_records]
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
        # Normalize axes list
        if hasattr(axes, 'ravel'):
            axes_list = axes.ravel().tolist()
        else:
            axes_list = [axes]
        for ax, vol in zip(axes_list, vols):
            subset = [r for r in records if r['volume_ul']==vol and (('avg_accuracy_ul' in r) or ('avg_dev_ul' in r))]
            devs = [r['avg_accuracy_ul'] if 'avg_accuracy_ul' in r else r.get('avg_dev_ul', math.nan) for r in subset]
            ax.plot(range(1,len(devs)+1), devs, marker='o', ms=3)
            ax.set_ylabel(f'{int(vol)}uL')
        axes_list[0].set_title('Accuracy Deviation by Volume')
        axes_list[-1].set_xlabel('Iteration')
        return self._save_fig(fig, 'deviation_by_volume.png')

    def plot_time_by_volume(self, records: List[Dict[str, Any]]):
        if not records:
            return None
        vols = sorted(set(r['volume_ul'] for r in records))
        fig, axes = plt.subplots(len(vols), 1, figsize=(5.0, 2.2*len(vols)), sharex=True)
        axes_list = axes.ravel().tolist() if hasattr(axes,'ravel') else [axes]
        for ax, vol in zip(axes_list, vols):
            subset = [r for r in records if r['volume_ul']==vol and 'avg_time_s' in r]
            times = [r['avg_time_s'] for r in subset]
            ax.plot(range(1,len(times)+1), times, marker='o', ms=3, color='tab:orange')
            ax.set_ylabel(f'{int(vol)}uL')
        axes_list[0].set_title('Time by Volume')
        axes_list[-1].set_xlabel('Iteration')
        return self._save_fig(fig, 'time_by_volume.png')

    def plot_replicate_volume_scatter(self, records: List[Dict[str, Any]]):
        # Combined replicate-level scatter akin to legacy measured volume over time.
        # We flatten replicate_vols_ul if present.
        pts = []
        for r in records:
            vol_ul = r.get('volume_ul')
            reps = r.get('replicate_vols_ul')
            if vol_ul is None or not reps:
                continue
            for rv in reps:
                pts.append((vol_ul, rv))
        if not pts:
            return None
        fig, ax = plt.subplots(figsize=(8,5))
        vols = sorted(set(p[0] for p in pts))
        colors = plt.cm.tab10(range(len(vols)))
        idx = 0
        for i, vol in enumerate(vols):
            vol_pts = [p[1] for p in pts if p[0]==vol]
            ax.scatter(range(idx, idx+len(vol_pts)), vol_pts, s=30, color=colors[i], alpha=0.75, label=f'{int(vol)}uL target')
            ax.axhline(y=vol, linestyle='--', color=colors[i], linewidth=1.2, alpha=0.8)
            idx += len(vol_pts)
        ax.set_xlabel('Measurement Index (across volumes)')
        ax.set_ylabel('Measured Volume (uL)')
        ax.set_title('Replicate Measured Volume (All Volumes)')
        ax.legend(fontsize='small')
        ax.grid(alpha=0.3)
        return self._save_fig(fig, 'replicate_volume_scatter.png')

    def plot_replicate_time_scatter(self, records: List[Dict[str, Any]]):
        pts = []
        for r in records:
            vol_ul = r.get('volume_ul')
            times = r.get('replicate_times_s')
            if vol_ul is None or not times:
                continue
            for t in times:
                pts.append((vol_ul, t))
        if not pts:
            return None
        fig, ax = plt.subplots(figsize=(8,5))
        vols = sorted(set(p[0] for p in pts))
        colors = plt.cm.tab10(range(len(vols)))
        idx = 0
        for i, vol in enumerate(vols):
            vol_pts = [p[1] for p in pts if p[0]==vol]
            ax.scatter(range(idx, idx+len(vol_pts)), vol_pts, s=30, color=colors[i], alpha=0.75, label=f'{int(vol)}uL target')
            idx += len(vol_pts)
        ax.set_xlabel('Measurement Index (across volumes)')
        ax.set_ylabel('Replicate Time (s)')
        ax.set_title('Replicate Time (All Volumes)')
        ax.legend(fontsize='small')
        ax.grid(alpha=0.3)
        return self._save_fig(fig, 'replicate_time_scatter.png')

    def generate_all_plots(self, records: List[Dict[str, Any]]):
        outputs = {}
        for name, fn in [
            ('time_vs_deviation', self.plot_time_vs_deviation),
            ('deviation_by_volume', self.plot_volume_groups),
            ('time_by_volume', self.plot_time_by_volume),
            ('replicate_volume_scatter', self.plot_replicate_volume_scatter),
            ('replicate_time_scatter', self.plot_replicate_time_scatter),
        ]:
            try:
                outputs[name] = fn(records)
            except KeyError as e:
                print(f"[analyzer] Skipping plot {name} missing key: {e}")
            except Exception as e:
                print(f"[analyzer] Plot {name} failed: {e}")
        return outputs
