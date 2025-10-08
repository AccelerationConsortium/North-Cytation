from __future__ import annotations
import yaml, os, math, json, time
from typing import Dict, Any
from .robot_adapter import RobotAdapter
from .optimizer import SimpleBayesLikeOptimizer
from .analyzer import Analyzer


def load_config(path: str) -> Dict[str, Any]:
    with open(path,'r') as f:
        return yaml.safe_load(f)


def scale_criteria(volume_ul: float, criteria_cfg: Dict[str, Any]):
    vol_ratio = volume_ul / 100.0  # relative to 100 uL
    return {
        'max_dev_ul': criteria_cfg['base_deviation_ul'] * (1 + criteria_cfg['deviation_scaling_per_100ul'] * vol_ratio),
        'max_time_s': criteria_cfg['base_time_s'] * (1 + criteria_cfg['time_scaling_per_100ul'] * vol_ratio),
        'max_var_ul': criteria_cfg['base_variation_ul'] * (1 + criteria_cfg['variation_scaling_per_100ul'] * vol_ratio),
    }


def objective_from_metrics(avg_dev_ul: float, avg_time_s: float, std_dev_ul: float):
    # Weighted sum; tune as needed
    return avg_dev_ul + 0.2 * avg_time_s + 0.5 * std_dev_ul


def evaluate_tolerances(dev_ul: float, std_ul: float, avg_time: float, vol_ul: float, policy: Dict[str, Any]):
    """Evaluate tolerance policy returning pass flags.

    Policy structure (subset):
    tolerance_policy:
      deviation: { absolute_ul, percent }
      variation: { stdev_ul }
      time: { seconds }
      logic: any_deviation | both
    """
    if not policy:
        return {
            'pass_deviation': True,
            'pass_variation': True,
            'pass_time': True,
            'pass_all': True,
            'deviation_mode': 'no_policy'
        }
    dev_cfg = policy.get('deviation', {})
    var_cfg = policy.get('variation', {})
    time_cfg = policy.get('time', {})
    logic = policy.get('logic', 'any_deviation')
    # Percent deviation
    pct_dev = (dev_ul / vol_ul * 100) if vol_ul else float('inf')
    abs_pass = dev_ul <= dev_cfg.get('absolute_ul', float('inf'))
    pct_pass = pct_dev <= dev_cfg.get('percent', float('inf'))
    if logic == 'both':
        pass_deviation = abs_pass and pct_pass
        deviation_mode = 'both'
    else:  # any_deviation (default)
        pass_deviation = abs_pass or pct_pass
        deviation_mode = 'any'
    pass_variation = std_ul <= var_cfg.get('stdev_ul', float('inf'))
    pass_time = avg_time <= time_cfg.get('seconds', float('inf'))
    pass_all = pass_deviation and pass_variation and pass_time
    return {
        'pass_deviation': pass_deviation,
        'pass_variation': pass_variation,
        'pass_time': pass_time,
        'pass_all': pass_all,
        'pct_dev': pct_dev,
        'deviation_mode': deviation_mode
    }


def main(config_path: str = 'next_gen_calibration/params.yaml'):
    cfg = load_config(config_path)
    simulate = cfg.get('simulate', True)
    out_base = cfg.get('output',{}).get('base_dir','output/next_gen_runs')
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_base, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    adapter = RobotAdapter(liquid=cfg.get('liquid','water'), simulate=simulate, seed=cfg.get('seed'))

    param_space = cfg['parameters']
    opt = SimpleBayesLikeOptimizer(param_space, seed=cfg.get('seed'))
    analyzer = Analyzer(run_dir)

    volumes_mL = cfg.get('volumes',[0.05])
    initial = cfg.get('initial_suggestions',5)
    batch_size = cfg.get('batch_size',1)
    precision_reps = cfg.get('precision_replicates',5)
    criteria_cfg = cfg['criteria']
    tolerance_policy = cfg.get('tolerance_policy', {})

    records = []
    for vol in volumes_mL:
        vol_ul = vol * 1000
        crit = scale_criteria(vol_ul, criteria_cfg)
        # initial suggestions
        suggestions = opt.suggest(initial)
        for params in suggestions:
            res = adapter.pipet_and_measure(vol, params, precision_reps)
            avg_vol = sum(res.calculated_volumes_ul)/len(res.calculated_volumes_ul)
            dev_ul = abs(avg_vol - vol_ul)
            std_ul = (sum((v-avg_vol)**2 for v in res.calculated_volumes_ul)/len(res.calculated_volumes_ul))**0.5
            avg_time = sum(res.replicate_times_s)/len(res.replicate_times_s)
            objective = objective_from_metrics(dev_ul, avg_time, std_ul)
            opt.observe(params, objective, {'volume_ul': vol_ul, 'avg_dev_ul': dev_ul, 'avg_time_s': avg_time, 'std_dev_ul': std_ul})
            tol = evaluate_tolerances(dev_ul, std_ul, avg_time, vol_ul, tolerance_policy)
            rec = {
                'volume_ul': vol_ul,
                'avg_dev_ul': dev_ul,
                'avg_time_s': avg_time,
                'std_dev_ul': std_ul,
                'objective': objective,
                'meets_dev': dev_ul <= crit['max_dev_ul'],
                'meets_time': avg_time <= crit['max_time_s'],
                'meets_var': std_ul <= crit['max_var_ul'],
                # tolerance policy flags
                **tol,
                'params': params,
            }
            records.append(rec)
        # iterative improvement (placeholder simple loop)
        for _ in range(10):
            best = opt.best_params() or {}
            for params in opt.suggest(batch_size):
                res = adapter.pipet_and_measure(vol, params, precision_reps)
                avg_vol = sum(res.calculated_volumes_ul)/len(res.calculated_volumes_ul)
                dev_ul = abs(avg_vol - vol_ul)
                std_ul = (sum((v-avg_vol)**2 for v in res.calculated_volumes_ul)/len(res.calculated_volumes_ul))**0.5
                avg_time = sum(res.replicate_times_s)/len(res.replicate_times_s)
                objective = objective_from_metrics(dev_ul, avg_time, std_ul)
                opt.observe(params, objective, {'volume_ul': vol_ul, 'avg_dev_ul': dev_ul, 'avg_time_s': avg_time, 'std_dev_ul': std_ul})
                tol = evaluate_tolerances(dev_ul, std_ul, avg_time, vol_ul, tolerance_policy)
                rec = {
                    'volume_ul': vol_ul,
                    'avg_dev_ul': dev_ul,
                    'avg_time_s': avg_time,
                    'std_dev_ul': std_ul,
                    'objective': objective,
                    'meets_dev': dev_ul <= crit['max_dev_ul'],
                    'meets_time': avg_time <= crit['max_time_s'],
                    'meets_var': std_ul <= crit['max_var_ul'],
                    **tol,
                    'params': params,
                }
                records.append(rec)

    analyzer.write_history_csv(records)
    summary = analyzer.summary(records)
    with open(os.path.join(run_dir,'summary.json'),'w') as f:
        json.dump(summary,f,indent=2)
    # Generate plots (minimal set)
    try:
        plot_paths = analyzer.generate_all_plots(records)
        with open(os.path.join(run_dir,'plots.json'),'w') as f:
            json.dump({k: os.path.basename(v) for k,v in plot_paths.items()}, f, indent=2)
    except Exception as e:
        print('Plot generation failed:', e)
    print('Run complete ->', run_dir)
    print('Summary:', summary)

if __name__ == '__main__':
    main()
