from __future__ import annotations
import yaml, os, json, time, importlib
from typing import Dict, Any, Callable, List
from .robot_adapter import RobotAdapter  # Fallback path only
from .analyzer import Analyzer
from .recommender import make_recommender


# Unified minimal protocol callable types (0.7.0)
ProtocolInitialize = Callable[..., dict]
ProtocolMeasure = Callable[[dict, float, dict, int], list]
ProtocolWrapup = Callable[[dict], None]


def load_config(path: str) -> Dict[str, Any]:
    with open(path,'r') as f:
        return yaml.safe_load(f)


def _select_module(cfg: Dict[str, Any], simulate: bool) -> str | None:
    proto_cfg = cfg.get('protocol', {})
    # Highest precedence: environment override
    env_mod = os.environ.get('CALIBRATION_PROTOCOL_MODULE')
    if env_mod:
        return env_mod
    # Explicit unified module overrides dual entries
    if proto_cfg.get('module'):
        return proto_cfg['module']
    # Dual module keys
    if simulate and proto_cfg.get('simulated_module'):
        return proto_cfg['simulated_module']
    if (not simulate) and proto_cfg.get('hardware_module'):
        return proto_cfg['hardware_module']
    return None


def _import_protocol(module_name: str, proto_cfg: Dict[str, Any]):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[orchestrator] Failed to import protocol module '{module_name}': {e}")
        return None
    class_name = proto_cfg.get('class_name')
    if class_name:
        try:
            cls = getattr(mod, class_name)
            inst = cls(**proto_cfg.get('kwargs', {}))
            return inst, getattr(inst, 'initialize', None), getattr(inst, 'measure', None), getattr(inst, 'wrapup', None)
        except Exception as e:
            print(f"[orchestrator] Failed to instantiate class '{class_name}' in '{module_name}': {e}")
    return mod, getattr(mod, 'initialize', None), getattr(mod, 'measure', None), getattr(mod, 'wrapup', None)


def _run_trial_measure(measure_fn: ProtocolMeasure, state: dict, vol_mL: float, params: dict, replicates: int) -> dict:
    measurements = measure_fn(state, vol_mL, params, replicates)
    vol_ul_target = vol_mL * 1000
    vols_ul: List[float] = []
    times_s: List[float] = []
    for m in measurements:
        v = m.get('volume')
        if v is None:
            continue
        vols_ul.append(v * 1000.0)
        times_s.append(m.get('elapsed_s', 0.0))
    if not vols_ul:
        return {'avg_vol_ul': 0, 'accuracy_dev_ul': float('inf'), 'std_precision_ul': float('inf'), 'avg_time_s': float('inf'), 'replicate_devs_ul': []}
    avg_vol = sum(vols_ul)/len(vols_ul)
    accuracy_dev_ul = abs(avg_vol - vol_ul_target)
    std_ul = (sum((v-avg_vol)**2 for v in vols_ul)/len(vols_ul))**0.5
    return {
        'avg_vol_ul': avg_vol,
        'accuracy_dev_ul': accuracy_dev_ul,
        'std_precision_ul': std_ul,
        'avg_time_s': sum(times_s)/len(times_s) if times_s else 0.0,
        'replicate_devs_ul': [abs(v - vol_ul_target) for v in vols_ul],
        'replicate_vols_ul': vols_ul,
        'replicate_times_s': times_s
    }


def _precision_test_trial(measure_fn: ProtocolMeasure, state: dict,
                          vol_mL: float, params: dict, band_abs_ul: float, replicates: int) -> dict:
    trial = _run_trial_measure(measure_fn, state, vol_mL, params, replicates)
    max_dev = max(trial['replicate_devs_ul']) if trial['replicate_devs_ul'] else float('inf')
    trial.update({'precision_max_dev_ul': max_dev, 'precision_pass': max_dev <= band_abs_ul})
    return trial


def main(config_path: str = 'next_gen_calibration/params.yaml'):
    cfg = load_config(config_path)
    simulate = cfg.get('simulate', True)
    out_base = cfg.get('output',{}).get('base_dir','output/next_gen_runs')
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(out_base, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    proto_cfg = cfg.get('protocol', {})
    module_name = _select_module(cfg, simulate)
    protocol_obj = None
    initialize: ProtocolInitialize | None = None
    measure: ProtocolMeasure | None = None
    wrapup: ProtocolWrapup | None = None
    state: dict = {}
    if module_name:
        protocol_obj, initialize, measure, wrapup = _import_protocol(module_name, proto_cfg) or (None, None, None, None, None)
        if all(callable(x) for x in (initialize, measure, wrapup)):
            # Attempt initialize with config first; fallback to no-arg
            tried_cfg = False
            try:
                init_result = initialize(cfg)
                tried_cfg = True
                if isinstance(init_result, dict):
                    state.update(init_result)
            except TypeError:
                # Signature likely doesn't accept cfg; try no-arg
                try:
                    init_result = initialize()
                    if isinstance(init_result, dict):
                        state.update(init_result)
                except Exception as e:
                    print(f"[orchestrator] Protocol initialize() failed (no-arg path): {e}; falling back to adapter.")
                    module_name = None
            except Exception as e:
                print(f"[orchestrator] Protocol initialize() failed ({'with cfg' if tried_cfg else 'attempt'}): {e}; falling back to adapter.")
                module_name = None
        else:
            print(f"[orchestrator] Protocol module '{module_name}' missing required hooks; falling back to adapter.")
            module_name = None

    if not module_name:
        adapter = RobotAdapter(liquid=cfg.get('liquid','water'), simulate=simulate, seed=cfg.get('random_seed'))

    recommender = make_recommender(cfg)
    analyzer = Analyzer(run_dir)

    volumes_mL = cfg.get('volumes',[0.05])
    thresholds = dict(cfg['thresholds'])
    if simulate and 'simulation_thresholds' in cfg:
        thresholds.update(cfg['simulation_thresholds'] or {})
    phases = cfg.get('phases', {})
    screening_cfg = phases.get('screening', {})
    optimization_cfg = phases.get('optimization', {})
    precision_cfg = phases.get('precision', {})
    precision_reps = precision_cfg.get('replicates', 5)
    opt_reps = optimization_cfg.get('replicates', 1)
    screening_trials = screening_cfg.get('n_trials', 5)
    opt_max_iters = optimization_cfg.get('max_iters', 10)
    max_trials_per_volume = phases.get('max_trials_per_volume', optimization_cfg.get('max_trials_per_volume', 999999))
    selective_cfg = cfg.get('volume_dependant_optimization', {})
    selective_enabled = selective_cfg.get('enable', False)
    vol_dep_params = set(selective_cfg.get('volume_dependent_params', []))

    successful_volume_params: list[tuple[float, dict]] = []
    records = []
    def run_trial(vol_mL: float, params: dict, reps: int):
        if module_name:
            return _run_trial_measure(measure, state, vol_mL, params, reps)
        else:
            vol_ul = vol_mL * 1000
            res = adapter.pipet_and_measure(vol_mL, params, reps)
            calculated = res.calculated_volumes_ul
            avg_vol = sum(calculated)/len(calculated)
            accuracy_dev_ul = abs(avg_vol - vol_ul)
            std_ul = (sum((v-avg_vol)**2 for v in calculated)/len(calculated))**0.5
            avg_time = sum(res.replicate_times_s)/len(res.replicate_times_s)
            return {
                'avg_vol_ul': avg_vol,
                'accuracy_dev_ul': accuracy_dev_ul,
                'std_precision_ul': std_ul,
                'avg_time_s': avg_time,
                'replicate_devs_ul': [abs(v - vol_ul) for v in calculated],
                'replicate_vols_ul': calculated,
                'replicate_times_s': res.replicate_times_s
            }

    def precision_test(vol_mL: float, params: dict, band_abs_ul: float, reps: int):
        if module_name:
            return _precision_test_trial(measure, state, vol_mL, params, band_abs_ul, reps)
        else:
            t = run_trial(vol_mL, params, reps)
            max_dev = max(t['replicate_devs_ul']) if t['replicate_devs_ul'] else float('inf')
            t.update({'precision_max_dev_ul': max_dev, 'precision_pass': max_dev <= band_abs_ul})
            return t

    print(f"[run] Volumes queued: {[int(v*1000) for v in volumes_mL]} uL | simulate={simulate} | protocol={module_name or 'adapter'}")
    for vol_idx, vol in enumerate(volumes_mL):
        print(f"[run] === Volume {vol_idx+1}/{len(volumes_mL)}: {vol*1000:.0f} uL ===")
        if vol_idx > 0 and selective_cfg.get('enable', False) and not selective_cfg.get('use_historical', False):
            recommender = make_recommender(cfg)
        vol_ul = vol * 1000
        volume_excess_ul = max(0, vol_ul - 100)
        scaling_factor = volume_excess_ul / 100.0
        accuracy_limit_ul = thresholds.get('base_accuracy_ul',1.0) + thresholds.get('accuracy_scaling_per_100ul',0.2) * scaling_factor
        time_limit_s = thresholds.get('base_time_s',20) + thresholds.get('time_scaling_per_100ul',1.0) * scaling_factor
        precision_band = thresholds.get('base_precision_ul',2.0) + thresholds.get('precision_scaling_per_100ul',0.2) * scaling_factor
        param_spec = cfg['parameters'].get('overaspirate_vol')
        if param_spec:
            frac = param_spec.get('max_fraction_of_target_volume')
            if frac is not None:
                dyn_high = vol * frac
                param_spec['bounds'][1] = min(param_spec['bounds'][1], dyn_high)

        fixed_params = {}
        if selective_enabled and successful_volume_params:
            last_params = successful_volume_params[-1][1]
            for p,v in last_params.items():
                if p not in vol_dep_params:
                    fixed_params[p] = v

        def apply_fixed(pdict: Dict[str, Any]):
            for k,v in fixed_params.items():
                pdict[k] = v
            return pdict

        trials_done = 0
        precision_passed = False
        attempted_precision_params: set[tuple] = set()

        print(f"[run] Screening: {screening_trials} trials (replicates={precision_reps})")
        for params in recommender.suggest(screening_trials):
            params = apply_fixed(params)
            trial = run_trial(vol, params, precision_reps)
            records.append({
                'phase': 'screening',
                'volume_ul': vol_ul,
                'avg_accuracy_ul': trial['accuracy_dev_ul'],
                'avg_time_s': trial['avg_time_s'],
                'std_precision_ul': trial['std_precision_ul'],
                'gate_accuracy_met': trial['accuracy_dev_ul'] <= accuracy_limit_ul,
                'gate_time_met': trial['avg_time_s'] <= time_limit_s,
                'replicate_vols_ul': trial.get('replicate_vols_ul'),
                'replicate_times_s': trial.get('replicate_times_s'),
                'params': params,
                'protocol': module_name or 'adapter'
            })
            recommender.observe(params, {'objective': trial['accuracy_dev_ul'], 'volume_ul': vol_ul, **trial})
            trials_done += 1
            if trials_done >= max_trials_per_volume:
                break

        achieved_gate = False
        print(f"[run] Optimization loop: max {opt_max_iters} iterations (replicates={opt_reps})")
        achieved_gate = False
        for i in range(opt_max_iters):
            if trials_done >= max_trials_per_volume:
                break
            params = apply_fixed(recommender.suggest(1)[0])
            trial = run_trial(vol, params, opt_reps)
            records.append({
                'phase': 'optimization',
                'iteration': i+1,
                'volume_ul': vol_ul,
                'avg_accuracy_ul': trial['accuracy_dev_ul'],
                'avg_time_s': trial['avg_time_s'],
                'std_precision_ul': trial['std_precision_ul'],
                'gate_accuracy_met': trial['accuracy_dev_ul'] <= accuracy_limit_ul,
                'gate_time_met': trial['avg_time_s'] <= time_limit_s,
                'replicate_vols_ul': trial.get('replicate_vols_ul'),
                'replicate_times_s': trial.get('replicate_times_s'),
                'params': params,
                'protocol': module_name or 'adapter'
            })
            recommender.observe(params, {'objective': trial['accuracy_dev_ul'], 'volume_ul': vol_ul, **trial})
            trials_done += 1
            if (trial['accuracy_dev_ul'] <= accuracy_limit_ul) and (trial['avg_time_s'] <= time_limit_s):
                achieved_gate = True
                break

        if achieved_gate:
            print(f"[run] Entering precision test band={precision_band:.2f} uL with best-so-far params")
        while achieved_gate and not precision_passed and trials_done < max_trials_per_volume:
            best_params = recommender.best_params() or params
            key = tuple(sorted(best_params.items()))
            if key in attempted_precision_params:
                best_params = apply_fixed(recommender.suggest(1)[0])
                key = tuple(sorted(best_params.items()))
            attempted_precision_params.add(key)
            p_res = precision_test(vol, best_params, precision_band, precision_reps)
            records.append({
                'phase': 'precision_test',
                'volume_ul': vol_ul,
                'precision_pass': p_res['precision_pass'],
                'precision_max_dev_ul': p_res['precision_max_dev_ul'],
                'precision_band_ul': precision_band,
                'replicate_vols_ul': p_res.get('replicate_vols_ul'),
                'replicate_times_s': p_res.get('replicate_times_s'),
                'params': best_params,
                'protocol': module_name or 'adapter'
            })
            trials_done += 1
            if p_res['precision_pass']:
                precision_passed = True
                print(f"[run] Precision PASS at {vol*1000:.0f} uL after {trials_done} trials")
                successful_volume_params.append((vol, best_params))
                break
            if trials_done >= max_trials_per_volume:
                break
            params = apply_fixed(recommender.suggest(1)[0])
            trial = run_trial(vol, params, opt_reps)
            records.append({
                'phase': 'optimization',
                'iteration': i+1,
                'volume_ul': vol_ul,
                'avg_accuracy_ul': trial['accuracy_dev_ul'],
                'avg_time_s': trial['avg_time_s'],
                'std_precision_ul': trial['std_precision_ul'],
                'gate_accuracy_met': trial['accuracy_dev_ul'] <= accuracy_limit_ul,
                'gate_time_met': trial['avg_time_s'] <= time_limit_s,
                'replicate_vols_ul': trial.get('replicate_vols_ul'),
                'replicate_times_s': trial.get('replicate_times_s'),
                'params': params,
                'protocol': module_name or 'adapter'
            })
            recommender.observe(params, {'objective': trial['accuracy_dev_ul'], 'volume_ul': vol_ul, **trial})
            trials_done += 1

    # Wrapup
    if module_name and callable(wrapup):
        try:
            wrapup(state)
        except Exception as e:
            print(f"[orchestrator] Protocol wrapup failed: {e}")
    analyzer.write_history_csv(records)
    summary = analyzer.summary(records)
    per_volume = analyzer.per_volume_summary(records)
    with open(os.path.join(run_dir,'summary.json'),'w') as f:
        json.dump(summary,f,indent=2)
    with open(os.path.join(run_dir,'per_volume_summary.json'),'w') as f:
        json.dump(per_volume,f,indent=2)
    try:
        plot_paths = analyzer.generate_all_plots(records)
        filtered = {k: os.path.basename(v) for k,v in plot_paths.items() if v}
        with open(os.path.join(run_dir,'plots.json'),'w') as f:
            json.dump(filtered, f, indent=2)
    except Exception as e:
        print('Plot generation failed:', e)
    print('Run complete ->', run_dir)
    print('Summary:', summary)

if __name__ == '__main__':
    main()
