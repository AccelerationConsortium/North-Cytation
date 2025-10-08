from __future__ import annotations
from typing import Dict, Any, List, Optional
import random

try:
    from ax.service.ax_client import AxClient  # type: ignore
    _AX_AVAILABLE = True
except Exception:  # pragma: no cover
    _AX_AVAILABLE = False

class BaseRecommender:
    """Abstract interface for calibration recommenders.

    Methods:
      suggest(n): return list of param dicts
      observe(params, metrics): record outcome (metrics contains fields used by optimizer)
      best_params(): optional best-so-far param dict
    """
    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        raise NotImplementedError
    def observe(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        raise NotImplementedError
    def best_params(self) -> Optional[Dict[str, Any]]:
        return None

class RandomLocalRecommender(BaseRecommender):
    """Localized random search similar to previous SimpleBayesLikeOptimizer."""
    def __init__(self, param_space: Dict[str, Dict[str, Any]], seed: Optional[int] = None):
        self.param_space = param_space
        self.history: List[Dict[str, Any]] = []
        if seed is not None:
            random.seed(seed)

    def _sample_param(self, name: str, spec: Dict[str, Any], center: Any | None = None):
        bounds = spec['bounds']
        low, high = bounds
        if center is not None:
            span = (high - low) * 0.3
            low2 = max(low, center - span/2)
            high2 = min(high, center + span/2)
            low, high = low2, high2
        if spec['type'] == 'int':
            return random.randint(int(low), int(high))
        if spec['type'] == 'float':
            return random.uniform(low, high)
        raise ValueError(f"Unsupported param type {spec['type']}")

    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        if not self.history:
            centers = {}
        else:
            best = min(self.history, key=lambda r: r['objective'])
            centers = best['params']
        out = []
        for _ in range(n):
            params = {name: self._sample_param(name, spec, centers.get(name)) for name, spec in self.param_space.items()}
            out.append(params)
        return out

    def observe(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        self.history.append({'params': params, **metrics})

    def best_params(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        best = min(self.history, key=lambda r: r['objective'])
        return best['params']

class AxRecommender(BaseRecommender):
    """Ax-based Bayesian optimization recommender.

    Creates a multi-metric experiment (deviation, time) but for simplicity combines into scalar objective for Ax (minimize).
    We still gate on raw deviation/time thresholds outside Ax for phase transitions.
    """
    def __init__(self, param_space: Dict[str, Dict[str, Any]], seed: Optional[int] = None):
        if not _AX_AVAILABLE:
            raise RuntimeError("Ax not installed; cannot use AxRecommender.")
        self.param_space = param_space
        self.ax = AxClient(random_seed=seed)
        parameters = []
        for name, spec in param_space.items():
            ptype = 'int' if spec['type'] == 'int' else 'float'
            parameters.append({
                'name': name,
                'type': ptype,
                'bounds': spec['bounds']
            })
        self.ax.create_experiment(
            name="calibration_bo",
            parameters=parameters,
            objective_name="objective",
            minimize=True
        )
        self._last_trial_index: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        # For simplicity generate n sequential single-trial suggestions
        suggestions: List[Dict[str, Any]] = []
        for _ in range(n):
            params, trial_index = self.ax.get_next_trial()
            self._last_trial_index = trial_index
            suggestions.append(params)
        return suggestions

    def observe(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        # metrics must contain 'objective'
        obj = metrics['objective']
        if self._last_trial_index is not None:
            self.ax.complete_trial(self._last_trial_index, {'objective': (obj, 0.0)})
            self._last_trial_index = None
        self.history.append({'params': params, **metrics})

    def best_params(self) -> Optional[Dict[str, Any]]:
        if not self.history:
            return None
        best = min(self.history, key=lambda r: r['objective'])
        return best['params']


def make_recommender(cfg: Dict[str, Any]):
    algo = cfg.get('algorithm', 'random_local')
    param_space = cfg['parameters']
    seed = cfg.get('random_seed')
    if algo == 'ax':
        if not _AX_AVAILABLE:
            print('[recommender] Ax not available, falling back to random_local')
            return RandomLocalRecommender(param_space, seed)
        return AxRecommender(param_space, seed)
    # default
    return RandomLocalRecommender(param_space, seed)
