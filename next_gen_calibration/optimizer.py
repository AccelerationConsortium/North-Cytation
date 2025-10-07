from __future__ import annotations
from typing import Dict, Any, List
import random

class SimpleBayesLikeOptimizer:
    """Lightweight placeholder for Ax or other BO framework.
    Maintains history and proposes random-guided improvements centering around best observed params.
    """
    def __init__(self, param_space: Dict[str, Dict[str, Any]], seed: int | None = None):
        self.param_space = param_space
        self.history: List[Dict[str, Any]] = []
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def _sample_param(self, name: str, spec: Dict[str, Any], center: Any | None = None):
        bounds = spec['bounds']
        if spec['type'] in ('int','float'):
            low, high = bounds
            if center is not None:
                span = (high - low) * 0.3
                low2 = max(low, center - span/2)
                high2 = min(high, center + span/2)
                low, high = low2, high2
            if spec['type']=='int':
                return random.randint(int(low), int(high))
            else:
                return random.uniform(low, high)
        raise ValueError(f"Unsupported param type {spec['type']}")

    def suggest(self, n: int = 1) -> List[Dict[str, Any]]:
        if not self.history:
            centers = {}
        else:
            # pick best by objective score (lower is better)
            best = min(self.history, key=lambda r: r['objective'])
            centers = {k: best['params'][k] for k in best['params']}
        suggestions = []
        for _ in range(n):
            params = {}
            for name, spec in self.param_space.items():
                params[name] = self._sample_param(name, spec, centers.get(name))
            suggestions.append(params)
        return suggestions

    def observe(self, params: Dict[str, Any], objective: float, meta: Dict[str, Any]):
        self.history.append({'params': params, 'objective': objective, **meta})

    def best_params(self) -> Dict[str, Any] | None:
        if not self.history:
            return None
        best = min(self.history, key=lambda r: r['objective'])
        return best['params']
