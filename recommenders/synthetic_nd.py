"""
N-dimensional version of the synthetic landscape used in
synthetic_3d_visualize.py. Identical math, generalized to D dims.

  f1 (turbidity): cone whose tip is at corner (1,1,...,1), broadest at
      that corner with half-radius SPIKE_R0, tapering linearly to a
      point at SPIKE_LENGTH along the inward diagonal.
      F1_LOW outside, F1_HIGH at the corner, F1_TIP at the tip.

  f2 (ratio): wedge whose corner is at (W,W,...,W) (W=WEDGE_W), facing
      the (0,0,...,0) corner. Inside the wedge (all xi <= W) the value
      is F2_HIGH (0.9); outside it is F2_LOW (0.7); transition has
      width EPS_SMOOTH on each of the D facing faces.

Use:
  from recommenders.synthetic_nd import build_landscape
  land = build_landscape(d=4)
  Y = land.evaluate(X)        # X: (n,d) -> Y: (n,2)
  F1, F2 = land.f1(X), land.f2(X)
"""

import numpy as np
from dataclasses import dataclass


# Shared constants (kept identical to synthetic_3d_visualize.py)
F1_LOW, F1_HIGH = 0.04, 1.00
F1_TIP = 0.35
SPIKE_R0 = 0.35
SPIKE_LENGTH = 0.95
EPS_SHARP = 0.03

F2_LOW, F2_HIGH = 0.70, 0.90
WEDGE_W = 0.7
EPS_SMOOTH = 0.05


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Landscape:
    d: int
    spike_corner: np.ndarray
    spike_axis: np.ndarray
    wedge_corner: np.ndarray

    def f1(self, X):
        Xs = X - self.spike_corner
        s = Xs @ self.spike_axis
        proj = np.outer(s, self.spike_axis)
        perp = np.linalg.norm(Xs - proj, axis=1)
        r_at_s = SPIKE_R0 * np.clip(1.0 - s / SPIKE_LENGTH, 0.0, 1.0)
        inside = _sigmoid((r_at_s - perp) / EPS_SHARP)
        # suppress behind corner and past the tip
        inside = np.where(s < -EPS_SHARP, 0.0, inside)
        inside = np.where(s > SPIKE_LENGTH + EPS_SHARP, 0.0, inside)
        t = np.clip(s / SPIKE_LENGTH, 0.0, 1.0)
        v_inside = F1_HIGH + (F1_TIP - F1_HIGH) * t
        return F1_LOW + (v_inside - F1_LOW) * inside

    def f2(self, X):
        d = X - self.wedge_corner
        h = np.max(d, axis=1)
        return F2_LOW + (F2_HIGH - F2_LOW) * _sigmoid(-h / EPS_SMOOTH)

    def evaluate(self, X):
        return np.stack([self.f1(X), self.f2(X)], axis=1)


def build_landscape(d: int) -> Landscape:
    spike_corner = np.ones(d)
    spike_axis = -np.ones(d) / np.sqrt(d)
    wedge_corner = np.full(d, WEDGE_W)
    return Landscape(d=d, spike_corner=spike_corner,
                     spike_axis=spike_axis, wedge_corner=wedge_corner)
