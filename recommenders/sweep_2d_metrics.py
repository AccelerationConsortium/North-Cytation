"""
Sweep all 2D ground-truth datasets through all transition recommenders and
build one combined metrics CSV plus a metrics explanation file.

Each (dataset, recommender) pair runs the existing test_real_ground_truth
pipeline; per-output and clumping rows are concatenated into one big CSV at
recommenders/test_outputs/sweep_metrics_summary.csv.

Run:
  python -m recommenders.sweep_2d_metrics
  python -m recommenders.sweep_2d_metrics --datasets DSS_BZT,SDS_TTAB
"""

import argparse
import os
import subprocess
import sys

import pandas as pd


OUT_ROOT = os.path.join(os.path.dirname(__file__), "test_outputs")
SUMMARY_CSV = os.path.join(OUT_ROOT, "sweep_metrics_summary.csv")
METRICS_DOC = os.path.join(OUT_ROOT, "METRICS_EXPLANATION.txt")

DEFAULT_DATASETS = ["surfactant2d", "CHAPS_BDDAC", "BDDAC_NaLS",
                    "DSS_BZT", "DSS_CTAB", "SDS_TTAB"]


METRICS_EXPLANATION = """\
Metrics in sweep_metrics_summary.csv
====================================

Each row is one (dataset, recommender, output) combination. "output" is one
of:
  ratio           - per-output boundary metrics for the ratio field
  turbidity_600   - per-output boundary metrics for the turbidity field
  _all            - boundary-AGNOSTIC clumping metrics over all picks

Coordinate convention
---------------------
All distances below are in NORMALIZED [0, 1]^2 coordinates: the (log10 mM)
input domain is mapped to the unit square first. Values are scale-free and
comparable across datasets.

Boundary definition
-------------------
For each output, the "boundary" on the fine ground-truth grid is the set of
cells in the top X% of |grad y|:
  ratio:         top 30% (broader transitions, smoother gradients)
  turbidity_600: top 10% (sharp narrow transitions)
A per-output presence check skips the output entirely (has_boundary=False)
when max(y) - min(y) < 0.10.

Boundary metrics (per output, where output != _all)
---------------------------------------------------
y_range
  max(y) - min(y) for this output across the grid. Sanity check.
has_boundary
  True if y_range >= 0.10. False rows have all metric columns as NaN.
n_boundary_cells
  Number of grid cells in the boundary mask. Reference value.
n_picks_total
  Total picks across all iterations (includes initial grid).
hit_rate
  PICK-CENTRIC: of my picks, what fraction touched the boundary?
  Higher = better. Caveat: high hit_rate can be achieved by clumping all
  picks on one part of the boundary; pair with coverage_auc.
coverage_auc
  BOUNDARY-CENTRIC: for r in [0, 1], plot
  frac{boundary cells with nearest-pick distance <= r} versus r and
  integrate. Higher = better. Hard to game by clumping. Single best
  metric for ranking recommenders on boundary tracing.

Clumping metrics (per recommender, where output == _all)
--------------------------------------------------------
n_picks_total
  Same as above; included for self-contained rows.
cv_nn
  std(nn_dist) / mean(nn_dist). Coefficient of variation of NN distances.
  Reference values:
    perfect grid          ~ 0.00
    uniform random in 2D  ~ 0.52
    clumpy distributions  > 0.52
  Lower = more uniform spread.
clumping_ratio
  median(nn_dist) / d_uniform, where d_uniform = (1/N)^(1/d) is the
  expected NN spacing for N points on a perfect grid in d dimensions.
  ~ 1.0 means roughly grid-like spacing; values further from 1 mean less
  uniform. (Strongly correlated with cv_nn but with a clearer absolute
  scale, hence kept alongside.)
min_pairwise
  Minimum nearest-neighbor distance over all picks. Worst single clump
  in the set. Largely independent of cv_nn / clumping_ratio (which
  describe the bulk distribution).

Rule of thumb
-------------
- coverage_auc        primary boundary-quality metric
- hit_rate            secondary, complementary view (pick-centric)
- cv_nn               primary spread-uniformity metric
- min_pairwise        secondary, worst-case clumping
- clumping_ratio      absolute scale check
"""


def run_one(name, reduce_mode):
    cmd = [sys.executable, "-m", "recommenders.test_real_ground_truth",
           "--name", name, "--reduce", reduce_mode]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n>>> Running {name} ...")
    log_path = os.path.join(OUT_ROOT, f"_sweep_log_{name}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                              text=True)
    print(f"    exit={proc.returncode}, log -> {log_path}")
    return proc.returncode


def collect():
    rows = []
    for name in os.listdir(OUT_ROOT):
        if not name.startswith("real_"):
            continue
        csv = os.path.join(OUT_ROOT, name, "metrics_summary.csv")
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        df.insert(0, "dataset", name[len("real_"):])
        rows.append(df)
    if not rows:
        print("No per-dataset metrics_summary.csv files found.")
        return None
    big = pd.concat(rows, ignore_index=True)
    # Keep recommender ordering stable
    rec_order = {"BayesianContrast": 0, "GradientUCB": 1, "Triangle": 2}
    big["_rec_ord"] = big["recommender"].map(rec_order).fillna(99)
    out_order = {"ratio": 0, "turbidity_600": 1, "_all": 2}
    big["_out_ord"] = big["output"].map(out_order).fillna(99)
    big = big.sort_values(["dataset", "_rec_ord", "_out_ord"])
    big = big.drop(columns=["_rec_ord", "_out_ord"])
    big.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}  ({len(big)} rows)")
    return big


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--reduce", default="standardized_sum",
                   choices=["sum", "max", "standardized_sum"])
    p.add_argument("--skip-run", action="store_true",
                   help="Skip running, just collect existing metrics_summary"
                        " files into the combined CSV.")
    args = p.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(METRICS_DOC, "w", encoding="utf-8") as f:
        f.write(METRICS_EXPLANATION)
    print(f"Wrote metrics explanation -> {METRICS_DOC}")

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    if not args.skip_run:
        for name in names:
            run_one(name, args.reduce)
    big = collect()
    if big is not None:
        print("\nFirst 12 rows:")
        print(big.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
