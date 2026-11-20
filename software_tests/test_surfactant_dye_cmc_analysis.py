import unittest

import numpy as np
import pandas as pd

from analysis.surfactant_dye_cmc_analysis import binding_model, dye_matrix, fit_cmc, stock_check


class SurfactantDyeCmcAnalysisTests(unittest.TestCase):
    def test_fit_recovers_cmc_and_k(self):
        s = np.repeat([0, 2, 5, 7, 7.6, 8, 8.4, 8.8, 9.5, 10, 11, 16, 22, 32], 3).astype(float)
        rng = np.random.default_rng(1)
        y = binding_model(s, 150, 2200, 8.0, 1.5, 0.05) * (1 + 0.01 * rng.standard_normal(len(s)))
        fit = fit_cmc(s, y, cmc_guess=8.6)
        self.assertAlmostEqual(fit["cmc_mM"], 8.0, delta=0.2)
        self.assertAlmostEqual(fit["k_per_mM"], 1.5, delta=0.3)

    def test_dye_matrix_background_and_linearity(self):
        rows = []
        for s, slope in ((0.0, 10.0), (16.0, 100.0)):
            for d in (0.0, 1.0, 4.0, 12.0, 32.0):
                signal = 50 + slope * (d if d < 32 else 16)  # top level falls to half the line
                rows.append(dict(block="dye_matrix", matrix="water", surfactant_mM=s, dye_uM=d,
                                 timepoint_min=0, f=signal))
        stats, slopes = dye_matrix(pd.DataFrame(rows), "f")
        self.assertTrue(np.allclose(stats.loc[stats.dye_uM == 0, "signal"], 0))
        by_s = slopes.set_index("surfactant_mM")
        self.assertAlmostEqual(by_s.loc[16.0, "slope_per_uM"], 100.0)
        self.assertAlmostEqual(by_s.loc[16.0, "top_vs_line"], 0.5)

    def test_stock_check_gives_real_concentration(self):
        dye = np.array([0, 1, 4, 12, 20, 32], float)
        absorbance = 0.04 + dye * 1e-6 * 50000 * 0.6 * 0.9  # stock is 90% of nominal
        data = pd.DataFrame(dict(block="stock_check", timepoint_min=0, dye_uM=dye, a=absorbance, f=dye * 10))
        summary, _ = stock_check(data, "f", "a", epsilon=50000, path_cm=0.6, nominal_stock_uM=2630)
        self.assertAlmostEqual(summary["actual_over_nominal"], 0.9)
        self.assertAlmostEqual(summary["measured_stock_uM"], 2367.0)


if __name__ == "__main__":
    unittest.main()
