import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scheduler.liquid_review import (coverage, current_parameters, estimate_optimum,
    matching_measurements, parse_log, environmental_history, past_results,
    past_result_measurements, past_result_parameters)


class LiquidReviewTests(unittest.TestCase):
    def test_log_counts_destinations_and_retains_explicit_class(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.log"
            path.write_text('\n'.join([
                'Dispensing 0.400 mL from stock to destination',
                'Liquid review context: {"liquid": "water", "target_volume_ml": 0.1}',
                'Selected small_tip for volume 0.100 mL',
                'Pipetting from vial stock, amount: 0.100 mL',
                'Pipetting into vial destination, amount: 0.050 mL',
                'Pipetting dispensing 0.050 mL from well plate index 2 of type 96',
                'Transferring 0.025 mL into well #3 of 96_STANDARD',
                'Liquid review context: {"liquid": null}',
                'Pipetting from vial water, amount: 0.100 mL',
                'Pipetting into vial destination, amount: 0.100 mL',
            ]))
            events = parse_log(path)
            self.assertEqual([e.volume_ul for e in events], [50, 50, 25, 100])
            self.assertEqual([e.liquid for e in events], ['water', 'water', 'water', 'Unidentified'])

    def test_old_logs_identify_fallback_class_without_vial_guess(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.log'
            path.write_text("Target volume 0.9mL for 'DMSO' in optimal_conditions_DMSO.csv\n"
                            "Pipetting from vial stock, amount: 0.900 mL\n"
                            "Pipetting into vial destination, amount: 0.900 mL\n")
            self.assertEqual(parse_log(path)[0].liquid, 'DMSO')

    def test_tip_boundary_never_bridges_gap(self):
        frame = pd.DataFrame({'volume_target': [5, 50, 200, 800]})
        self.assertEqual(coverage(frame, 150), 'Outside tip-specific coverage')
        self.assertEqual(coverage(frame, 200), 'Measured point')
        self.assertEqual(coverage(frame, 500), 'Interpolated coverage')
        self.assertEqual(coverage(frame, 50, 'large_tip'), 'Tip differs from calibration regime')

    def test_matching_excludes_other_settings_but_allows_overaspirate(self):
        frame = pd.DataFrame({'volume_target': [50], 'aspirate_speed': [15]})
        params = current_parameters(frame, 50)
        row = dict(params, target_volume_ml=.05, measured_volume_ml=.05,
                   liquid_type='water', validation_type='vial_to_vial', session_id='a')
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'pipetting_data'
            data.mkdir()
            pd.DataFrame([dict(row, overaspirate_vol=.001), dict(row, overaspirate_vol=.005),
                          dict(row, aspirate_speed=10)]).to_csv(data / 'master_pipetting_measurements.csv', index=False)
            matched, excluded = matching_measurements(folder, 'water', frame)
            self.assertEqual(len(matched), 2)
            self.assertEqual(excluded, {'Different aspirate_speed': 1})

    def test_exact_duplicate_measurement_is_not_counted_twice(self):
        frame = pd.DataFrame({'volume_target': [50], 'aspirate_speed': [15]})
        row = dict(current_parameters(frame, 50), target_volume_ml=.05,
                   measured_volume_ml=.051, liquid_type='water', validation_type='vial_to_vial',
                   session_id='a', timestamp='20261009_170251', replicate_number=1)
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'pipetting_data'
            data.mkdir()
            pd.DataFrame([row, row]).to_csv(data / 'master_pipetting_measurements.csv', index=False)
            matched, excluded = matching_measurements(folder, 'water', frame)
            self.assertEqual(len(matched), 1)
            self.assertEqual(excluded.get('Exact duplicate measurement'), 1)

    def test_estimation_and_single_setting(self):
        rows = [dict(session_id=str(s), overaspirate_vol=x / 1000,
                     target_volume_ml=.05, measured_volume_ml=(50 + x - 3 + s * .01) / 1000)
                for s in range(4) for x in [1, 3, 5]]
        data = pd.DataFrame(rows)
        estimate, _ = estimate_optimum(data)
        self.assertAlmostEqual(estimate[0], 2.985)
        self.assertIsNotNone(estimate[1])
        self.assertIsNone(estimate_optimum(data[data.overaspirate_vol == .001])[0])

    def test_legacy_environment_header(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'pipetting_data'
            data.mkdir()
            (data / 'parameter_corrections.csv').write_text('session_id,target_volume_ml\na,0.05,23,40,100000\n')
            self.assertEqual(environmental_history(folder).iloc[0].temp_c, '23')

    def test_past_results_use_optimum_and_validation_with_saved_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'sdl_pipette_calibration/output/pc_example'
            calibration = base / 'calibration/run_water'
            validation = base / 'validation/validation_run_20261008_water'
            calibration.mkdir(parents=True)
            validation.mkdir(parents=True)
            (calibration / 'raw_measurements.csv').write_text(
                'liquid,volume_target_ul,volume_measured_ul\nwater,50,90\n')
            (calibration / 'optimal_conditions_water.csv').write_text(
                'volume_target_ul,volume_measured_ul,calibration_overaspirate_vol,hardware_parameters_aspirate_speed\n'
                '50,50.2,0.003,12\n')
            (calibration / 'experiment_summary.json').write_text('{"simulation_mode": false}')
            (validation / 'raw_validation_data.csv').write_text(
                'volume_target_ul,volume_measured_ul,parameters_used\n'
                '50,49.8,"{\'aspirate_speed\': 12, \'overaspirate_vol\': 0.003}"\n')
            (validation / 'validation_analysis.json').write_text('{"simulation_mode": false}')
            results = past_results(folder, 'water')
            self.assertEqual({item.kind for item in results}, {'calibration', 'validation'})
            calibration_result = next(item for item in results if item.kind == 'calibration')
            validation_result = next(item for item in results if item.kind == 'validation')
            self.assertEqual(past_result_measurements(calibration_result).measured_ul.iloc[0], 50.2)
            self.assertEqual(past_result_parameters(past_result_measurements(calibration_result).iloc[0],
                                                    'calibration')['aspirate_speed'], 12)
            self.assertEqual(past_result_parameters(past_result_measurements(validation_result).iloc[0],
                                                    'validation')['overaspirate_vol'], .003)

    def test_legacy_calibration_optimum_uses_ml_columns_and_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / 'output/calibration_runs/20251009_183138_water_simulate'
            run.mkdir(parents=True)
            (run / 'optimal_conditions.csv').write_text(
                'target_volume_mL,average_obtained_volume_mL,aspirate_speed,overaspirate_vol\n'
                '0.05,0.051,14,0.003\n')
            (run / 'experiment_config.yaml').write_text('simulate: false\n')
            result, = past_results(folder, 'water')
            frame = past_result_measurements(result)
            self.assertEqual(frame.target_ul.iloc[0], 50)
            self.assertEqual(frame.measured_ul.iloc[0], 51)
            self.assertEqual(past_result_parameters(frame.iloc[0], 'calibration')['aspirate_speed'], 14)

    def test_past_results_exclude_simulated_and_unknown_even_if_config_claims_real(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / 'sdl_pipette_calibration/output'
            for name in ('real', 'simulated', 'unknown'):
                run = base / f'run_{name}_water'
                run.mkdir(parents=True)
                (run / 'optimal_conditions_water.csv').write_text(
                    'volume_target_ul,volume_measured_ul\n50,50\n')
                if name != 'unknown':
                    (run / 'experiment_config_used.yaml').write_text('experiment:\n  simulate: false\n')
                if name == 'simulated':
                    (run / 'experiment_summary.json').write_text('{"simulation_mode": true}')
            results = past_results(folder, 'water')
            self.assertEqual([result.path.parent.name for result in results], ['run_real_water'])

    def test_dialog_renders_log_and_missing_history(self):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PySide6.QtWidgets import QApplication
        from scheduler.liquid_review_gui import LiquidReviewDialog
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'pipetting_data').mkdir()
            (root / 'pipetting_data/optimal_conditions_water.csv').write_text('volume_target\n50\n')
            log = root / 'run.log'
            log.write_text('Liquid review context: {"liquid":"water"}\nPipetting from vial stock, amount: 0.050 mL\nPipetting into vial dest, amount: 0.050 mL\n')
            dialog = LiquidReviewDialog(root, {'log_file': str(log)})
            self.assertEqual(dialog.tabs.count(), 1)
            self.assertEqual(dialog.tabs.widget(0).pages.count(), 6)
            self.assertEqual([dialog.tabs.widget(0).pages.tabText(i) for i in range(6)],
                             ['Calibration Summary', 'Coverage', 'Past Calibration Results', 'Overvolume Validation',
                              'Environment', 'Recommendations'])
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            summary = dialog.tabs.widget(0).pages.widget(0)
            axes = [canvas.figure.axes[0] for canvas in summary.findChildren(FigureCanvasQTAgg)]
            self.assertEqual(len(axes), 2)
            self.assertEqual(axes[1].get_ylabel(), 'Measured V_M (µL)')
            from PySide6.QtWidgets import QTableWidget, QTextEdit
            self.assertEqual(len(summary.findChildren(QTableWidget)), 0)
            coverage_page = dialog.tabs.widget(0).pages.widget(1)
            self.assertEqual(coverage_page.findChildren(QTableWidget)[0].rowCount(), 1)
            self.assertIn('Log references', coverage_page.findChildren(QTextEdit)[0].toPlainText())
            dialog.close()
            dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
