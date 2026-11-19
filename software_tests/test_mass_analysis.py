"""Mass analyzer shared by aspirate_from_vial, dispense_into_vial and dispense_into_vial_from_reservoir."""
import logging

import pytest

from North_Safe import North_Robot
from pipetting_data.mass_analysis import analyze_mass_readings, empty_stability


def readings(pre, during, post, phase='dispensing'):
    """[(mass_g, steady)] per phase -> recorded readings."""
    rows = [('baseline_pre', m, s) for m, s in pre] + [(phase, m, s) for m, s in during] + \
           [('baseline_post', m, s) for m, s in post]
    return [{'timestamp': i, 'time_relative': i * 0.2, 'mass_g': m, 'phase': p, 'steady_status': s}
            for i, (p, m, s) in enumerate(rows)]


def test_dispense_uses_steady_readings_and_reports_mass_added():
    data = readings(pre=[(10.0, True), (10.2, False), (10.0, True)], during=[(10.05, False)],
                    post=[(10.1, True), (10.1, True)])
    result = analyze_mass_readings(data, 'dispense')
    assert result['mass_g'] == pytest.approx(0.1)
    assert (result['pre_method'], result['post_method']) == ('STABLE', 'STABLE')
    stats = result['stability']
    assert (stats['pre_stable_count'], stats['pre_total_count']) == (2, 3)
    assert (stats['post_stable_count'], stats['post_total_count']) == (2, 2)
    assert stats['pre_baseline_std'] == pytest.approx(0.11547, abs=1e-5)  # all pre readings, sample std
    assert stats['post_baseline_std'] == 0.0


def test_aspiration_reports_mass_removed_as_positive():
    data = readings(pre=[(10.0, True)] * 2, during=[(9.95, False)], post=[(9.9, True)] * 2, phase='aspiration')
    assert analyze_mass_readings(data, 'aspiration')['mass_g'] == pytest.approx(0.1)


def test_no_steady_readings_falls_back_to_all_readings_in_the_phase():
    data = readings(pre=[(10.0, False), (10.2, False)], during=[], post=[(10.3, False), (10.5, False)])
    result = analyze_mass_readings(data, 'reservoir')
    assert result['mass_g'] == pytest.approx(0.3)
    assert (result['pre_method'], result['post_method']) == ('TIME-BASED', 'TIME-BASED')


def test_too_few_readings_gives_zero_mass_and_no_baselines():
    result = analyze_mass_readings(readings(pre=[(10.0, True)], during=[], post=[(10.1, True)] * 2), 'dispense')
    assert result['mass_g'] == 0.0
    assert result['pre_baseline_g'] is None
    assert result['stability'] == empty_stability()


def test_robot_helper_updates_stability_and_saves_csv_and_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    stability = empty_stability()
    stability['failure_reason'] = 'kept'  # keys set elsewhere survive the update
    data = readings(pre=[(10.0, True)] * 3, during=[(10.05, False)], post=[(10.1, True)] * 3)
    mass = robot._analyze_mass_readings(data, 'reservoir', 4, 0.1, stability, save=True,
                                        file_prefix="reservoir_mass_", file_stem="4_res1")
    assert mass == pytest.approx(0.1)
    assert stability['pre_total_count'] == 3 and stability['failure_reason'] == 'kept'
    saved = sorted(p.name for p in (tmp_path / 'output' / 'mass_measurements').rglob('*.*'))
    assert len(saved) == 2
    assert saved[0].startswith('reservoir_mass_data_4_res1_') and saved[0].endswith('.csv')
    assert saved[1].startswith('reservoir_mass_plot_4_res1_') and saved[1].endswith('.png')


def test_robot_helper_saves_csv_only_when_too_few_readings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = object.__new__(North_Robot)
    robot.logger = logging.getLogger(__name__)
    data = readings(pre=[(10.0, True)], during=[], post=[(10.1, True)])
    assert robot._analyze_mass_readings(data, 'dispense', 'water', 0.1, empty_stability(), save=True,
                                        file_prefix="mass_", file_stem="water") == 0.0
    saved = [p.name for p in (tmp_path / 'output' / 'mass_measurements').rglob('*.*')]
    assert len(saved) == 1 and saved[0].startswith('mass_data_water_')
