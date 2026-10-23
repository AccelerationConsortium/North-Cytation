from unittest.mock import Mock, patch

import pytest

from pipetting_data.embedded_liquid_validations import run_liquid_validations, volume_groups


def test_volume_groups_keep_200_with_large_tip_and_use_planned_values():
    assert volume_groups([0, 5, 25, 195, 200, 400, 800]) == {
        "small": [5, 25, 195],
        "large": [200, 400, 800],
    }


def test_validations_use_exact_liquid_names_and_separate_sessions(tmp_path):
    lash = Mock(simulate=True)
    lash.nr_robot.resolve_pipetting_setup.side_effect = lambda volume, liquid, **kwargs: (
        None, 'small_tip' if volume < .2 else 'large_tip')
    with patch("pipetting_data.embedded_calibration_validation.validate_pipetting_accuracy",
               return_value={"success": True}) as validate:
        result = run_liquid_validations(
            lash, [("water", "water", [5, 195, 200, 400]),
                   ("dye_vial", "DMSO", [5])], tmp_path)
    assert len(result) == 3
    assert [call.kwargs["liquid_type"] for call in validate.call_args_list] == ["water", "water", "DMSO"]
    assert [call.kwargs["volumes_ml"] for call in validate.call_args_list] == [
        [0.005, 0.195], [0.2, 0.4], [0.005]]
    assert all(call.kwargs["compensate_overvolume"] is False
               for call in validate.call_args_list)


def test_unknown_liquid_fails_before_any_validation(tmp_path):
    lash = Mock(simulate=True)
    lash.nr_robot.resolve_pipetting_setup.side_effect = lambda volume, liquid, **kwargs: (None, 'small_tip')
    with patch("pipetting_data.embedded_calibration_validation.validate_pipetting_accuracy") as validate:
        with pytest.raises(ValueError, match="Unknown liquid 'BDDAC'"):
            run_liquid_validations(lash, [("water", "water", [5]),
                                          ("BDDAC_stock", "BDDAC", [50])], tmp_path)
    validate.assert_not_called()


def test_two_surfactant_sources_get_distinct_sessions_with_sds_parameters(tmp_path):
    lash = Mock(simulate=True)
    lash.nr_robot.resolve_pipetting_setup.side_effect = lambda volume, liquid, **kwargs: (
        None, 'small_tip' if volume < .2 else 'large_tip')
    with patch("pipetting_data.embedded_calibration_validation.validate_pipetting_accuracy",
               return_value={"success": True}) as validate:
        results = run_liquid_validations(
            lash, [("SDS_stock", "SDS", [50, 200]),
                   ("BDDAC_stock", "SDS", [60, 200])], tmp_path)
    assert len(results) == 4
    assert [call.kwargs["source_vial"] for call in validate.call_args_list] == [
        "SDS_stock", "SDS_stock", "BDDAC_stock", "BDDAC_stock"]
    assert all(call.kwargs["liquid_type"] == "SDS" for call in validate.call_args_list)


def test_195_ul_can_be_validated_with_large_tip(tmp_path):
    lash = Mock(simulate=True)
    lash.nr_robot.resolve_pipetting_setup.side_effect = lambda volume, liquid, **kwargs: (
        None, 'large_tip' if volume == .195 else 'small_tip')
    with patch("pipetting_data.embedded_calibration_validation.validate_pipetting_accuracy",
               return_value={"success": True}) as validate:
        run_liquid_validations(lash, [("DMSO", "DMSO", [5, 195])], tmp_path)
    assert [call.kwargs['tip_type'] for call in validate.call_args_list] == [
        'small_tip', 'large_tip']
    assert validate.call_args_list[1].kwargs['volumes_ml'] == [.195]
