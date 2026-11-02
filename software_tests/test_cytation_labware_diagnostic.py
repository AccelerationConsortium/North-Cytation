import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from workflows import cytation_labware_test as diagnostic


def procedure(name="96 WELL PLATE", wavelength="600"):
    return (f'<BTIProcedure Version="1.00"><Labware><Type>Microplate</Type>'
            f'<Name>{name}</Name><UseLid>false</UseLid></Labware><StepList>'
            f'<ReadStep><Detection>Absorbance</Detection><ReadType>EndPoint</ReadType><Measurements>'
            f'<Measurement><Wavelength>{wavelength}</Wavelength></Measurement>'
            f'</Measurements></ReadStep></StepList></BTIProcedure>')


def plate_with_readback(readback):
    return SimpleNamespace(
        get_procedure=Mock(side_effect=[procedure(), readback]),
        get_modifiable_procedure=Mock(return_value=(
            '<BTIModifiableProcedure><Labware><Name Modifiable="Yes">'
            '96 WELL PLATE</Name></Labware></BTIModifiableProcedure>')),
        set_procedure=Mock(), set_modifiable_procedure=Mock(),
        validate_procedure=Mock(return_value=(1, None)),
    )


def test_replacement_requires_gen5_readback(tmp_path):
    plate = plate_with_readback(procedure("48 WELL PLATE"))
    result = diagnostic.prepare_plate(
        plate, "48 WELL PLATE", "replace_labware", tmp_path, logging.getLogger(__name__))
    assert result["after"]["name"] == "48 WELL PLATE"
    requested = plate.set_procedure.call_args.args[0]
    assert diagnostic._labware(requested)["name"] == "48 WELL PLATE"
    assert diagnostic._steps(requested) == diagnostic._steps(procedure())
    assert (tmp_path / "readback.xml").read_text() == procedure("48 WELL PLATE")


def test_swallowed_setter_failure_is_detected(tmp_path):
    plate = plate_with_readback(procedure())
    with pytest.raises(RuntimeError, match="Labware readback"):
        diagnostic.prepare_plate(
            plate, "48 WELL PLATE", "replace_labware", tmp_path, logging.getLogger(__name__))


def test_changed_measurements_are_rejected(tmp_path):
    plate = plate_with_readback(procedure("48 WELL PLATE", "450"))
    with pytest.raises(RuntimeError, match="StepList changed"):
        diagnostic.prepare_plate(
            plate, "48 WELL PLATE", "replace_labware", tmp_path, logging.getLogger(__name__))


def test_roundtrip_reapplies_exact_original(tmp_path):
    plate = plate_with_readback(procedure())
    diagnostic.prepare_plate(plate, "96 WELL PLATE", "roundtrip", tmp_path, logging.getLogger(__name__))
    plate.set_procedure.assert_called_once_with(procedure())


def test_validation_rejection_is_preserved(tmp_path):
    plate = plate_with_readback(procedure())
    plate.validate_procedure.side_effect = [(1, None), (-42, "unsupported plate")]
    with pytest.raises(RuntimeError, match="unsupported plate"):
        diagnostic.prepare_plate(
            plate, "96 WELL PLATE", "baseline", tmp_path, logging.getLogger(__name__))
    plate.set_procedure.assert_not_called()


def test_simulation_transfers_one_plate_without_gen5(monkeypatch, tmp_path):
    locations = {"source_stack": ["unrelated-source-plate"], "pipetting_area": ["plate1"],
                 "cytation_tray": [], "gripper": []}
    plate_at = Mock(return_value="plate1")
    inv = SimpleNamespace(
        plate_at=plate_at, contents=lambda name: locations[name],
        top=lambda name: locations[name][-1], attachments={},
        definitions={"96 WELL PLATE": {"name_in_cytation": "96 WELL PLATE"}})
    cfg = {k: v for k, v in vars(diagnostic).items() if k.isupper() and not k.startswith("_")}
    cfg["OUTPUT_ROOT"] = str(tmp_path)
    calls = []

    def move(source, destination):
        locations[destination].append(locations[source].pop())
        calls.append((source, destination))

    def measure(protocol, wells, **kwargs):
        assert protocol == cfg["PROTOCOL_FILE"] and wells == [0]
        assert callable(kwargs["plate_setup"])
        move("pipetting_area", "cytation_tray")
        move("cytation_tray", "pipetting_area")
        return None

    lash = SimpleNamespace(
        _workflow_should_continue=True, workflow_config=cfg, simulate=True,
        logger=logging.getLogger("diagnostic-simulation"),
        nr_track=SimpleNamespace(inventory=inv),
        measure_wellplate=Mock(side_effect=measure))
    monkeypatch.setattr(diagnostic, "Lash_E", lambda **kw: lash)
    result = diagnostic.execute(config=cfg, show_gui=False)
    assert result["completed"] and result["simulation"]
    assert result["final_location"] == "pipetting_area"
    plate_at.assert_called_once_with("pipetting_area", "96 WELL PLATE")
    assert locations["source_stack"] == ["unrelated-source-plate"]
    assert calls == [("pipetting_area", "cytation_tray"),
                     ("cytation_tray", "pipetting_area")]
    assert len(list(tmp_path.glob("*/summary.json"))) == 1


@pytest.mark.parametrize("mode", [None, "roundtrip", "replace_labware"])
def test_xml_hook_runs_inside_existing_coordinator_and_reader_path(tmp_path, mode):
    from biotek_new import Biotek_Wrapper

    events = []
    xml = [procedure()]

    def set_xml(value):
        events.append("set_xml")
        xml[0] = value

    plate = SimpleNamespace(
        get_procedure=lambda: xml[0],
        get_modifiable_procedure=lambda: "<BTIModifiableProcedure/>",
        validate_procedure=lambda _: (1, None),
        set_procedure=set_xml,
        keep_plate_in_after_read=lambda: events.append("keep_plate_in"),
        set_partial_plate=lambda _: events.append("set_wells"),
        start_read=lambda: events.append("start_read") or object(),
        read_status=5,
        get_raw_data=lambda: (events.append("get_raw_data") or 2, {"value": [0.0473]}))
    experiment = SimpleNamespace(
        protocol_type="Standard", plates=SimpleNamespace(add=lambda: events.append("add_plate") or plate))
    reader = object.__new__(Biotek_Wrapper)
    reader.simulate = False
    reader.logger = logging.getLogger(__name__)
    reader.biotek = SimpleNamespace(
        app=SimpleNamespace(data_export_enabled=False),
        new_experiment=lambda _: events.append("new_experiment") or experiment)
    reader.build_bti_partial_plate_xml = lambda **kw: "<partial/>"

    lash = object.__new__(diagnostic.Lash_E)
    lash.simulate = False
    lash.logger = logging.getLogger(__name__)
    lash.cytation = reader
    plate_type = "48 WELL PLATE" if mode == "replace_labware" else "96 WELL PLATE"
    lash._resolve_plate = lambda *a: ("plate1", plate_type, False, plate_type)
    lash.nr_robot = SimpleNamespace(move_home=lambda: events.append("home"))
    lash.nr_track = SimpleNamespace(
        inventory=SimpleNamespace(definitions={plate_type: {"num_wells": 48 if mode == "replace_labware" else 96}}),
        origin=lambda: events.append("origin"))
    lash.move_wellplate_to_cytation = lambda *a, **kw: events.append("load")
    lash.move_wellplate_back_from_cytation = lambda *a, **kw: events.append("return")

    def setup(p):
        assert lash._cytation_reading is True
        diagnostic.prepare_plate(p, plate_type, mode, tmp_path, lash.logger)

    options = {} if mode is None else {"plate_setup": setup}
    data = lash.measure_wellplate("test.prt", [0], plate_type=plate_type, **options)
    expected = ["home", "load", "new_experiment", "add_plate"]
    if mode is not None:
        expected.append("set_xml")
    expected += ["keep_plate_in", "set_wells", "start_read", "get_raw_data", "return", "origin"]
    assert events == expected
    assert data.iloc[0, 0] == 0.0473
    assert lash._cytation_reading is False
