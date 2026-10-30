import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import cytation_xml_only as probe


ORIGINAL = '''<BTIProcedure Version="1.00"><Labware><Type>Microplate</Type>
<Name>96 WELL PLATE</Name><UseLid>false</UseLid></Labware>
<WellSelection>Runtime</WellSelection><StepList><ReadStep>
<Detection>Absorbance</Detection><ReadType>EndPoint</ReadType>
<Measurements><Measurement><Wavelength>600</Wavelength></Measurement></Measurements>
</ReadStep></StepList><Options><EjectPlateOnCompletion>true</EjectPlateOnCompletion>
<SkipLoadPlateDialog>false</SkipLoadPlateDialog></Options></BTIProcedure>'''


def test_changes_only_name_and_known_rejected_tokens():
    import xml.etree.ElementTree as ET
    result = ET.fromstring(probe.candidate_xml(ORIGINAL, "48 WELL PLATE"))
    assert result.findtext("Labware/Name") == "48 WELL PLATE"
    assert result.find("WellSelection") is None
    assert result.find("Options") is None
    result.find("Labware/Name").text = "96 WELL PLATE"
    before = ET.fromstring(ORIGINAL)
    before.remove(before.find("WellSelection"))
    before.remove(before.find("Options"))
    assert ET.canonicalize(ET.tostring(result, encoding="unicode"), strip_text=True) == ET.canonicalize(
        ET.tostring(before, encoding="unicode"), strip_text=True)


def test_existing_run_protocol_can_probe_without_any_read(tmp_path):
    xml = [ORIGINAL]
    plate = SimpleNamespace(
        get_procedure=lambda: xml[0],
        set_procedure=lambda value: xml.__setitem__(0, value),
        _invoke_property_get=lambda name, **kw: {"MaxRows": 6, "MaxColumns": 8}[name],
        set_partial_plate=Mock(), validate_procedure=lambda _: (1, None))
    reader = object.__new__(probe.ProcedureOnlyReader)
    reader.simulate = False
    reader.logger = logging.getLogger("test-xml-only")
    # No carrier, read, abort or close methods exist on this fake driver/plate.
    reader.biotek = SimpleNamespace(
        app=SimpleNamespace(data_export_enabled=False),
        new_experiment=lambda _: SimpleNamespace(
            protocol_type="Standard", plates=SimpleNamespace(add=lambda: plate)))
    reader.build_bti_partial_plate_xml = lambda **kw: "<BTIPartialPlate/>"
    summary = {"plate_type": "48 WELL PLATE"}
    reader.run_protocol("same.prt", [0], plate_type="48 WELL PLATE",
                        plate_setup=lambda p: probe.inspect_plate(
                            p, reader, "48 WELL PLATE", tmp_path, summary, probe.Capture()))
    assert summary["passed"] and summary["dimensions_match"]
    assert summary["steps_unchanged"] and summary["partial_selection_accepted"]
    plate.set_partial_plate.assert_called_once()
    probe.write_report(tmp_path, summary)
    assert "Outcome: PASS" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "original.xml").read_text() == ORIGINAL
    assert (tmp_path / "readback.xml").is_file()


def test_setter_failure_saves_empty_readback_without_parse_error(tmp_path):
    capture = probe.Capture()
    logger = logging.getLogger("test-setter-error")
    logger.addHandler(capture)
    plate = SimpleNamespace(get_procedure=Mock(side_effect=[ORIGINAL, ""]),
                            set_procedure=lambda _: logger.error("Token is not supported"))
    summary = {"plate_type": "48 WELL PLATE"}
    try:
        with pytest.raises(RuntimeError, match="No procedure returned"):
            probe.inspect_plate(plate, SimpleNamespace(logger=logger), "48 WELL PLATE", tmp_path, summary, capture)
    finally:
        logger.removeHandler(capture)
    assert summary["setter_errors"] == ["Token is not supported"]
    assert (tmp_path / "readback.xml").read_text() == ""
    probe.write_report(tmp_path, summary)
    assert "Outcome: STOPPED" in (tmp_path / "report.md").read_text()
    assert "NOT ESTABLISHED" in (tmp_path / "report.md").read_text()


def test_report_does_not_claim_measurement_success(tmp_path):
    probe.write_report(tmp_path, {"plate_type": "48 WELL PLATE", "phase": "validate", "error": "rejected"})
    report = (tmp_path / "report.md").read_text()
    assert "rejected" in report and "cannot establish actual measurement" in report


def test_embedded_wells_preserved_without_runtime_selection(tmp_path):
    import xml.etree.ElementTree as ET
    xml = [ORIGINAL]
    plate = SimpleNamespace(
        get_procedure=lambda: xml[0],
        set_procedure=lambda value: xml.__setitem__(0, value),
        _invoke_property_get=lambda name, **kw: {"MaxRows": 6, "MaxColumns": 8}[name],
        validate_procedure=lambda _: (1, None))
    # No set_partial_plate method: this path must not call it.
    source = ET.fromstring(ORIGINAL)
    ET.SubElement(source.find("StepList/ReadStep"), "Wells").text = "Full Plate"
    xml[0] = ET.tostring(source, encoding="unicode")
    summary = {"plate_type": "48 WELL PLATE", "step_wells": "A1..A3"}
    probe.inspect_plate(plate, SimpleNamespace(logger=logging.getLogger(__name__)),
                        "48 WELL PLATE", tmp_path, summary, probe.Capture(), "A1..A3")
    assert summary["passed"] and summary["embedded_selection_accepted"]
    assert summary["returned_step_wells"] == ["A1..A3"]
    assert summary["steps_match_requested"] and not summary["steps_unchanged"]
