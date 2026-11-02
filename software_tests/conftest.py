"""Keep tests away from the live overaspirate corrections file."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_overaspirate_corrections(tmp_path, monkeypatch):
    from pipetting_data import overaspirate_corrections
    monkeypatch.setattr(overaspirate_corrections, "CORRECTIONS_FILE",
                        tmp_path / "overaspirate_corrections.csv")
