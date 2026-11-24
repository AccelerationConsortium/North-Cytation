"""Plate/batch splitting with randomization inside each block (workflows/helpers/plate_blocks.py)."""
import numpy as np
import pandas as pd
import pytest

from workflows.helpers.plate_blocks import assign_blocks


def samples(n):
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)]})


def test_each_batch_stays_in_its_block_and_is_shuffled():
    out = assign_blocks(samples(96), 24, 96, 12, rng=np.random.default_rng(42))
    for b in range(4):
        wells = out[out.batch == b].well_index
        assert sorted(wells) == list(range(24 * b, 24 * b + 24))
    assert list(out.well_index) != list(range(96))


def test_no_rng_keeps_input_order():
    out = assign_blocks(samples(30), 24, 96, 12)
    assert list(out.well_index) == list(range(30))
    assert list(out.batch) == [0] * 24 + [1] * 6
    assert out.well_position.iloc[0] == "A1" and out.well_position.iloc[29] == "C6"


def test_new_plate_and_batch_never_spans_plates():
    out = assign_blocks(samples(100), 40, 96, 12, rng=np.random.default_rng(1))
    sizes = out.groupby("batch").size().tolist()
    assert sizes == [40, 40, 16, 4]
    assert out.groupby("batch").plate.nunique().max() == 1
    assert list(out[out.plate == 1].well_index.sort_values()) == [0, 1, 2, 3]


def test_same_seed_same_layout():
    a = assign_blocks(samples(48), 24, 96, 12, rng=np.random.default_rng(7))
    b = assign_blocks(samples(48), 24, 96, 12, rng=np.random.default_rng(7))
    assert a.equals(b)


def test_metadata_stays_with_its_sample():
    s = samples(24).assign(conc=np.arange(24.0))
    out = assign_blocks(s, 24, 96, 12, rng=np.random.default_rng(3))
    assert (out.conc == out.sample_id.str[1:].astype(float)).all()


def test_bad_batch_size():
    with pytest.raises(ValueError):
        assign_blocks(samples(3), 0, 96, 12)
