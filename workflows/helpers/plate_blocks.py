"""Split samples into plates and batches; randomize wells within each batch's block.

Pure code, no hardware. Each batch takes the next block of consecutive wells (row by row,
e.g. 24 wells = rows A-B of a 96-well plate) and its samples are shuffled within that
block only, so one batch is made and read as one compact block. A batch never spans two
plates: when the plate has fewer free wells than batch_size, the last batch on that plate
is shortened to fill it, and the next batch starts on a new plate.
"""
import pandas as pd


def well_position(index, columns):
    return f"{chr(65 + index // columns)}{index % columns + 1}"


def assign_blocks(samples, batch_size, wells_per_plate, columns, rng=None):
    """Return samples (in input order) with plate, batch, well_index, well_position.

    samples: DataFrame, one row per well to make (replicates already expanded).
    rng: numpy Generator to shuffle within each block, or None to keep input order.
    """
    if batch_size < 1 or wells_per_plate < 1:
        raise ValueError("batch_size and wells_per_plate must be positive")
    out = samples.reset_index(drop=True).copy()
    plates, batches, wells = [], [], []
    plate, batch, next_well, start = 0, 0, 0, 0
    while start < len(out):
        if next_well >= wells_per_plate:
            plate, next_well = plate + 1, 0
        size = min(batch_size, wells_per_plate - next_well, len(out) - start)
        block = list(range(next_well, next_well + size))
        if rng is not None:
            block = [int(i) for i in rng.permutation(block)]
        plates += [plate] * size
        batches += [batch] * size
        wells += block
        start, next_well, batch = start + size, next_well + size, batch + 1
    out["plate"] = plates
    out["batch"] = batches
    out["well_index"] = wells
    out["well_position"] = [well_position(i, columns) for i in wells]
    return out
