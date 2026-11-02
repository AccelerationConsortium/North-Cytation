from biotek_new import Biotek_Wrapper


def test_24_well_groups_stop_at_six_column_row_boundaries():
    reader = object.__new__(Biotek_Wrapper)
    reader.logger = None

    assert reader.group_wells([7, 0, 6, 5, 4, 23], "24 WELL PLATE") == [
        ["A1"], ["A5", "A6"], ["B1", "B2"], ["D6"],
    ]
    assert reader.group_wells(list(range(24)), "24 WELL PLATE") == [
        [f"{row}{column}" for column in range(1, 7)]
        for row in "ABCD"
    ]
