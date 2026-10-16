"""Exercise real coordinator and reader grouping without hardware drivers."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import pandas as pd

from biotek_new import Biotek_Wrapper


class PreparationProtocolTests(unittest.TestCase):
    def test_shake_executes_once_before_all_well_groups(self):
        source = Path(__file__).resolve().parents[1] / "master_usdl_coordinator.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Lash_E")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "measure_wellplate")
        namespace = {"pd": pd, "os": os}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for simulate, preparation in ((False, "shake_5.prt"), (False, None), (True, "shake_5.prt")):
            with self.subTest(simulate=simulate, preparation=preparation):
                lash = MagicMock()
                lash.simulate = simulate
                lash.nr_robot.get_config_parameter.return_value = "96 WELL PLATE"
                reader = object.__new__(Biotek_Wrapper)
                reader.simulate = False
                reader.biotek = MagicMock()
                reader._log = MagicMock()
                reader.determine_read_type = MagicMock(return_value="read")
                reader.run_plate = MagicMock(side_effect=lambda *args, **kwargs: pd.DataFrame({"334_373": [1.0]}))
                lash.cytation.run_protocol.side_effect = reader.run_protocol
                namespace["measure_wellplate"](lash, "fluorescence.prt", list(range(45)),
                                               repeats=2, preparation_protocol=preparation)
                calls = lash.cytation.run_protocol.call_args_list
                if simulate:
                    self.assertEqual(calls, [])
                else:
                    self.assertEqual([c.args[0] for c in calls],
                                     ([preparation] if preparation else []) + ["fluorescence.prt"] * 2)
                    # Two fluorescence reads each span four groups; shaking adds only one.
                    self.assertEqual(reader.run_plate.call_count, 9 if preparation else 8)
                    if preparation:
                        self.assertEqual(calls[0].args[1], [0])
                lash.move_wellplate_to_cytation.assert_called_once()
                lash.move_wellplate_back_from_cytation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
