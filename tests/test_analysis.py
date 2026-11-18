import unittest

import numpy as np
import pandas as pd

from scripts.run_analysis import diff_ci, uplift_table


class AnalysisSmokeTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "treatment": [0, 0, 0, 1, 1, 1],
                "visit": [0, 0, 1, 0, 1, 1],
                "conversion": [0, 0, 0, 0, 0, 1],
            }
        )

    def test_itt_is_treated_minus_control(self):
        self.assertAlmostEqual(diff_ci(self.frame, "visit")["lift"], 1 / 3)

    def test_uplift_table_has_budget_rows(self):
        table = uplift_table(self.frame, np.arange(6, dtype=float), "visit", 5.0)
        self.assertEqual(table["target_pct"].tolist(), [1, 5, 10, 20, 30, 50, 100])
        self.assertTrue((table["n"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
