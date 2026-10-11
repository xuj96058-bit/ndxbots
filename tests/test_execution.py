"""Daily target decisions feeding the sole funded-budget execution model."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from ndxbots.strategy.targets import decide_quantity_targets


class TargetDecisionTests(unittest.TestCase):
    def test_review_targets_use_signal_close_and_own_nav(self):
        decision = decide_quantity_targets(pd.Timestamp("2024-01-02"), np.array([1, -1]),
            np.zeros(2), np.array([10., 20.]), 1_000., 1., 1.)
        np.testing.assert_allclose(decision.quantities, [50., -25.])

    def test_off_review_keeps_quantities_and_only_cuts_or_exits(self):
        decision = decide_quantity_targets(pd.Timestamp("2024-01-03"), np.array([1, 0, 1]),
            np.array([10., 10., 0.]), np.array([15., 5., 10.]), 1_000., .6, 1.)
        np.testing.assert_allclose(decision.quantities, [6., 0., 0.])
        self.assertEqual(decision.reasons[1], "mandatory_selection_exit")
        retry = decide_quantity_targets(pd.Timestamp("2024-01-04"), np.array([1, 0, 1]),
            np.array([10., 10., 0.]), np.array([15., 5., 10.]), 1_000., .6, .6, decision.reduction_targets)
        np.testing.assert_allclose(retry.quantities, [6., 0., 0.])

    def test_off_review_direction_flip_closes_without_new_direction(self):
        decision = decide_quantity_targets(pd.Timestamp("2024-01-03"), np.array([-1]),
            np.array([10.]), np.array([10.]), 1_000., 1., 1.)
        np.testing.assert_allclose(decision.quantities, [0.])

    def test_missing_signal_quote_preserves_existing_quantity(self):
        decision = decide_quantity_targets(pd.Timestamp("2024-01-02"), np.array([1]),
            np.array([3.]), np.array([np.nan]), 1_000., 1., 1.)
        np.testing.assert_allclose(decision.quantities, [3.])


if __name__ == "__main__":
    unittest.main()
