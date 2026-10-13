import unittest

import numpy as np
import pandas as pd

from ndxbots.analysis.sentiment_qqq import correlations, relationship_table
from ndxbots.config import load_settings


class RelationshipTests(unittest.TestCase):
    def inputs(self):
        days = pd.bdate_range("2025-01-01", periods=80)
        qqq = pd.DataFrame({"date": days, "close": np.arange(100.0, 180.0)})
        regime = pd.DataFrame({"date": days, "sentiment": .5, "sentiment_trade": np.linspace(.2, .8, 80),
                               "crowding": .5, "regime_data_valid": True, "regime_data_reason": "ok"})
        return regime, qqq, load_settings()

    def test_returns_calculated_before_cropping_and_forward_tail_is_missing(self):
        regime, qqq, settings = self.inputs()
        table = relationship_table(regime, qqq, settings, qqq.date.iloc[10])
        self.assertAlmostEqual(table.qqq_ret_5d.iloc[0], 110 / 105 - 1)
        self.assertAlmostEqual(table.qqq_forward_5d.iloc[0], 115 / 110 - 1)
        self.assertEqual(table.qqq_forward_1d.notna().sum(), 69)
        self.assertEqual(table.qqq_forward_5d.notna().sum(), 65)
        self.assertEqual(table.qqq_forward_21d.notna().sum(), 49)
        self.assertTrue(pd.isna(table.qqq_forward_1d.iloc[-1]))
        self.assertTrue(pd.notna(table.sentiment_trade_smooth.iloc[0]))

    def test_changed_future_prices_do_not_change_prior_sentiment(self):
        regime, qqq, settings = self.inputs()
        original = relationship_table(regime, qqq, settings, qqq.date.iloc[0])
        altered = qqq.copy()
        altered.loc[40:, "close"] *= 2
        changed = relationship_table(regime, altered, settings, qqq.date.iloc[0])
        pd.testing.assert_series_equal(original.sentiment_trade_smooth, changed.sentiment_trade_smooth)
        self.assertNotEqual(original.qqq_forward_21d.iloc[30], changed.qqq_forward_21d.iloc[30])

    def test_correlations_pairwise_counts_and_rank_ties(self):
        table = pd.DataFrame({"sentiment_trade_smooth": [1, 1, 2, 3, np.nan],
                              "qqq_close": [10, 10, 20, 30, 40]})
        for column in ("qqq_ret_1d", "qqq_ret_5d", "qqq_ret_21d", "qqq_forward_1d", "qqq_forward_5d", "qqq_forward_21d"):
            table[column] = table.qqq_close
        summary = correlations(table)
        self.assertTrue(summary.n.eq(4).all())
        self.assertTrue(np.allclose(summary.pearson, 1))
        self.assertTrue(np.allclose(summary.spearman, 1))

    def test_flat_score_has_undefined_correlation(self):
        table = pd.DataFrame({"sentiment": [0.5] * 4, "qqq_close": [1, 2, 3, 4]})
        for column in ("qqq_ret_1d", "qqq_ret_5d", "qqq_ret_21d", "qqq_forward_1d", "qqq_forward_5d", "qqq_forward_21d"):
            table[column] = table.qqq_close
        self.assertTrue(correlations(table).pearson.isna().all())

    def test_duplicate_or_nonpositive_qqq_is_rejected(self):
        regime, qqq, settings = self.inputs()
        with self.assertRaisesRegex(ValueError, "unique"):
            relationship_table(regime, pd.concat([qqq, qqq.iloc[:1]]), settings)
        qqq.loc[0, "close"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            relationship_table(regime, qqq, settings)


if __name__ == "__main__":
    unittest.main()
