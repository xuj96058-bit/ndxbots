from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
with patch("dotenv.load_dotenv", return_value=False):
    from ndxbots.regime.compute import _breadth, compute_regime
    from ndxbots.regime.membership import member_mask, member_mask_with_expected


def settings(**overrides):
    values = dict(benchmark="US.QQQ", regime_ma_window=5, regime_nhnl_window=3,
                  regime_rsi_window=3, regime_min_names=2, regime_min_coverage=0.8,
                  regime_ew_window=3, regime_top_amount=1, regime_window=10,
                  regime_min_periods=10, regime_breadth_weight=0.7,
                  regime_leverage_weight=0.3, regime_hot=0.8, regime_cold=0.2,
                  regime_split=0.5, regime_leverage_lag_days=21,
                  regime_leverage_max_age_days=90)
    values.update(overrides)
    return SimpleNamespace(**values)


def intervals(codes=("A", "B", "C")):
    return pd.DataFrame({"code": list(codes), "start": pd.Timestamp("1990-01-01"), "end": pd.NaT})


def panel(periods=80, with_volume=True):
    days = pd.bdate_range("2020-01-01", periods=periods)
    t = np.arange(periods)
    rows = []
    for code, close in {
        "A": 100 + t * 0.5 + np.sin(t / 2),
        "B": 200 - t * 0.2 + np.cos(t / 3),
        "C": 150 + np.sin(t / 3) * 4,
        "US.QQQ": 100 + t * 0.1 + np.cos(t / 4),
    }.items():
        volume = 1000 + (t % 11) * (20 if code == "A" else 80)
        frame = pd.DataFrame({"date": days, "code": code, "close": close, "turnover": volume * close})
        if with_volume:
            frame["volume"] = volume
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def compute_fixture(prices, config=None):
    with patch("ndxbots.regime.membership.load_membership_intervals", return_value=(intervals(), "point_in_time:test")), \
         patch("ndxbots.regime.compute.load_margin_debt", return_value=(pd.DataFrame(columns=["month", "margin_debt"]), "")):
        return compute_regime(prices, config or settings())


class RegimeComputeTests(unittest.TestCase):
    def test_each_component_requires_actual_contributors(self):
        days = pd.bdate_range("2020-01-01", periods=60)
        close = pd.DataFrame(np.nan, index=days, columns=["OLD"] + [f"NEW{i}" for i in range(39)])
        close["OLD"] = 100 + np.arange(60)
        close.iloc[-1, 1:] = 10
        members = pd.DataFrame(True, index=days, columns=close.columns)
        result = _breadth(close, None, members, settings(regime_min_names=40, regime_ma_window=50, regime_nhnl_window=20, regime_rsi_window=14))
        last = result.iloc[-1]
        self.assertEqual(last.n_members, 40)
        self.assertEqual(last.n_breadth_ma50, 1)
        self.assertEqual(last.n_breadth_nhnl, 1)
        self.assertEqual(last.n_breadth_rsi, 1)
        self.assertEqual(last.n_breadth_upvol, 0)
        self.assertTrue(last[["breadth_ma50", "breadth_nhnl", "breadth_upvol", "breadth_rsi"]].isna().all())

    def test_volume_requires_directional_returns_and_positive_finite_volume(self):
        days = pd.bdate_range("2020-01-01", periods=8)
        close = pd.DataFrame({"A": 10 + np.arange(8), "B": 20 - np.arange(8), "C": np.full(8, 10)}, index=days)
        volume = pd.DataFrame(100.0, index=days, columns=close.columns)
        volume.iloc[-1, 1] = -10
        result = _breadth(close, volume, pd.DataFrame(True, index=days, columns=close.columns), settings())
        self.assertEqual(result.n_breadth_upvol.iloc[-1], 1)
        self.assertTrue(np.isnan(result.breadth_upvol.iloc[-1]))

    def test_expected_members_include_missing_codes_and_deduplicate_intervals(self):
        history = pd.concat([intervals(("A", "B", "C", "D", "US.QQQ")), intervals(("A",))], ignore_index=True)
        days = pd.bdate_range("2020-01-01", periods=3)
        with patch("ndxbots.regime.membership.load_membership_intervals", return_value=(history, "point_in_time:test")):
            mask, source, expected = member_mask_with_expected(days, ["A", "B", "US.QQQ"], settings(), "US.QQQ")
            old_mask, old_source = member_mask(days, ["A", "B", "US.QQQ"], settings(), "US.QQQ")
        self.assertTrue(expected.eq(4).all())
        self.assertTrue(mask.sum(axis=1).eq(2).all())
        assert_frame_equal(mask, old_mask)
        self.assertEqual(source, old_source)

    def test_known_membership_and_component_coverage_gate_scores(self):
        with patch("ndxbots.regime.membership.load_membership_intervals", return_value=(intervals(("A", "B", "C", "MISSING")), "point_in_time:test")), \
             patch("ndxbots.regime.compute.load_margin_debt", return_value=(pd.DataFrame(columns=["month", "margin_debt"]), "")):
            result = compute_regime(panel(), settings())
        last = result.iloc[-1]
        self.assertEqual(last.n_expected_members, 4)
        self.assertEqual(last.membership_coverage, 0.75)
        self.assertTrue(last.membership_inadequate)
        self.assertFalse(last.regime_data_valid)
        self.assertTrue(np.isnan(last.sentiment_trade))
        self.assertIn("membership_coverage", last.regime_data_reason)

    def test_explicit_trade_score_excludes_volume_and_tracks_degradation(self):
        result = compute_fixture(panel())
        expected = result[["pct_ma50", "pct_nhnl", "pct_rsi"]].mean(axis=1, skipna=False)
        assert_series_equal(result.sentiment_trade, expected, check_names=False)
        self.assertTrue(result.sentiment_trade_degraded.iloc[-1])
        self.assertTrue(result.regime_data_valid.iloc[-1])
        self.assertNotEqual(result.sentiment.iloc[-1], result.sentiment_trade.iloc[-1])
        without_volume = compute_fixture(panel(with_volume=False))
        self.assertTrue(np.isnan(without_volume.sentiment.iloc[-1]))
        self.assertTrue(without_volume.regime_data_valid.iloc[-1])
        assert_series_equal(result.sentiment_trade, without_volume.sentiment_trade)

    def test_stale_leverage_is_not_scored_and_has_reason(self):
        prices = panel()
        days = pd.DatetimeIndex(sorted(prices.date.unique()))
        leverage = pd.DataFrame({"leverage_yoy": 0.2, "leverage_reference_month": "2019-01",
                                 "leverage_available_on": pd.Timestamp("2019-02-21"),
                                 "leverage_age_days": 400, "leverage_stale": True,
                                 "leverage_availability_source": "assumed_lag"}, index=days)
        with patch("ndxbots.regime.compute.leverage_on_trading_days", return_value=leverage):
            result = compute_fixture(prices)
        self.assertTrue(result.leverage_yoy.isna().all())
        self.assertTrue(result.leverage_pct.isna().all())
        self.assertTrue(result.sentiment_trade_degraded.iloc[-1])
        self.assertIn("leverage_stale", result.regime_data_reason.iloc[-1])

    def test_static_snapshot_has_unknown_coverage_and_flag(self):
        with patch("ndxbots.regime.membership.load_membership_intervals", return_value=(None, "static_snapshot")), \
             patch("ndxbots.regime.compute.load_margin_debt", return_value=(pd.DataFrame(columns=["month", "margin_debt"]), "")):
            result = compute_regime(panel(), settings())
        self.assertTrue(result.n_expected_members.isna().all())
        self.assertTrue(result.membership_coverage.isna().all())
        self.assertTrue(result.membership_static_snapshot.all())
        self.assertFalse(result.membership_coverage_known.any())
        self.assertIn("static_membership", result.regime_data_reason.iloc[-1])

    def test_compute_prefix_and_changed_future_prices_preserve_history(self):
        prices = panel()
        cutoff = sorted(prices.date.unique())[49]
        baseline = compute_fixture(prices)
        prefix = compute_fixture(prices[prices.date <= cutoff])
        changed = prices.copy()
        changed.loc[changed.date > cutoff, ["close", "volume", "turnover"]] *= 20
        changed_result = compute_fixture(changed)
        assert_frame_equal(baseline.iloc[:50], prefix, check_exact=True)
        assert_frame_equal(baseline.iloc[:50], changed_result.iloc[:50], check_exact=True)


if __name__ == "__main__":
    unittest.main()
