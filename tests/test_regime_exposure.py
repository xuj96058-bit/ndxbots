from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ndxbots.config import load_settings
from ndxbots.regime.exposure import exposure_by_date, gross_by_date, load_regime_table
from ndxbots.strategy.build import _print_regime
from test_intraday_engine import fixture, run_fixture
from test_regime_compute import compute_fixture, panel, settings as compute_settings


def table(values, *, days=None, crowding=.5):
    days = pd.bdate_range("2024-01-02", periods=len(values)) if days is None else pd.DatetimeIndex(days)
    return pd.DataFrame({"date": days, "sentiment": .99,
                         "sentiment_trade": values, "crowding": crowding,
                         "regime_data_valid": True, "regime_data_reason": "ok"})


class RegimeExposureTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(load_settings(), gross_smooth=1)

    def test_original_ladder_boundaries_and_dual_hot_are_preserved(self):
        data = table([.350001, .35, .30, .25, .20, .01, .80], crowding=.80)
        result = exposure_by_date(data, self.config)
        np.testing.assert_allclose(result.gross_target, [1., .9, .8, .7, .6, .6, .6])
        self.assertEqual(result.gross_reason.iloc[-1], "dual_hot")
        self.assertEqual(result.gross_reason.iloc[0], "normal")

    def test_legacy_cache_uses_three_complete_components_not_display_or_upvol(self):
        days = pd.bdate_range("2024-01-02", periods=2)
        legacy = pd.DataFrame({"date": days, "pct_ma50": .3, "pct_nhnl": .3,
                               "pct_rsi": .3, "pct_upvol": [0., 1.],
                               "leverage_pct": .3, "sentiment": [.1, .99], "crowding": .5})
        result = exposure_by_date(legacy, self.config)
        np.testing.assert_allclose(result.sentiment_trade, .3)
        np.testing.assert_allclose(result.gross_target, .8)
        self.assertTrue(result.regime_quality_known.eq(False).all())
        incomplete = legacy.drop(columns="pct_rsi")
        self.assertTrue(gross_by_date(incomplete, self.config).eq(.6).all())
        legacy.loc[0, "leverage_pct"] = 2.
        self.assertEqual(gross_by_date(legacy, self.config).iloc[0], .6)

    def test_missing_cache_and_missing_sessions_use_floor_and_reset_smoothing(self):
        days = pd.bdate_range("2024-01-02", periods=4)
        missing = exposure_by_date(None, self.config, dates=days)
        self.assertTrue(missing.gross_target.eq(.6).all())
        self.assertTrue(missing.gross_reason.str.contains("regime_cache_missing").all())
        config = replace(self.config, gross_smooth=2)
        data = table([.7, .7, .7], days=days[[0, 2, 3]])
        result = exposure_by_date(data, config, dates=days)
        np.testing.assert_allclose(result.gross_target, [.6, .6, .6, 1.])
        self.assertIn("regime_date_missing", result.gross_reason.iloc[1])
        self.assertIn("smoothing_warmup_or_gap", result.gross_reason.iloc[2])

    def test_invalid_scores_or_quality_flags_cannot_become_full_risk(self):
        for name in ("sentiment_trade", "crowding"):
            for value in (np.nan, np.inf, -.1, 1.1, "bad"):
                with self.subTest(name=name, value=value):
                    data = table([.7])
                    data[name] = value
                    result = exposure_by_date(data, self.config)
                    self.assertEqual(result.gross_target.iloc[0], .6)
                    self.assertTrue(result.gross_data_fallback.iloc[0])
        data = table([.7])
        data["regime_data_valid"] = False
        data["regime_data_reason"] = "membership_coverage"
        data["membership_coverage"] = .5
        result = exposure_by_date(data, self.config)
        self.assertEqual(result.gross_target.iloc[0], .6)
        self.assertIn("membership_coverage", result.gross_reason.iloc[0])
        self.assertEqual(result.membership_coverage.iloc[0], .5)

    def test_error_policy_checks_required_interval_without_rejecting_preparation(self):
        config = replace(self.config, gross_smooth=2, regime_missing_policy="error")
        data = table([.7, .7, .7])
        exposure_by_date(data, config, required_dates=pd.DatetimeIndex(data.date.iloc[1:]))
        with self.assertRaisesRegex(ValueError, "smoothing_warmup"):
            gross_by_date(data, config)
        with self.assertRaisesRegex(ValueError, "regime_cache_missing"):
            gross_by_date(None, config, dates=pd.DatetimeIndex(data.date))

    def test_future_values_do_not_change_past_exposure(self):
        config = replace(self.config, gross_smooth=2)
        data = table([.7, .7, .2, .2])
        original = exposure_by_date(data, config)
        changed = data.copy()
        changed.loc[3, ["sentiment_trade", "crowding"]] = [.99, .99]
        assert_frame_equal(original.iloc[:3], exposure_by_date(changed, config).iloc[:3])

    def test_duplicate_cache_dates_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".regime-test-", dir=Path(__file__).resolve().parent) as folder:
            config = replace(self.config, data_root=Path(folder))
            path = config.data_root / "regime"
            path.mkdir()
            data = table([.7, .7], days=["2024-01-02", "2024-01-02"])
            data.to_parquet(path / "daily.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_regime_table(config)

    def test_actual_compute_to_exposure_pipeline_is_causal_and_auditable(self):
        config = replace(self.config, gross_smooth=3)
        prices = panel(periods=80)
        regime = compute_fixture(prices, compute_settings())
        result = exposure_by_date(regime, config)
        self.assertTrue(result.gross_target.between(.6, 1).all())
        self.assertTrue(result.regime_data_degraded.iloc[-1])
        self.assertTrue(result.regime_data_valid.iloc[-1])
        self.assertFalse(result.gross_data_fallback.iloc[-1])
        shorter = compute_fixture(prices.loc[prices.date.le(regime.date.iloc[69])], compute_settings())
        assert_frame_equal(result.iloc[:70].reset_index(drop=True),
                           exposure_by_date(shorter, config).reset_index(drop=True))

    def test_actual_cached_signal_reduces_quantity_at_next_session_open(self):
        parts = fixture()
        with tempfile.TemporaryDirectory(prefix=".regime-test-", dir=Path(__file__).resolve().parent) as folder:
            config = replace(parts[0], data_root=Path(folder), use_gross_ladder=True, gross_smooth=1)
            path = config.data_root / "regime"
            path.mkdir()
            data = table([.7, .2, .7, .7], days=parts[3].date)
            data.to_parquet(path / "daily.parquet", index=False)
            result = run_fixture((config, *parts[1:]))
        reduced = result["fills"].loc[result["fills"].action.eq("exit")].iloc[0]
        self.assertEqual(reduced.signal_date, pd.Timestamp("2024-01-03"))
        self.assertEqual(reduced.date, pd.Timestamp("2024-01-04"))
        self.assertEqual(reduced.time, 570)
        self.assertAlmostEqual(reduced.quantity, -40.)
        self.assertEqual(result["curve"].gross_reason.iloc[1], "cold_ladder")
        self.assertEqual(result["curve"].sentiment_trade.iloc[1], .2)
        self.assertEqual(result["targets"].gross_reason.iloc[1], "cold_ladder")

    def test_backtest_missing_cache_uses_floor_and_error_policy_is_explicit(self):
        parts = fixture()
        with tempfile.TemporaryDirectory(prefix=".regime-test-", dir=Path(__file__).resolve().parent) as folder:
            config = replace(parts[0], data_root=Path(folder), use_gross_ladder=True)
            result = run_fixture((config, *parts[1:]))
            self.assertTrue(result["curve"].gross_target.eq(.6).all())
            self.assertEqual(result["audit"]["regime_fallback_signal_days"], 4)
            with self.assertRaisesRegex(ValueError, "Regime exposure unavailable"):
                run_fixture((replace(config, regime_missing_policy="error"), *parts[1:]))

    def test_reporting_uses_exact_signal_date_and_actual_trade_reason(self):
        data = exposure_by_date(table([.7, .2]), self.config)
        out = io.StringIO()
        with redirect_stdout(out):
            _print_regime(self.config, data.date.iloc[0], data)
        self.assertIn("2024-01-02", out.getvalue())
        self.assertIn("交易情緒=0.700", out.getvalue())
        self.assertIn("觸發=normal", out.getvalue())
        self.assertNotIn("交易情緒=0.200", out.getvalue())


if __name__ == "__main__":
    unittest.main()
