from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndxbots.config import Settings, load_settings
from ndxbots.strategy.scores import _apply_factor_scores, build_score_table, latest_pool


def score_settings(**changes):
    values = dict(strategy_factors=("f1", "f2"), strategy_invert=(),
                  strategy_quality_weight=0.5)
    values.update(changes)
    return SimpleNamespace(**values)


def build_settings(**changes):
    with patch("ndxbots.config._load_yaml", return_value={}):
        original = load_settings()
    values = dict(strategy_factors=("f1", "f2"), strategy_quality_weight=0.5,
                  strategy_top_n=10, qqq_regime_gate=False, require_above_ma200=False,
                  allow_short=True, space_col="", short_space_col="", min_atr_pct=0,
                  rsi_gate=False, macd_gate=False, bb_width_gate=False)
    values.update(changes)
    return replace(original, **values)


def factor_frame(day="2024-01-02"):
    # The two trend ranks cancel to T=2/3 for every stock. Quality remains signed.
    return pd.DataFrame({"date": pd.to_datetime([day] * 3), "code": ["A", "B", "C"],
                         "f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0],
                         "my_efficiency_21": [-1.0, 0.0, 1.0]})


class FactorScoreTests(unittest.TestCase):
    def test_half_weight_is_equal_group_weight_not_equal_three_factor_weight(self):
        result = _apply_factor_scores(factor_frame(), score_settings())
        np.testing.assert_allclose(result.trend_score, [2 / 3] * 3)
        np.testing.assert_allclose(result.quality_long_score, [1 / 3, 2 / 3, 1])
        np.testing.assert_allclose(result.quality_short_score, [1, 2 / 3, 1 / 3])
        np.testing.assert_allclose(result.long_score, [1 / 2, 2 / 3, 5 / 6])
        np.testing.assert_allclose(result.short_score, [2 / 3, 1 / 2, 1 / 3])
        pd.testing.assert_series_equal(result.score, result.long_score, check_names=False)

    def test_explicit_quarter_weight_has_hand_calculated_scores(self):
        result = _apply_factor_scores(factor_frame(), score_settings(strategy_quality_weight=0.25))
        np.testing.assert_allclose(result.long_score, [7 / 12, 2 / 3, 3 / 4])
        np.testing.assert_allclose(result.short_score, [1 / 2, 5 / 12, 1 / 3])

    def test_short_percentile_endpoint_is_not_complement_of_long(self):
        result = _apply_factor_scores(factor_frame(), score_settings())
        self.assertEqual(result.loc[0, "quality_short_score"], 1.0)
        self.assertEqual(result.loc[2, "quality_short_score"], 1 / 3)
        np.testing.assert_allclose(result.short_score - (1 - result.long_score), [1 / 6] * 3)

    def test_quality_ties_use_average_percentiles_in_both_directions(self):
        frame = factor_frame()
        frame["my_efficiency_21"] = [0.0, 0.0, 1.0]
        result = _apply_factor_scores(frame, score_settings())
        np.testing.assert_allclose(result.quality_long_score, [1 / 2, 1 / 2, 1])
        np.testing.assert_allclose(result.quality_short_score, [5 / 6, 5 / 6, 1 / 3])

    def test_missing_component_never_uses_skipna_to_create_a_score(self):
        frame = factor_frame()
        frame.loc[0, "f2"] = np.nan
        frame.loc[1, "my_efficiency_21"] = np.nan
        result = _apply_factor_scores(frame, score_settings())
        self.assertTrue(result.loc[[0, 1], ["long_score", "short_score", "score"]].isna().all().all())
        self.assertTrue(pd.isna(result.loc[0, "trend_score"]))
        self.assertTrue(pd.isna(result.loc[1, "quality_long_score"]))
        self.assertTrue(pd.isna(result.loc[1, "quality_short_score"]))
        # Each component ranks its observed values; a missing row is not rank zero.
        self.assertEqual(result.loc[0, "quality_long_score"], 0.5)
        self.assertAlmostEqual(result.loc[2, "long_score"], 0.875)
        self.assertAlmostEqual(result.loc[2, "short_score"], 0.375)

    def test_zero_quality_weight_reproduces_legacy_formula_and_inversion(self):
        frame = factor_frame().drop(columns="my_efficiency_21")
        result = _apply_factor_scores(frame, score_settings(strategy_quality_weight=0,
                                                           strategy_invert=("f2",)))
        np.testing.assert_allclose(result.long_score, [1 / 6, 1 / 2, 5 / 6])
        np.testing.assert_allclose(result.short_score, [5 / 6, 1 / 2, 1 / 6])
        self.assertTrue(result[["quality_long_score", "quality_short_score"]].isna().all().all())

    def test_zero_quality_weight_ignores_quality_values_and_absent_column(self):
        frame = factor_frame()
        frame["my_efficiency_21"] = np.nan
        with_column = _apply_factor_scores(frame, score_settings(strategy_quality_weight=0))
        without_column = _apply_factor_scores(factor_frame().drop(columns="my_efficiency_21"),
                                             score_settings(strategy_quality_weight=0))
        columns = ["trend_score", "long_score", "short_score", "score"]
        pd.testing.assert_frame_equal(with_column[columns], without_column[columns])

    def test_zero_quality_weight_still_requires_all_legacy_factors_per_row(self):
        frame = factor_frame().drop(columns="my_efficiency_21")
        frame.loc[0, "f2"] = np.nan
        result = _apply_factor_scores(frame, score_settings(strategy_quality_weight=0))
        self.assertTrue(result.loc[0, ["trend_score", "long_score", "short_score"]].isna().all())

    def test_full_quality_weight_uses_signed_quality_percentiles(self):
        result = _apply_factor_scores(factor_frame(), score_settings(strategy_quality_weight=1))
        np.testing.assert_allclose(result.long_score, [1 / 3, 2 / 3, 1])
        np.testing.assert_allclose(result.short_score, [1, 2 / 3, 1 / 3])

    def test_missing_quality_column_requests_factor_recomputation(self):
        with self.assertRaisesRegex(SystemExit, "my_efficiency_21.*python -m ndxbots.factors.compute"):
            _apply_factor_scores(factor_frame().drop(columns="my_efficiency_21"), score_settings())

    def test_missing_legacy_factor_requests_factor_recomputation(self):
        with self.assertRaisesRegex(SystemExit, "f2.*python -m ndxbots.factors.compute"):
            _apply_factor_scores(factor_frame().drop(columns="f2"), score_settings())

    def test_direct_score_settings_reject_invalid_weight(self):
        for weight in [-0.01, 1.01, np.nan, np.inf, -np.inf]:
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "quality_weight"):
                _apply_factor_scores(factor_frame(), score_settings(strategy_quality_weight=weight))


class ScoreTableTests(unittest.TestCase):
    def test_normalized_date_ranking_excludes_benchmark_before_ranking(self):
        frame = factor_frame()
        frame["date"] = pd.to_datetime(["2024-01-02 09:00", "2024-01-02 12:00", "2024-01-02 16:00"])
        benchmark = pd.DataFrame([dict(date=pd.Timestamp("2024-01-02 20:00"), code="US.QQQ",
                                        f1=100, f2=100, my_efficiency_21=100)])
        original = pd.concat([frame, benchmark], ignore_index=True)
        before = original.copy(deep=True)
        result = build_score_table(original, build_settings()).set_index("code")
        self.assertEqual(result.index.tolist(), ["A", "B", "C"])
        self.assertEqual(result.date.unique().tolist(), [pd.Timestamp("2024-01-02")])
        np.testing.assert_allclose(result.long_score, [1 / 2, 2 / 3, 5 / 6])
        np.testing.assert_allclose(result.short_score, [2 / 3, 1 / 2, 1 / 3])
        pd.testing.assert_frame_equal(original, before)

    def test_entry_gate_does_not_change_cross_section_rank(self):
        frame = factor_frame()
        frame["my_rsi_14"] = [20.0, 20.0, 100.0]
        result = build_score_table(frame, build_settings(rsi_gate=True, rsi_long_max=80)).set_index("code")
        self.assertFalse(result.loc["C", "eligible"])
        self.assertFalse(result.loc["C", "in_pool_long"])
        self.assertEqual(result.loc["C", "quality_long_score"], 1.0)
        self.assertAlmostEqual(result.loc["A", "long_score"], 0.5)
        self.assertAlmostEqual(result.loc["B", "long_score"], 2 / 3)

    def test_missing_quality_row_cannot_enter_pool(self):
        frame = factor_frame()
        frame.loc[2, "my_efficiency_21"] = np.nan
        result = build_score_table(frame, build_settings()).set_index("code")
        self.assertTrue(pd.isna(result.loc["C", "long_score"]))
        self.assertTrue(pd.isna(result.loc["C", "short_score"]))
        self.assertFalse(result.loc["C", "eligible"])
        self.assertFalse(result.loc["C", "in_pool"])
        self.assertFalse(result.loc["C", "in_hold"])

    def test_short_pool_uses_separately_ranked_short_score(self):
        frame = factor_frame()
        frame["my_ma200_gap"] = [-0.1, 0.1, 0.1]
        result = build_score_table(frame, build_settings()).set_index("code")
        self.assertEqual(result.loc["A", "pool_side"], "short")
        self.assertAlmostEqual(result.loc["A", "side_score"], 2 / 3)
        self.assertAlmostEqual(result.loc["A", "long_score"], 0.5)
        self.assertTrue(result.loc["A", "in_pool_short"])

    def test_future_rows_values_and_new_names_do_not_change_past_scores_or_selection(self):
        history = factor_frame()
        past = build_score_table(history, build_settings())
        future = factor_frame("2024-01-03")
        future["f1"] = [1e6, -1e6, 0]
        future["f2"] = [-1e6, 1e6, 0]
        future["my_efficiency_21"] = [1, -1, 0]
        newcomer = pd.DataFrame([dict(date=pd.Timestamp("2024-01-03"), code="D",
                                       f1=1e9, f2=1e9, my_efficiency_21=1)])
        extended = build_score_table(pd.concat([future, newcomer, history], ignore_index=True),
                                     build_settings())
        past_from_extended = extended.loc[extended.date.eq(pd.Timestamp("2024-01-02"))]
        columns = ["date", "code", "trend_score", "quality_long_score", "quality_short_score",
                   "long_score", "short_score", "combo_rank", "in_pool", "in_hold"]
        pd.testing.assert_frame_equal(past[columns].reset_index(drop=True),
                                      past_from_extended[columns].reset_index(drop=True))

    def test_explicit_legacy_weight_builds_with_stale_factor_table(self):
        frame = factor_frame().drop(columns="my_efficiency_21")
        result = build_score_table(frame, build_settings(strategy_quality_weight=0))
        np.testing.assert_allclose(result.long_score, [2 / 3] * 3)
        np.testing.assert_allclose(result.short_score, [1 / 3] * 3)
        with self.assertRaisesRegex(SystemExit, "my_efficiency_21.*ndxbots.factors.compute"):
            build_score_table(frame, build_settings())

    def test_latest_pool_exports_group_diagnostics(self):
        frame = factor_frame().rename(columns={"f1": "my_ma50_gap", "f2": "my_struct_gap"})
        result = latest_pool(build_score_table(frame, build_settings(
            strategy_factors=("my_ma50_gap", "my_struct_gap"))))
        self.assertTrue({"trend_score", "quality_long_score", "quality_short_score",
                         "my_efficiency_21", "my_ma50_gap", "my_struct_gap"}.issubset(result.columns))


class QualityWeightConfigurationTests(unittest.TestCase):
    def test_empty_and_legacy_yaml_adopt_new_half_quality_default(self):
        for raw in [{}, {"strategy": {"factors": ["my_ma50_gap", "my_struct_gap"], "invert": []}}]:
            with self.subTest(raw=raw), patch("ndxbots.config._load_yaml", return_value=raw):
                settings = load_settings()
                self.assertEqual(settings.strategy_quality_weight, 0.5)
                self.assertEqual(settings.strategy_factors, ("my_ma50_gap", "my_struct_gap"))
                self.assertEqual(settings.strategy_invert, ())

    def test_old_settings_constructor_uses_new_default_without_argument(self):
        with patch("ndxbots.config._load_yaml", return_value={}):
            settings = load_settings()
        old_arguments = {field.name: getattr(settings, field.name) for field in fields(settings)
                         if field.name != "strategy_quality_weight"}
        self.assertEqual(Settings(**old_arguments).strategy_quality_weight, 0.5)

    def test_explicit_valid_weights_include_legacy_and_quality_endpoints(self):
        for weight in [0, 0.25, 0.5, 1, "0.75"]:
            with self.subTest(weight=weight), patch("ndxbots.config._load_yaml",
                                                   return_value={"strategy": {"quality_weight": weight}}):
                self.assertEqual(load_settings().strategy_quality_weight, float(weight))

    def test_yaml_rejects_nonfinite_out_of_range_and_nonnumeric_weights(self):
        for weight in [-0.01, 1.01, np.nan, np.inf, -np.inf, "invalid"]:
            with self.subTest(weight=weight), patch("ndxbots.config._load_yaml",
                                                   return_value={"strategy": {"quality_weight": weight}}):
                with self.assertRaises(ValueError):
                    load_settings()


if __name__ == "__main__":
    unittest.main()
