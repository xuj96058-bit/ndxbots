from dataclasses import fields, replace
import unittest
from unittest.mock import patch

from ndxbots.config import Settings, load_settings


class RegimeConfigTests(unittest.TestCase):
    def test_defaults_and_old_constructor_are_compatible(self):
        with patch("ndxbots.config._load_yaml", return_value={}):
            config = load_settings()
        self.assertEqual(config.regime_missing_policy, "floor")
        self.assertEqual(config.regime_min_coverage, .8)
        self.assertEqual(config.regime_leverage_max_age_days, 90)
        new_fields = {"regime_missing_policy", "regime_min_coverage", "regime_leverage_max_age_days"}
        old = {field.name: getattr(config, field.name) for field in fields(Settings)
               if field.name not in new_fields}
        self.assertEqual(Settings(**old).regime_missing_policy, "floor")

    def test_explicit_policy_and_quality_settings(self):
        with patch("ndxbots.config._load_yaml", return_value={"regime": {
                "missing_policy": "error", "min_coverage": .9,
                "leverage_max_age_days": 60}}):
            config = load_settings()
        self.assertEqual(config.regime_missing_policy, "error")
        self.assertEqual(config.regime_min_coverage, .9)
        self.assertEqual(config.regime_leverage_max_age_days, 60)

    def test_yaml_rejects_invalid_regime_parameters(self):
        invalid = [
            {"breadth_weight": .8, "leverage_weight": .3},
            {"breadth_weight": float("nan")}, {"leverage_weight": float("inf")},
            {"breadth_weight": -.1}, {"breadth_weight": True},
            {"cold": .5}, {"split": .8}, {"hot": .1},
            {"min_coverage": 0}, {"min_coverage": 1.1},
            {"leverage_max_age_days": 0}, {"leverage_max_age_days": 1.5},
            {"leverage_lag_days": -1}, {"leverage_lag_days": True},
            {"gross_smooth": 0}, {"gross_smooth": 2.5},
            {"gross_ladder_step": 0}, {"gross_step": 0},
            {"gross_floor": -1}, {"gross_floor": float("nan")},
            {"dual_hot": 1.1}, {"gross_ladder_start": float("inf")},
            {"min_periods": 253}, {"ma_window": 0},
            {"missing_policy": "full"}, {"missing_policy": None},
        ]
        for values in invalid:
            with self.subTest(values=values), patch("ndxbots.config._load_yaml",
                    return_value={"regime": values}):
                with self.assertRaises(ValueError):
                    load_settings()

    def test_direct_settings_updates_are_validated(self):
        config = load_settings()
        for values in ({"regime_missing_policy": "normal"}, {"gross_smooth": 0},
                       {"regime_hot": None}, {"regime_breadth_weight": .9}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config, **values)


if __name__ == "__main__":
    unittest.main()
