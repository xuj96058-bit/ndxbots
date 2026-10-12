"""Configuration selects the requested historical Hurst semantics explicitly."""
from dataclasses import fields
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ndxbots.config import Settings, load_settings


class HurstConfigTests(unittest.TestCase):
    def test_omitted_mode_keeps_strict_entry_default(self):
        with patch("ndxbots.config._load_yaml", return_value={}):
            settings = load_settings()
        self.assertEqual(settings.hurst_gate_mode, "strict_entry")
        old_arguments = {field.name: getattr(settings, field.name)
                         for field in fields(Settings) if field.name != "hurst_gate_mode"}
        self.assertEqual(Settings(**old_arguments).hurst_gate_mode, "strict_entry")

    def test_explicit_supported_modes(self):
        for mode in ("strict_entry", "legacy_regime"):
            with self.subTest(mode=mode), patch("ndxbots.config._load_yaml",
                    return_value={"strategy": {"hurst_gate_mode": mode}}):
                self.assertEqual(load_settings().hurst_gate_mode, mode)

    def test_invalid_modes_are_rejected(self):
        for mode in (None, "", "legacy", "STRICT_ENTRY", 0, True):
            with self.subTest(mode=mode), patch("ndxbots.config._load_yaml",
                    return_value={"strategy": {"hurst_gate_mode": mode}}):
                with self.assertRaisesRegex(ValueError, "hurst_gate_mode"):
                    load_settings()


if __name__ == "__main__":
    unittest.main()
