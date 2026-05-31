"""
Unit Tests — RulesExtractor

Tests surrogate decision tree fitting, rule extraction structure,
band statistics computation, and JSON serialization/deserialization.
"""

import os
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.ml.rules import RulesExtractor


def _make_synthetic_data(n: int = 500) -> tuple:
    """Generate synthetic preprocessed feature data and probabilities."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "EXT_SOURCE_2":       rng.uniform(0.0, 1.0, n),
        "EXT_SOURCE_3":       rng.uniform(0.0, 1.0, n),
        "EXT_SOURCE_MEAN":    rng.uniform(0.0, 1.0, n),
        "AGE_YEARS":          rng.uniform(20.0, 70.0, n),
        "EMPLOYMENT_YEARS":   rng.uniform(0.0, 30.0, n),
        "CREDIT_INCOME_RATIO": rng.uniform(0.5, 8.0, n),
        "AMT_CREDIT":         rng.uniform(100_000.0, 1_000_000.0, n),
        "AMT_INCOME_TOTAL":   rng.uniform(50_000.0, 500_000.0, n),
        "ANNUITY_INCOME_RATIO": rng.uniform(0.01, 0.5, n),
    })
    # Simulate LightGBM probabilities (higher for low EXT_SOURCE)
    y_prob = 1.0 - X["EXT_SOURCE_2"].values * 0.6 - X["EXT_SOURCE_MEAN"].values * 0.3
    y_prob = np.clip(y_prob + rng.normal(0, 0.1, n), 0.01, 0.99)
    y_true = pd.Series((y_prob >= 0.5).astype(int))
    return X, y_prob, y_true


class TestRulesExtractorStructure(unittest.TestCase):
    """Tests for the structural integrity of extracted rules."""

    @classmethod
    def setUpClass(cls):
        X, y_prob, y_true = _make_synthetic_data(500)
        cls.extractor = RulesExtractor(max_depth=3, min_samples_leaf=50)
        cls.result = cls.extractor.extract(X, y_prob, y_true)

    def test_result_has_required_keys(self):
        required = {"raw_rules_text", "structured_rules", "band_statistics",
                    "surrogate_fidelity", "tree_depth", "n_leaves", "n_rules"}
        self.assertEqual(required, required.intersection(set(self.result.keys())))

    def test_rules_list_is_not_empty(self):
        self.assertGreater(len(self.result["structured_rules"]), 0)

    def test_n_rules_matches_list_length(self):
        self.assertEqual(self.result["n_rules"], len(self.result["structured_rules"]))

    def test_fidelity_is_reasonable(self):
        """Surrogate tree fidelity to LightGBM should be > 0.5."""
        self.assertGreater(self.result["surrogate_fidelity"], 0.5)
        self.assertLessEqual(self.result["surrogate_fidelity"], 1.0)

    def test_tree_depth_bounded(self):
        self.assertLessEqual(self.result["tree_depth"], 3)


class TestRulesStructuredFormat(unittest.TestCase):
    """Tests for the structured_rules list entries."""

    @classmethod
    def setUpClass(cls):
        X, y_prob, y_true = _make_synthetic_data(500)
        extractor = RulesExtractor(max_depth=3, min_samples_leaf=50)
        result = extractor.extract(X, y_prob, y_true)
        cls.rules = result["structured_rules"]

    def test_each_rule_has_required_fields(self):
        required_fields = {
            "rule_id", "conditions", "rule_text", "predicted_class",
            "risk_band", "risk_color", "n_samples", "n_defaulted", "n_repaid",
            "confidence", "support_pct", "avg_default_probability", "description",
        }
        for rule in self.rules:
            with self.subTest(rule_id=rule.get("rule_id")):
                for field in required_fields:
                    self.assertIn(field, rule, f"Field '{field}' missing in rule {rule.get('rule_id')}")

    def test_risk_band_values_valid(self):
        valid_bands = {"Low", "Medium", "High"}
        for rule in self.rules:
            self.assertIn(rule["risk_band"], valid_bands)

    def test_confidence_in_range(self):
        for rule in self.rules:
            self.assertGreaterEqual(rule["confidence"], 0.0)
            self.assertLessEqual(rule["confidence"], 100.0)

    def test_support_in_range(self):
        for rule in self.rules:
            self.assertGreaterEqual(rule["support_pct"], 0.0)
            self.assertLessEqual(rule["support_pct"], 100.0)

    def test_avg_probability_in_range(self):
        for rule in self.rules:
            self.assertGreaterEqual(rule["avg_default_probability"], 0.0)
            self.assertLessEqual(rule["avg_default_probability"], 100.0)

    def test_conditions_is_list(self):
        for rule in self.rules:
            self.assertIsInstance(rule["conditions"], list)

    def test_n_samples_consistency(self):
        """n_defaulted + n_repaid should equal n_samples."""
        for rule in self.rules:
            self.assertEqual(
                rule["n_defaulted"] + rule["n_repaid"], rule["n_samples"],
                f"n_samples mismatch in rule {rule['rule_id']}"
            )


class TestBandStatistics(unittest.TestCase):
    """Tests for the band_statistics section of the result."""

    @classmethod
    def setUpClass(cls):
        X, y_prob, y_true = _make_synthetic_data(500)
        extractor = RulesExtractor(max_depth=3, min_samples_leaf=50)
        result = extractor.extract(X, y_prob, y_true)
        cls.band_stats = result["band_statistics"]

    def test_all_three_bands_present(self):
        self.assertIn("Low", self.band_stats)
        self.assertIn("Medium", self.band_stats)
        self.assertIn("High", self.band_stats)

    def test_band_counts_sum_to_total(self):
        total = sum(v["count"] for v in self.band_stats.values())
        self.assertGreater(total, 0)

    def test_pct_sum_approximately_100(self):
        pct_total = sum(v["pct_of_portfolio"] for v in self.band_stats.values())
        self.assertAlmostEqual(pct_total, 100.0, delta=2.0)

    def test_avg_probability_present(self):
        for band, stats in self.band_stats.items():
            self.assertIn("avg_predicted_probability", stats)
            self.assertGreaterEqual(stats["avg_predicted_probability"], 0.0)
            self.assertLessEqual(stats["avg_predicted_probability"], 100.0)

    def test_actual_default_rate_present_when_y_true_given(self):
        for band, stats in self.band_stats.items():
            self.assertIn("actual_default_rate", stats)
            self.assertIsNotNone(stats["actual_default_rate"])


class TestRulesSaveLoad(unittest.TestCase):
    """Tests for JSON serialization and deserialization of rules."""

    def test_save_and_load_roundtrip(self):
        X, y_prob, y_true = _make_synthetic_data(200)
        extractor = RulesExtractor(max_depth=2, min_samples_leaf=20)
        extractor.extract(X, y_prob, y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rules.json")
            extractor.save(path)

            self.assertTrue(os.path.exists(path))

            loaded = RulesExtractor.load(path)
            self.assertIn("structured_rules", loaded)
            self.assertEqual(loaded["n_rules"], len(loaded["structured_rules"]))

    def test_load_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            RulesExtractor.load("/nonexistent/path/rules.json")

    def test_saved_json_is_valid(self):
        X, y_prob, y_true = _make_synthetic_data(200)
        extractor = RulesExtractor(max_depth=2, min_samples_leaf=20)
        extractor.extract(X, y_prob, y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rules.json")
            extractor.save(path)
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)


class TestGetRuleTable(unittest.TestCase):
    """Tests for the get_rule_table DataFrame output."""

    @classmethod
    def setUpClass(cls):
        X, y_prob, y_true = _make_synthetic_data(300)
        cls.extractor = RulesExtractor(max_depth=3, min_samples_leaf=30)
        cls.extractor.extract(X, y_prob, y_true)

    def test_returns_dataframe(self):
        df = self.extractor.get_rule_table()
        self.assertIsInstance(df, pd.DataFrame)

    def test_dataframe_not_empty(self):
        df = self.extractor.get_rule_table()
        self.assertFalse(df.empty)

    def test_expected_columns_present(self):
        df = self.extractor.get_rule_table()
        for col in ["Rule #", "Risk Band", "Support (%)", "Confidence (%)"]:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
