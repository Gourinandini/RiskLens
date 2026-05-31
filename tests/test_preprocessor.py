"""
Unit Tests — DataPreprocessor

Tests feature engineering correctness, imputation contract, encoding strategy
selection, schema enforcement, and the fit→transform pipeline contract.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.preprocessor import DataPreprocessor


def _make_minimal_df(n: int = 100) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame with the essential raw columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "SK_ID_CURR":         np.arange(1, n + 1),
        "TARGET":             rng.integers(0, 2, n),
        "AMT_INCOME_TOTAL":   rng.uniform(50_000, 500_000, n),
        "AMT_CREDIT":         rng.uniform(100_000, 1_000_000, n),
        "AMT_ANNUITY":        rng.uniform(5_000, 50_000, n),
        "AMT_GOODS_PRICE":    rng.uniform(100_000, 900_000, n),
        "CODE_GENDER":        rng.choice(["M", "F"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "NAME_INCOME_TYPE":   rng.choice(["Working", "Pensioner", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE":rng.choice(["Secondary / secondary special", "Higher education"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Single / not married", "Married"], n),
        "NAME_HOUSING_TYPE":  rng.choice(["House / apartment", "Rented apartment"], n),
        "NAME_OCCUPATION_TYPE": rng.choice(["Laborers", "Core staff", "Managers"], n),
        "DAYS_BIRTH":         rng.integers(-25000, -6000, n),
        "DAYS_EMPLOYED":      rng.integers(-10000, -100, n),
        "CNT_CHILDREN":       rng.integers(0, 5, n),
        "CNT_FAM_MEMBERS":    rng.uniform(1.0, 6.0, n),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.0, 0.1, n),
        "REGION_RATING_CLIENT": rng.integers(1, 4, n),
        "EXT_SOURCE_1":       rng.uniform(0.0, 1.0, n),
        "EXT_SOURCE_2":       rng.uniform(0.0, 1.0, n),
        "EXT_SOURCE_3":       rng.uniform(0.0, 1.0, n),
    })


class TestDataPreprocessorFitTransform(unittest.TestCase):
    """Tests for the fit_transform pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_minimal_df(200)
        cls.preprocessor = DataPreprocessor()
        cls.X_train, cls.X_val, cls.y_train, cls.y_val = cls.preprocessor.fit_transform(cls.df.copy())

    # ── Shape & split tests ──────────────────────────────────────────────────

    def test_80_20_split_shapes(self):
        self.assertEqual(len(self.X_train), 160)
        self.assertEqual(len(self.X_val),   40)
        self.assertEqual(len(self.y_train), 160)
        self.assertEqual(len(self.y_val),   40)

    def test_no_target_column_in_output(self):
        self.assertNotIn("TARGET", self.X_train.columns)
        self.assertNotIn("TARGET", self.X_val.columns)

    def test_no_id_column_in_output(self):
        self.assertNotIn("SK_ID_CURR", self.X_train.columns)

    # ── Engineered features ──────────────────────────────────────────────────

    def test_age_years_engineered(self):
        self.assertIn("AGE_YEARS", self.X_train.columns)

    def test_employment_years_engineered(self):
        self.assertIn("EMPLOYMENT_YEARS", self.X_train.columns)

    def test_credit_income_ratio_engineered(self):
        self.assertIn("CREDIT_INCOME_RATIO", self.X_train.columns)

    def test_annuity_income_ratio_engineered(self):
        self.assertIn("ANNUITY_INCOME_RATIO", self.X_train.columns)

    def test_credit_term_months_engineered(self):
        self.assertIn("CREDIT_TERM_MONTHS", self.X_train.columns)

    def test_ext_source_mean_engineered(self):
        self.assertIn("EXT_SOURCE_MEAN", self.X_train.columns)

    # ── No NaN in output ────────────────────────────────────────────────────

    def test_no_nan_in_train(self):
        self.assertFalse(self.X_train.isnull().any().any(), "NaNs found in X_train after preprocessing")

    def test_no_nan_in_val(self):
        self.assertFalse(self.X_val.isnull().any().any(), "NaNs found in X_val after preprocessing")

    # ── Data type consistency ────────────────────────────────────────────────

    def test_all_numeric_train(self):
        for col in self.X_train.columns:
            self.assertTrue(
                np.issubdtype(self.X_train[col].dtype, np.number),
                f"Column '{col}' is not numeric in X_train"
            )

    # ── Feature name consistency ─────────────────────────────────────────────

    def test_train_val_same_columns(self):
        self.assertListEqual(
            sorted(self.X_train.columns.tolist()),
            sorted(self.X_val.columns.tolist()),
            "X_train and X_val column sets differ"
        )

    def test_get_feature_names_matches_train(self):
        self.assertEqual(
            sorted(self.preprocessor.get_feature_names()),
            sorted(self.X_train.columns.tolist())
        )


class TestDataPreprocessorTransform(unittest.TestCase):
    """Tests for the transform-only (inference) path."""

    @classmethod
    def setUpClass(cls):
        df_train = _make_minimal_df(200)
        cls.preprocessor = DataPreprocessor()
        cls.preprocessor.fit_transform(df_train.copy())

        # Build a single-row inference DataFrame (no TARGET column)
        cls.single_row = pd.DataFrame([{
            "SK_ID_CURR":            999999,
            "AMT_INCOME_TOTAL":      150_000.0,
            "AMT_CREDIT":            500_000.0,
            "AMT_ANNUITY":           25_000.0,
            "AMT_GOODS_PRICE":       450_000.0,
            "CODE_GENDER":           "F",
            "NAME_CONTRACT_TYPE":    "Cash loans",
            "NAME_INCOME_TYPE":      "Working",
            "NAME_EDUCATION_TYPE":   "Higher education",
            "NAME_FAMILY_STATUS":    "Married",
            "NAME_HOUSING_TYPE":     "House / apartment",
            "NAME_OCCUPATION_TYPE":  "Core staff",
            "DAYS_BIRTH":            -12_000,
            "DAYS_EMPLOYED":         -2_000,
            "CNT_CHILDREN":          1,
            "CNT_FAM_MEMBERS":       3.0,
            "REGION_POPULATION_RELATIVE": 0.02,
            "REGION_RATING_CLIENT":  2,
            "EXT_SOURCE_1":          0.7,
            "EXT_SOURCE_2":          0.6,
            "EXT_SOURCE_3":          0.5,
        }])
        cls.X_single = cls.preprocessor.transform(cls.single_row)

    def test_output_one_row(self):
        self.assertEqual(len(self.X_single), 1)

    def test_output_columns_match_training(self):
        self.assertListEqual(
            sorted(self.X_single.columns.tolist()),
            sorted(self.preprocessor.get_feature_names())
        )

    def test_no_nan_in_inference_output(self):
        self.assertFalse(self.X_single.isnull().any().any())

    def test_target_column_safely_ignored(self):
        """transform() should silently drop TARGET if present (robustness check)."""
        row_with_target = self.single_row.copy()
        row_with_target["TARGET"] = 0
        result = self.preprocessor.transform(row_with_target)
        self.assertNotIn("TARGET", result.columns)
        self.assertEqual(len(result), 1)


class TestDataPreprocessorMissingHandling(unittest.TestCase):
    """Tests that missing values in inference data are imputed gracefully."""

    @classmethod
    def setUpClass(cls):
        df_train = _make_minimal_df(200)
        cls.preprocessor = DataPreprocessor()
        cls.preprocessor.fit_transform(df_train.copy())

    def test_missing_numerical_imputed(self):
        row = pd.DataFrame([{
            "SK_ID_CURR":            1,
            "AMT_INCOME_TOTAL":      None,   # deliberately missing
            "AMT_CREDIT":            500_000.0,
            "AMT_ANNUITY":           None,   # deliberately missing
            "CODE_GENDER":           "M",
            "NAME_CONTRACT_TYPE":    "Cash loans",
            "NAME_INCOME_TYPE":      "Working",
            "NAME_EDUCATION_TYPE":   "Higher education",
            "NAME_FAMILY_STATUS":    "Married",
            "NAME_HOUSING_TYPE":     "House / apartment",
            "DAYS_BIRTH":            -10_000,
            "DAYS_EMPLOYED":         -1_500,
            "EXT_SOURCE_2":          0.5,
        }])
        result = self.preprocessor.transform(row)
        self.assertFalse(result.isnull().any().any(), "NaNs present after imputation of missing numerics")

    def test_unseen_categorical_handled(self):
        """handle_unknown='ignore' in OHE — unseen categories should not raise."""
        row = pd.DataFrame([{
            "SK_ID_CURR":            2,
            "AMT_INCOME_TOTAL":      200_000.0,
            "AMT_CREDIT":            400_000.0,
            "AMT_ANNUITY":           20_000.0,
            "CODE_GENDER":           "XYZ",  # unseen gender
            "NAME_CONTRACT_TYPE":    "UNSEEN_TYPE",
            "NAME_INCOME_TYPE":      "Student",  # unseen category
            "NAME_EDUCATION_TYPE":   "Higher education",
            "NAME_FAMILY_STATUS":    "Single / not married",
            "NAME_HOUSING_TYPE":     "House / apartment",
            "DAYS_BIRTH":            -9_000,
            "DAYS_EMPLOYED":         -500,
            "EXT_SOURCE_2":          0.4,
        }])
        try:
            result = self.preprocessor.transform(row)
            self.assertFalse(result.isnull().any().any())
        except Exception as e:
            self.fail(f"transform raised an exception on unseen categories: {e}")


class TestDataPreprocessorRequireTarget(unittest.TestCase):
    """Tests that fit_transform raises ValueError when TARGET is absent."""

    def test_missing_target_raises_value_error(self):
        preprocessor = DataPreprocessor()
        df_no_target = _make_minimal_df(100).drop(columns=["TARGET"])
        with self.assertRaises(ValueError):
            preprocessor.fit_transform(df_no_target)


if __name__ == "__main__":
    unittest.main()
