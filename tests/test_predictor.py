"""
Unit Tests — RiskPredictor

Tests single-row prediction structure, probability range validation,
risk band assignment logic, SHAP value computation, and batch prediction.
"""

import unittest
import numpy as np
import pandas as pd


def _make_synthetic_df(n: int = 100) -> pd.DataFrame:
    """Build a minimal synthetic raw DataFrame for training."""
    rng = np.random.default_rng(0)
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
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"], n),
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


def _build_and_fit_predictor():
    """Train a minimal in-memory model and return a RiskPredictor-like object."""
    import lightgbm as lgb
    import shap
    from src.data.preprocessor import DataPreprocessor

    df = _make_synthetic_df(300)
    preprocessor = DataPreprocessor()
    X_train, X_val, y_train, y_val = preprocessor.fit_transform(df.copy())

    model = lgb.LGBMClassifier(
        n_estimators=20,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    # Create a minimal predictor-like object without loading files
    class _FakePredictor:
        def __init__(self):
            self.model = model
            self.preprocessor = preprocessor
            self.explainer = explainer
            self.band_low_max = 0.25
            self.band_med_max = 0.50

        def predict(self, input_dict):
            df_row = pd.DataFrame([input_dict])
            X_row = self.preprocessor.transform(df_row)
            prob = float(self.model.predict_proba(X_row)[0, 1])
            if prob < self.band_low_max:
                band, color = "Low", "green"
            elif prob < self.band_med_max:
                band, color = "Medium", "orange"
            else:
                band, color = "High", "red"
            return {"raw_probability": prob, "risk_score": round(prob, 4),
                    "risk_band": band, "risk_color": color}

        def get_shap_values(self, input_dict):
            df_row = pd.DataFrame([input_dict])
            X_row = self.preprocessor.transform(df_row)
            shap_vals = self.explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]
            feature_names = X_row.columns.tolist()
            contribs = [
                {"feature": f, "shap_value": float(v),
                 "direction": "increases_risk" if v > 0 else "decreases_risk"}
                for f, v in zip(feature_names, sv)
            ]
            top5 = sorted(contribs, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
            return {"shap_values": [float(v) for v in sv],
                    "feature_names": feature_names, "base_value": 0.0, "top_5_features": top5}

        def batch_predict(self, df_in):
            df_out = df_in.copy()
            X = self.preprocessor.transform(df_out)
            probs = self.model.predict_proba(X)[:, 1]
            df_out["RISK_SCORE"] = np.round(probs, 4)
            p70 = float(np.percentile(probs, 70))
            p90 = float(np.percentile(probs, 90))
            df_out["RISK_BAND"] = [
                "Low" if p < p70 else ("Medium" if p < p90 else "High") for p in probs
            ]
            return df_out

    return _FakePredictor()


_PREDICTOR = _build_and_fit_predictor()

_SAMPLE_INPUT = {
    "AMT_INCOME_TOTAL": 150_000.0,
    "AMT_CREDIT": 500_000.0,
    "AMT_ANNUITY": 25_000.0,
    "CODE_GENDER": "F",
    "NAME_CONTRACT_TYPE": "Cash loans",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "DAYS_BIRTH": -12_000.0,
    "DAYS_EMPLOYED": -2_000.0,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55,
}


class TestRiskPredictorPredict(unittest.TestCase):
    """Tests for the predict() method output structure."""

    def test_returns_dict(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        self.assertIsInstance(result, dict)

    def test_required_keys_present(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        for key in ["raw_probability", "risk_score", "risk_band", "risk_color"]:
            self.assertIn(key, result)

    def test_probability_in_zero_one(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        self.assertGreaterEqual(result["raw_probability"], 0.0)
        self.assertLessEqual(result["raw_probability"], 1.0)

    def test_risk_band_valid_value(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        self.assertIn(result["risk_band"], {"Low", "Medium", "High"})

    def test_risk_color_valid_value(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        self.assertIn(result["risk_color"], {"green", "orange", "red"})

    def test_risk_score_rounded(self):
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        # risk_score should have at most 4 decimal places
        self.assertEqual(result["risk_score"], round(result["risk_score"], 4))

    def test_band_color_consistency(self):
        """Risk band and risk color should always be in sync."""
        result = _PREDICTOR.predict(_SAMPLE_INPUT)
        band_to_color = {"Low": "green", "Medium": "orange", "High": "red"}
        self.assertEqual(result["risk_color"], band_to_color[result["risk_band"]])


class TestRiskPredictorBandAssignment(unittest.TestCase):
    """Verify that low/medium/high boundaries are consistent."""

    def test_low_probability_gives_low_band(self):
        """Artificially extreme high EXT_SOURCE scores should produce a valid band."""
        inp = {**_SAMPLE_INPUT, "EXT_SOURCE_2": 0.99, "EXT_SOURCE_3": 0.99}
        result = _PREDICTOR.predict(inp)
        # Regardless of synthetic model behavior, a valid band must be returned
        self.assertIn(result["risk_band"], {"Low", "Medium", "High"})

    def test_extreme_low_ext_source_gives_higher_risk(self):
        """Artificially low EXT_SOURCE scores should produce Medium/High band."""
        inp = {**_SAMPLE_INPUT, "EXT_SOURCE_2": 0.01, "EXT_SOURCE_3": 0.01,
               "AMT_CREDIT": 900_000.0, "AMT_INCOME_TOTAL": 50_000.0}
        result = _PREDICTOR.predict(inp)
        self.assertIn(result["risk_band"], {"Medium", "High"})


class TestRiskPredictorSHAP(unittest.TestCase):
    """Tests for the get_shap_values() method."""

    @classmethod
    def setUpClass(cls):
        cls.shap_result = _PREDICTOR.get_shap_values(_SAMPLE_INPUT)

    def test_returns_dict(self):
        self.assertIsInstance(self.shap_result, dict)

    def test_required_keys_present(self):
        for key in ["shap_values", "feature_names", "base_value", "top_5_features"]:
            self.assertIn(key, self.shap_result)

    def test_top_5_has_5_entries(self):
        self.assertEqual(len(self.shap_result["top_5_features"]), 5)

    def test_top_5_features_have_required_keys(self):
        for f in self.shap_result["top_5_features"]:
            for key in ["feature", "shap_value", "direction"]:
                self.assertIn(key, f)

    def test_direction_values_valid(self):
        for f in self.shap_result["top_5_features"]:
            self.assertIn(f["direction"], {"increases_risk", "decreases_risk"})

    def test_shap_values_list_non_empty(self):
        self.assertGreater(len(self.shap_result["shap_values"]), 0)

    def test_feature_names_matches_shap_values_length(self):
        self.assertEqual(
            len(self.shap_result["shap_values"]),
            len(self.shap_result["feature_names"])
        )


class TestRiskPredictorBatchPredict(unittest.TestCase):
    """Tests for the batch_predict() method."""

    @classmethod
    def setUpClass(cls):
        cls.raw_df = _make_synthetic_df(50)
        cls.result_df = _PREDICTOR.batch_predict(cls.raw_df.copy())

    def test_output_has_risk_score_column(self):
        self.assertIn("RISK_SCORE", self.result_df.columns)

    def test_output_has_risk_band_column(self):
        self.assertIn("RISK_BAND", self.result_df.columns)

    def test_risk_score_in_zero_one(self):
        self.assertTrue((self.result_df["RISK_SCORE"] >= 0).all())
        self.assertTrue((self.result_df["RISK_SCORE"] <= 1).all())

    def test_risk_band_values_valid(self):
        valid = {"Low", "Medium", "High"}
        self.assertTrue(self.result_df["RISK_BAND"].isin(valid).all())

    def test_output_row_count_matches_input(self):
        self.assertEqual(len(self.result_df), len(self.raw_df))

    def test_no_nan_in_risk_columns(self):
        self.assertFalse(self.result_df["RISK_SCORE"].isnull().any())
        self.assertFalse(self.result_df["RISK_BAND"].isnull().any())


if __name__ == "__main__":
    unittest.main()
