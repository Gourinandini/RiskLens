import unittest
import os
import pandas as pd
import numpy as np
from src.utils.config import DATA_PATH, DB_PATH, MODEL_PATH
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.ml.predict import RiskPredictor
from src.talk_to_data.query_runner import QueryRunner

class TestCreditRiskPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests to optimize runtime."""
        cls.loader = DataLoader()
        cls.df = cls.loader.load_application_train(nrows=10000)

    def test_config_paths(self):
        """Test that configuration paths exist."""
        self.assertTrue(bool(DATA_PATH))
        self.assertTrue(bool(DB_PATH))
        self.assertTrue(bool(MODEL_PATH))

    def test_data_loader(self):
        """Test DataLoader operations and summaries."""
        self.assertFalse(self.df.empty)
        self.assertIn("TARGET", self.df.columns)
        self.assertIn("SK_ID_CURR", self.df.columns)
        
        summary = self.loader.get_dataset_summary()
        self.assertGreater(summary["rows"], 0)
        self.assertGreater(summary["columns"], 0)
        self.assertGreaterEqual(summary["missing_percent"], 0.0)

    def test_preprocessor(self):
        """Test feature engineering, scaling, split, and mapping functions."""
        # Use a small slice for speed
        df_sample = self.df.head(200).copy()
        
        preprocessor = DataPreprocessor()
        X_train, X_val, y_train, y_val = preprocessor.fit_transform(df_sample)
        
        # Test shape ratios (80/20 train-val split)
        self.assertEqual(len(X_train), 160)
        self.assertEqual(len(X_val), 40)
        self.assertEqual(len(y_train), 160)
        self.assertEqual(len(y_val), 40)
        
        # Verify engineered column outputs
        self.assertIn("AGE_YEARS", X_train.columns)
        self.assertIn("EMPLOYMENT_YEARS", X_train.columns)
        self.assertIn("CREDIT_INCOME_RATIO", X_train.columns)
        self.assertIn("ANNUITY_INCOME_RATIO", X_train.columns)
        self.assertIn("CREDIT_TERM_MONTHS", X_train.columns)
        self.assertIn("EXT_SOURCE_MEAN", X_train.columns)

    def test_predictor(self):
        """Test risk probability prediction and local SHAP explanations."""
        predictor = RiskPredictor()
        
        # Define mock applicant payload matching form values
        input_payload = {
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "CODE_GENDER": "F",
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "DAYS_BIRTH": -12000.0,
            "DAYS_EMPLOYED": -2000.0,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.5
        }
        
        res = predictor.predict(input_payload)
        self.assertIn("raw_probability", res)
        self.assertIn("risk_band", res)
        self.assertIn(res["risk_band"], ["Low", "Medium", "High"])
        
        # Test SHAP value computation
        shap_res = predictor.get_shap_values(input_payload)
        self.assertIn("shap_values", shap_res)
        self.assertIn("top_5_features", shap_res)
        self.assertEqual(len(shap_res["top_5_features"]), 5)

    def test_query_runner_validation(self):
        """Test SQLite security rules block non-SELECT queries."""
        runner = QueryRunner()
        
        # Standard SELECT queries should be allowed
        self.assertTrue(runner.validate_sql("SELECT * FROM applications LIMIT 10"))
        self.assertTrue(runner.validate_sql("SELECT AVG(AMT_INCOME_TOTAL) FROM applications WHERE TARGET = 1;"))
        
        # DDL/DML destructive statements must be blocked
        self.assertFalse(runner.validate_sql("DROP TABLE applications;"))
        self.assertFalse(runner.validate_sql("UPDATE applications SET TARGET = 0;"))
        self.assertFalse(runner.validate_sql("DELETE FROM applications;"))
        self.assertFalse(runner.validate_sql("SELECT * FROM applications; DROP TABLE applications;"))

    def test_query_runner_execution(self):
        """Test QueryRunner can execute SELECT statement on DB."""
        runner = QueryRunner()
        res = runner.execute("SELECT SK_ID_CURR, TARGET, RISK_BAND FROM applications LIMIT 5")
        self.assertIsNone(res["error"])
        self.assertEqual(len(res["columns"]), 3)
        self.assertEqual(res["row_count"], 5)
        self.assertEqual(len(res["rows"]), 5)

if __name__ == "__main__":
    unittest.main()
