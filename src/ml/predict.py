"""
Model Inference and SHAP Explanation Module.

Defines the RiskPredictor class responsible for loading the trained model,
running inference (single-row or batch), mapping results to risk categories,
and calculating SHAP explanations for predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RiskPredictor:
    """
    RiskPredictor executes predictions using the trained LightGBM model,
    calculates risk bands, and extracts SHAP explainability matrices.
    """

    def __init__(self) -> None:
        """
        Initialize RiskPredictor. Loads model and preprocessor artifacts,
        initializes the SHAP TreeExplainer, and derives band thresholds from
        the model's calibrated optimal_threshold stored in metrics.json.
        """
        logger.info("Initializing RiskPredictor...")
        
        model_dir = os.path.dirname(MODEL_PATH)
        preproc_path = os.path.join(model_dir, "preprocessor.joblib")
        
        # Load artifacts
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model artifact not found at {MODEL_PATH}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        if not os.path.exists(preproc_path):
            error_msg = f"Preprocessor artifact not found at {preproc_path}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully.")
            
            self.preprocessor = joblib.load(preproc_path)
            logger.info("Preprocessor loaded successfully.")
            
            # Load TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized successfully.")

            # Derive band thresholds from the calibrated optimal_threshold in metrics.json.
            # The optimal_threshold represents the score at which the model best separates
            # defaulters from repaid loans. We use it to anchor the Medium/High boundary,
            # and place the Low/Medium boundary at half that value.
            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                opt_thresh = float(metrics.get("optimal_threshold", 0.5))
            else:
                opt_thresh = 0.5
            # Low  : score < opt_thresh / 2
            # Medium: opt_thresh / 2  <= score < opt_thresh
            # High : score >= opt_thresh
            self.band_low_max = round(opt_thresh / 2, 4)      # e.g. 0.3408 for thresh=0.6815
            self.band_med_max = round(opt_thresh, 4)           # e.g. 0.6815
            logger.info(
                f"Band thresholds — Low: <{self.band_low_max}, "
                f"Medium: {self.band_low_max}–{self.band_med_max}, High: >={self.band_med_max}"
            )
            
        except Exception as e:
            logger.error(f"Error loading predictor artifacts: {str(e)}")
            raise e

    def predict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the default probability and assign a risk band for a single applicant.
        Risk bands use thresholds derived from the model's calibrated optimal_threshold:
          - Low    : probability < optimal_threshold / 2
          - Medium : probability in [optimal_threshold / 2, optimal_threshold)
          - High   : probability >= optimal_threshold

        Args:
            input_dict (Dict[str, Any]): Dictionary of applicant feature values.

        Returns:
            Dict[str, Any]: Probability, risk score, risk band, and risk color.
        """
        try:
            df_row = pd.DataFrame([input_dict])
            X_row = self.preprocessor.transform(df_row)
            
            # Predict probability
            prob = float(self.model.predict_proba(X_row)[0, 1])
            
            # Determine band and color using calibrated thresholds
            if prob < self.band_low_max:
                band = "Low"
                color = "green"
            elif prob < self.band_med_max:
                band = "Medium"
                color = "orange"
            else:
                band = "High"
                color = "red"
                
            return {
                "raw_probability": prob,
                "risk_score": round(prob, 4),
                "risk_band": band,
                "risk_color": color
            }
        except Exception as e:
            logger.error(f"Error predicting risk score: {str(e)}")
            raise e

    def get_shap_values(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SHAP attribution values for a single applicant row.

        Args:
            input_dict (Dict[str, Any]): Dictionary of applicant feature values.

        Returns:
            Dict[str, Any]: SHAP values, feature names, base value, and top 5 features list.
        """
        try:
            df_row = pd.DataFrame([input_dict])
            X_row = self.preprocessor.transform(df_row)
            
            # Compute SHAP values
            shap_vals = self.explainer.shap_values(X_row)
            
            # Resolve SHAP array dimensions depending on SHAP API formatting
            if isinstance(shap_vals, list):
                shap_vals_single = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                shap_vals_single = shap_vals[0, :, 1]
            else:
                shap_vals_single = shap_vals[0]
                
            # Base value (expected value)
            base_val = self.explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_value = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
            else:
                base_value = float(base_val)
                
            feature_names = X_row.columns.tolist()
            
            # Build list of feature contributions
            contributions = []
            for feat, val in zip(feature_names, shap_vals_single):
                direction = "increases_risk" if val > 0 else "decreases_risk"
                contributions.append({
                    "feature": feat,
                    "shap_value": float(val),
                    "direction": direction
                })
                
            # Top 5 by absolute contribution
            top_5 = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
            
            return {
                "shap_values": [float(v) for v in shap_vals_single],
                "feature_names": feature_names,
                "base_value": base_value,
                "top_5_features": top_5
            }
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            raise e

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction function. Adds RISK_SCORE and RISK_BAND columns to the input dataframe.
        Risk bands are assigned using percentile-based thresholds on the predicted score
        distribution so that the population is always meaningfully segmented:
          - Low    : bottom 70% of predicted scores
          - Medium : 70th–90th percentile
          - High   : top 10% (above 90th percentile)

        Args:
            df (pd.DataFrame): Batch DataFrame of applicants.

        Returns:
            pd.DataFrame: Modified DataFrame with RISK_SCORE and RISK_BAND columns.
        """
        logger.info("Executing batch risk prediction...")
        try:
            df_out = df.copy()
            X = self.preprocessor.transform(df_out)
            
            # Predict probabilities
            probs = self.model.predict_proba(X)[:, 1]
            df_out["RISK_SCORE"] = np.round(probs, 4)

            # Use percentile-based thresholds so the distribution is always segmented
            # regardless of how the model is calibrated.
            p70 = float(np.percentile(probs, 70))   # Low / Medium boundary
            p90 = float(np.percentile(probs, 90))   # Medium / High boundary
            logger.info(f"Batch band thresholds — Low:<{p70:.4f}, Medium:{p70:.4f}–{p90:.4f}, High:>={p90:.4f}")

            df_out["RISK_BAND"] = df_out["RISK_SCORE"].apply(
                lambda x: "Low" if x < p70 else ("Medium" if x < p90 else "High")
            )
            return df_out
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            raise e
