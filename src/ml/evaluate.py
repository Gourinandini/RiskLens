"""
Model Evaluation Module.

Defines the ModelEvaluator class that computes performance metrics
(ROC-AUC, PR-AUC, F1, optimal threshold, confusion matrix) and extracts
feature importances from the trained LightGBM model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    ModelEvaluator handles computing and formatting of validation metrics
    for the trained credit risk classifier.
    """

    def evaluate(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model predictions against validation labels.

        Args:
            model (Any): The trained classifier model.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.

        Returns:
            Dict[str, Any]: Dictionary containing ROC-AUC, PR-AUC, F1 Score,
                classification report, confusion matrix, and optimal threshold.
        """
        logger.info("Evaluating classifier performance on validation set...")
        
        try:
            # Predict probabilities
            probs = model.predict_proba(X_val)[:, 1]
            
            # ROC AUC
            roc_auc = float(roc_auc_score(y_val, probs))
            
            # PR AUC (Average Precision)
            pr_auc = float(average_precision_score(y_val, probs))
            
            # Find optimal threshold to maximize F1
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
            
            best_idx = np.argmax(f1_scores)
            optimal_threshold = float(thresholds[best_idx])
            best_f1 = float(f1_scores[best_idx])
            
            logger.info(f"Optimal threshold found at {optimal_threshold:.4f} with F1-Score {best_f1:.4f}")
            
            # Compute classification report and confusion matrix at optimal threshold
            preds = (probs >= optimal_threshold).astype(int)
            class_report = classification_report(y_val, preds)
            conf_mat = confusion_matrix(y_val, preds).tolist()
            
            metrics = {
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "f1_score": round(best_f1, 4),
                "classification_report": class_report,
                "confusion_matrix": conf_mat,
                "optimal_threshold": round(optimal_threshold, 4)
            }
            
            logger.info("Evaluation metrics calculated successfully.")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise e

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract the top 20 most important features from the model.

        Args:
            model (Any): The trained LightGBM classifier.
            feature_names (List[str]): List of column names corresponding to training features.

        Returns:
            pd.DataFrame: DataFrame with columns 'feature' and 'importance'.
        """
        logger.info("Calculating model feature importances...")
        try:
            importances = model.feature_importances_
            
            # Match feature names and values
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
            
            # Sort and take top 20
            top_importance = (
                importance_df.sort_values(by="importance", ascending=False)
                .head(20)
                .reset_index(drop=True)
            )
            return top_importance
            
        except Exception as e:
            logger.error(f"Error retrieving feature importances: {str(e)}")
            raise e
