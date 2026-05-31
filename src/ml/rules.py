"""
Decision Rules Extraction Module.

Fits a shallow DecisionTreeClassifier as a surrogate model on the LightGBM
predicted probabilities and extracts human-readable if/then rules for business
stakeholders. Also provides risk band statistics and rule confidence tables.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RulesExtractor:
    """
    RulesExtractor trains a shallow surrogate DecisionTreeClassifier on the
    LightGBM model predictions and extracts interpretable decision rules.

    The surrogate tree learns to mimic the LightGBM model's classification
    boundary using human-readable feature thresholds, producing business-grade
    if/then decision rules with support and confidence metrics.
    """

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 200) -> None:
        """
        Initialize RulesExtractor.

        Args:
            max_depth (int): Maximum depth of the surrogate decision tree.
            min_samples_leaf (int): Minimum samples required at each leaf node.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree: DecisionTreeClassifier | None = None
        self.feature_names: List[str] = []
        self.rules_list: List[Dict[str, Any]] = []
        self.band_stats: Dict[str, Any] = {}

    def extract(self, X: pd.DataFrame, y_pred_proba: np.ndarray, y_true: pd.Series = None) -> Dict[str, Any]:
        """
        Fit surrogate decision tree and extract interpretable rules.

        Args:
            X (pd.DataFrame): Feature matrix (preprocessed).
            y_pred_proba (np.ndarray): Predicted probabilities from LightGBM model.
            y_true (pd.Series, optional): Ground truth TARGET labels for statistics.

        Returns:
            Dict[str, Any]: Extracted rules, band statistics, and metadata.
        """
        logger.info("Starting surrogate decision tree rule extraction...")

        self.feature_names = X.columns.tolist()

        # Binarize probabilities using 0.5 for surrogate training
        y_surrogate = (y_pred_proba >= 0.5).astype(int)

        # Fit a shallow decision tree as surrogate
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion="gini",
            random_state=42
        )
        self.tree.fit(X, y_surrogate)

        surrogate_accuracy = self.tree.score(X, y_surrogate)
        logger.info(f"Surrogate tree fitted. Fidelity to LightGBM: {surrogate_accuracy:.4f}")

        # Extract text rules
        raw_rules_text = export_text(
            self.tree,
            feature_names=self.feature_names,
            show_weights=True
        )

        # Parse rules into structured format
        self.rules_list = self._parse_rules_from_tree(X, y_pred_proba, y_surrogate)

        # Compute risk band statistics
        self.band_stats = self._compute_band_stats(y_pred_proba, y_true)

        result = {
            "raw_rules_text": raw_rules_text,
            "structured_rules": self.rules_list,
            "band_statistics": self.band_stats,
            "surrogate_fidelity": round(surrogate_accuracy, 4),
            "tree_depth": self.tree.get_depth(),
            "n_leaves": self.tree.get_n_leaves(),
            "n_rules": len(self.rules_list),
        }

        logger.info(f"Extracted {len(self.rules_list)} interpretable decision rules.")
        return result

    def _parse_rules_from_tree(
        self,
        X: pd.DataFrame,
        y_pred_proba: np.ndarray,
        y_surrogate: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Walk the decision tree and extract human-readable rules for each leaf.

        Args:
            X (pd.DataFrame): Feature matrix.
            y_pred_proba (np.ndarray): Model predicted probabilities.
            y_surrogate (np.ndarray): Binarized surrogate labels.

        Returns:
            List[Dict[str, Any]]: Structured list of rules with conditions and statistics.
        """
        tree_ = self.tree.tree_
        feature_names = self.feature_names
        rules = []

        # Load optimal_threshold to assign risk bands dynamically
        model_dir = os.path.dirname(MODEL_PATH)
        metrics_path = os.path.join(model_dir, "metrics.json")
        opt_thresh = 0.5
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                opt_thresh = float(metrics.get("optimal_threshold", 0.5))
            except Exception as e:
                logger.warning(f"Could not load optimal_threshold from metrics.json: {e}")

        low_max = opt_thresh / 2
        med_max = opt_thresh

        def recurse(node: int, conditions: List[str]) -> None:
            """Recursive walk of decision tree nodes."""
            if tree_.feature[node] == -2:
                # Leaf node — extract rule
                n_samples = int(tree_.n_node_samples[node])
                # tree_.value stores weighted counts; round to nearest integer
                # and ensure they sum exactly to n_samples
                raw_vals = tree_.value[node][0]  # [count_class0, count_class1]
                total_raw = raw_vals.sum()
                if total_raw > 0:
                    n_repaid = int(round(raw_vals[0] / total_raw * n_samples))
                    n_default = n_samples - n_repaid
                else:
                    n_repaid = n_samples
                    n_default = 0

                if n_samples == 0:
                    return

                predicted_class = 1 if raw_vals[1] > raw_vals[0] else 0
                confidence = round((max(n_default, n_repaid) / n_samples) * 100, 1)
                support = round((n_samples / len(X)) * 100, 1)

                # Get node leaf mask for average probability
                node_ids = self.tree.apply(X)  # pass DataFrame to preserve feature names
                mask = (node_ids == node)
                avg_prob = float(y_pred_proba[mask].mean()) if mask.sum() > 0 else 0.0

                # Assign risk band and color based on calibrated thresholds
                if avg_prob < low_max:
                    risk_band = "Low"
                    risk_color = "#00E676"
                elif avg_prob < med_max:
                    risk_band = "Medium"
                    risk_color = "#FFB300"
                else:
                    risk_band = "High"
                    risk_color = "#FF1744"

                # Format human-readable rule
                rule_text = " AND ".join(conditions) if conditions else "All applicants"
                
                rules.append({
                    "rule_id": len(rules) + 1,
                    "conditions": conditions.copy(),
                    "rule_text": rule_text,
                    "predicted_class": predicted_class,
                    "risk_band": risk_band,
                    "risk_color": risk_color,
                    "n_samples": n_samples,
                    "n_defaulted": n_default,
                    "n_repaid": n_repaid,
                    "confidence": confidence,
                    "support_pct": support,
                    "avg_default_probability": round(avg_prob * 100, 2),
                    "description": _build_description(conditions, risk_band, confidence, support, avg_prob),
                })
            else:
                # Internal node — recurse left (≤) and right (>)
                feat = feature_names[tree_.feature[node]]
                thresh = round(tree_.threshold[node], 4)

                # Make feature names more human-readable
                feat_display = _humanize_feature(feat)

                left_cond = f"{feat_display} ≤ {thresh}"
                right_cond = f"{feat_display} > {thresh}"

                recurse(tree_.children_left[node], conditions + [left_cond])
                recurse(tree_.children_right[node], conditions + [right_cond])

        recurse(0, [])

        # Sort rules by support (most common first)
        rules.sort(key=lambda r: r["support_pct"], reverse=True)
        return rules

    def _compute_band_stats(
        self,
        y_pred_proba: np.ndarray,
        y_true: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Compute statistical summary for each risk band.

        Args:
            y_pred_proba (np.ndarray): Predicted probabilities from LightGBM.
            y_true (pd.Series, optional): Ground truth labels.

        Returns:
            Dict[str, Any]: Statistics per risk band.
        """
        # Use calibrated percentile-based thresholds
        p40 = float(np.percentile(y_pred_proba, 40))
        p75 = float(np.percentile(y_pred_proba, 75))

        bands = {
            "Low": y_pred_proba < p40,
            "Medium": (y_pred_proba >= p40) & (y_pred_proba < p75),
            "High": y_pred_proba >= p75,
        }

        stats = {}
        total = len(y_pred_proba)

        for band_name, mask in bands.items():
            n = int(mask.sum())
            avg_prob = float(y_pred_proba[mask].mean()) if n > 0 else 0.0
            
            actual_default_rate = None
            if y_true is not None:
                y_true_arr = y_true.values
                actual_default_rate = float(y_true_arr[mask].mean()) if n > 0 else 0.0

            stats[band_name] = {
                "count": n,
                "pct_of_portfolio": round((n / total) * 100, 1) if total > 0 else 0.0,
                "avg_predicted_probability": round(avg_prob * 100, 2),
                "actual_default_rate": round(actual_default_rate * 100, 2) if actual_default_rate is not None else None,
            }

        return stats

    def get_rule_table(self) -> pd.DataFrame:
        """
        Return rules as a formatted DataFrame for display.

        Returns:
            pd.DataFrame: Rules table with key metrics.
        """
        if not self.rules_list:
            return pd.DataFrame()

        rows = []
        for r in self.rules_list:
            rows.append({
                "Rule #": r["rule_id"],
                "Risk Band": r["risk_band"],
                "Support (%)": r["support_pct"],
                "Confidence (%)": r["confidence"],
                "Avg Prob (%)": r["avg_default_probability"],
                "# Applicants": r["n_samples"],
                "# Defaulted": r["n_defaulted"],
                "Rule Summary": r["rule_text"][:80] + "..." if len(r["rule_text"]) > 80 else r["rule_text"],
            })

        return pd.DataFrame(rows)

    def save(self, output_path: str) -> None:
        """
        Save extracted rules to a JSON file.

        Args:
            output_path (str): Path to save the rules JSON file.
        """
        if not self.rules_list:
            logger.warning("No rules to save. Run extract() first.")
            return

        payload = {
            "structured_rules": self.rules_list,
            "band_statistics": self.band_stats,
            "n_rules": len(self.rules_list),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Rules saved to {output_path}")

    @staticmethod
    def load(rules_path: str) -> Dict[str, Any]:
        """
        Load previously extracted rules from a JSON file.

        Args:
            rules_path (str): Path to rules JSON file.

        Returns:
            Dict[str, Any]: Loaded rules data.
        """
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rules file not found at: {rules_path}")

        with open(rules_path, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded {data.get('n_rules', 0)} rules from {rules_path}")
        return data


def _humanize_feature(feature_name: str) -> str:
    """Convert internal feature names to human-readable labels."""
    mapping = {
        "EXT_SOURCE_2": "External Credit Score 2",
        "EXT_SOURCE_3": "External Credit Score 3",
        "EXT_SOURCE_1": "External Credit Score 1",
        "EXT_SOURCE_MEAN": "Avg External Credit Score",
        "EXT_SOURCE_MIN": "Min External Credit Score",
        "AGE_YEARS": "Age (years)",
        "EMPLOYMENT_YEARS": "Employment Duration (years)",
        "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio",
        "ANNUITY_INCOME_RATIO": "Annuity-to-Income Ratio",
        "CREDIT_TERM_MONTHS": "Credit Term (months)",
        "AMT_CREDIT": "Loan Amount",
        "AMT_INCOME_TOTAL": "Annual Income",
        "AMT_ANNUITY": "Monthly Annuity",
        "INCOME_PER_PERSON": "Income Per Person",
        "DOCUMENT_COUNT": "Document Count",
        "REGION_RATING_CLIENT": "Region Risk Rating",
        "REGION_POPULATION_RELATIVE": "Region Population Density",
        "CNT_CHILDREN": "Number of Children",
        "CNT_FAM_MEMBERS": "Family Members Count",
    }
    return mapping.get(feature_name, feature_name.replace("_", " ").title())


def _build_description(
    conditions: List[str],
    risk_band: str,
    confidence: float,
    support: float,
    avg_prob: float,
) -> str:
    """Build a plain English description of a decision rule."""
    cond_str = " and ".join(conditions) if conditions else "all criteria"
    prob_pct = round(avg_prob * 100, 1)

    desc_map = {
        "Low": f"Applicants where {cond_str} are likely to repay. "
               f"This rule covers {support:.1f}% of the portfolio with {confidence:.1f}% confidence. "
               f"Average default probability: {prob_pct:.1f}%.",
        "Medium": f"Applicants where {cond_str} fall into a borderline zone requiring manual review. "
                  f"This rule covers {support:.1f}% of the portfolio with {confidence:.1f}% confidence. "
                  f"Average default probability: {prob_pct:.1f}%.",
        "High": f"Applicants where {cond_str} present elevated default risk. "
                f"This rule covers {support:.1f}% of the portfolio with {confidence:.1f}% confidence. "
                f"Average default probability: {prob_pct:.1f}%.",
    }
    return desc_map.get(risk_band, "")
