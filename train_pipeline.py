"""
Unified Training Pipeline Entrypoint.

Runs the complete model training and rule extraction pipeline:
  1. Load raw data from CSV
  2. Preprocess features (imputation, encoding, engineering)
  3. Train LightGBM classifier with early stopping
  4. Evaluate and save metrics
  5. Extract interpretable decision rules via surrogate tree
  6. Save all artifacts to ./models/

Usage:
    python train_pipeline.py
    python train_pipeline.py --nrows 50000  # smaller sample for testing
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Ensure PYTHONPATH includes project root
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH, DATA_PATH
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.ml.train import ModelTrainer
from src.ml.rules import RulesExtractor

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Credit Risk Intelligence Platform — Training Pipeline"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to load from CSV (default: all rows, ~307k)"
    )
    parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Skip surrogate rule extraction step"
    )
    return parser.parse_args()


def run_training_pipeline(nrows: int = None) -> None:
    """Run model training pipeline."""
    logger.info("=" * 60)
    logger.info("  RiskLens Credit Risk Platform — Training Pipeline")
    logger.info("=" * 60)

    # Patch nrows into the ModelTrainer if provided
    if nrows is not None:
        logger.info(f"Training on a subset of {nrows:,} rows.")
        # Monkey-patch the loader call
        original_train = None

    trainer = ModelTrainer()
    if nrows is not None:
        # Override internal nrows
        trainer.loader.load_application_train = lambda nrows=nrows: \
            pd.read_csv(os.path.join(DATA_PATH, "application_train.csv"), nrows=nrows)

    trainer.train()
    logger.info("✅ Model training and evaluation complete.")


def run_rules_extraction() -> None:
    """Run surrogate rule extraction on the trained model."""
    logger.info("\n" + "=" * 60)
    logger.info("  Extracting Interpretable Decision Rules")
    logger.info("=" * 60)

    model_dir = os.path.dirname(MODEL_PATH)
    preproc_path = os.path.join(model_dir, "preprocessor.joblib")

    if not os.path.exists(MODEL_PATH):
        logger.error("Model artifact not found. Run training first.")
        return

    if not os.path.exists(preproc_path):
        logger.error("Preprocessor artifact not found. Run training first.")
        return

    import joblib

    logger.info("Loading model and preprocessor artifacts...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(preproc_path)

    # Load sample data for rule extraction (use 30k rows for speed)
    csv_path = os.path.join(DATA_PATH, "application_train.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Training CSV not found at {csv_path}")
        return

    logger.info("Loading sample data for rule extraction (30,000 rows)...")
    df = pd.read_csv(csv_path, nrows=30000)

    y_true = df["TARGET"].copy()
    df_features = df.drop(columns=["TARGET"])

    logger.info("Preprocessing features...")
    X = preprocessor.transform(df_features)

    logger.info("Generating LightGBM predictions...")
    y_pred_proba = model.predict_proba(X)[:, 1]

    logger.info("Fitting surrogate decision tree and extracting rules...")
    extractor = RulesExtractor(max_depth=4, min_samples_leaf=200)
    result = extractor.extract(X, y_pred_proba, y_true)

    # Save rules to JSON
    rules_path = os.path.join(model_dir, "rules.json")
    extractor.save(rules_path)

    # Print summary
    logger.info("\n" + "─" * 50)
    logger.info(f"  Surrogate Tree Fidelity: {result['surrogate_fidelity']:.4f}")
    logger.info(f"  Tree Depth: {result['tree_depth']}")
    logger.info(f"  Leaf Nodes (Rules): {result['n_rules']}")
    logger.info(f"  Rules saved to: {rules_path}")
    logger.info("\nRisk Band Portfolio Summary:")
    for band, stats in result["band_statistics"].items():
        default_str = f"  | Actual Default Rate: {stats['actual_default_rate']:.1f}%" \
            if stats.get("actual_default_rate") is not None else ""
        logger.info(
            f"  {band:8s}: {stats['count']:6,} applicants ({stats['pct_of_portfolio']:.1f}%)"
            f"  | Avg Prob: {stats['avg_predicted_probability']:.1f}%{default_str}"
        )
    logger.info("─" * 50)
    logger.info("✅ Rule extraction complete.")


def main() -> None:
    args = parse_args()

    try:
        run_training_pipeline(nrows=args.nrows)

        if not args.skip_rules:
            run_rules_extraction()

        logger.info("\n" + "=" * 60)
        logger.info("  ✅ All pipeline steps completed successfully!")
        logger.info("  📂 Artifacts saved to: ./models/")
        logger.info("  🚀 Run: streamlit run app.py")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
