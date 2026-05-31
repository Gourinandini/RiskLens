"""
Standalone script to extract decision rules from the already-trained LightGBM model.
Run: python extract_rules.py
"""
import os
import sys
sys.path.insert(0, '.')

import joblib
import numpy as np
import pandas as pd

from src.utils.config import MODEL_PATH, DATA_PATH
from src.ml.rules import RulesExtractor

def main():
    print("Loading model artifacts...")
    model_dir = os.path.dirname(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    print("  -> Artifacts loaded OK")

    print("Loading sample data (30k rows)...")
    csv_path = os.path.join(DATA_PATH, "application_train.csv")
    df = pd.read_csv(csv_path, nrows=30000)
    y_true = df["TARGET"].copy()
    X = preprocessor.transform(df.drop(columns=["TARGET"]))
    print(f"  -> Features shape: {X.shape}")

    print("Generating LightGBM predictions...")
    y_prob = model.predict_proba(X)[:, 1]
    print(f"  -> Prob range: {y_prob.min():.4f} – {y_prob.max():.4f}")

    print("Fitting surrogate decision tree and extracting rules...")
    extractor = RulesExtractor(max_depth=4, min_samples_leaf=200)
    result = extractor.extract(X, y_prob, y_true)

    rules_path = os.path.join(model_dir, "rules.json")
    extractor.save(rules_path)

    print("\n" + "="*50)
    fidelity = result["surrogate_fidelity"]
    n_rules = result["n_rules"]
    depth = result["tree_depth"]
    print(f"Surrogate Fidelity: {fidelity:.4f}")
    print(f"Rules extracted:    {n_rules}")
    print(f"Tree depth:         {depth}")
    print("\nRisk Band Portfolio Summary:")
    for band, stats in result["band_statistics"].items():
        cnt = stats["count"]
        pct = stats["pct_of_portfolio"]
        avg_p = stats["avg_predicted_probability"]
        dr = stats.get("actual_default_rate")
        dr_str = f" | Actual Default Rate: {dr:.1f}%" if dr is not None else ""
        print(f"  {band:8s}: {cnt:6,} ({pct:.1f}%) | Avg Prob: {avg_p:.1f}%{dr_str}")
    print("="*50)
    print(f"Rules saved to: {rules_path}")
    print("SUCCESS")

if __name__ == "__main__":
    main()
