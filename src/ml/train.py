"""
Model Training Module.

Orchestrates loading, preprocessing, model configuration, fitting with early
stopping, and model/preprocessor artifact saving.
"""

import os
import joblib
import lightgbm as lgb
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger(__name__)

class ModelTrainer:
    """
    ModelTrainer trains a LightGBM classification model on preprocessed credit risk data
    and saves the trained model and fitting pipeline artifacts.
    """

    def __init__(self) -> None:
        """Initialize ModelTrainer with loader and preprocessor instances."""
        self.loader: DataLoader = DataLoader()
        self.preprocessor: DataPreprocessor = DataPreprocessor()
        self.model: lgb.LGBMClassifier | None = None

    def train(self) -> None:
        """
        Run the training pipeline: load data, preprocess, compute weights,
        fit LightGBM with early stopping, and save artifacts.
        """
        logger.info("Starting model training pipeline...")
        
        # Load dataset (limit nrows to avoid out-of-memory errors)
        df = self.loader.load_application_train(nrows=100000)
        
        # Preprocess features
        X_train, X_val, y_train, y_val = self.preprocessor.fit_transform(df)
        
        # Calculate class weight ratio for imbalance adjustment
        num_neg = int((y_train == 0).sum())
        num_pos = int((y_train == 1).sum())
        scale_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
        
        logger.info(f"Class breakdown: TARGET 0: {num_neg}, TARGET 1: {num_pos}")
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Setup LGBMClassifier
        # NOTE: early_stopping_rounds must NOT be set in the constructor —
        # it is silently ignored there. It must go into fit() via callbacks.
        # NOTE: metric='auc' is set explicitly to DISABLE the default
        # binary_logloss metric. With high scale_pos_weight, binary_logloss
        # degrades while AUC improves — causing early stopping to fire at
        # iteration 1 if both metrics are tracked.
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=7,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            metric='auc',           # <-- only track AUC; disables binary_logloss
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        
        # Fit with validation set and proper early stopping via callbacks
        logger.info("Fitting LightGBM classifier with early stopping on validation split...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=callbacks,
        )
        logger.info(f"Model fitted. Best iteration: {self.model.best_iteration_}")


        
        # Create output directory structure if necessary
        model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and preprocessor joblib packages
        logger.info(f"Saving model to {MODEL_PATH}...")
        joblib.dump(self.model, MODEL_PATH)
        
        preproc_path = os.path.join(model_dir, "preprocessor.joblib")
        logger.info(f"Saving preprocessor to {preproc_path}...")
        joblib.dump(self.preprocessor, preproc_path)
        
        # Evaluate model and save metrics to file
        try:
            from src.ml.evaluate import ModelEvaluator
            import json
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(self.model, X_val, y_val)
            metrics_path = os.path.join(model_dir, "metrics.json")
            logger.info(f"Saving validation metrics to {metrics_path}...")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            
            # Save feature importances to CSV
            importance_df = evaluator.get_feature_importance(self.model, self.preprocessor.get_feature_names())
            importance_path = os.path.join(model_dir, "feature_importances.csv")
            logger.info(f"Saving feature importances to {importance_path}...")
            importance_df.to_csv(importance_path, index=False)
        except Exception as e:
            logger.error(f"Error during validation evaluation saving: {str(e)}")
            
        logger.info("Training process complete. Artifacts saved successfully.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
