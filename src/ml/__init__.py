"""
Machine Learning package.

Exposes ModelTrainer, ModelEvaluator, RiskPredictor, and RulesExtractor classes.
"""

def __getattr__(name: str):
    if name == "ModelTrainer":
        from src.ml.train import ModelTrainer
        return ModelTrainer
    elif name == "ModelEvaluator":
        from src.ml.evaluate import ModelEvaluator
        return ModelEvaluator
    elif name == "RiskPredictor":
        from src.ml.predict import RiskPredictor
        return RiskPredictor
    elif name == "RulesExtractor":
        from src.ml.rules import RulesExtractor
        return RulesExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

