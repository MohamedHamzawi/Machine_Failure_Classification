from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor


def build_model_pipeline(model):
    """
    Build full ML pipeline:
    1. Feature engineering
    2. Preprocessing
    3. Model training
    """

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    add_engineered_features,
                    validate=False
                )
            ),
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    return pipeline