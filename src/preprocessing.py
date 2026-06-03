from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

BASE_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_NUMERIC_FEATURES = [
    "Temp_diff [K]",
    "Power [W]",
    "Torque_Wear",
]

CATEGORICAL_FEATURES = [
    "Type",
]

def build_preprocessor():
    """
    Build preprocessing pipeline for numeric and categorical features.
    """

    numeric_features = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor