from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder


def build_preprocessor(numerical_features, categorical_features):
    """
    Build preprocessing pipeline for numerical and categorical features.

    Numerical features:
    - StandardScaler

    Categorical features:
    - OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot',OneHotEncoder())
    ])
    
    preprocessor=ColumnTransformer(transformers=[
        ("num",numeric_transformer,numerical_features),
        ("cat",categorical_transformer,categorical_features)
    ])
    
    return preprocessor