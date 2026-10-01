"""Preprocessing shared by training, evaluation, and inference."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .feature_engineering import CATEGORICAL_COLUMNS


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    categorical = [name for name in CATEGORICAL_COLUMNS if name in feature_names]
    numeric = [name for name in feature_names if name not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
