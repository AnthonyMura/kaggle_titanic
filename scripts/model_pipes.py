from typing import List
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_pipes import build_preprocessing_pipeline


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    estimator: BaseEstimator
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator)
        ]
    )


def get_random_forest_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
    return build_model_pipeline(preprocessor, estimator)


def get_gradient_boosting_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    estimator = GradientBoostingClassifier(random_state=42)
    return build_model_pipeline(preprocessor, estimator)