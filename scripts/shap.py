import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

import shap


def extract_shap_importance(
        shap_values: shap.Explanation
        ) -> pd.DataFrame:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(
            {
                "feature"      : shap_values.feature_names,
                "mean_abs_shap": mean_abs_shap
                }
            )
    return importance_df.sort_values(
            by="mean_abs_shap",
            ascending=False
            ).reset_index(drop=True)


def _get_prediction_function(classifier: Any) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function
    return classifier.predict


def excecute_shap(
        model_data: Dict[str, Any],
        features: pd.DataFrame,
        target: pd.Series,
        background_sample_size: int = 100,
        explanation_sample_size: Optional[int] = None
        ) -> shap.Explanation:
    pipeline = model_data["best_pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    f_preprocessed = preprocessor.transform(features)
    transformed_names = preprocessor.get_feature_names_out().tolist()

    df_background = pd.DataFrame(f_preprocessed, columns=transformed_names)

    if background_sample_size < df_background.shape[0]:
        df_background = df_background.sample(
                n=background_sample_size,
                random_state=42
                )

    df_explanation = df_background.copy()
    if explanation_sample_size is not None and explanation_sample_size < df_background.shape[0]:
        df_explanation = df_background.sample(
                n=explanation_sample_size,
                random_state=42
                )

    predict_fn = _get_prediction_function(classifier)

    def model_wrapper(x: np.ndarray) -> np.ndarray:
        return predict_fn(x)

    masker = shap.maskers.Independent(
            data=df_background.to_numpy(),
            max_samples=df_background.shape[0]
            )

    explainer = shap.Explainer(
            model=model_wrapper,
            masker=masker,
            feature_names=transformed_names
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        shap_values = explainer(df_explanation.to_numpy())

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        shap.summary_plot(
                shap_values,
                feature_names=transformed_names,
                max_display=30,
                show=True
                )

    return shap_values