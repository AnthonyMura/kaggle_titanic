import shap
import pandas as pd
import numpy as np
from typing import Dict, Any


def extract_shap_importance(
    shap_values: shap.Explanation
) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": mean_abs_shap
    })
    return importance_df.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


def excecute_shap(
        model_data: Dict[str, Any],
        features: pd.DataFrame,
        target: pd.Series
        ):
    pipeline = model_data["best_pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    f_preprocessd = preprocessor.transform(features)
    transformed_names = preprocessor.get_feature_names_out()

    df = pd.DataFrame(
        f_preprocessd,
        columns=transformed_names,
        index=features.index
    )

    explainer = shap.Explainer(model=classifier, masker=df)
    shap_values = explainer(df)

    shap.summary_plot(
        shap_values,
        max_display=30,
        show=True
    )

    return shap_values
