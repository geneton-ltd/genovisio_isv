import enum
import os
import sys
from dataclasses import dataclass
from typing import Any

import annotation
import joblib
import pandas as pd
import shap
import xgboost as xgb

from isv.src import constants, core, features


class ACMGClassification(enum.StrEnum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VOUS = "VOUS"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


def get_shap_values(loaded_model: Any, input_df: pd.DataFrame) -> dict[str, float]:
    explainer_cnvs = shap.Explainer(loaded_model)
    shap_values = explainer_cnvs(input_df).values[0]

    return {attr: float(shap_val) for shap_val, attr in zip(shap_values, loaded_model.feature_names)}


def get_shap_scores(shap_values: dict[str, float]) -> dict[str, float]:
    shap_scores: dict[str, float] = {}
    for attribute in shap_values.keys():
        shap_scores[attribute] = shap_values[attribute] * 2 - 1
    return shap_scores


def get_isv_score(prediction: float) -> float:
    isv_score = (prediction * 2) - 1
    return isv_score


def get_acmg_classification(isv_score: float) -> ACMGClassification:
    if isv_score >= 0.99:
        return ACMGClassification.PATHOGENIC
    elif isv_score >= 0.9:
        return ACMGClassification.LIKELY_PATHOGENIC
    elif isv_score <= -0.99:
        return ACMGClassification.BENIGN
    elif isv_score <= -0.9:
        return ACMGClassification.LIKELY_BENIGN
    else:
        return ACMGClassification.VOUS


@dataclass
class Prediction:
    isv_prediction: float
    isv_score: float
    isv_classification: ACMGClassification
    isv_shap_values: dict[str, float]
    isv_shap_scores: dict[str, float]
    isv_features: features.ISVFeatures


def format_model_path(cnvtype: annotation.enums.CNVType) -> str:
    models_name = f"isv2_{cnvtype}.json"
    return os.path.join(core.MODELS_DIR, models_name)


def get_attributes(cnvtype: annotation.enums.CNVType) -> list[str]:
    if cnvtype == annotation.enums.CNVType.LOSS:
        return constants.LOSS_ATTRIBUTES
    elif cnvtype == annotation.enums.CNVType.GAIN:
        return constants.GAIN_ATTRIBUTES
    else:
        raise ValueError("Invalid CNV type")


def prepare_dataframe(annotated_cnv: features.ISVFeatures, cnv_type: annotation.enums.CNVType) -> pd.DataFrame:
    attributes = get_attributes(cnv_type)

    cnv_dct = annotated_cnv.as_dict_of_attributes()
    annotated_cnv_floats = {col: float(cnv_dct[col]) for col in cnv_dct if col in attributes}
    df = pd.DataFrame.from_dict(annotated_cnv_floats, orient="index").T
    return df[attributes]


def predict(annotated_cnv: features.ISVFeatures, cnv_type: annotation.enums.CNVType) -> Prediction:
    model_path = format_model_path(cnv_type)
    print(f"Loading model from {model_path=}", file=sys.stderr)
    loaded_model = joblib.load(model_path)

    input_df = prepare_dataframe(annotated_cnv, cnv_type)

    dmat_cnvs = xgb.DMatrix(input_df)
    predicted_probability = float(loaded_model.predict(dmat_cnvs)[0])
    print(f"{predicted_probability=}", file=sys.stderr)
    isv_score = get_isv_score(predicted_probability)

    shap_values = get_shap_values(loaded_model, input_df)

    return Prediction(
        isv_prediction=predicted_probability,
        isv_score=isv_score,
        isv_classification=get_acmg_classification(isv_score),
        isv_shap_values=shap_values,
        isv_shap_scores=get_shap_scores(shap_values),
        isv_features=annotated_cnv,
    )
