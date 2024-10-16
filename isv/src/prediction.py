import enum
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import annotation
import joblib
import pandas as pd
import shap
import xgboost as xgb

from isv.src import constants, isv_attributes


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

    def store_as_json(self, path: str) -> None:
        path = os.path.abspath(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def format_model_path(cnvtype: annotation.enums.CNVType) -> str:
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    models_name = f"isv2_{cnvtype}.json"
    return os.path.join(models_dir, models_name)


def get_attributes(cnvtype: annotation.enums.CNVType) -> list[str]:
    if cnvtype == annotation.enums.CNVType.LOSS:
        return constants.LOSS_ATTRIBUTES
    elif cnvtype == annotation.enums.CNVType.GAIN:
        return constants.GAIN_ATTRIBUTES
    else:
        raise ValueError("Invalid CNV type")


def prepare_dataframe(annotated_cnv: isv_attributes.ISVAnnotValues, cnv_type: annotation.enums.CNVType) -> pd.DataFrame:
    attributes = get_attributes(cnv_type)

    cnv_dct = annotated_cnv.as_dict_of_attributes()
    annotated_cnv_floats = {col: float(cnv_dct[col]) for col in cnv_dct if col in attributes}
    df = pd.DataFrame.from_dict(annotated_cnv_floats, orient="index").T
    return df[attributes]


def predict(annotated_cnv: isv_attributes.ISVAnnotValues, cnv_type: annotation.enums.CNVType) -> Prediction:
    model_path = format_model_path(cnv_type)
    print(f"Loading model from {model_path=}", file=sys.stderr)
    loaded_model = joblib.load(model_path)

    input_df = prepare_dataframe(annotated_cnv, cnv_type)

    dmat_cnvs = xgb.DMatrix(input_df)
    prediction_cnvs = loaded_model.predict(dmat_cnvs)  # TODO rework later/simplify
    print(f"{prediction_cnvs=}", file=sys.stderr)
    predictions_df = pd.DataFrame(prediction_cnvs, columns=["isv2_predictions"])
    print(f"{predictions_df=}", file=sys.stderr)

    # isv score from prediction
    predictions_df["isv2_score"] = predictions_df["isv2_predictions"].apply(get_isv_score)
    predictions_df["isv2_classification"] = predictions_df["isv2_score"].apply(get_acmg_classification)

    shap_values = get_shap_values(loaded_model, input_df)

    return Prediction(
        isv_prediction=predictions_df["isv2_predictions"].iloc[0].item(),
        isv_score=predictions_df["isv2_score"].iloc[0].item(),
        isv_classification=predictions_df["isv2_classification"].iloc[0],
        isv_shap_values=shap_values,
        isv_shap_scores=get_shap_scores(shap_values),
    )
