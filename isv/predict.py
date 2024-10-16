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

from isv.src import constants


@dataclass
class ISVAnnotValues:
    gencode_genes: int
    protein_coding: int
    pseudogenes: int
    mirna: int
    lncrna: int
    rrna: int
    snrna: int
    morbid_genes: int
    disease_associated_genes: int
    hi_genes: int
    regions_HI: int
    regions_TS: int
    regulatory: int
    regulatory_enhancer: int
    regulatory_silencer: int
    regulatory_transcriptional_cis_regulatory_region: int
    regulatory_promoter: int
    regulatory_DNase_I_hypersensitive_site: int
    regulatory_enhancer_blocking_element: int
    regulatory_TATA_box: int

    def as_dict_of_attributes(self) -> dict[str, int]:
        return asdict(self)


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


def prepare_dataframe(annotated_cnv: ISVAnnotValues, cnv_type: annotation.enums.CNVType) -> pd.DataFrame:
    attributes = get_attributes(cnv_type)

    cnv_dct = annotated_cnv.as_dict_of_attributes()
    annotated_cnv_floats = {col: float(cnv_dct[col]) for col in cnv_dct if col in attributes}
    df = pd.DataFrame.from_dict(annotated_cnv_floats, orient="index").T
    return df[attributes]


def predict(annotated_cnv: ISVAnnotValues, cnv_type: annotation.enums.CNVType) -> Prediction:
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


def get_annotation_attributes(annot: annotation.Annotation) -> ISVAnnotValues:
    gene_type_counts = annot.count_gene_types()
    all_genes = (
        gene_type_counts["protein_coding"]
        + gene_type_counts["pseudogenes"]
        + gene_type_counts["mirna"]
        + gene_type_counts["lncrna"]
        + gene_type_counts["rrna"]
        + gene_type_counts["snrna"]
    )

    regulatory_counts = annot.count_regulatory_types()
    regulatory_sum = (
        regulatory_counts["CTCF_binding_site"]
        + regulatory_counts["enhancer"]
        + regulatory_counts["silencer"]
        + regulatory_counts["transcriptional_cis_regulatory_region"]
        + regulatory_counts["promoter"]
        + regulatory_counts["DNase_I_hypersensitive_site"]
        + regulatory_counts["enhancer_blocking_element"]
        + regulatory_counts["TATA_box"]
    )

    hi_genes = annot.get_haploinsufficient_gene_names(overlap_type=annotation.enums.OverlapType.all)
    hi_regions = annot.get_haploinsufficient_regions(overlap_type=annotation.enums.OverlapType.all)
    ts_regions = annot.get_triplosensitivity_regions(overlap_type=annotation.enums.OverlapType.all)

    annotated_genes = annot.get_annotated_genes()

    return ISVAnnotValues(
        gencode_genes=all_genes,
        protein_coding=gene_type_counts["protein_coding"],
        pseudogenes=gene_type_counts["pseudogenes"],
        mirna=gene_type_counts["mirna"],
        lncrna=gene_type_counts["lncrna"],
        rrna=gene_type_counts["rrna"],
        snrna=gene_type_counts["snrna"],
        morbid_genes=len(annotated_genes["morbid_genes"]),
        disease_associated_genes=len(annotated_genes["associated_with_disease"]),
        hi_genes=len(hi_genes),
        regions_HI=len(hi_regions),
        regions_TS=len(ts_regions),
        regulatory=regulatory_sum,
        regulatory_enhancer=regulatory_counts["enhancer"],
        regulatory_silencer=regulatory_counts["silencer"],
        regulatory_transcriptional_cis_regulatory_region=regulatory_counts["transcriptional_cis_regulatory_region"],
        regulatory_promoter=regulatory_counts["promoter"],
        regulatory_DNase_I_hypersensitive_site=regulatory_counts["DNase_I_hypersensitive_site"],
        regulatory_enhancer_blocking_element=regulatory_counts["enhancer_blocking_element"],
        regulatory_TATA_box=regulatory_counts["TATA_box"],
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict pathogenicity from annotated CNV.")
    parser.add_argument("input", help="Annotated CNV stored as json")
    parser.add_argument("--output", help="Path to store the prediction JSON. Else prints to stdout.", default=None)
    args = parser.parse_args()

    annot = annotation.Annotation.load_from_json(args.input)
    isv_annot = get_annotation_attributes(annot)

    prediction = predict(isv_annot, annot.cnv.cnv_type)

    if args.output:
        prediction.store_as_json(args.output)
    else:
        print(json.dumps(asdict(prediction), indent=2), file=sys.stdout)


if __name__ == "__main__":
    main()
