import annotation
import scipy.special

from isv.src import constants, features, prediction


def test_predict():
    annot_values = features.ISVFeatures(
        gencode_genes=80,
        protein_coding=39,
        pseudogenes=3,
        mirna=0,
        lncrna=0,
        rrna=0,
        snrna=0,
        morbid_genes=10,
        disease_associated_genes=13,
        hi_genes=2,
        regions_HI=0,
        regions_TS=0,
        regulatory=681,
        regulatory_enhancer=164,
        regulatory_silencer=0,
        regulatory_transcriptional_cis_regulatory_region=0,
        regulatory_promoter=100,
        regulatory_DNase_I_hypersensitive_site=0,
        regulatory_enhancer_blocking_element=0,
        regulatory_TATA_box=0,
        regulatory_open_chromatin_region=0,
        regulatory_flanking_region=0,
        regulatory_CTCF_binding_site=0,
        regulatory_TF_binding_site=0,
        regulatory_curated=0,
    )

    result = prediction.predict(annot_values, annotation.enums.CNVType.LOSS)
    total_shap = sum(result.isv_shap_values.values())
    total_shap_odds = scipy.special.expit(result.explainer_base_value + total_shap)
    assert abs(total_shap_odds - result.isv_prediction) < 1e-6
    assert result.isv_prediction > 0.6
    assert result.isv_score > 0.3
    assert result.isv_classification == prediction.ACMGClassification.VOUS

    for attr in constants.LOSS_ATTRIBUTES:
        assert attr in result.isv_features.as_dict_of_attributes().keys()

    assert constants.LOSS_ATTRIBUTES == list(result.isv_shap_values.keys())


def test_shap_values():
    annotated_values = features.ISVFeatures(
        gencode_genes=300,
        protein_coding=0,
        pseudogenes=1000,
        mirna=0,
        lncrna=0,
        rrna=0,
        snrna=0,
        morbid_genes=100,
        disease_associated_genes=100,
        hi_genes=5,
        regions_HI=5,
        regions_TS=25,
        regulatory=500,
        regulatory_enhancer=0,
        regulatory_silencer=200,
        regulatory_transcriptional_cis_regulatory_region=0,
        regulatory_promoter=0,
        regulatory_DNase_I_hypersensitive_site=0,
        regulatory_enhancer_blocking_element=0,
        regulatory_TATA_box=0,
        regulatory_open_chromatin_region=0,
        regulatory_flanking_region=0,
        regulatory_CTCF_binding_site=0,
        regulatory_TF_binding_site=0,
        regulatory_curated=0,
    )

    result = prediction.predict(annotated_values, annotation.enums.CNVType.GAIN)
    total_shap = sum(result.isv_shap_values.values())
    total_shap_odds = scipy.special.expit(result.explainer_base_value + total_shap)
    assert abs(total_shap_odds - result.isv_prediction) < 1e-6
    assert result.isv_prediction > 0.75
    assert result.isv_score > 0.5
    assert result.isv_classification == prediction.ACMGClassification.VOUS

    for attr in constants.GAIN_ATTRIBUTES:
        assert attr in result.isv_features.as_dict_of_attributes().keys()

    assert constants.GAIN_ATTRIBUTES == list(result.isv_shap_values.keys())
