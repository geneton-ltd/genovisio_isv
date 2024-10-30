import annotation

from isv.src import features, prediction


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
    )

    result = prediction.predict(annot_values, annotation.enums.CNVType.GAIN)

    print("SHAP VALUES", type(result.isv_shap_values))
    print(result.isv_shap_values)
    assert result == prediction.Prediction(
        isv_prediction=0.9919388890266418,
        isv_score=0.9838777780532837,
        isv_classification=prediction.ACMGClassification.LIKELY_PATHOGENIC,
        isv_shap_values={
            "gencode_genes": -0.007487309580706873,
            "protein_coding": -0.0016489525987545682,
            "pseudogenes": -0.0045932165271384325,
            "mirna": -0.0024935691382905897,
            "lncrna": -0.01954475841461909,
            "rrna": 0.0,
            "snrna": -0.002116625520270135,
            "morbid_genes": 0.016454276332696374,
            "disease_associated_genes": 0.008784748053452786,
            "hi_genes": 0.007162925711143261,
            "regions_HI": -0.007827842236345042,
            "regions_TS": -0.014850730914687346,
            "regulatory": -0.015093982115859892,
            "regulatory_enhancer": -0.0013117994192155952,
            "regulatory_silencer": 0.0,
            "regulatory_transcriptional_cis_regulatory_region": 0.0,
            "regulatory_promoter": -0.010893374268655569,
            "regulatory_DNase_I_hypersensitive_site": 0.0,
            "regulatory_enhancer_blocking_element": 0.0,
            "regulatory_TATA_box": 0.0,
        },
        isv_shap_scores={
            "gencode_genes": -0.007487309580706873,
            "protein_coding": -0.0016489525987545682,
            "pseudogenes": -0.0045932165271384325,
            "mirna": -0.0024935691382905897,
            "lncrna": -0.01954475841461909,
            "rrna": 0.0,
            "snrna": -0.002116625520270135,
            "morbid_genes": 0.016454276332696374,
            "disease_associated_genes": 0.008784748053452786,
            "hi_genes": 0.007162925711143261,
            "regions_HI": -0.007827842236345042,
            "regions_TS": -0.014850730914687346,
            "regulatory": -0.015093982115859892,
            "regulatory_enhancer": -0.0013117994192155952,
            "regulatory_silencer": 0.0,
            "regulatory_transcriptional_cis_regulatory_region": 0.0,
            "regulatory_promoter": -0.010893374268655569,
            "regulatory_DNase_I_hypersensitive_site": 0.0,
            "regulatory_enhancer_blocking_element": 0.0,
            "regulatory_TATA_box": 0.0,
        },
        isv_features=annot_values,
    )


def test_shap_values():
    annotated_values = features.ISVFeatures(
        gencode_genes=1,
        protein_coding=1,
        pseudogenes=0,
        mirna=0,
        lncrna=0,
        rrna=0,
        snrna=0,
        morbid_genes=1,
        disease_associated_genes=1,
        hi_genes=0,
        regions_HI=0,
        regions_TS=0,
        regulatory=1,
        regulatory_enhancer=0,
        regulatory_silencer=0,
        regulatory_transcriptional_cis_regulatory_region=0,
        regulatory_promoter=0,
        regulatory_DNase_I_hypersensitive_site=0,
        regulatory_enhancer_blocking_element=0,
        regulatory_TATA_box=0,
    )

    result = prediction.predict(annotated_values, annotation.enums.CNVType.LOSS)

    assert result.isv_shap_values == {
        "gencode_genes": -0.001927492717935655,
        "protein_coding": -0.020810622836816832,
        "pseudogenes": -0.014682043028594239,
        "mirna": -0.0007550389840000536,
        "lncrna": 0.002913866574548539,
        "rrna": 0.0,
        "snrna": -0.007335019845483958,
        "morbid_genes": 0.013905657070214352,
        "disease_associated_genes": 0.02139180318485674,
        "hi_genes": -0.07510238397080769,
        "regions_HI": -0.002166717122879159,
        "regions_TS": -0.019773419437143613,
        "regulatory": -0.015584879749943232,
        "regulatory_enhancer": 0.012059786718962509,
        "regulatory_silencer": 0.0,
        "regulatory_transcriptional_cis_regulatory_region": 0.0,
        "regulatory_promoter": 0.02675586934182097,
        "regulatory_DNase_I_hypersensitive_site": 0.0,
        "regulatory_enhancer_blocking_element": 0.0,
        "regulatory_TATA_box": 0.0,
    }
