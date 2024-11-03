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

    assert result == prediction.Prediction(
        isv_prediction=0.9915704727172852,
        isv_score=0.9831409454345703,
        isv_classification=prediction.ACMGClassification.LIKELY_PATHOGENIC,
        isv_shap_values={
            "gencode_genes": -0.0069333390283776485,
            "pseudogenes": -0.0023958869803474654,
            "mirna": 0.0004557597314434351,
            "snrna": -0.0036613690787413726,
            "morbid_genes": 0.02454142752716766,
            "disease_associated_genes": 0.019304866009577246,
            "hi_genes": -0.005989121384606385,
            "regions_HI": -0.005114040226455132,
            "regions_TS": -0.033045386377829125,
            "regulatory": -0.0072940625459038755,
            "regulatory_enhancer": -0.019394158791678426,
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
    print(result.isv_shap_values)

    assert result.isv_shap_values == {
        "gencode_genes": -0.010741195378547204,
        "protein_coding": -0.02688071150344005,
        "pseudogenes": -0.02470392766259786,
        "mirna": -0.0017974743092537251,
        "lncrna": 0.008163900879669991,
        "morbid_genes": 0.021527917712255706,
        "disease_associated_genes": 0.0342692272460824,
        "hi_genes": -0.1071287399346984,
        "regions_HI": 0.0019592892535194163,
        "regions_TS": -0.015655114137433174,
        "regulatory": -0.010401324343676138,
        "regulatory_enhancer": 0.022800187439036702,
        "regulatory_promoter": 0.029876231495716707,
    }
