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
        isv_prediction=0.994540274143219,
        isv_score=0.989080548286438,
        isv_classification=prediction.ACMGClassification.LIKELY_PATHOGENIC,
        isv_shap_values={
            "gencode_genes": 0.5013508200645447,
            "protein_coding": 0.5173912644386292,
            "pseudogenes": 0.01303008571267128,
            "mirna": 0.0,
            "lncrna": 0.0,
            "rrna": 0.0,
            "snrna": 0.0,
            "morbid_genes": -0.06586112082004547,
            "disease_associated_genes": 0.014231479726731777,
            "hi_genes": 0.7603306174278259,
            "regions_HI": -0.15067468583583832,
            "regions_TS": 0.10014545917510986,
            "regulatory": 3.760049343109131,
            "regulatory_enhancer": 0.8933524489402771,
            "regulatory_silencer": 0.0,
            "regulatory_transcriptional_cis_regulatory_region": 0.0,
            "regulatory_promoter": 0.2965705096721649,
            "regulatory_DNase_I_hypersensitive_site": 0.0,
            "regulatory_enhancer_blocking_element": 0.0,
            "regulatory_TATA_box": 0.0,
        },
        isv_shap_scores={
            "gencode_genes": 0.0027016401290893555,
            "protein_coding": 0.0347825288772583,
            "pseudogenes": -0.9739398285746574,
            "mirna": -1.0,
            "lncrna": -1.0,
            "rrna": -1.0,
            "snrna": -1.0,
            "morbid_genes": -1.131722241640091,
            "disease_associated_genes": -0.9715370405465364,
            "hi_genes": 0.5206612348556519,
            "regions_HI": -1.3013493716716766,
            "regions_TS": -0.7997090816497803,
            "regulatory": 6.520098686218262,
            "regulatory_enhancer": 0.7867048978805542,
            "regulatory_silencer": -1.0,
            "regulatory_transcriptional_cis_regulatory_region": -1.0,
            "regulatory_promoter": -0.40685898065567017,
            "regulatory_DNase_I_hypersensitive_site": -1.0,
            "regulatory_enhancer_blocking_element": -1.0,
            "regulatory_TATA_box": -1.0,
        },
        isv_features=annot_values,
    )
