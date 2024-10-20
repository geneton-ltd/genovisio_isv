from dataclasses import asdict, dataclass

import annotation


@dataclass
class ISVFeatures:
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


def get(annot: annotation.Annotation) -> ISVFeatures:
    gene_type_counts = annot.count_gene_types()
    all_genes = (
        gene_type_counts["protein_coding"]
        + gene_type_counts["pseudogenes"]
        + gene_type_counts["mirna"]
        + gene_type_counts["lncrna"]
        + gene_type_counts["rrna"]
        + gene_type_counts["snrna"]
        + gene_type_counts["other"]
    )

    regulatory_counts = annot.count_regulatory_types()
    regulatory_sum = (
        regulatory_counts["CTCF_binding_site"]
        + regulatory_counts["enhancer"]
        + regulatory_counts["silencer"]
        + regulatory_counts["open_chromatin_region"]
        + regulatory_counts["TF_binding_site"]
        + regulatory_counts["curated"]
        + regulatory_counts["flanking_region"]
        + regulatory_counts["transcriptional_cis_regulatory_region"]
        + regulatory_counts["promoter"]
        + regulatory_counts["DNase_I_hypersensitive_site"]
        + regulatory_counts["enhancer_blocking_element"]
        + regulatory_counts["TATA_box"]
        + regulatory_counts["other"]
    )

    hi_genes = annot.get_haploinsufficient_gene_names(overlap_type=annotation.enums.Overlap.ANY, valid_scores=[1, 2, 3])
    hi_regions = annot.get_haploinsufficient_regions(overlap_type=annotation.enums.Overlap.ANY, valid_scores=[1, 2, 3])
    ts_regions = annot.get_triplosensitivity_regions(overlap_type=annotation.enums.Overlap.ANY, valid_scores=[1, 2, 3])

    annotated_genes = annot.get_annotated_genes()

    return ISVFeatures(
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
