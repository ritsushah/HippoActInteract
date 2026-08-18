"""Unit tests for the evidence atlas (local fixtures; no live download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evidence_atlas import (
    aggregate_hits,
    atlas_stats,
    build_atlas_table,
    classify_pair,
    dominant_channel,
    extract_uniprot_accessions,
    localization_overlap,
    location_buckets,
    parse_biogrid_tab3,
    parse_intact_mitab,
    parse_string_channel_frame,
    phase2_gate,
    physical_edges_from_curated,
)

BIOGRID_TAB3 = """#BioGRID Interaction ID\tEntrez Gene Interactor A\tEntrez Gene Interactor B\tBioGRID ID Interactor A\tBioGRID ID Interactor B\tSystematic Name Interactor A\tSystematic Name Interactor B\tOfficial Symbol Interactor A\tOfficial Symbol Interactor B\tSynonyms Interactor A\tSynonyms Interactor B\tExperimental System\tExperimental System Type\tAuthor\tPublication Source\tOrganism ID Interactor A\tOrganism ID Interactor B\tThroughput\tScore\tModification\tQualifications\tTags\tSource Database\tSWISS-PROT Accessions Interactor A\tTREMBL Accessions Interactor A\tREFSEQ Accessions Interactor A\tSWISS-PROT Accessions Interactor B\tTREMBL Accessions Interactor B\tREFSEQ Accessions Interactor B
1\t1\t2\t1\t2\t-\t-\tYAP1\tACTB\t-\t-\tTwo-hybrid\tphysical\tSmith\tPUBMED:1111\t9606\t9606\tLow Throughput\t-\t-\t-\t-\tBIOGRID\tP46937\t-\t-\tP60709\t-\t-
2\t3\t4\t3\t4\t-\t-\tCBK1\tACT1\t-\t-\tDosage Rescue\tgenetic\tLee\tPUBMED:2222\t559292\t559292\tHigh Throughput\t-\t-\t-\t-\tBIOGRID\tP53894\t-\t-\tP60010\t-\t-
3\t5\t6\t5\t6\t-\t-\tOUT\tSIDE\t-\t-\tAffinity Capture-MS\tphysical\tX\tPUBMED:3333\t9606\t9606\tHigh Throughput\t-\t-\t-\t-\tBIOGRID\tP99999\t-\t-\tP88888\t-\t-
"""

INTACT_MITAB = (
    "uniprotkb:P46937\tuniprotkb:P18206\tuniprotkb:P46937\tuniprotkb:P18206\t"
    "psi-mi:yap1\tpsi-mi:vcl\tpsi-mi:\"MI:0018\"(two hybrid)\tSmith et al.\tpubmed:4444\t"
    "taxid:9606\ttaxid:9606\tpsi-mi:\"MI:0407\"(direct interaction)\tpsi-mi:intact\tintact:EBI-1\t-\n"
    "uniprotkb:P53894\tuniprotkb:P60010\t-\t-\tcbk1\tact1\tpsi-mi:\"MI:0006\"(anti bait coip)\t"
    "Lee\tpubmed:5555\ttaxid:559292\ttaxid:559292\tpsi-mi:\"MI:0915\"(physical association)\t"
    "psi-mi:intact\tintact:EBI-2\t-\n"
)


def _proteins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "string_id": [
                "9606.YAP1",
                "9606.ACTB",
                "9606.VCL",
                "9606.NF2",
                "4932.CBK1",
                "4932.ACT1",
                "4932.KIC1",
            ],
            "preferred_name": ["YAP1", "ACTB", "VCL", "NF2", "CBK1", "ACT1", "KIC1"],
            "species_id": [9606, 9606, 9606, 9606, 4932, 4932, 4932],
            "species_name": [
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Homo sapiens",
                "Saccharomyces cerevisiae",
                "Saccharomyces cerevisiae",
                "Saccharomyces cerevisiae",
            ],
            "compartment": ["hippo", "actin", "actin", "hippo", "hippo", "actin", "hippo"],
            "uniprot_accession": ["P46937", "P60709", "P18206", "P35240", "P53894", "P60010", "P38692"],
        }
    )


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_string_id": ["9606.YAP1", "9606.ACTB", "4932.ACT1"],
            "target_string_id": ["9606.NF2", "9606.VCL", "4932.KIC1"],
            "combined_score": [0.9, 0.8, 0.75],
        }
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_extract_uniprot_from_mitab_and_pipe_cells() -> None:
    assert extract_uniprot_accessions("uniprotkb:P46937|uniprotkb:Q9H0") == {"P46937"}
    assert extract_uniprot_accessions("P46937|P60709") == {"P46937", "P60709"}
    assert extract_uniprot_accessions("-") == set()


def test_parse_biogrid_keeps_wanted_physical_and_genetic(tmp_path: Path) -> None:
    path = _write(tmp_path / "biogrid.tab3.txt", BIOGRID_TAB3)
    wanted = {"P46937", "P60709", "P53894", "P60010"}
    hits = parse_biogrid_tab3(path, wanted)
    keys = {(h.accession_a, h.accession_b, h.experimental_system_type) for h in hits}
    assert ("P46937", "P60709", "physical") in keys
    assert ("P53894", "P60010", "genetic") in keys
    assert all(h.accession_a in wanted and h.accession_b in wanted for h in hits)


def test_parse_intact_mitab_direct_and_association(tmp_path: Path) -> None:
    path = _write(tmp_path / "intact.mitab", INTACT_MITAB)
    wanted = {"P46937", "P18206", "P53894", "P60010"}
    hits = parse_intact_mitab(path, wanted)
    assert len(hits) == 2
    yap_vcl = next(h for h in hits if {h.accession_a, h.accession_b} == {"P46937", "P18206"})
    assert "MI:0018" in yap_vcl.detection_method
    assert "MI:0407" in yap_vcl.interaction_type


def test_classify_physical_wins_over_hub_and_textmining() -> None:
    assert (
        classify_pair(
            biogrid_physical=True,
            intact=False,
            string_combined=0.2,
            dominant="textmining",
            localization="conflicting",
            degree_product=10_000,
            degree_product_cutoff=100,
        )
        == "physical_curated"
    )


def test_classify_string_only_unreported_and_artifact() -> None:
    assert (
        classify_pair(
            biogrid_physical=False,
            intact=False,
            string_combined=0.8,
            dominant="experiments",
            localization="compatible",
            degree_product=10,
            degree_product_cutoff=100,
        )
        == "string_functional_only"
    )
    assert (
        classify_pair(
            biogrid_physical=False,
            intact=False,
            string_combined=0.2,
            dominant="",
            localization="unclear",
            degree_product=10,
            degree_product_cutoff=100,
        )
        == "unreported"
    )
    assert (
        classify_pair(
            biogrid_physical=False,
            intact=False,
            string_combined=0.8,
            dominant="textmining",
            localization="compatible",
            degree_product=10,
            degree_product_cutoff=100,
        )
        == "artifact_risk"
    )
    assert (
        classify_pair(
            biogrid_physical=False,
            intact=False,
            string_combined=0.1,
            dominant="",
            localization="conflicting",
            degree_product=10,
            degree_product_cutoff=100,
        )
        == "artifact_risk"
    )


def test_localization_overlap_rules() -> None:
    assert localization_overlap({"cytoplasm"}, {"cytoplasm", "nucleus"}) == "compatible"
    assert localization_overlap({"nucleus"}, {"extracellular"}) == "conflicting"
    assert localization_overlap({"nucleus"}, {"cytoskeleton"}) == "unclear"
    assert localization_overlap(set(), {"nucleus"}) == "unclear"
    buckets = location_buckets("SUBCELLULAR LOCATION: Nucleus. Cytoplasm.", "C:cytoskeleton")
    assert "nucleus" in buckets and "cytoplasm" in buckets


def test_dominant_channel_picks_experiments() -> None:
    assert dominant_channel({"nscore": 0.1, "escore": 0.6, "tscore": 0.2}) == "experiments"
    assert dominant_channel({"nscore": 0.0, "escore": 0.0}) == ""


def test_build_atlas_and_stats_from_fixtures(tmp_path: Path) -> None:
    biogrid = parse_biogrid_tab3(_write(tmp_path / "b.tab3.txt", BIOGRID_TAB3), set(_proteins()["uniprot_accession"]))
    intact = parse_intact_mitab(_write(tmp_path / "i.mitab", INTACT_MITAB), set(_proteins()["uniprot_accession"]))
    curated = aggregate_hits(biogrid + intact)
    channels = parse_string_channel_frame(
        pd.DataFrame(
            {
                "stringId_A": ["9606.YAP1"],
                "stringId_B": ["9606.NF2"],
                "score": [0.9],
                "nscore": [0.0],
                "fscore": [0.0],
                "pscore": [0.0],
                "ascore": [0.0],
                "escore": [0.4],
                "dscore": [0.2],
                "tscore": [0.1],
            }
        )
    )
    locations = pd.DataFrame(
        {
            "uniprot_accession": ["P46937", "P60709", "P18206", "P35240", "P53894", "P60010", "P38692"],
            "subcellular_location": [
                "Nucleus. Cytoplasm.",
                "Cytoplasm, cytoskeleton.",
                "Cell membrane. Cytoplasm.",
                "Nucleus. Plasma membrane.",
                "Cytoplasm.",
                "Cytoplasm, cytoskeleton.",
                "Bud neck.",
            ],
            "go_cc": [""] * 7,
        }
    )
    graphsage = pd.DataFrame(
        {
            "string_id_a": ["9606.ACTB", "4932.ACT1"],
            "string_id_b": ["9606.YAP1", "4932.KIC1"],
            "probability": [0.92, 0.95],
            "rank": [2, 1],
        }
    )
    atlas = build_atlas_table(
        _proteins(),
        _interactions(),
        curated,
        channels,
        locations,
        graphsage,
        degree_quantile=0.99,
    )
    human_yap_actb = atlas[(atlas.protein_a == "ACTB") & (atlas.protein_b == "YAP1")]
    assert len(human_yap_actb) == 1
    assert human_yap_actb.iloc[0].evidence_class == "physical_curated"
    yeast_cbk_act = atlas[(atlas.protein_a == "ACT1") & (atlas.protein_b == "CBK1")]
    assert bool(yeast_cbk_act.iloc[0].intact)
    assert bool(yeast_cbk_act.iloc[0].genetic_only) is False
    stats = atlas_stats(atlas)
    assert stats["n_pairs"] == 6
    assert stats["n_physical_curated"] >= 2
    physical = physical_edges_from_curated(_proteins(), curated)
    assert not physical.empty
    assert set(physical.string_id_a) | set(physical.string_id_b)


def test_phase2_gate_stops_when_almost_all_absent_are_physical() -> None:
    run, reason = phase2_gate(
        n_absent=156,
        frac_physical_absent=0.95,
        frac_unreported_absent=0.02,
        spearman_degree=0.1,
    )
    assert run is False
    assert "resource" in reason
    run2, _ = phase2_gate(
        n_absent=156,
        frac_physical_absent=0.2,
        frac_unreported_absent=0.5,
        spearman_degree=0.05,
    )
    assert run2 is True
    run3, reason3 = phase2_gate(
        n_absent=156,
        frac_physical_absent=0.2,
        frac_unreported_absent=0.05,
        spearman_degree=0.6,
    )
    assert run3 is True
    assert "degree" in reason3
