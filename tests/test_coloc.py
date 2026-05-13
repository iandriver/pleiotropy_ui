"""Unit tests for pleio.coloc — mechanism-tier classifier.

These tests exercise the curated/offline path (cis-pQTL instruments ×
disease GWAS tables via mr_engine). The OT GraphQL fallback path is
network-dependent and is not unit-tested here — covered by integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pleio import coloc


# ---------------------------------------------------------------------------
# Pure tier-decision logic
# ---------------------------------------------------------------------------
def test_tier_decision_t1_requires_pqtl_and_directional_consistency():
    assert coloc._decide_tier(
        has_pqtl_coloc=True, has_eqtl_coloc=False,
        has_pav=False, has_gwas=True,
        directional_consistency="concordant",
    ) == "T1"


def test_tier_decision_pqtl_without_consistency_falls_to_t2():
    assert coloc._decide_tier(
        has_pqtl_coloc=True, has_eqtl_coloc=False,
        has_pav=False, has_gwas=True,
        directional_consistency="inconsistent",
    ) == "T2"


def test_tier_decision_eqtl_only_is_t2():
    assert coloc._decide_tier(
        has_pqtl_coloc=False, has_eqtl_coloc=True,
        has_pav=False, has_gwas=True,
        directional_consistency="concordant",
    ) == "T2"


def test_tier_decision_pav_only_is_t3():
    assert coloc._decide_tier(
        has_pqtl_coloc=False, has_eqtl_coloc=False,
        has_pav=True, has_gwas=False,
        directional_consistency="n/a",
    ) == "T3"


def test_tier_decision_gwas_only_is_t4():
    assert coloc._decide_tier(
        has_pqtl_coloc=False, has_eqtl_coloc=False,
        has_pav=False, has_gwas=True,
        directional_consistency="n/a",
    ) == "T4"


def test_tier_decision_no_evidence_is_none():
    assert coloc._decide_tier(
        has_pqtl_coloc=False, has_eqtl_coloc=False,
        has_pav=False, has_gwas=False,
        directional_consistency="n/a",
    ) == "none"


def test_tier_decision_prefers_higher_tier_evidence():
    """PAV + GWAS + pQTL coloc + dir-consistency must produce T1, not T3."""
    assert coloc._decide_tier(
        has_pqtl_coloc=True, has_eqtl_coloc=True,
        has_pav=True, has_gwas=True,
        directional_consistency="concordant",
    ) == "T1"


# ---------------------------------------------------------------------------
# Disease ↔ outcome-code crosswalk
# ---------------------------------------------------------------------------
def test_outcome_for_disease_by_efo_id():
    assert coloc.outcome_for_disease("EFO_0001645", None) == "CAD"
    assert coloc.outcome_for_disease("MONDO_0005148", None) == "T2D"


def test_outcome_for_disease_by_name_fallback():
    assert coloc.outcome_for_disease(None, "coronary artery disease") == "CAD"
    assert coloc.outcome_for_disease(None, "Type 2 Diabetes Mellitus") == "T2D"
    # Capitalised / messy input still resolves.
    assert coloc.outcome_for_disease(None, "  Triglycerides  ") == "TG"


def test_outcome_for_disease_unknown_returns_none():
    assert coloc.outcome_for_disease("EFO_9999999", "asthma") is None


# ---------------------------------------------------------------------------
# Curated coloc / MR layer — golden biology cases
# ---------------------------------------------------------------------------
def test_pcsk9_cad_is_pqtl_coloc_with_directional_consistency():
    """PCSK9 LoF lowers protein and lowers CAD risk → T1, concordant."""
    ev = coloc.curated_coloc_evidence("PCSK9", "CAD")
    assert ev.has_pqtl_coloc is True
    assert ev.has_eqtl_coloc is False
    assert ev.instrument_source == "pQTL"
    assert ev.significant, f"PCSK9→CAD MR should be significant; got p={ev.mr_p}"
    assert ev.directional_consistency == "concordant"
    # Sign: lower PCSK9 → lower CAD risk, so β > 0 (raising PCSK9 raises risk).
    assert ev.mr_beta is not None and ev.mr_beta > 0


def test_il6r_cad_is_pqtl_coloc_with_directional_consistency():
    """Soluble IL6R-lowering alleles lower CAD risk → T1, concordant."""
    ev = coloc.curated_coloc_evidence("IL6R", "CAD")
    assert ev.has_pqtl_coloc is True
    assert ev.directional_consistency == "concordant"
    assert ev.mr_beta is not None and ev.mr_beta > 0


def test_lpa_cad_is_pqtl_coloc_concordant_with_large_effect():
    """LPA raises Lp(a) and raises CAD risk strongly → T1, concordant."""
    ev = coloc.curated_coloc_evidence("LPA", "CAD")
    assert ev.has_pqtl_coloc is True
    assert ev.directional_consistency == "concordant"
    # Large effect: log-OR per SD protein should be sizeable.
    assert abs(ev.mr_beta) > 0.2


def test_hmgcr_cad_is_eqtl_coloc():
    """HMGCR instruments are GTEx-eQTL only → eQTL-coloc path, not pQTL."""
    ev = coloc.curated_coloc_evidence("HMGCR", "CAD")
    assert ev.has_eqtl_coloc is True
    assert ev.has_pqtl_coloc is False
    assert ev.instrument_source == "eQTL"


def test_apoc3_tg_is_pqtl_coloc_concordant():
    """APOC3 LoF lowers protein and dramatically lowers triglycerides."""
    ev = coloc.curated_coloc_evidence("APOC3", "TG")
    assert ev.has_pqtl_coloc is True
    assert ev.directional_consistency == "concordant"
    # APOC3-raising direction should raise TG (β > 0).
    assert ev.mr_beta is not None and ev.mr_beta > 0


def test_uncurated_target_returns_empty_evidence():
    ev = coloc.curated_coloc_evidence("NOT_A_REAL_GENE", "CAD")
    assert ev.has_pqtl_coloc is False
    assert ev.has_eqtl_coloc is False
    assert ev.mr_beta is None
    assert any("not in the curated" in n for n in ev.notes)


def test_unknown_outcome_returns_empty_evidence():
    ev = coloc.curated_coloc_evidence("PCSK9", "MIGRAINE")
    assert ev.has_pqtl_coloc is False
    assert ev.mr_beta is None


def test_curated_evidence_object_has_notes():
    ev = coloc.curated_coloc_evidence("PCSK9", "CAD")
    assert isinstance(ev.notes, list) and len(ev.notes) >= 1


def test_sign_concordance_is_reported():
    """sign_concordance should be 1.0 for a clean target like PCSK9/CAD,
    where every instrument's Wald ratio shares the IVW sign."""
    ev = coloc.curated_coloc_evidence("PCSK9", "CAD")
    assert ev.sign_concordance is not None
    assert ev.sign_concordance == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# v0.2 expansion — new curated proteins / outcomes
# ---------------------------------------------------------------------------
def test_angptl3_lowers_ldl_t1_concordant():
    ev = coloc.curated_coloc_evidence("ANGPTL3", "LDL-C")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    # Higher ANGPTL3 → higher LDL-C, so β > 0 in our consistent encoding.
    assert ev.mr_beta is not None and ev.mr_beta > 0


def test_angptl4_lowers_tg_t1_concordant():
    ev = coloc.curated_coloc_evidence("ANGPTL4", "TG")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"


def test_ldlr_lowers_ldl_t1_concordant():
    """Higher LDLR clears more LDL → lower LDL-C; protein-LOWERING raises LDL."""
    ev = coloc.curated_coloc_evidence("LDLR", "LDL-C")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    # Higher LDLR → lower LDL-C, so β should be negative.
    assert ev.mr_beta < 0


def test_cetp_raises_hdl_t1_concordant():
    """CETP-lowering raises HDL — canonical CETP-inhibitor biology."""
    ev = coloc.curated_coloc_evidence("CETP", "HDL-C")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    # Higher CETP → lower HDL, so β should be negative.
    assert ev.mr_beta < 0


def test_apob_raises_cad_t1_concordant():
    ev = coloc.curated_coloc_evidence("APOB", "CAD")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    # Higher APOB → higher CAD risk, β > 0.
    assert ev.mr_beta > 0


def test_tyk2_protects_against_ibd_t1_concordant():
    """TYK2 P1104A LoF protects against IBD (deucravacitinib mechanism)."""
    ev = coloc.curated_coloc_evidence("TYK2", "IBD")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    # Higher TYK2 → higher IBD risk; lowering protects → β > 0.
    assert ev.mr_beta > 0


def test_tyk2_protects_against_ra_t1_concordant():
    ev = coloc.curated_coloc_evidence("TYK2", "RA")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"


def test_il6r_lowers_ra_t1_concordant():
    """sIL-6R-lowering → lower RA (tocilizumab/sarilumab mechanism)."""
    ev = coloc.curated_coloc_evidence("IL6R", "RA")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"


def test_crp_self_loop_works():
    """CRP cis-pQTL → CRP outcome should be a perfect positive control."""
    ev = coloc.curated_coloc_evidence("CRP", "CRP")
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"


def test_outcome_for_disease_v02_outcomes():
    assert coloc.outcome_for_disease("EFO_0004458", None) == "CRP"
    assert coloc.outcome_for_disease("EFO_0000685", None) == "RA"
    assert coloc.outcome_for_disease("EFO_0003767", None) == "IBD"
    assert coloc.outcome_for_disease("EFO_0003885", None) == "MS"
    # Disease-name fallback
    assert coloc.outcome_for_disease(None, "Rheumatoid arthritis") == "RA"
    assert coloc.outcome_for_disease(None, "Crohn's disease") == "IBD"


def test_curated_protein_set_size():
    """v0.2 should have at least 13 curated proteins."""
    from pleio import pqtl_instruments
    proteins = pqtl_instruments.available_proteins()
    assert len(proteins) >= 13
    for must_have in ("PCSK9", "LPA", "ANGPTL3", "CETP", "TYK2"):
        assert must_have in proteins


# ---------------------------------------------------------------------------
# coloc_evidence_from_data — new pure-data engine used by both paths
# ---------------------------------------------------------------------------
def test_coloc_evidence_from_data_handles_empty_inputs():
    import pandas as pd
    ev = coloc.coloc_evidence_from_data(
        pd.DataFrame(), pd.DataFrame(),
        instrument_kind="pQTL", outcome_is_binary=True,
    )
    assert not ev.has_pqtl_coloc
    assert ev.mr_beta is None


def test_coloc_evidence_from_data_runs_on_synthetic_inputs():
    """Build a synthetic instrument set + outcome that should produce T1."""
    import pandas as pd
    instr = pd.DataFrame([
        dict(rsid=f"rs{i}", ea="A", oa="G", eaf=0.3, beta=0.3, se=0.04, p=1e-12)
        for i in range(8)
    ])
    out = pd.DataFrame([
        dict(rsid=f"rs{i}", ea="A", oa="G", eaf=0.3, beta=0.10, se=0.02)
        for i in range(8)
    ])
    ev = coloc.coloc_evidence_from_data(
        instr, out, instrument_kind="pQTL",
        outcome_is_binary=True, label="synthetic",
    )
    assert ev.has_pqtl_coloc
    assert ev.directional_consistency == "concordant"
    assert ev.mr_beta is not None and ev.mr_beta > 0


# ---------------------------------------------------------------------------
# Tier label / colour metadata is self-consistent
# ---------------------------------------------------------------------------
def test_tier_metadata_complete():
    for tier in ("T1", "T2", "T3", "T4", "none"):
        assert tier in coloc.TIER_LABELS
        assert tier in coloc.TIER_CONFIDENCE
        assert tier in coloc.TIER_COLOR
    # Confidence ordering: T1 > T2 > T3 > T4 > none
    confs = [coloc.TIER_CONFIDENCE[t] for t in ("T1", "T2", "T3", "T4", "none")]
    assert confs == sorted(confs, reverse=True)
