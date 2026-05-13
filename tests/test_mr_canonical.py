"""Canonical MR sanity tests — does the engine reproduce published results
on real (curated) PCSK9 / LPA / IL6R / APOC3 instruments?

These aren't claims about the precise magnitudes (which depend on the
release we curated against), just that the *direction and order* of the
known biology comes out. If any of these breaks, something fundamental
in the engine has regressed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pleio import disease_gwas, mr_engine, pqtl_instruments


def _run(gene: str, outcome: str):
    exp = pqtl_instruments.get_instruments(gene)
    out = disease_gwas.get_outcome_effects(outcome, exp["rsid"].tolist())
    harm = mr_engine.harmonize(exp, out)
    res = mr_engine.run_mr(harm, outcome_is_binary=disease_gwas.OUTCOMES[outcome]["binary"])
    return res, harm


def test_pcsk9_lowers_cad_risk():
    """Lower PCSK9 protein → lower CAD risk. β should be POSITIVE (protein up → risk up)."""
    res, _ = _run("PCSK9", "CAD")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"PCSK9→CAD IVW β = {ivw.estimate:.3f} ± {ivw.se:.3f}, p={ivw.pvalue:.2e}, "
          f"OR={ivw.or_ci()[0]:.2f} per SD protein")
    # Higher protein -> higher risk (i.e. lowering protein is therapeutic)
    assert ivw.estimate > 0.05
    # ... and significant
    assert ivw.pvalue < 0.01


def test_pcsk9_lowers_ldl():
    """Higher PCSK9 → higher LDL-C (well-known; should be very strong)."""
    res, _ = _run("PCSK9", "LDL-C")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"PCSK9→LDL-C IVW β = {ivw.estimate:.3f}, p={ivw.pvalue:.2e}")
    assert ivw.estimate > 0.20
    assert ivw.pvalue < 1e-8


def test_lpa_raises_cad():
    """Higher Lp(a) → higher CAD risk."""
    res, _ = _run("LPA", "CAD")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"LPA→CAD IVW β = {ivw.estimate:.3f}, OR={ivw.or_ci()[0]:.2f} per SD")
    assert ivw.estimate > 0.15
    assert ivw.pvalue < 1e-8


def test_il6r_lowers_cad():
    """Lower IL6R signaling (proxied by D358A) → lower CAD risk → positive β."""
    res, _ = _run("IL6R", "CAD")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"IL6R→CAD IVW β = {ivw.estimate:.3f}, OR={ivw.or_ci()[0]:.2f} per SD")
    assert ivw.estimate > 0.02
    # Modest effect — don't demand small p
    assert ivw.pvalue < 0.20


def test_apoc3_lowers_tg_strongly():
    """APOC3 LoF → big TG decrease → positive β (higher APOC3 → higher TG)."""
    res, _ = _run("APOC3", "TG")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"APOC3→TG IVW β = {ivw.estimate:.3f}, p={ivw.pvalue:.2e}")
    assert ivw.estimate > 0.5
    assert ivw.pvalue < 1e-8


def test_apoc3_lowers_cad():
    """APOC3 LoF → lower CAD; higher APOC3 → higher CAD risk."""
    res, _ = _run("APOC3", "CAD")
    assert "error" not in res, res
    ivw = res["ivw"]
    print(f"APOC3→CAD IVW β = {ivw.estimate:.3f}, OR={ivw.or_ci()[0]:.2f}")
    assert ivw.estimate > 0.0
