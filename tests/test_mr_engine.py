"""Unit tests for pleio.mr_engine — math sanity, not biology."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pleio import mr_engine as mr


# -----------------------------------------------------------------------------
# Simulation helpers
# -----------------------------------------------------------------------------
def _simulate_mr(
    true_beta: float,
    n_snps: int = 15,
    pleiotropy_frac: float = 0.0,
    pleiotropy_size: float = 0.0,
    n_exp: int = 50_000,
    n_out: int = 200_000,
    seed: int = 7,
):
    """Simulate two-sample summary statistics with a known causal effect.

    Each SNP has true exposure effect bx_i drawn from N(0.05, 0.15), and
    observed effects with SE 1/sqrt(2*MAF*(1-MAF)*N).
    """
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.45, n_snps)
    bx_true = rng.normal(0.05, 0.15, n_snps)
    # exposure-side observed effects
    se_x = 1.0 / np.sqrt(2 * maf * (1 - maf) * n_exp)
    bx_obs = rng.normal(bx_true, se_x)
    # outcome-side: causal + optional pleiotropic offset
    pleio_mask = rng.random(n_snps) < pleiotropy_frac
    pleio_offset = pleio_mask * rng.normal(pleiotropy_size, 0.01, n_snps)
    by_true = true_beta * bx_true + pleio_offset
    se_y = 0.7 / np.sqrt(2 * maf * (1 - maf) * n_out)
    by_obs = rng.normal(by_true, se_y)
    return bx_obs, se_x, by_obs, se_y


# -----------------------------------------------------------------------------
# IVW recovers a known causal effect
# -----------------------------------------------------------------------------
def test_ivw_recovers_true_effect():
    bx, sx, by, sy = _simulate_mr(true_beta=0.30, n_snps=20)
    r = mr.ivw(bx, sx, by, sy)
    assert r.method.startswith("IVW")
    assert abs(r.estimate - 0.30) < 0.05, r.estimate
    assert r.pvalue < 1e-6
    assert r.n_snps == 20


def test_ivw_null_effect_is_not_significant():
    bx, sx, by, sy = _simulate_mr(true_beta=0.0, n_snps=20)
    r = mr.ivw(bx, sx, by, sy)
    assert abs(r.estimate) < 0.1, r.estimate


# -----------------------------------------------------------------------------
# Weighted median is robust to 30% invalid instruments
# -----------------------------------------------------------------------------
def test_weighted_median_robust_to_minority_pleiotropy():
    bx, sx, by, sy = _simulate_mr(
        true_beta=0.30, n_snps=20,
        pleiotropy_frac=0.30, pleiotropy_size=0.6,  # 30% of SNPs biased upward
    )
    r_ivw = mr.ivw(bx, sx, by, sy)
    r_wm = mr.weighted_median(bx, sx, by, sy, n_boot=300)
    # IVW should be biased upward; WM should be closer to 0.30
    assert abs(r_wm.estimate - 0.30) < abs(r_ivw.estimate - 0.30)


# -----------------------------------------------------------------------------
# MR-Egger intercept ≈ 0 when no directional pleiotropy
# -----------------------------------------------------------------------------
def test_egger_intercept_near_zero_no_pleiotropy():
    bx, sx, by, sy = _simulate_mr(true_beta=0.30, n_snps=25, pleiotropy_frac=0.0)
    slope, intercept = mr.egger(bx, sx, by, sy)
    assert abs(intercept.estimate) < 0.05, intercept.estimate


# -----------------------------------------------------------------------------
# Harmonization: detects flipped alleles, drops palindromes when EAF unknown
# -----------------------------------------------------------------------------
def test_harmonize_flips_swapped_alleles():
    exp = pd.DataFrame([
        {"rsid": "rs1", "ea": "A", "oa": "G", "beta": 0.1, "se": 0.01},
        {"rsid": "rs2", "ea": "C", "oa": "T", "beta": 0.2, "se": 0.01},
    ])
    # rs2 has ea/oa swapped in outcome → expect beta flipped
    out = pd.DataFrame([
        {"rsid": "rs1", "ea": "A", "oa": "G", "beta": 0.05, "se": 0.005},
        {"rsid": "rs2", "ea": "T", "oa": "C", "beta": 0.04, "se": 0.005},
    ])
    h = mr.harmonize(exp, out)
    assert set(h["rsid"]) == {"rs1", "rs2"}
    rs2 = h.set_index("rsid").loc["rs2"]
    assert rs2["action"] == "flipped"
    assert rs2["beta_out"] == pytest.approx(-0.04)


def test_harmonize_drops_palindromic_when_eaf_unknown():
    exp = pd.DataFrame([{"rsid": "rs1", "ea": "A", "oa": "T", "beta": 0.1, "se": 0.01}])
    out = pd.DataFrame([{"rsid": "rs1", "ea": "A", "oa": "T", "beta": 0.05, "se": 0.005}])
    h = mr.harmonize(exp, out)
    assert h.iloc[0]["action"] == "ambiguous_palindrome"


# -----------------------------------------------------------------------------
# End-to-end: run_mr returns the full battery
# -----------------------------------------------------------------------------
def test_run_mr_returns_battery():
    bx, sx, by, sy = _simulate_mr(true_beta=0.25, n_snps=15)
    harmonized = pd.DataFrame({
        "rsid": [f"rs{i}" for i in range(len(bx))],
        "ea": ["A"] * len(bx), "oa": ["G"] * len(bx),
        "beta_exp": bx, "se_exp": sx, "beta_out": by, "se_out": sy,
        "action": ["kept"] * len(bx),
    })
    res = mr.run_mr(harmonized)
    assert "ivw" in res and "weighted_median" in res and "egger_slope" in res
    assert res["n_snps"] == len(bx)
    assert abs(res["ivw"].estimate - 0.25) < 0.07


def test_interpret_direction():
    assert "inhibit" in mr.interpret_direction(0.3, outcome_is_binary=True).lower()
    assert "activate" in mr.interpret_direction(-0.3, outcome_is_binary=True).lower()
    assert "no causal" in mr.interpret_direction(0.001, outcome_is_binary=True).lower()
