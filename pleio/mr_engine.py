"""Two-sample Mendelian randomization — IVW, weighted median, MR-Egger.

Implements the standard two-sample MR estimators using only numpy / scipy so
the code is auditable and unit-testable. Assumes harmonized inputs:

    beta_exp, se_exp:  exposure (here: cis-pQTL) SNP effects on the protein
    beta_out, se_out:  outcome (here: disease) SNP effects on the trait/disease
                       both on the SAME effect allele.

All standard errors assume two-sample independence (samples non-overlapping)
and no measurement error in the exposure (a known limitation of IVW which
MR-Egger and weighted median partially correct for).

References
----------
* Burgess S, Butterworth A, Thompson SG. Genet Epidemiol 2013 (IVW).
* Bowden J, Davey Smith G, Burgess S. Int J Epidemiol 2015 (Egger).
* Bowden J et al. Genet Epidemiol 2016 (weighted median).
* Hemani G et al. eLife 2018 (sensitivity tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# Harmonization
# -----------------------------------------------------------------------------
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _is_palindromic(ea: str, oa: str) -> bool:
    return {ea.upper(), oa.upper()} in ({"A", "T"}, {"C", "G"})


@dataclass
class HarmonizedSNP:
    rsid: str
    ea: str  # harmonized effect allele (matches exposure)
    oa: str
    beta_exp: float
    se_exp: float
    beta_out: float
    se_out: float
    eaf_exp: float | None = None
    eaf_out: float | None = None
    action: str = "kept"  # "kept" | "flipped" | "ambiguous_palindrome" | "allele_mismatch"


def harmonize(
    exposure: pd.DataFrame,
    outcome: pd.DataFrame,
    palindrome_eaf_threshold: float = 0.42,
) -> pd.DataFrame:
    """Align outcome effects to the exposure effect allele.

    Both inputs must have columns:
      rsid, ea, oa, beta, se, eaf (eaf optional).

    Palindromic SNPs (A/T or C/G) are kept only if both EAFs are clearly
    on the same side of 0.5 and outside the palindrome_eaf_threshold window;
    otherwise they're dropped as ambiguous.
    """
    cols_required = {"rsid", "ea", "oa", "beta", "se"}
    if not cols_required.issubset(exposure.columns) or not cols_required.issubset(outcome.columns):
        missing = cols_required - set(exposure.columns) | cols_required - set(outcome.columns)
        raise ValueError(f"Missing columns: {missing}")

    e = exposure.set_index("rsid")
    o = outcome.set_index("rsid")
    common = e.index.intersection(o.index)
    rows = []
    for rsid in common:
        er, or_ = e.loc[rsid], o.loc[rsid]
        ea_e, oa_e = er["ea"].upper(), er["oa"].upper()
        ea_o, oa_o = or_["ea"].upper(), or_["oa"].upper()

        # Direct match
        if (ea_e, oa_e) == (ea_o, oa_o):
            action = "kept"
            beta_out = float(or_["beta"])
        # Swapped alleles
        elif (ea_e, oa_e) == (oa_o, ea_o):
            action = "flipped"
            beta_out = -float(or_["beta"])
        # Strand flip
        elif (ea_e, oa_e) == (_COMPLEMENT.get(ea_o, "?"), _COMPLEMENT.get(oa_o, "?")):
            action = "kept"
            beta_out = float(or_["beta"])
        elif (ea_e, oa_e) == (_COMPLEMENT.get(oa_o, "?"), _COMPLEMENT.get(ea_o, "?")):
            action = "flipped"
            beta_out = -float(or_["beta"])
        else:
            rows.append(HarmonizedSNP(rsid, ea_e, oa_e, float(er["beta"]), float(er["se"]),
                                      np.nan, np.nan, action="allele_mismatch"))
            continue

        # Palindrome ambiguity check
        if _is_palindromic(ea_e, oa_e):
            eaf_e = er.get("eaf", np.nan)
            eaf_o = or_.get("eaf", np.nan)
            if pd.isna(eaf_e) or pd.isna(eaf_o):
                rows.append(HarmonizedSNP(rsid, ea_e, oa_e, float(er["beta"]), float(er["se"]),
                                          beta_out, float(or_["se"]),
                                          float(eaf_e) if not pd.isna(eaf_e) else None,
                                          float(eaf_o) if not pd.isna(eaf_o) else None,
                                          action="ambiguous_palindrome"))
                continue
            # if EAFs are on opposite sides of 0.5 it's likely the wrong strand
            same_side = (eaf_e - 0.5) * (eaf_o - 0.5) > 0
            in_window = abs(eaf_e - 0.5) < palindrome_eaf_threshold or abs(eaf_o - 0.5) < palindrome_eaf_threshold
            if not same_side or in_window:
                rows.append(HarmonizedSNP(rsid, ea_e, oa_e, float(er["beta"]), float(er["se"]),
                                          beta_out, float(or_["se"]),
                                          float(eaf_e), float(eaf_o),
                                          action="ambiguous_palindrome"))
                continue

        rows.append(HarmonizedSNP(
            rsid=rsid, ea=ea_e, oa=oa_e,
            beta_exp=float(er["beta"]), se_exp=float(er["se"]),
            beta_out=beta_out, se_out=float(or_["se"]),
            eaf_exp=float(er["eaf"]) if "eaf" in er else None,
            eaf_out=float(or_["eaf"]) if "eaf" in or_ else None,
            action=action,
        ))

    return pd.DataFrame([s.__dict__ for s in rows])


# -----------------------------------------------------------------------------
# Estimators
# -----------------------------------------------------------------------------
@dataclass
class MRResult:
    method: str
    estimate: float
    se: float
    pvalue: float
    n_snps: int
    extra: dict = field(default_factory=dict)

    def or_ci(self) -> tuple[float, float, float]:
        """exp(estimate) and 95% CI bounds — for binary outcomes."""
        z = 1.959963984540054
        return (
            float(np.exp(self.estimate)),
            float(np.exp(self.estimate - z * self.se)),
            float(np.exp(self.estimate + z * self.se)),
        )


def _f_statistic(beta_exp: np.ndarray, se_exp: np.ndarray) -> np.ndarray:
    """Per-SNP F-statistic, F = (beta/se)^2. Rule of thumb F > 10 = strong."""
    return (beta_exp / se_exp) ** 2


def ivw(beta_exp: np.ndarray, se_exp: np.ndarray,
        beta_out: np.ndarray, se_out: np.ndarray) -> MRResult:
    """Inverse-variance weighted; no intercept; uses 2nd-order weights (1/se_out^2)."""
    w = 1.0 / (se_out ** 2)
    # Weighted least squares through the origin: slope = Σ(w·βx·βy) / Σ(w·βx²)
    slope = float(np.sum(w * beta_exp * beta_out) / np.sum(w * beta_exp ** 2))
    # Standard error
    n = len(beta_exp)
    se = float(np.sqrt(1.0 / np.sum(w * beta_exp ** 2)))
    # Cochran's Q for heterogeneity
    q_terms = w * (beta_out - slope * beta_exp) ** 2
    q = float(q_terms.sum())
    df = n - 1
    q_p = float(1 - stats.chi2.cdf(q, df)) if df > 0 else float("nan")
    # Random-effects SE (only if Q > df)
    re_se = float(se * np.sqrt(max(q / df, 1.0))) if df > 0 else se
    z = slope / re_se if re_se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else float("nan")
    return MRResult(
        method="IVW (random-effects)",
        estimate=slope, se=re_se, pvalue=p, n_snps=n,
        extra={"fixed_se": se, "Q": q, "Q_df": df, "Q_p": q_p},
    )


def weighted_median(beta_exp: np.ndarray, se_exp: np.ndarray,
                    beta_out: np.ndarray, se_out: np.ndarray,
                    n_boot: int = 1000, seed: int = 42) -> MRResult:
    """Weighted-median estimator (Bowden 2016).

    Robust to up to 50% of weight coming from invalid instruments.
    SE via parametric bootstrap.
    """
    n = len(beta_exp)
    ratio = beta_out / beta_exp
    w = (se_out / beta_exp) ** -2  # 2nd-order weights, equivalent to 1/var(ratio)

    def _wmed(r: np.ndarray, w_: np.ndarray) -> float:
        order = np.argsort(r)
        rs = r[order]
        ws = w_[order]
        cumw = np.cumsum(ws) - 0.5 * ws
        cumw /= cumw[-1]
        # piecewise linear interpolation at 0.5
        idx = np.searchsorted(cumw, 0.5)
        if idx == 0:
            return float(rs[0])
        if idx >= n:
            return float(rs[-1])
        x0, x1 = cumw[idx - 1], cumw[idx]
        y0, y1 = rs[idx - 1], rs[idx]
        return float(y0 + (0.5 - x0) / (x1 - x0) * (y1 - y0))

    point = _wmed(ratio, w)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        bx = rng.normal(beta_exp, se_exp)
        by = rng.normal(beta_out, se_out)
        # avoid divide-by-zero in resampled betas
        ok = np.abs(bx) > 1e-6
        if ok.sum() < 3:
            boot[b] = np.nan
            continue
        r = by[ok] / bx[ok]
        wb = (se_out[ok] / bx[ok]) ** -2
        boot[b] = _wmed(r, wb)
    se = float(np.nanstd(boot, ddof=1))
    z = point / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else float("nan")
    return MRResult(method="Weighted median", estimate=point, se=se, pvalue=p, n_snps=n)


def egger(beta_exp: np.ndarray, se_exp: np.ndarray,
          beta_out: np.ndarray, se_out: np.ndarray) -> tuple[MRResult, MRResult]:
    """MR-Egger: weighted linear regression with an intercept.

    Returns (slope, intercept). Intercept is the directional pleiotropy test;
    p < 0.05 suggests horizontal pleiotropy.
    """
    n = len(beta_exp)
    # Orient all exposures positive (per Bowden 2015).
    sign = np.sign(beta_exp)
    bx = beta_exp * sign
    by = beta_out * sign

    w = 1.0 / (se_out ** 2)
    X = np.column_stack([np.ones(n), bx])
    W = np.diag(w)
    # WLS: beta = (X' W X)^{-1} X' W y
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ by
    coef = np.linalg.solve(XtWX, XtWy)
    intercept, slope = float(coef[0]), float(coef[1])
    resid = by - X @ coef
    # residual variance / dispersion (use max(1, ...) — random-effects style)
    df = n - 2
    sigma2 = float((w * resid ** 2).sum() / df) if df > 0 else 1.0
    cov = sigma2 * np.linalg.inv(XtWX)
    se_intercept = float(np.sqrt(cov[0, 0]))
    se_slope = float(np.sqrt(cov[1, 1]))
    z_s = slope / se_slope if se_slope > 0 else float("nan")
    p_s = float(2 * stats.norm.sf(abs(z_s))) if not np.isnan(z_s) else float("nan")
    z_i = intercept / se_intercept if se_intercept > 0 else float("nan")
    p_i = float(2 * stats.norm.sf(abs(z_i))) if not np.isnan(z_i) else float("nan")
    return (
        MRResult(method="MR-Egger slope", estimate=slope, se=se_slope, pvalue=p_s, n_snps=n,
                 extra={"sigma2": sigma2}),
        MRResult(method="MR-Egger intercept", estimate=intercept, se=se_intercept,
                 pvalue=p_i, n_snps=n,
                 extra={"interpretation": "p<0.05 → directional pleiotropy"}),
    )


def wald_ratios(beta_exp: np.ndarray, se_exp: np.ndarray,
                beta_out: np.ndarray, se_out: np.ndarray) -> pd.DataFrame:
    """Per-SNP Wald ratio + delta-method SE. Useful for forest plots."""
    ratio = beta_out / beta_exp
    # delta-method, ignoring covariance (two-sample)
    se = np.sqrt((se_out / beta_exp) ** 2 + (beta_out ** 2 * se_exp ** 2) / (beta_exp ** 4))
    return pd.DataFrame({"ratio": ratio, "se": se, "F": _f_statistic(beta_exp, se_exp)})


# -----------------------------------------------------------------------------
# Convenience: run the standard battery on a harmonized DataFrame
# -----------------------------------------------------------------------------
def run_mr(harmonized: pd.DataFrame, outcome_is_binary: bool = True) -> dict:
    """Run IVW, WM, Egger on a harmonized table.

    Drops SNPs marked ambiguous_palindrome or allele_mismatch.
    """
    df = harmonized[~harmonized["action"].isin(["ambiguous_palindrome", "allele_mismatch"])]
    df = df.dropna(subset=["beta_exp", "se_exp", "beta_out", "se_out"])
    if len(df) < 3:
        return {"error": f"Only {len(df)} valid SNPs after harmonization; need ≥3.",
                "harmonized": harmonized}
    bx = df["beta_exp"].to_numpy()
    sx = df["se_exp"].to_numpy()
    by = df["beta_out"].to_numpy()
    sy = df["se_out"].to_numpy()

    ivw_r = ivw(bx, sx, by, sy)
    wm_r = weighted_median(bx, sx, by, sy)
    eg_slope, eg_int = egger(bx, sx, by, sy)
    per_snp = wald_ratios(bx, sx, by, sy)
    per_snp.insert(0, "rsid", df["rsid"].values)
    f_stats = _f_statistic(bx, sx)

    return {
        "n_snps": int(len(df)),
        "mean_F": float(np.mean(f_stats)),
        "min_F": float(np.min(f_stats)),
        "ivw": ivw_r,
        "weighted_median": wm_r,
        "egger_slope": eg_slope,
        "egger_intercept": eg_int,
        "per_snp": per_snp,
        "harmonized": df.reset_index(drop=True),
        "outcome_is_binary": outcome_is_binary,
    }


def interpret_direction(estimate: float, outcome_is_binary: bool = True) -> str:
    """Plain-English read-out of a causal estimate.

    For binary outcomes: positive log-OR = protein-raising → risk-raising →
    therapeutic strategy is *inhibition*. Negative log-OR = *activation*.
    """
    if not np.isfinite(estimate):
        return "indeterminate"
    if abs(estimate) < 0.02:
        return "no causal effect detected"
    if outcome_is_binary:
        if estimate > 0:
            return (
                "Higher protein → higher disease risk. "
                "Therapeutic strategy: **inhibit / antagonize** the target."
            )
        return (
            "Higher protein → lower disease risk. "
            "Therapeutic strategy: **activate / agonize** the target."
        )
    return ("Higher protein → higher trait" if estimate > 0
            else "Higher protein → lower trait")
