"""Paper-methods functions for the pleiotropy analysis.

Two groups of reusable functions, lifted out of the notebooks so the
analysis is importable, testable, and consistent across notebooks:

1. **Therapeutic-area assignment** (manuscript §"Therapeutic area
   assignment to studies"). Maps each disease EFO to one of 22
   therapeutic areas by propagating EFO ontology ancestry to the
   Supplementary-Table-9 root terms, resolving multi-TA diseases by the
   ST9 priority order. ``measurement`` traits are flagged for exclusion
   from the TA-based pleiotropy count; unmapped diseases fall to
   ``other``.

2. **Sweet-spot modelling.** Tools to test the manuscript's central
   claim — that intermediate pleiotropy is a therapeutic "sweet spot",
   i.e. drug-clinical-success rate is an inverted-U in pleiotropy:
   * :func:`success_rate_curve` — binned success rate + Wilson CIs.
   * :func:`fit_quadratic_sweet_spot` — logistic regression with a
     quadratic pleiotropy term; a significant *negative* quadratic
     coefficient is direct evidence of an inverted-U, and the vertex
     gives the estimated peak-pleiotropy location with a CI.
   * :func:`categorical_contrasts` — intermediate-vs-both-tails odds
     ratios.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ===========================================================================
# §1 — Therapeutic-area assignment (Supplementary Table 9)
# ===========================================================================
# ST9 therapeutic-area root terms in PRIORITY ORDER (row 0 = highest
# priority). A disease whose EFO ancestry reaches several roots is assigned
# the highest-priority one — e.g. ovarian cancer -> "cancer or benign tumor"
# rather than "reproductive system" or "endocrine system disease".
ST9_THERAPEUTIC_AREAS: list[tuple[str, str]] = [
    ("EFO_0001444",  "measurement"),
    ("MONDO_0045024", "cancer or benign tumor"),
    ("OTAR_0000018", "genetic, familial or congenital disease"),
    ("EFO_0005741",  "infectious disease"),
    ("OTAR_0000009", "injury, poisoning or other complication"),
    ("OTAR_0000014", "pregnancy or perinatal disease"),
    ("MONDO_0024458", "disorder of visual system"),
    ("EFO_0000319",  "cardiovascular disease"),
    ("EFO_0009605",  "pancreas disease"),
    ("EFO_0010282",  "gastrointestinal disease"),
    ("OTAR_0000017", "reproductive system or breast disease"),
    ("EFO_0010285",  "integumentary system disease"),
    ("EFO_0001379",  "endocrine system disease"),
    ("OTAR_0000010", "respiratory or thoracic disease"),
    ("EFO_0009690",  "urinary system disease"),
    ("OTAR_0000006", "musculoskeletal or connective tissue disease"),
    ("MONDO_0021205", "disorder of ear"),
    ("EFO_0000540",  "immune system disease"),
    ("EFO_0005803",  "hematologic disease"),
    ("EFO_0000618",  "nervous system disease"),
    ("MONDO_0002025", "psychiatric disorder"),
    ("OTAR_0000020", "nutritional or metabolic disease"),
    ("EFO_0003765",  "sign or symptom"),
]
_TA_RANK: dict[str, int] = {efo: i for i, (efo, _) in enumerate(ST9_THERAPEUTIC_AREAS)}
_TA_NAME: dict[str, str] = dict(ST9_THERAPEUTIC_AREAS)
MEASUREMENT_TA = "measurement"
OTHER_TA = "other"


def _ancestor_set(ancestors) -> set[str]:
    """Coerce a ``disease.ancestors`` cell (list / np.ndarray / None) to a set."""
    if ancestors is None:
        return set()
    if isinstance(ancestors, np.ndarray):
        return set(ancestors.tolist())
    try:
        return set(ancestors)
    except TypeError:
        return set()


def assign_therapeutic_area(disease_id: str, ancestors) -> str:
    """Map one disease to its single therapeutic area.

    ``ancestors`` is the disease's EFO ontology ancestry (the
    ``disease.ancestors`` field). The disease itself is included in the
    lookup so a TA root term maps to itself. Returns the ST9 TA name, or
    ``"other"`` if no root is reached.
    """
    anc = _ancestor_set(ancestors) | {disease_id}
    hits = [r for r in anc if r in _TA_RANK]
    if not hits:
        return OTHER_TA
    return _TA_NAME[min(hits, key=_TA_RANK.__getitem__)]


def build_ta_map(disease_df: pd.DataFrame) -> pd.DataFrame:
    """Build the disease -> therapeutic-area lookup for a whole release.

    ``disease_df`` must have ``id`` and ``ancestors`` columns (the OT
    ``disease`` parquet). Returns a DataFrame with columns
    ``diseaseId``, ``ta``, ``is_measurement``.
    """
    anc_map = dict(zip(disease_df["id"], disease_df["ancestors"]))
    out = pd.DataFrame({"diseaseId": disease_df["id"].to_numpy()})
    out["ta"] = [
        assign_therapeutic_area(d, anc_map.get(d)) for d in out["diseaseId"]
    ]
    out["is_measurement"] = out["ta"] == MEASUREMENT_TA
    return out


# ===========================================================================
# Pleiotropy buckets
# ===========================================================================
BUCKET_ORDER: list[str] = ["none", "specific", "intermediate", "highly_pleiotropic"]
BUCKET_COLOR: dict[str, str] = {
    "none":               "#bdbdbd",
    "specific":           "#27ae60",
    "intermediate":       "#e3b505",
    "highly_pleiotropic": "#c0392b",
}


def bucket_by_n_tas(n: int | float) -> str:
    """Therapeutic-area pleiotropy bucket — the manuscript's 2-5-TA
    "sweet spot" defines the intermediate band."""
    if n is None or (isinstance(n, float) and np.isnan(n)) or n == 0:
        return "none"
    if n == 1:
        return "specific"
    if 2 <= n <= 5:
        return "intermediate"
    return "highly_pleiotropic"


def bucket_by_n_diseases(n: int | float) -> str:
    """Legacy bucket on the raw unique-EFO-disease count (duplicative;
    kept for back-compatibility / comparison)."""
    if n is None or (isinstance(n, float) and np.isnan(n)) or n == 0:
        return "none"
    if n == 1:
        return "specific"
    if 2 <= n <= 20:
        return "intermediate"
    return "highly_pleiotropic"


# ===========================================================================
# Sweet-spot modelling
# ===========================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    More reliable than the normal approximation in the small-cell and
    near-0/near-1 regimes the pleiotropy tails fall into.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def success_rate_curve(
    df: pd.DataFrame,
    pleiotropy_col: str,
    outcome_col: str,
    by: str | None = None,
    max_x: int | None = None,
    min_cell: int = 10,
) -> pd.DataFrame:
    """Per-pleiotropy-value success rate with Wilson 95% CIs.

    For each distinct value of ``pleiotropy_col`` (optionally split by the
    categorical ``by`` column), compute the fraction of ``outcome_col`` ==
    True and its Wilson interval. Cells with fewer than ``min_cell`` genes
    are dropped (too noisy to plot). This is the descriptive view of the
    inverted-U.
    """
    d = df[[pleiotropy_col, outcome_col] + ([by] if by else [])].copy()
    d = d.dropna(subset=[pleiotropy_col, outcome_col])
    if max_x is not None:
        d = d[d[pleiotropy_col] <= max_x]
    group_cols = [pleiotropy_col] + ([by] if by else [])
    rows = []
    for keys, sub in d.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        k = int(sub[outcome_col].astype(bool).sum())
        if n < min_cell:
            continue
        lo, hi = wilson_ci(k, n)
        row = {pleiotropy_col: keys[0], "n": n, "k": k, "rate": k / n,
               "ci_lo": lo, "ci_hi": hi}
        if by:
            row[by] = keys[1]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def binned_outcome_rate(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Outcome rate (+ Wilson 95% CI) across bins of any feature.

    A feature with few distinct values (e.g. a 0/1 flag, or a small
    integer count) is grouped by value; a continuous feature is split
    into ``n_bins`` quantile bins (ranked, so ties don't collapse).
    Returns ``x`` (bin-median feature value), ``n``, ``k``, ``rate``,
    ``ci_lo``, ``ci_hi`` — ready to plot as an axis-vs-outcome curve.
    """
    d = df[[feature, outcome]].dropna().copy()
    d[outcome] = d[outcome].astype(int)
    if d[feature].nunique() <= n_bins:
        d["_grp"] = d[feature]
    else:
        d["_grp"] = pd.qcut(d[feature].rank(method="first"),
                            n_bins, labels=False)
    rows = []
    for _, sub in d.groupby("_grp", observed=True):
        n = len(sub)
        k = int(sub[outcome].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"x": float(sub[feature].median()), "n": n, "k": k,
                     "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def fit_quadratic_sweet_spot(
    df: pd.DataFrame,
    outcome: str,
    pleiotropy_col: str = "n_TAs",
    covariates: Sequence[str] | None = None,
) -> dict:
    """Logistic regression with a quadratic pleiotropy term.

    Fits ``logit(outcome) ~ x + x² (+ covariates)`` where ``x`` is
    ``pleiotropy_col``. The test for a therapeutic sweet spot is:

      * the quadratic coefficient β₂ is **negative** (concave — an
        inverted-U rather than a monotone trend), and
      * β₂ is statistically significant.

    The vertex ``x* = -β₁ / (2·β₂)`` estimates the peak-pleiotropy
    location; its 95% CI is obtained by the delta method on the
    (β₁, β₂) covariance.

    Returns a dict: ``model`` (fitted statsmodels result), ``beta1``,
    ``beta2``, ``beta2_pvalue``, ``concave``, ``vertex``, ``vertex_se``,
    ``vertex_ci``, ``n``, ``pseudo_r2``.
    """
    import statsmodels.formula.api as smf

    cov = list(covariates or [])
    d = df[[outcome, pleiotropy_col, *cov]].copy()
    d = d.dropna()
    d["_x"] = d[pleiotropy_col].astype(float)
    d["_x2"] = d["_x"] ** 2
    d["_y"] = d[outcome].astype(int)

    rhs = ["_x", "_x2", *cov]
    formula = "_y ~ " + " + ".join(rhs)
    res = smf.logit(formula, data=d).fit(disp=0)

    b1 = float(res.params["_x"])
    b2 = float(res.params["_x2"])
    p2 = float(res.pvalues["_x2"])

    vertex = -b1 / (2 * b2) if b2 != 0 else float("nan")
    # Delta method: Var(x*) for x* = -b1/(2 b2).
    #   ∂x*/∂b1 = -1/(2 b2);  ∂x*/∂b2 =  b1/(2 b2²)
    cm = res.cov_params()
    v11 = float(cm.loc["_x", "_x"])
    v22 = float(cm.loc["_x2", "_x2"])
    v12 = float(cm.loc["_x", "_x2"])
    if b2 != 0:
        g1 = -1.0 / (2 * b2)
        g2 = b1 / (2 * b2 * b2)
        var_v = g1 * g1 * v11 + g2 * g2 * v22 + 2 * g1 * g2 * v12
        se_v = float(np.sqrt(var_v)) if var_v > 0 else float("nan")
    else:
        se_v = float("nan")
    vertex_ci = (vertex - 1.96 * se_v, vertex + 1.96 * se_v)

    return {
        "model": res,
        "formula": formula,
        "beta1": b1,
        "beta2": b2,
        "beta2_pvalue": p2,
        "concave": b2 < 0,
        "vertex": vertex,
        "vertex_se": se_v,
        "vertex_ci": vertex_ci,
        "n": int(len(d)),
        "pseudo_r2": float(res.prsquared),
    }


def zscore(s: pd.Series) -> pd.Series:
    """Standardise a numeric series (mean 0, sd 1); NaNs preserved."""
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s - s.mean()


def multivariate_logit(
    df: pd.DataFrame,
    outcome: str,
    features: Sequence[str],
    standardize: bool = True,
) -> dict:
    """Multivariate logistic regression of a binary outcome on many features.

    With ``standardize=True`` every feature is z-scored first, so the
    coefficients are directly comparable — ``OR_per_SD`` is the odds-ratio
    for a one-standard-deviation increase. Returns a dict with ``model``
    (fitted result) and ``table`` — a tidy per-feature DataFrame
    (``feature``, ``coef``, ``OR_per_SD``, ``z``, ``pvalue``) sorted by
    descending |z|.
    """
    import statsmodels.formula.api as smf

    d = df[[outcome, *features]].copy()
    d[outcome] = d[outcome].astype(int)
    used = []
    for f in features:
        col = f"z_{f}" if standardize else f
        d[col] = zscore(d[f]) if standardize else pd.to_numeric(d[f], errors="coerce")
        used.append(col)
    d = d.dropna(subset=[outcome, *used])

    res = smf.logit(f"{outcome} ~ " + " + ".join(used), data=d).fit(disp=0)
    rows = []
    for f, col in zip(features, used):
        rows.append({
            "feature": f,
            "coef": float(res.params[col]),
            "OR_per_SD": float(np.exp(res.params[col])),
            "z": float(res.tvalues[col]),
            "pvalue": float(res.pvalues[col]),
        })
    table = (pd.DataFrame(rows)
             .sort_values("z", key=lambda s: s.abs(), ascending=False)
             .reset_index(drop=True))
    return {"model": res, "table": table,
            "pseudo_r2": float(res.prsquared), "n": int(len(d))}


def mediation(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    covariates: Sequence[str] | None = None,
    standardize: bool = True,
) -> dict:
    """Single-mediator analysis for a binary ``outcome``.

    Tests whether ``mediator`` explains the ``treatment`` -> ``outcome``
    association (e.g. does expression breadth mediate pleiotropy ->
    safety). All variables are z-scored by default so paths are
    comparable.

    Returns: ``total`` (treatment coef in outcome~treatment), ``direct``
    (treatment coef in outcome~treatment+mediator), ``a_path`` (mediator
    ~ treatment, OLS), ``b_path`` (mediator coef in the direct model),
    ``indirect_ab`` (a·b), ``indirect_diff`` (total-direct), and
    ``prop_mediated`` ((total-direct)/total).

    Note: for a logistic outcome the difference method and the product
    method don't coincide exactly (non-collapsibility); both are
    reported, and ``prop_mediated`` uses the difference method — the
    standard reported quantity.
    """
    import statsmodels.formula.api as smf

    cov = list(covariates or [])
    allv = [outcome, treatment, mediator, *cov]
    d = df[allv].copy()
    d[outcome] = d[outcome].astype(int)
    t = f"z_{treatment}" if standardize else treatment
    m = f"z_{mediator}" if standardize else mediator
    cz = []
    for c in [treatment, mediator, *cov]:
        col = f"z_{c}" if standardize else c
        d[col] = zscore(d[c]) if standardize else pd.to_numeric(d[c], errors="coerce")
        if c in cov:
            cz.append(col)
    d = d.dropna(subset=[outcome, t, m, *cz])

    cterm = (" + " + " + ".join(cz)) if cz else ""
    m_tot = smf.logit(f"{outcome} ~ {t}{cterm}", data=d).fit(disp=0)
    m_dir = smf.logit(f"{outcome} ~ {t} + {m}{cterm}", data=d).fit(disp=0)
    a_mod = smf.ols(f"{m} ~ {t}{cterm}", data=d).fit()

    total = float(m_tot.params[t])
    direct = float(m_dir.params[t])
    a = float(a_mod.params[t])
    b = float(m_dir.params[m])
    return {
        "n": int(len(d)),
        "total": total,
        "direct": direct,
        "a_path": a,
        "b_path": b,
        "indirect_ab": a * b,
        "indirect_diff": total - direct,
        "prop_mediated": (total - direct) / total if total != 0 else float("nan"),
    }


def categorical_contrasts(
    df: pd.DataFrame,
    outcome: str,
    bucket_col: str = "bucket_ta",
    reference: str = "intermediate",
    covariates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Odds ratios for each pleiotropy bucket vs the reference bucket.

    With ``reference="intermediate"`` the manuscript's claim is that
    **both** the ``specific`` and ``highly_pleiotropic`` ORs are below 1
    (and significant) — i.e. the intermediate bucket beats both tails,
    the discrete signature of a sweet spot.

    Returns a tidy DataFrame: ``bucket``, ``OR``, ``OR_lcl``, ``OR_ucl``,
    ``pvalue``, ``n``.
    """
    import statsmodels.formula.api as smf

    cov = list(covariates or [])
    d = df[[outcome, bucket_col, *cov]].copy().dropna()
    d["_y"] = d[outcome].astype(int)
    d["_b"] = pd.Categorical(d[bucket_col].astype(str))

    rhs = [f"C(_b, Treatment(reference='{reference}'))", *cov]
    res = smf.logit("_y ~ " + " + ".join(rhs), data=d).fit(disp=0)

    rows = []
    conf = res.conf_int()
    for name in res.params.index:
        if not name.startswith("C(_b"):
            continue
        # term looks like  C(_b, Treatment(...))[T.<bucket>]
        bucket = name.split("[T.", 1)[1].rstrip("]")
        beta = float(res.params[name])
        lo, hi = float(conf.loc[name, 0]), float(conf.loc[name, 1])
        rows.append({
            "bucket": bucket,
            "OR": float(np.exp(beta)),
            "OR_lcl": float(np.exp(lo)),
            "OR_ucl": float(np.exp(hi)),
            "pvalue": float(res.pvalues[name]),
            "n": int((d[bucket_col].astype(str) == bucket).sum()),
        })
    out = pd.DataFrame(rows)
    out.attrs["reference"] = reference
    out.attrs["n_reference"] = int((d[bucket_col].astype(str) == reference).sum())
    return out
