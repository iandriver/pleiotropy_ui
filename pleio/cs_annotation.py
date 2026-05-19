"""Credible-set lead-variant annotation (paper §"Annotation of CS lead
variants by MAF and beta rescaling").

Joins each GWAS credible-set lead variant onto:
  * the major-ancestry MAF from the local gnomAD v4.1 joint-AF table
    (produced by ``scripts/fetch_gnomad_af.py``)
  * the linked study's sample composition (binary vs quantitative, total
    sample size ``n``, case fraction ``K``)

Derives a rescaled effect size via the paper's formulas:

* Binary / logistic trait:    ``SE(β) = 1 / √(2·n·f·(1−f)·K·(1−K))``
* Quantitative / linear trait: ``SE(β) = 1 / √(2·n·f·(1−f))``
* ``Z = sign · √(χ²₁ quantile at 1−p)``  — magnitude from the reported
  p-value through the χ²₁ inverse-survival function; sign from the
  original ``beta`` (or ``zScore``) when available.
* ``β_rescaled = SE · Z``

For top-hits with no reported sign we keep the magnitude of ``Z`` /
``β_rescaled`` and set ``direction_known = False`` — those variants must
be excluded from any downstream analysis that depends on direction.

The major-ancestry population per study is chosen by the paper's
three-rule cascade against ``study.ldPopulationStructure``:

  1. single-population sample → that population;
  2. uneven mixture → the population with the largest relative sample size;
  3. evenly distributed mixture → prefer NFE, else the first reported.

Output: ``data/credible_set_annotated.parquet``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# The five populations we have gnomAD v4.1 joint-AF coverage for, matching
# every ldPopulation code that actually appears in OT 26.03 study metadata.
ALLOWED_POPS: tuple[str, ...] = ("nfe", "eas", "afr", "amr", "fin")
_AF_COLS: dict[str, str] = {pop: f"af_joint_{pop}" for pop in ALLOWED_POPS}


def _latest_release_dir(data_dir: Path) -> Path:
    rels = sorted(data_dir.glob("ot_release_*"))
    if not rels:
        sys.exit(f"ERROR: no OT release under {data_dir}.")
    return rels[-1]


# ---------------------------------------------------------------------------
# Major-ancestry population selection — paper's (1)/(2)/(3) cascade.
# ---------------------------------------------------------------------------
def select_major_population(
    pop_struct,
    allowed: Iterable[str] = ALLOWED_POPS,
) -> str | None:
    """Return the major-ancestry population code for one study.

    Implements the manuscript rule:

      (1) single population → use it.
      (2) populations with uneven proportions → the one with the largest
          relative sample size.
      (3) populations evenly distributed → prefer ``nfe``, else the first
          population reported by the authors.

    Populations not in ``allowed`` (i.e. ones we don't have gnomAD AF for)
    are dropped before the cascade runs; if nothing remains, returns
    ``None``.

    ``pop_struct`` is the ``study.ldPopulationStructure`` list:
    ``[{'ldPopulation': 'nfe', 'relativeSampleSize': 0.7}, ...]``.
    """
    if pop_struct is None:
        return None
    try:
        items = list(pop_struct)
    except TypeError:
        return None
    allowed_set = set(allowed)
    items = [d for d in items
             if d is not None and d.get("ldPopulation") in allowed_set]
    if not items:
        return None
    if len(items) == 1:
        return items[0]["ldPopulation"]

    sizes = [float(d.get("relativeSampleSize") or 0.0) for d in items]
    # Evenly distributed iff all relative sample sizes coincide.
    even = all(abs(s - sizes[0]) < 1e-9 for s in sizes)
    if even:
        for d in items:
            if d["ldPopulation"] == "nfe":
                return "nfe"
        return items[0]["ldPopulation"]
    return items[int(np.argmax(sizes))]["ldPopulation"]


# ---------------------------------------------------------------------------
# Helpers — p-value to Z, EAF picker
# ---------------------------------------------------------------------------
def p_value_from_components(mantissa: pd.Series, exponent: pd.Series) -> pd.Series:
    """Reassemble p = mantissa · 10**exponent, robust to extreme exponents.

    OT stores p-values as (mantissa, exponent) precisely so we can keep
    p ≤ 10⁻³⁰⁰ without underflowing. Returns a Float64 ``Series`` in
    [0, 1] (NaN where either component is missing).
    """
    m = pd.to_numeric(mantissa, errors="coerce")
    e = pd.to_numeric(exponent, errors="coerce")
    # Clip the mantissa to safe range and let the exponent ride.
    out = m * np.power(10.0, e)
    # Anything that overflowed to inf or under-/over-shot is clipped.
    out = out.clip(lower=0.0, upper=1.0)
    return out


def z_magnitude_from_p(p: pd.Series) -> pd.Series:
    """|Z| = √(χ²₁ inverse-CDF at 1−p) — equivalent to |Φ⁻¹(p/2)| for two-sided p."""
    p = pd.to_numeric(p, errors="coerce").clip(lower=np.finfo(float).tiny,
                                               upper=1.0)
    chi2 = stats.chi2.isf(p.to_numpy(), df=1)  # = ppf(1-p, df=1) but stable in the tail
    return pd.Series(np.sqrt(chi2), index=p.index)


def pick_eaf(df_joined: pd.DataFrame) -> pd.Series:
    """Vectorised: pick the AF column matching each row's ``major_pop``."""
    out = pd.Series(np.nan, index=df_joined.index, dtype="float64")
    for pop, col in _AF_COLS.items():
        if col not in df_joined.columns:
            continue
        mask = df_joined["major_pop"].eq(pop)
        out.loc[mask] = df_joined.loc[mask, col].astype("float64")
    return out


# ---------------------------------------------------------------------------
# Main annotation pipeline
# ---------------------------------------------------------------------------
def annotate(
    release_dir: Path,
    gnomad_af_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Compute the §2 annotation for every GWAS credible-set lead variant.

    Reads ``credible_set``, ``study``, and the local gnomAD AF table.
    Writes the annotated parquet to ``out_path`` and returns it.
    """
    print(f"Loading credible_set / study / gnomAD AF …")
    cs = pd.read_parquet(
        release_dir / "credible_set",
        columns=[
            "studyLocusId", "studyId", "variantId",
            "chromosome", "position",
            "beta", "zScore", "pValueMantissa", "pValueExponent",
            "sampleSize", "studyType",
            "effectAlleleFrequencyFromSource",
        ],
    )
    cs = cs[cs["studyType"] == "gwas"].copy()
    print(f"  {len(cs):,} GWAS credible-set rows")

    study = pd.read_parquet(
        release_dir / "study",
        columns=["studyId", "studyType",
                 "ldPopulationStructure",
                 "nCases", "nControls", "nSamples"],
    )
    study = study[study["studyType"] == "gwas"].copy()
    print(f"  {len(study):,} GWAS studies")

    af = pd.read_parquet(
        gnomad_af_path,
        columns=["variantId", "match",
                 "af_joint",
                 "af_joint_nfe", "af_joint_eas", "af_joint_afr",
                 "af_joint_amr", "af_joint_fin"],
    )
    print(f"  {len(af):,} gnomAD AF rows")

    # ----- per-study annotations -------------------------------------------
    print("Selecting major-ancestry population per study …")
    study["major_pop"] = study["ldPopulationStructure"].apply(
        select_major_population
    )
    pop_counts = study["major_pop"].value_counts(dropna=False)
    print("  major_pop distribution: "
          + ", ".join(f"{k}:{v:,}" for k, v in pop_counts.items()))

    nc = pd.to_numeric(study["nCases"], errors="coerce").fillna(0)
    nk = pd.to_numeric(study["nControls"], errors="coerce").fillna(0)
    ns = pd.to_numeric(study["nSamples"], errors="coerce").fillna(0)
    study["is_binary"] = (nc > 0) & (nk > 0)
    study["n_total"] = np.where(study["is_binary"], nc + nk, ns)
    study["K_case_fraction"] = np.where(
        study["is_binary"] & ((nc + nk) > 0),
        nc / (nc + nk),
        np.nan,
    )
    print(f"  binary studies: {int(study['is_binary'].sum()):,};  "
          f"quantitative: {int((~study['is_binary']).sum()):,}")

    # ----- join everything --------------------------------------------------
    print("Joining CS × study × gnomAD …")
    df = cs.merge(
        study[["studyId", "major_pop", "is_binary",
               "n_total", "K_case_fraction"]],
        on="studyId", how="left",
    )
    df = df.merge(af, on="variantId", how="left")

    df["eaf"] = pick_eaf(df)
    df["maf"] = np.minimum(df["eaf"], 1.0 - df["eaf"])

    # Use the credible-set's own sample size where reported, else fall
    # back to the study-level total. n_total is preferred because the
    # paper says "n is the sample size reported by the study".
    cs_n = pd.to_numeric(df["sampleSize"], errors="coerce")
    df["n_used"] = df["n_total"].where(df["n_total"] > 0, cs_n)

    # ----- standard errors --------------------------------------------------
    print("Computing rescaled SE (binary / quantitative formulas) …")
    f = df["maf"].astype("float64")
    n = df["n_used"].astype("float64")
    K = df["K_case_fraction"].astype("float64")

    common = 2.0 * n * f * (1.0 - f)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_quant = 1.0 / np.sqrt(common)
        se_binary = 1.0 / np.sqrt(common * K * (1.0 - K))
    df["se_rescaled"] = np.where(df["is_binary"], se_binary, se_quant)

    # ----- p-value → |Z| → signed Z → β_rescaled ---------------------------
    print("Computing Z from p-value via χ²₁ inverse-survival …")
    p = p_value_from_components(df["pValueMantissa"], df["pValueExponent"])
    df["p_value"] = p
    z_abs = z_magnitude_from_p(p)

    # Sign from beta (preferred) else zScore else unknown.
    sign_beta = np.sign(pd.to_numeric(df["beta"], errors="coerce").fillna(0))
    sign_z    = np.sign(pd.to_numeric(df["zScore"], errors="coerce").fillna(0))
    sign = np.where(sign_beta != 0, sign_beta, sign_z)
    df["direction_known"] = sign != 0
    df["z_score"] = sign * z_abs
    df.loc[~df["direction_known"], "z_score"] = z_abs.loc[~df["direction_known"]]

    df["beta_rescaled"] = df["se_rescaled"] * df["z_score"]

    # ----- QC flags --------------------------------------------------------
    df["has_maf"] = df["maf"].notna() & (df["maf"] > 0) & (df["maf"] < 1)
    df["has_n"]   = df["n_used"].fillna(0) > 0
    df["has_p"]   = df["p_value"].notna() & (df["p_value"] > 0)
    if "K_case_fraction" in df.columns:
        df["has_K_ok"] = ~df["is_binary"] | (
            (df["K_case_fraction"] > 0) & (df["K_case_fraction"] < 1)
        )
    else:
        df["has_K_ok"] = True
    df["annotation_ok"] = (
        df["has_maf"] & df["has_n"] & df["has_p"] & df["has_K_ok"]
        & df["se_rescaled"].notna()
        & np.isfinite(df["beta_rescaled"])
    )

    # ----- save ------------------------------------------------------------
    keep_cols = [
        "studyLocusId", "studyId", "variantId", "chromosome", "position",
        "beta", "zScore", "pValueMantissa", "pValueExponent",
        "sampleSize", "effectAlleleFrequencyFromSource",
        "major_pop", "is_binary", "n_used", "K_case_fraction",
        "match", "eaf", "maf",
        "p_value", "z_score", "se_rescaled", "beta_rescaled",
        "direction_known",
        "has_maf", "has_n", "has_p", "has_K_ok", "annotation_ok",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].rename(columns={
        "beta":  "beta_raw",
        "zScore": "z_score_raw",
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    n_total = len(out)
    n_ok    = int(out["annotation_ok"].sum())
    n_dir   = int((out["annotation_ok"] & out["direction_known"]).sum())
    print()
    print(f"Wrote {n_total:,} rows -> {out_path}")
    print(f"  annotation_ok: {n_ok:,} ({100*n_ok/n_total:.1f}%)")
    print(f"  + direction_known: {n_dir:,} ({100*n_dir/n_total:.1f}%)")
    print(f"  rejection reasons (rows failing each gate):")
    for col in ["has_maf", "has_n", "has_p", "has_K_ok"]:
        if col in out.columns:
            print(f"    {col} = False: {int((~out[col]).sum()):,}")
    print(f"  |β_rescaled| median (rows ok): "
          f"{out.loc[out['annotation_ok'], 'beta_rescaled'].abs().median():.4f}")
    print(f"  |β_rescaled| 95th pct (rows ok): "
          f"{out.loc[out['annotation_ok'], 'beta_rescaled'].abs().quantile(0.95):.4f}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--release-dir", default=None,
                   help="OT release dir. Default: latest under data/.")
    p.add_argument("--gnomad-af",
                   default=str(_DATA_DIR / "gnomad_v4.1_joint_af.parquet"),
                   help="Path to the gnomAD AF parquet from "
                        "scripts/fetch_gnomad_af.py.")
    p.add_argument("--out",
                   default=str(_DATA_DIR / "credible_set_annotated.parquet"),
                   help="Output parquet path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    release_dir = (Path(args.release_dir) if args.release_dir
                   else _latest_release_dir(_DATA_DIR))
    annotate(release_dir, Path(args.gnomad_af), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
