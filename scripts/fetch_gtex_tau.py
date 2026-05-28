"""Fetch the GTEx v8 bulk median-TPM matrix and compute a continuous
tissue-specificity index (tau) per gene.

Why this exists
---------------
The §12 multivariate model's mediation test asks whether pleiotropy's
association with drug-safety risk is *explained by* expression breadth.
The local Open Targets ``tissueSpecificity`` / ``tissueDistribution``
features are coarse 4-level scores and the mediation came back ~0%.
This pulls the real thing: GTEx v8 median TPM across 54 tissues, from
which we compute the Yanai tau index — a continuous 0..1
tissue-specificity measure (0 = uniformly/broadly expressed,
1 = expressed in a single tissue).

Output: ``data/gtex_v8_tissue_specificity.parquet`` keyed by ``geneId``
(unversioned Ensembl), with columns ``gtex_tau``,
``gtex_n_tissues_expr`` (tissues with TPM > 1), ``gtex_max_log_tpm``,
``gtex_max_tissue``.

The download is a single ~6 MB gzipped GCT — cheap, one-time.

Usage:
    python scripts/fetch_gtex_tau.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# GTEx v8 gene median TPM by tissue (GCT). The canonical, stable release.
GTEX_V8_MEDIAN_TPM_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
)


def compute_tau(median_tpm: pd.DataFrame, tissue_cols: list[str]) -> pd.DataFrame:
    """Yanai tissue-specificity index tau, per gene.

    tau = Σ(1 − x̂_i) / (n − 1), where x̂_i = x_i / max_i(x_i), computed
    on log2(TPM + 1) expression. tau ∈ [0, 1]: 0 = uniformly expressed
    across all tissues (housekeeping-like), 1 = expressed in exactly one
    tissue. Genes not expressed anywhere get tau = NaN.
    """
    expr = median_tpm[tissue_cols].to_numpy(dtype=float)
    log_expr = np.log2(expr + 1.0)
    n = len(tissue_cols)

    row_max = log_expr.max(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xhat = log_expr / row_max[:, None]
    tau = np.nansum(1.0 - xhat, axis=1) / (n - 1)
    tau[row_max == 0.0] = np.nan  # not expressed anywhere → undefined

    max_idx = np.argmax(log_expr, axis=1)
    out = pd.DataFrame({
        "geneId": median_tpm["Name"].str.split(".").str[0].to_numpy(),
        "gtex_tau": tau,
        "gtex_n_tissues_expr": (expr > 1.0).sum(axis=1),
        "gtex_max_log_tpm": row_max,
        "gtex_max_tissue": [tissue_cols[i] for i in max_idx],
    })
    # Collapse any duplicate Ensembl ids (rare PAR genes) — keep the
    # most-expressed copy.
    out = (out.sort_values("gtex_max_log_tpm", ascending=False)
              .drop_duplicates("geneId", keep="first")
              .reset_index(drop=True))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=GTEX_V8_MEDIAN_TPM_URL,
                   help="GTEx median-TPM GCT URL.")
    p.add_argument("--out",
                   default=str(_DATA_DIR / "gtex_v8_tissue_specificity.parquet"),
                   help="Output parquet path.")
    args = p.parse_args(argv)

    print(f"Reading GTEx median-TPM GCT:\n  {args.url}")
    # GCT format: line 1 '#1.2', line 2 '<n_genes>\t<n_tissues>',
    # line 3 column header, then the data — so skip the first 2 lines.
    try:
        gct = pd.read_csv(args.url, sep="\t", skiprows=2, compression="gzip")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"ERROR: could not read GTEx GCT — {exc}")

    tissue_cols = [c for c in gct.columns if c not in ("Name", "Description")]
    print(f"  {len(gct):,} genes x {len(tissue_cols)} tissues")

    tau = compute_tau(gct, tissue_cols)
    n_ok = tau["gtex_tau"].notna().sum()
    print(f"Computed tau for {n_ok:,} expressed genes "
          f"({n_ok / len(tau):.0%} of {len(tau):,}).")
    print(f"  tau distribution: "
          f"median {tau['gtex_tau'].median():.3f}, "
          f"broad (tau<0.3) {int((tau['gtex_tau'] < 0.3).sum()):,}, "
          f"specific (tau>0.85) {int((tau['gtex_tau'] > 0.85).sum()):,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tau.to_parquet(out_path, index=False)
    print(f"Wrote {len(tau):,} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
