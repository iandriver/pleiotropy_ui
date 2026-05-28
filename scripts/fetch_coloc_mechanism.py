"""Fetch the Open Targets colocalisation dataset and derive a per-gene
*regulatory-mechanism* feature table.

Why this exists
---------------
The §12 multivariate model needs a "mechanism" axis: is a gene's
disease association *regulatory* (the GWAS signal colocalises with a
molecular-QTL — the disease acts through the gene's expression /
splicing / abundance) as opposed to a bare statistical association?

The OT ``colocalisation`` dataset (~16 GB, 200 parquet parts, ~218 M
credible-set-pair rows) holds every pairwise colocalisation. ~77% are
GWAS-GWAS pairs we don't need. This script streams the parts
**column-projected** (only the 5 columns we use), keeps just the
GWAS↔QTL pairs above an H4 threshold, never stores a full part, then
joins both sides to genes:

  * left  (GWAS credible set) -> gene via ``l2g_prediction`` (top L2G gene)
  * right (QTL credible set)  -> gene via ``credible_set`` -> ``study.geneId``

and keeps the *same-gene* (cis) colocalisations — the GWAS signal
acting through that gene's own molecular QTL.

Output: ``data/coloc_mechanism.parquet`` keyed by ``geneId`` with
per-QTL-type coloc counts, a ``has_*_coloc`` flag set, and the best H4.

Usage:
    python scripts/fetch_coloc_mechanism.py
    python scripts/fetch_coloc_mechanism.py --h4-min 0.8
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

COLOC_BASE = ("https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/"
              "26.03/output/colocalisation/")
QTL_TYPES = ("eqtl", "pqtl", "sqtl", "tuqtl", "sceqtl")
# Only these columns are read from each remote parquet part (projection).
KEEP_COLS = ["leftStudyLocusId", "rightStudyLocusId",
             "rightStudyType", "h4", "colocalisationMethod"]


def _latest_release_dir() -> Path:
    rels = sorted(_DATA_DIR.glob("ot_release_*"))
    if not rels:
        sys.exit(f"ERROR: no OT release under {_DATA_DIR}.")
    return rels[-1]


def list_coloc_parts() -> list[str]:
    """Scrape the colocalisation directory for its parquet part files."""
    r = requests.get(COLOC_BASE, timeout=60)
    r.raise_for_status()
    parts = sorted(set(re.findall(r'href="(part-[^"]+\.parquet)"', r.text)))
    if not parts:
        sys.exit("ERROR: no parquet parts found at the colocalisation URL.")
    return parts


def _download_part(url: str, dest: Path,
                   timeout: int = 120, retries: int = 4) -> bool:
    """Download one part with a per-read timeout and retries.

    A plain ``pd.read_parquet(url)`` over HTTP has no network timeout, so
    a stalled EBI connection hangs the whole run forever. Streaming the
    bytes through ``requests`` with ``timeout`` makes a stall raise
    (and retry) instead of hanging.
    """
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=2 ** 20):
                        if chunk:
                            fh.write(chunk)
            return True
        except (requests.RequestException, OSError) as exc:
            dest.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            print(f"  WARN: download failed after {retries} retries — "
                  f"{url}: {exc}", flush=True)
            return False
    return False


def stream_filter_parts(parts: list[str], tmp_dir: Path,
                        h4_min: float) -> pd.DataFrame:
    """Download each part (timed + retried), keep GWAS↔QTL coloc rows
    with H4 >= h4_min, discard the raw part. Resumable — a part's
    filtered output is cached, so a re-run skips finished parts and
    retries any that failed."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    qtl = set(QTL_TYPES)
    raw = tmp_dir / "_raw_part.tmp"
    done = 0
    for i, part in enumerate(parts, 1):
        out_c = tmp_dir / f"{part}.filt.parquet"
        if out_c.exists() and out_c.stat().st_size > 0:
            done += 1
            continue
        if not _download_part(COLOC_BASE + part, raw):
            continue  # resumable — a later re-run retries this part
        df = pd.read_parquet(raw, columns=KEEP_COLS)
        keep = df[df["rightStudyType"].isin(qtl) & (df["h4"] >= h4_min)]
        keep.to_parquet(out_c, index=False)
        raw.unlink(missing_ok=True)
        done += 1
        print(f"  [{done:>3}/{len(parts)}] {part}  "
              f"kept {len(keep):,} of {len(df):,}", flush=True)

    have = [tmp_dir / f"{p}.filt.parquet" for p in parts]
    have = [p for p in have if p.exists() and p.stat().st_size > 0]
    if len(have) < len(parts):
        print(f"  NOTE: {len(parts) - len(have)} part(s) still missing — "
              "re-run to retry them.")
    return pd.concat([pd.read_parquet(p) for p in have], ignore_index=True)


def per_gene_mechanism(coloc: pd.DataFrame, release_dir: Path) -> pd.DataFrame:
    """Join both sides of each GWAS↔QTL coloc to genes and aggregate.

    Keeps same-gene (cis) colocalisations — the GWAS signal acting
    through that gene's own QTL — and counts them per QTL type per gene.
    """
    # left side: GWAS credible set -> its top-L2G gene
    l2g = pd.read_parquet(release_dir / "l2g_prediction",
                          columns=["studyLocusId", "geneId", "score"])
    left_gene = (l2g.sort_values("score", ascending=False)
                    .drop_duplicates("studyLocusId", keep="first")
                    .rename(columns={"studyLocusId": "leftStudyLocusId",
                                     "geneId": "left_gene"})
                    [["leftStudyLocusId", "left_gene"]])

    # right side: QTL credible set -> its study -> the QTL's gene
    cs = pd.read_parquet(release_dir / "credible_set",
                         columns=["studyLocusId", "studyId", "studyType"])
    cs_qtl = cs[cs["studyType"].isin(QTL_TYPES)]
    study = pd.read_parquet(release_dir / "study",
                            columns=["studyId", "geneId"])
    right_gene = (cs_qtl.merge(study, on="studyId", how="left")
                        .rename(columns={"studyLocusId": "rightStudyLocusId",
                                         "geneId": "right_gene"})
                        [["rightStudyLocusId", "right_gene"]])

    m = (coloc.merge(left_gene, on="leftStudyLocusId", how="inner")
              .merge(right_gene, on="rightStudyLocusId", how="inner"))
    print(f"  GWAS↔QTL coloc rows with both genes resolved: {len(m):,}")
    cis = m[m["left_gene"] == m["right_gene"]].copy()
    cis["geneId"] = cis["left_gene"]
    print(f"  same-gene (cis) colocalisations: {len(cis):,}")

    # per-gene aggregation
    rows = []
    for gene, sub in cis.groupby("geneId"):
        rec = {"geneId": gene,
               "n_qtl_coloc": len(sub),
               "max_h4_coloc": float(sub["h4"].max())}
        for qt in QTL_TYPES:
            rec[f"n_{qt}_coloc"] = int((sub["rightStudyType"] == qt).sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    for qt in QTL_TYPES:
        out[f"has_{qt}_coloc"] = (out[f"n_{qt}_coloc"] > 0).astype(int)
    out["has_qtl_coloc"] = 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h4-min", type=float, default=0.8,
                   help="Minimum COLOC H4 posterior to count a "
                        "colocalisation. Default 0.8.")
    p.add_argument("--tmp-dir", default=str(_DATA_DIR / "coloc_tmp"),
                   help="Scratch dir for per-part filtered output "
                        "(resumable).")
    p.add_argument("--out", default=str(_DATA_DIR / "coloc_mechanism.parquet"),
                   help="Output per-gene mechanism parquet.")
    args = p.parse_args(argv)

    release_dir = _latest_release_dir()
    print(f"OT release: {release_dir}")

    print("Listing colocalisation parts …")
    parts = list_coloc_parts()
    print(f"  {len(parts)} parquet parts")

    print(f"Streaming + filtering (GWAS↔QTL, H4 ≥ {args.h4_min}) …")
    coloc = stream_filter_parts(parts, Path(args.tmp_dir), args.h4_min)
    print(f"  {len(coloc):,} GWAS↔QTL colocalisations above threshold")

    print("Joining both sides to genes …")
    mech = per_gene_mechanism(coloc, release_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mech.to_parquet(out_path, index=False)
    print(f"\nWrote {len(mech):,} genes -> {out_path}")
    print(f"  genes with eQTL coloc:  {int(mech['has_eqtl_coloc'].sum()):,}")
    print(f"  genes with pQTL coloc:  {int(mech['has_pqtl_coloc'].sum()):,}")
    print(f"  genes with sQTL coloc:  {int(mech['has_sqtl_coloc'].sum()):,}")
    print(f"  median n_qtl_coloc / gene: {mech['n_qtl_coloc'].median():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
