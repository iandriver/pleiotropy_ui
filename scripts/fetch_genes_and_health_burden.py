"""Fetch the Genes & Health 44k ExWAS gene-burden sumstats and derive a
per-gene *rare-variant burden* feature table for the §12 model.

Why this exists
---------------
The §12 multivariate safety model has a common-variant pleiotropy axis
(`n_TAs`, GWAS-derived) and a functional priors axis (GTEx τ, mouseKO,
genetic constraint). It does **not** have a rare-variant burden axis —
the kind of "is this gene tolerant of loss of function in living
humans?" signal that the Genes & Health 44k ExWAS (Kim, DeBoever,
Walter, van Heel et al. 2026, *Nat Genet* 58:821) explicitly built.

Their headline finding: among antagonist-mode drugs, those targeting
genes with a *biallelic pLoF* (human knockout) in 44k G&H exomes have
**OR = 2.16, P = 5.3 × 10⁻⁵ for Phase 1 → Phase 2+** progression — the
exact transition where safety/tolerability decides progression. That's a
clean external prior on our `has_safety_event` outcome.

This script pulls two data products from the paper's public GCS bucket
``gs://genesandhealth_publicdatasets/results_44k_ExWAS/``:

1. **Per-gene burden sumstats** — all ``*_genetests_*.regenie.gz`` files
   under ``44kExWAS_quanttraits/regenie/``,
   ``44kExWAS_binarytraits/3digitICD10/regenie/``, and
   ``…/customphenotypes/regenie/``. Each file is REGENIE gene-based
   burden output across 4 functional masks (pLoF-HC, pLoF+pDM, all
   pLoF+missense, synonymous control) × 4 AF cutoffs (singleton,
   <0.01%, <0.1%, <1%) for ~20k genes. ~645 files × ~3 MB ≈ 1.5 GB
   one-time download, cached under ``data/gh_burden_tmp/``.

2. **Supplementary table xlsx** — ``medrxiv_preprint_supp_media-2.xlsx``
   (~15 MB) — for the human-knockout gene table (ST15: per-gene
   het/hom/comphet carrier counts across G&H + 5 reference cohorts) and
   the phenotype-to-EFO map (ST6: 730 traits, each tagged with EFO id).

Per-gene features written to ``data/gh_burden_features.parquet``
(keyed by Ensembl ``geneId``):

* ``gh_n_burden_hits``  — # traits where the gene has any non-synonymous
  burden (mask A/B/C × any AF cutoff) with log10P ≥ 6.54 (P < 2.89e-7,
  the paper's permutation-FDR-5% threshold).
* ``gh_n_burden_TAs``   — # distinct ST9 therapeutic areas hit (rare-
  variant analogue of the GWAS-based ``n_TAs``).
* ``gh_min_burden_p``   — strongest burden P across all traits.
* ``gh_max_log10p``     — its log10P (numerical-stability copy).
* ``gh_is_human_ko``    — gene has ≥1 biallelic pLoF carrier in G&H.
* ``gh_n_ko_carriers``  — # biallelic carriers (hom + comphet).
* ``gh_ko_only_in_gh``  — KO seen *only* in G&H among 6 reference
  cohorts (the maximally novel-knockout subset).

Usage:
    python scripts/fetch_genes_and_health_burden.py
    python scripts/fetch_genes_and_health_burden.py --max-files 5  # dev/sanity
    python scripts/fetch_genes_and_health_burden.py --skip-download  # re-aggregate cached
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Import the existing ST9 therapeutic-area mapping so the rare-variant
# `gh_n_burden_TAs` is on the same TA scale as the GWAS `n_TAs`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pleio.pleiotropy import _TA_RANK, _TA_NAME, MEASUREMENT_TA, OTHER_TA  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

GCS_LIST = ("https://storage.googleapis.com/storage/v1/b/"
            "genesandhealth_publicdatasets/o")
GCS_BLOB = ("https://storage.googleapis.com/"
            "genesandhealth_publicdatasets/")

# Three sub-trees that contain gene-test regenie outputs.
TRAIT_PREFIXES = [
    ("quant",       "results_44k_ExWAS/44kExWAS_quanttraits/regenie/"),
    ("icd10",       "results_44k_ExWAS/44kExWAS_binarytraits/3digitICD10/regenie/"),
    ("custom",      "results_44k_ExWAS/44kExWAS_binarytraits/customphenotypes/regenie/"),
]
SUPP_XLSX_BLOB = "results_44k_ExWAS/medrxiv_preprint_supp_media-2.xlsx"

# Paper §"Rare variant association analyses under the additive model":
# permutation-derived FDR-5% threshold P_add < 2.89e-7 → -log10p = 6.54.
FDR_LOG10P = 6.54

# Match `SYMBOL(ENSG00000000000).MASK_X.<af_cutoff>` in the ID column.
_ID_RE = re.compile(r"^(?P<sym>[^()]+)\((?P<ensg>ENSG\d+)\)\.(?P<mask>MASK_[A-D])\.(?P<af>\S+)$")


# ---------------------------------------------------------------------------
# GCS listing + download
# ---------------------------------------------------------------------------
def _list_gcs(prefix: str) -> list[dict]:
    """Page through the GCS JSON API for every blob under `prefix`."""
    out, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        r = requests.get(GCS_LIST, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        out.extend(d.get("items", []))
        token = d.get("nextPageToken")
        if not token:
            break
    return out


def list_genetest_files() -> list[tuple[str, str, str]]:
    """Return [(trait_kind, trait_code, blob_name), …] for every
    `*_genetests_*.regenie.gz` file across the three trait sub-trees."""
    found = []
    for kind, prefix in TRAIT_PREFIXES:
        items = _list_gcs(prefix)
        for it in items:
            name = it["name"]
            if "_genetests_" not in name or not name.endswith(".regenie.gz"):
                continue
            # Filename pattern: <date>_<trait>_GNH_genetests_<trait>.regenie.gz
            base = name.rsplit("/", 1)[-1]
            m = re.match(r"\d{4}_\d{2}_\d{2}_(?P<trait>.+?)_GNH_genetests_.+\.regenie\.gz$", base)
            if not m:
                continue
            found.append((kind, m.group("trait"), name))
    return found


def _download(url: str, dest: Path, timeout: int = 120, retries: int = 3) -> bool:
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=2 ** 20):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
            return True
        except (requests.RequestException, OSError):
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return False
    return False


def download_files(files: list[tuple[str, str, str]], tmp_dir: Path,
                   workers: int = 8) -> list[Path]:
    """Download (or skip if cached) every gene-test file. Returns paths
    that exist on disk after the run."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    plan: list[tuple[str, Path]] = []
    for kind, trait, blob in files:
        # `data/gh_burden_tmp/<kind>/<basename>`
        local = tmp_dir / kind / blob.rsplit("/", 1)[-1]
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and local.stat().st_size > 0:
            continue
        plan.append((GCS_BLOB + blob, local))
    if not plan:
        print(f"  all {len(files)} files already cached")
    else:
        print(f"  {len(plan)} new file(s) to download (parallel={workers})")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_download, url, dest): dest for url, dest in plan}
            for f in as_completed(futs):
                dest = futs[f]
                ok = f.result()
                done += 1
                if done % 25 == 0 or done == len(plan):
                    print(f"    [{done}/{len(plan)}] last={dest.name} ok={ok}",
                          flush=True)

    # Re-walk for final paths
    have = []
    for kind, trait, blob in files:
        local = tmp_dir / kind / blob.rsplit("/", 1)[-1]
        if local.exists() and local.stat().st_size > 0:
            have.append(local)
    return have


# ---------------------------------------------------------------------------
# Parse a single gene-test regenie file → per-gene strongest hit
# ---------------------------------------------------------------------------
def parse_burden_file(path: Path) -> pd.DataFrame:
    """Read one ``*_genetests_*.regenie.gz`` file and collapse it to one
    row per gene = the strongest (max-log10P) non-synonymous ADD burden
    across all masks and AF cutoffs in this file.

    Returns a DataFrame with columns ``geneId``, ``symbol``, ``max_log10p``,
    plus the mask and AF that produced it (for inspection).
    """
    # REGENIE writes a `##MASKS=<…>` header line then the column header.
    with gzip.open(path, "rt") as fh:
        first = fh.readline()
        if not first.startswith("##"):
            # Some files may have no comment line — rewind via seek
            # impossible on gzip stream, so re-open.
            df = pd.read_csv(path, sep=r"\s+", compression="gzip")
        else:
            df = pd.read_csv(fh, sep=r"\s+")
    if df.empty:
        return pd.DataFrame(columns=["geneId", "symbol", "max_log10p", "mask", "af"])

    df = df[df["TEST"] == "ADD"]
    if df.empty:
        return pd.DataFrame(columns=["geneId", "symbol", "max_log10p", "mask", "af"])

    parsed = df["ID"].str.extract(_ID_RE)
    df = df.assign(**parsed.to_dict(orient="series"))
    # Drop unparseable rows + MASK_D (synonymous = negative control)
    df = df.dropna(subset=["ensg", "mask"])
    df = df[df["mask"] != "MASK_D"]
    if df.empty:
        return pd.DataFrame(columns=["geneId", "symbol", "max_log10p", "mask", "af"])

    # Coerce log10P, drop NA
    df["LOG10P"] = pd.to_numeric(df["LOG10P"], errors="coerce")
    df = df.dropna(subset=["LOG10P"])
    if df.empty:
        return pd.DataFrame(columns=["geneId", "symbol", "max_log10p", "mask", "af"])

    # Per-gene strongest hit in this trait
    idx = df.groupby("ensg")["LOG10P"].idxmax()
    top = df.loc[idx, ["ensg", "sym", "LOG10P", "mask", "af"]].rename(
        columns={"ensg": "geneId", "sym": "symbol", "LOG10P": "max_log10p"})
    return top.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trait → EFO → ST9 TA mapping
# ---------------------------------------------------------------------------
def _latest_release_dir() -> Path:
    rels = sorted(_DATA_DIR.glob("ot_release_*"))
    if not rels:
        sys.exit(f"ERROR: no OT release under {_DATA_DIR}.")
    return rels[-1]


def build_efo_to_ta(release_dir: Path) -> dict[str, str]:
    """Map every EFO/MONDO/etc. id → ST9 therapeutic area name, using
    the same ancestry-propagation as `pleio.pleiotropy`. We propagate
    each disease's `therapeuticAreas` field (already the ST9 roots in
    OT 26.03) to a single TA via the existing rank table."""
    d = pd.read_parquet(release_dir / "disease",
                        columns=["id", "name", "therapeuticAreas"])
    out: dict[str, str] = {}
    for row in d.itertuples(index=False):
        tas = row.therapeuticAreas if row.therapeuticAreas is not None else []
        try:
            tas = list(tas)
        except TypeError:
            continue
        # Pick the highest-priority TA from the disease's own list.
        ranked = sorted(((_TA_RANK.get(t, 1_000), t) for t in tas),
                        key=lambda x: x[0])
        if not ranked or ranked[0][0] >= 1_000:
            out[row.id] = OTHER_TA
        else:
            efo_root = ranked[0][1]
            out[row.id] = _TA_NAME.get(efo_root, OTHER_TA)
    return out


def build_icd10_to_efo(release_dir: Path) -> dict[str, str]:
    """Map 3-digit ICD-10 code (e.g. 'E11') → the most-specific EFO id
    whose `dbXRefs` contains `ICD10CM:<code>` or `ICD10WHO:<code>`.

    Prefer EFOs that are ICD10 "headers" — exact match on the 3-digit
    code, not a subcode. Falls through to the first EFO whose ICD10
    starts with the 3-digit code."""
    d = pd.read_parquet(release_dir / "disease",
                        columns=["id", "name", "dbXRefs"])
    exact: dict[str, str] = {}
    prefix: dict[str, str] = {}
    for row in d.itertuples(index=False):
        refs = row.dbXRefs if row.dbXRefs is not None else []
        try:
            refs = list(refs)
        except TypeError:
            continue
        for r in refs:
            if not isinstance(r, str):
                continue
            for tag in ("ICD10CM:", "ICD10WHO:"):
                if r.startswith(tag):
                    code = r[len(tag):]
                    # 3-digit prefix, e.g. "E11" from "E11.9"
                    head = code.split(".")[0][:3]
                    if not head:
                        continue
                    if code == head and head not in exact:
                        exact[head] = row.id
                    elif head not in prefix:
                        prefix[head] = row.id
    return {**prefix, **exact}  # exact wins over prefix


def build_trait_to_ta(release_dir: Path, supp_xlsx: Path) -> dict[tuple[str, str], str]:
    """Return a dict keyed by ``(trait_kind, trait_code)`` → ST9 TA.

    Strategy:
      * Map every EFO → ST9 TA via the OT disease parquet (same path
        the GWAS pipeline uses).
      * Read ST6 of the supp xlsx and walk every row, trying to match
        the phenotype's filename code to one of:
          quant   — exact match on filename trait code (strip `.residual`),
                    using a small alias table (ST6 has full names, not codes);
          icd10   — `(icd10, <3-digit-code>)` via OT dbXRefs;
          custom  — exact match on underscored phenotype name vs ST6 Phenotype.
    """
    efo_to_ta = build_efo_to_ta(release_dir)
    icd_to_efo = build_icd10_to_efo(release_dir)

    st6 = pd.read_excel(supp_xlsx, "ST6")
    out: dict[tuple[str, str], str] = {}

    # 1. ICD10 mapping (independent of ST6): every 3-digit ICD10 → TA
    for code, efo in icd_to_efo.items():
        ta = efo_to_ta.get(efo, OTHER_TA)
        out[("icd10", code)] = ta

    # 2. ST6 rows give every phenotype an EFO id directly (when known).
    # Build (phenotype_name_underscored → TA) for custom + (canonical
    # short code → TA) for quant.
    for row in st6.itertuples(index=False):
        pheno_name = str(getattr(row, "Phenotype", "")).strip()
        efo = getattr(row, "_11", None)  # EFO ID column index varies
        # Re-resolve EFO ID column robustly
        try:
            efo = row._asdict().get("EFO ID")
        except Exception:
            pass
        if not isinstance(efo, str) or not efo.strip():
            continue
        ta = efo_to_ta.get(efo.strip(), OTHER_TA)

        # 2a. Custom binary — match by underscored name
        custom_key = pheno_name.replace(" ", "_")
        out[("custom", custom_key)] = ta

        # 2b. Quant trait short code: use the trait's "Equivalent
        # Phenotype Group" or fall back to inferring an abbreviation —
        # but the ST6 doesn't carry the file code. We rely instead on a
        # post-hoc match on the parsed trait code (handled at call
        # time): see resolve_trait_ta() below.
        if getattr(row, "QT", 0) == 1.0:
            out[("quant", pheno_name)] = ta  # by name
            # add a stripped/lowered version for fuzzy match
            out[("quant", pheno_name.replace(" ", "_"))] = ta

    return out


def resolve_trait_ta(kind: str, code: str,
                     trait_map: dict[tuple[str, str], str],
                     st6: pd.DataFrame) -> str:
    """Resolve a trait file's (kind, code) to a ST9 TA, with kind-
    specific fallbacks."""
    if kind == "icd10":
        # File code is e.g. 'A01' — direct ICD10 lookup
        return trait_map.get(("icd10", code), OTHER_TA)
    if kind == "custom":
        # File code is underscored phenotype name
        return trait_map.get(("custom", code), OTHER_TA)
    if kind == "quant":
        # File code is e.g. 'AFP.residual' — strip `.residual`,
        # then try multiple matching strategies against ST6 names.
        short = code.replace(".residual", "").replace("_residual", "")
        # Try direct keys first
        for key in (short, short.replace("_", " ")):
            ta = trait_map.get(("quant", key))
            if ta:
                return ta
        # Fuzzy: match by initials of multi-word ST6 phenotype names
        for row in st6.itertuples(index=False):
            name = str(getattr(row, "Phenotype", "")).strip()
            if not name:
                continue
            initials = "".join(w[0].upper() for w in re.split(r"\s+", name) if w)
            if initials and initials == short.upper():
                return trait_map.get(("quant", name), OTHER_TA)
            # also try name with spaces removed
            if name.replace(" ", "").lower() == short.replace(".", "").lower():
                return trait_map.get(("quant", name), OTHER_TA)
        return OTHER_TA
    return OTHER_TA


# ---------------------------------------------------------------------------
# Aggregation + KO axis
# ---------------------------------------------------------------------------
def aggregate_burden(parsed: list[tuple[str, str, pd.DataFrame]],
                     trait_to_ta: dict[tuple[str, str], str],
                     st6: pd.DataFrame) -> pd.DataFrame:
    """Combine per-file per-gene strongest hits into one row per gene.

    Args:
        parsed: [(trait_kind, trait_code, per_gene_top_df), …]
        trait_to_ta: dict from build_trait_to_ta
        st6: Supplementary Table 6 (Phenotypes and counts)
    """
    rows = []
    for kind, code, df in parsed:
        if df.empty:
            continue
        ta = resolve_trait_ta(kind, code, trait_to_ta, st6)
        # Drop measurement-TA quant traits so they don't bloat
        # gh_n_burden_TAs the same way the GWAS pipeline excludes them.
        for r in df.itertuples(index=False):
            rows.append((r.geneId, r.symbol, kind, code, ta,
                         float(r.max_log10p)))
    if not rows:
        return pd.DataFrame(columns=[
            "geneId", "gh_n_burden_hits", "gh_n_burden_TAs",
            "gh_max_log10p", "gh_min_burden_p"])
    rec = pd.DataFrame(rows, columns=[
        "geneId", "symbol", "trait_kind", "trait_code", "ta", "log10p"])

    sig = rec[rec["log10p"] >= FDR_LOG10P].copy()
    # Drop measurement TA from the rare-variant TA count (parallel to GWAS).
    sig_ta = sig[~sig["ta"].isin({MEASUREMENT_TA, OTHER_TA})]

    by_gene = (rec.groupby("geneId")
                  .agg(gh_max_log10p=("log10p", "max"),
                       symbol=("symbol", "first"))
                  .reset_index())
    by_gene["gh_min_burden_p"] = 10.0 ** (-by_gene["gh_max_log10p"])

    hits = (sig.groupby("geneId").size()
              .rename("gh_n_burden_hits").reset_index())
    by_gene = by_gene.merge(hits, on="geneId", how="left")
    by_gene["gh_n_burden_hits"] = by_gene["gh_n_burden_hits"].fillna(0).astype(int)

    n_tas = (sig_ta.groupby("geneId")["ta"].nunique()
                  .rename("gh_n_burden_TAs").reset_index())
    by_gene = by_gene.merge(n_tas, on="geneId", how="left")
    by_gene["gh_n_burden_TAs"] = by_gene["gh_n_burden_TAs"].fillna(0).astype(int)

    return by_gene


def load_ko_table(supp_xlsx: Path) -> pd.DataFrame:
    """Parse ST15 (gene-level pLoF/pDM genotype counts) → per-gene KO
    features keyed by Ensembl id."""
    st15 = pd.read_excel(supp_xlsx, "ST15")
    # Robust column names
    col_ensg = next(c for c in st15.columns if c.upper() == "ENSG")
    col_sym = next(c for c in st15.columns if c.lower() == "symbol")
    col_hom = next(c for c in st15.columns if "homozygous" in c.lower()
                   and "compound" not in c.lower())
    col_chet = next(c for c in st15.columns if "compound heterozygous" in c.lower())

    # "In G&H" / "In gnomAD v4" / etc. are booleans for each cohort the KO
    # is present in. "Only in G&H" = In G&H True AND all others False.
    other_cohort_cols = [c for c in st15.columns
                         if c.lower().startswith("in ") and c != "In G&H"]

    ko = pd.DataFrame({
        "geneId": st15[col_ensg].astype(str),
        "symbol": st15[col_sym].astype(str),
        "n_hom": st15[col_hom].fillna(0).astype(int),
        "n_chet": st15[col_chet].fillna(0).astype(int),
        "in_gh": st15["In G&H"].fillna(False).astype(bool),
    })
    ko["gh_n_ko_carriers"] = ko["n_hom"] + ko["n_chet"]
    ko["gh_is_human_ko"] = (ko["gh_n_ko_carriers"] > 0).astype(int)

    # "Only in G&H" — present in G&H but not in any of the other listed cohorts
    any_other = pd.Series(False, index=st15.index)
    for c in other_cohort_cols:
        any_other = any_other | st15[c].fillna(False).astype(bool)
    ko["gh_ko_only_in_gh"] = (ko["in_gh"] & ~any_other).astype(int)

    return ko[["geneId", "gh_is_human_ko", "gh_n_ko_carriers",
               "gh_ko_only_in_gh"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tmp-dir", default=str(_DATA_DIR / "gh_burden_tmp"),
                   help="Local cache for downloaded regenie files.")
    p.add_argument("--out",
                   default=str(_DATA_DIR / "gh_burden_features.parquet"),
                   help="Per-gene burden feature parquet.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download threads.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap on # files (sanity / sample run).")
    p.add_argument("--skip-download", action="store_true",
                   help="Use whatever's already in --tmp-dir; do not "
                        "fetch new files.")
    p.add_argument("--supp-xlsx",
                   default=str(_DATA_DIR / "gh_supp_tables.xlsx"),
                   help="Where to cache the supplementary xlsx.")
    args = p.parse_args(argv)

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    supp_path = Path(args.supp_xlsx)

    # 1. Supplementary xlsx (small, always cached)
    if not supp_path.exists():
        print(f"Fetching supp xlsx …")
        ok = _download(GCS_BLOB + SUPP_XLSX_BLOB, supp_path)
        if not ok:
            sys.exit("ERROR: could not fetch supplementary xlsx.")
    print(f"  supp xlsx: {supp_path}  ({supp_path.stat().st_size / 1e6:.1f} MB)")

    # 2. List + download gene-test files
    print("Listing gene-test files on GCS …")
    files = list_genetest_files()
    print(f"  found {len(files):,} gene-test files "
          f"({sum(1 for k,_,_ in files if k=='quant'):,} quant, "
          f"{sum(1 for k,_,_ in files if k=='icd10'):,} icd10, "
          f"{sum(1 for k,_,_ in files if k=='custom'):,} custom)")
    if args.max_files:
        files = files[: args.max_files]
        print(f"  --max-files cap: {len(files)} files")
    if not args.skip_download:
        print("Downloading (cached) …")
        download_files(files, tmp_dir, workers=args.workers)
    else:
        print("  --skip-download: aggregating from cache only")

    # 3. Parse + aggregate
    print("Parsing each gene-test file → per-gene strongest burden …")
    parsed: list[tuple[str, str, pd.DataFrame]] = []
    skipped = 0
    for i, (kind, code, blob) in enumerate(files, 1):
        local = tmp_dir / kind / blob.rsplit("/", 1)[-1]
        if not local.exists() or local.stat().st_size == 0:
            skipped += 1
            continue
        try:
            df = parse_burden_file(local)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: parse failed for {local.name}: {exc}", flush=True)
            skipped += 1
            continue
        parsed.append((kind, code, df))
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] parsed; "
                  f"running rows so far: {sum(len(d) for _,_,d in parsed):,}",
                  flush=True)
    if skipped:
        print(f"  NOTE: {skipped} file(s) missing or unparseable — re-run "
              "without --skip-download to retry the downloads.")

    print("Building trait → ST9 TA map …")
    release_dir = _latest_release_dir()
    trait_map = build_trait_to_ta(release_dir, supp_path)
    st6 = pd.read_excel(supp_path, "ST6")

    print("Aggregating per-gene across all traits …")
    burden = aggregate_burden(parsed, trait_map, st6)
    print(f"  per-gene burden rows: {len(burden):,}")

    # 4. KO axis from ST15
    print("Loading ST15 → per-gene human-KO features …")
    ko = load_ko_table(supp_path)
    print(f"  ST15 rows (genes): {len(ko):,}")
    print(f"  gh_is_human_ko==1: {int(ko['gh_is_human_ko'].sum()):,} "
          f"(paper headline: 2,991 genes)")
    print(f"  gh_ko_only_in_gh==1: {int(ko['gh_ko_only_in_gh'].sum()):,} "
          f"(paper headline: 1,669 vs UKB; 546 vs 5 cohorts)")

    # 5. Merge + write
    print("Merging burden + KO …")
    merged = burden.merge(ko, on="geneId", how="outer")
    # Sensible defaults for genes with no signal in either side
    for c, default in [
            ("gh_n_burden_hits", 0),
            ("gh_n_burden_TAs", 0),
            ("gh_max_log10p", 0.0),
            ("gh_min_burden_p", 1.0),
            ("gh_is_human_ko", 0),
            ("gh_n_ko_carriers", 0),
            ("gh_ko_only_in_gh", 0)]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(default)
    merged["gh_n_burden_hits"] = merged["gh_n_burden_hits"].astype(int)
    merged["gh_n_burden_TAs"] = merged["gh_n_burden_TAs"].astype(int)
    merged["gh_is_human_ko"] = merged["gh_is_human_ko"].astype(int)
    merged["gh_n_ko_carriers"] = merged["gh_n_ko_carriers"].astype(int)
    merged["gh_ko_only_in_gh"] = merged["gh_ko_only_in_gh"].astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["geneId", "gh_n_burden_hits", "gh_n_burden_TAs",
            "gh_max_log10p", "gh_min_burden_p",
            "gh_is_human_ko", "gh_n_ko_carriers", "gh_ko_only_in_gh"]
    merged[cols].to_parquet(out_path, index=False)

    print(f"\nWrote {len(merged):,} genes -> {out_path}")
    print(f"  genes with ≥1 FDR-sig burden hit: "
          f"{int((merged['gh_n_burden_hits'] > 0).sum()):,}")
    print(f"  median gh_max_log10p:  {merged['gh_max_log10p'].median():.2f}")
    print(f"  human-KO genes:        {int(merged['gh_is_human_ko'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
