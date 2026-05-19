"""Fetch gnomAD v4.1 *joint* per-ancestry allele frequencies for the Open
Targets credible-set GWAS lead variants — via remote bcftools range queries, so
no multi-GB VCF is ever stored locally.

Why this exists
---------------
The manuscript's "Annotation of CS lead variants by MAF and beta rescaling"
step needs the major-ancestry MAF of each credible-set lead variant, taken
from the gnomAD v4.1 joint allele-frequency release. OT 26.03's
``credible_set.effectAlleleFrequencyFromSource`` is only ~38% populated, so
we re-annotate from gnomAD.

The local GWAS credible-set universe is ~480k unique lead variants and the
linked studies only ever use 5 genetic-ancestry groups (nfe, eas, afr, amr,
fin), so the extract is tiny — a single ``data/gnomad_v4.1_joint_af.parquet``
with 5 population-AF columns. We get there without downloading the joint
VCFs whole: gnomAD's bgzipped VCFs are tabix-indexed and HTTP-range
accessible, so ``bcftools query -R <regions.bed> <remote-url>`` fetches
only the bgzf blocks covering our variant positions.

Usage
-----
    # Full run (remote range queries; resumable per-chromosome)
    python scripts/fetch_gnomad_af.py

    # Subset of chromosomes (e.g. a quick test)
    python scripts/fetch_gnomad_af.py --chroms 21 22

    # If you already downloaded the joint sites VCFs locally
    python scripts/fetch_gnomad_af.py --gnomad-dir /path/to/gnomad_joint_vcfs

    # Just dump the variant universe + verify the gnomAD INFO field names
    python scripts/fetch_gnomad_af.py --dry-run

Requires ``bcftools`` (htslib) on PATH, built with libcurl for remote
access (the Homebrew / bioconda builds are). ``bcftools query`` does the
region-filtering *and* INFO-field extraction in C, so Python only ever
sees a small pre-digested TSV — the alternative (tabix + Python parsing
of multi-KB gnomAD VCF lines) is ~100x slower. Output:
``data/gnomad_v4.1_joint_af.parquet`` keyed by ``variantId``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# gnomAD v4.1 joint (exome+genome) sites-only VCFs on the GCS public bucket.
GNOMAD_JOINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "vcf/joint/gnomad.joint.v4.1.sites.chr{chrom}.vcf.bgz"
)

# Genetic-ancestry groups that actually appear in OT 26.03 study
# ldPopulationStructure. Field names verified against the v4.1 joint header.
POP_AF_FIELD: dict[str, str] = {
    "nfe": "AF_joint_nfe",
    "eas": "AF_joint_eas",
    "afr": "AF_joint_afr",
    "amr": "AF_joint_amr",
    "fin": "AF_joint_fin",
}
OVERALL_AF_FIELD = "AF_joint"
_ALL_AF_FIELDS = [OVERALL_AF_FIELD, *POP_AF_FIELD.values()]

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_bcftools() -> str:
    bcftools = shutil.which("bcftools")
    if bcftools is None:
        sys.exit(
            "ERROR: `bcftools` not found on PATH. Install it — e.g.\n"
            "  brew install bcftools       (macOS)\n"
            "  conda install -c bioconda bcftools\n"
            "It must be built with libcurl for remote range access "
            "(the Homebrew / bioconda builds are)."
        )
    return bcftools


def _latest_release_dir(data_dir: Path) -> Path:
    rels = sorted(data_dir.glob("ot_release_*"))
    if not rels:
        sys.exit(f"ERROR: no OT release under {data_dir}. "
                 "Run scripts/download_ot_release.py first.")
    return rels[-1]


def variant_universe(release_dir: Path) -> pd.DataFrame:
    """Unique GWAS credible-set lead variants, parsed to chrom/pos/ref/alt.

    variantId is the OT/Gentropy ``chrom_pos_ref_alt`` string on GRCh38.
    """
    cs = pd.read_parquet(
        release_dir / "credible_set",
        columns=["variantId", "studyType"],
    )
    gwas = cs[(cs["studyType"] == "gwas") & cs["variantId"].notna()]
    vids = pd.Series(gwas["variantId"].unique(), name="variantId")
    n_before = len(vids)

    # A well-formed variantId is exactly chrom_pos_ref_alt (3 underscores).
    # Structural-variant / malformed ids are dropped.
    vids = vids[vids.str.count("_") == 3]
    parts = vids.str.split("_", expand=True)
    vdf = pd.DataFrame({
        "variantId": vids.values,
        "chrom": parts[0].values,
        "pos": pd.to_numeric(parts[1], errors="coerce").values,
        "ref": parts[2].values,
        "alt": parts[3].values,
    })
    vdf = vdf[vdf["pos"].notna()].copy()
    vdf["pos"] = vdf["pos"].astype(int)
    if len(vdf) < n_before:
        print(f"  dropped {n_before - len(vdf):,} malformed variantIds")
    return vdf.reset_index(drop=True)


def discover_af_fields(bcftools: str, sample_chrom: str = "22") -> None:
    """Fetch the gnomAD VCF header and confirm our INFO fields exist.

    Doesn't change behaviour — just fails loudly if gnomAD renamed a field
    between releases, instead of silently producing all-NaN columns.
    """
    url = GNOMAD_JOINT_URL.format(chrom=sample_chrom)
    print(f"Verifying INFO fields against {url} …")
    try:
        out = subprocess.run(
            [bcftools, "view", "-h", url],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.exit(f"ERROR: could not fetch gnomAD header — {exc}\n"
                 "Check network access and that bcftools has libcurl support.")
    header_ids = {
        line.split("ID=", 1)[1].split(",", 1)[0]
        for line in out.splitlines()
        if line.startswith("##INFO=<ID=")
    }
    missing = [f for f in _ALL_AF_FIELDS if f not in header_ids]
    if missing:
        sys.exit(
            f"ERROR: expected gnomAD INFO field(s) not in the v4.1 joint "
            f"header: {missing}\nThe release schema may have changed — "
            "inspect `bcftools view -h` output and update POP_AF_FIELD."
        )
    print(f"  OK — all {len(_ALL_AF_FIELDS)} AF fields present "
          f"({', '.join(_ALL_AF_FIELDS)})")


def _merge_to_windows(positions: list[int], gap: int) -> list[tuple[int, int]]:
    """Collapse a sorted position list into [start, end] windows, merging
    any two positions within ``gap`` bp of each other.

    Remote-access latency is dominated by the *number* of regions, not their
    width — htslib reads in ~64 KB bgzf blocks regardless — so coarsening
    hundreds of thousands of 1-bp queries into a few hundred windows per
    chromosome cuts round-trips dramatically while skipping variant-free
    genomic deserts entirely.
    """
    positions = sorted(set(positions))
    windows: list[tuple[int, int]] = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p - prev > gap:
            windows.append((start, prev))
            start = p
        prev = p
    windows.append((start, prev))
    return windows


# bcftools query format — emits exactly the columns we need as a clean TSV,
# extracted in C. Order must match `_AF_OUT_COLS` below.
_QUERY_FMT = (
    "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t"
    + "\t".join(f"%INFO/{f}" for f in _ALL_AF_FIELDS)
    + "\n"
)
_AF_OUT_COLS = ["af_joint"] + [f"af_joint_{p}" for p in POP_AF_FIELD]


def _to_af(v: str) -> float:
    """bcftools query emits '.' for a missing INFO value."""
    if v in (".", ""):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def fetch_chrom(
    chrom: str,
    vdf_chrom: pd.DataFrame,
    bcftools: str,
    tmp_dir: Path,
    gnomad_dir: Path | None,
    merge_gap: int,
) -> pd.DataFrame:
    """Range-query one chromosome's gnomAD joint VCF for our variant set.

    ``bcftools query -R <bed> -f <fmt>`` does the region filtering and
    INFO-field extraction in C; Python only receives an 11-column TSV.
    Matches on exact (pos, ref, alt); also recognises ref/alt-swapped
    records (recording AF as 1 - gnomAD_AF). Returns one row per input
    variant, with NaN AFs for sites gnomAD doesn't carry.
    """
    if gnomad_dir is not None:
        src = str(gnomad_dir / f"gnomad.joint.v4.1.sites.chr{chrom}.vcf.bgz")
    else:
        src = GNOMAD_JOINT_URL.format(chrom=chrom)

    # bcftools -R wants a BED file (0-based, half-open). gnomAD contigs are
    # chr-prefixed. Positions are merged into windows to cut round-trips.
    windows = _merge_to_windows(vdf_chrom["pos"].tolist(), merge_gap)
    bed = tmp_dir / f"chr{chrom}.bed"
    with open(bed, "w") as fh:
        for start, end in windows:
            fh.write(f"chr{chrom}\t{start - 1}\t{end}\n")

    # Only keep gnomAD records that actually match a wanted variant — bounds
    # the dict even when a window spans many irrelevant sites.
    want = set(zip(vdf_chrom["pos"], vdf_chrom["ref"], vdf_chrom["alt"]))
    want_swapped = set(zip(vdf_chrom["pos"], vdf_chrom["alt"], vdf_chrom["ref"]))

    af_cols = _AF_OUT_COLS
    fwd: dict[tuple[int, str, str], dict] = {}

    proc = subprocess.Popen(
        [bcftools, "query", "-f", _QUERY_FMT, "-R", str(bed), src],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # TSV: CHROM POS REF ALT FILTER AF_joint AF_joint_nfe ... (11 cols)
        f = line.rstrip("\n").split("\t")
        if len(f) != 5 + len(_ALL_AF_FIELDS):
            continue
        pos = int(f[1])
        ref, alt = f[2], f[3]
        key = (pos, ref, alt)
        if key not in want and key not in want_swapped:
            continue
        rec = {"gnomad_filter": f[4]}
        for i, col in enumerate(af_cols):
            rec[col] = _to_af(f[5 + i])
        fwd[key] = rec
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read().strip() if proc.stderr else "")[:400]
        raise RuntimeError(
            f"bcftools failed for chr{chrom} (rc={proc.returncode}): {err}"
        )
    rows = []
    for r in vdf_chrom.itertuples(index=False):
        key = (r.pos, r.ref, r.alt)
        swp = (r.pos, r.alt, r.ref)
        if key in fwd:
            rec = fwd[key]
            rows.append({"variantId": r.variantId, "match": "exact",
                         "gnomad_filter": rec["gnomad_filter"],
                         **{c: rec[c] for c in af_cols}})
        elif swp in fwd:
            rec = fwd[swp]
            # gnomAD carries the ref/alt-swapped record — our allele's
            # frequency is 1 - gnomAD's reported (alt) frequency.
            rows.append({"variantId": r.variantId, "match": "swapped",
                         "gnomad_filter": rec["gnomad_filter"],
                         **{c: (1.0 - rec[c]) if pd.notna(rec[c]) else rec[c]
                            for c in af_cols}})
        else:
            rows.append({"variantId": r.variantId, "match": "missing",
                         "gnomad_filter": None,
                         **{c: float("nan") for c in af_cols}})
    return pd.DataFrame(rows)


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
    p.add_argument("--out", default=str(_DATA_DIR / "gnomad_v4.1_joint_af.parquet"),
                   help="Output parquet path.")
    p.add_argument("--tmp-dir", default=str(_DATA_DIR / "gnomad_af_tmp"),
                   help="Scratch dir for per-chromosome intermediates "
                        "(resumable — existing chrom parquets are reused).")
    p.add_argument("--gnomad-dir", default=None,
                   help="Use local gnomad.joint.v4.1.sites.chr*.vcf.bgz files "
                        "from this dir instead of remote range queries.")
    p.add_argument("--chroms", nargs="+", default=CHROMS, choices=CHROMS,
                   help="Subset of chromosomes to fetch. Default: all 24.")
    p.add_argument("--jobs", type=int, default=4,
                   help="Parallel chromosome workers (remote I/O bound). "
                        "Default 4.")
    p.add_argument("--merge-gap", type=int, default=250_000,
                   help="Variant positions within this many bp are merged "
                        "into one bcftools region. Larger = fewer "
                        "round-trips, more bytes streamed. Default 250000.")
    p.add_argument("--dry-run", action="store_true",
                   help="Dump the variant universe + verify gnomAD field "
                        "names, then exit without fetching.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bcftools = _require_bcftools()

    release_dir = (Path(args.release_dir) if args.release_dir
                   else _latest_release_dir(_DATA_DIR))
    print(f"OT release: {release_dir}")

    print("Building variant universe from credible_set …")
    vdf = variant_universe(release_dir)
    print(f"  {len(vdf):,} unique GWAS credible-set lead variants")
    per_chrom = vdf["chrom"].value_counts()
    print("  per chromosome: "
          + ", ".join(f"{c}:{per_chrom.get(c, 0):,}" for c in args.chroms))

    gnomad_dir = Path(args.gnomad_dir) if args.gnomad_dir else None
    if gnomad_dir is None:
        discover_af_fields(bcftools)
    else:
        print(f"Using local gnomAD VCFs from {gnomad_dir}")

    if args.dry_run:
        print("\n--dry-run: stopping before fetch.")
        return 0

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for chrom in args.chroms:
        out_c = tmp_dir / f"chr{chrom}.parquet"
        if out_c.exists() and out_c.stat().st_size > 0:
            print(f"  chr{chrom}: cached ({out_c.name}) — skipping")
            continue
        vdf_c = vdf[vdf["chrom"] == chrom]
        if vdf_c.empty:
            print(f"  chr{chrom}: no variants — skipping")
            continue
        todo.append((chrom, vdf_c))

    def _work(chrom: str, vdf_c: pd.DataFrame) -> str:
        with tempfile.TemporaryDirectory(dir=tmp_dir) as scratch:
            res = fetch_chrom(chrom, vdf_c, bcftools, Path(scratch),
                              gnomad_dir, args.merge_gap)
        out_c = tmp_dir / f"chr{chrom}.parquet"
        res.to_parquet(out_c, index=False)
        n_hit = (res["match"] != "missing").sum()
        return (f"  chr{chrom}: {len(res):,} variants, "
                f"{n_hit:,} annotated ({100 * n_hit / len(res):.1f}%)")

    if todo:
        print(f"\nFetching {len(todo)} chromosome(s) with {args.jobs} worker(s)…")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_work, c, v): c for c, v in todo}
            for fut in cf.as_completed(futs):
                chrom = futs[fut]
                try:
                    print(fut.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"  chr{chrom}: FAILED — {exc}", file=sys.stderr)

    # Concatenate all per-chromosome intermediates that exist.
    parts = []
    for chrom in args.chroms:
        out_c = tmp_dir / f"chr{chrom}.parquet"
        if out_c.exists() and out_c.stat().st_size > 0:
            parts.append(pd.read_parquet(out_c))
    if not parts:
        print("No chromosome results to concatenate.", file=sys.stderr)
        return 1

    full = pd.concat(parts, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    n_hit = (full["match"] != "missing").sum()
    n_swap = (full["match"] == "swapped").sum()
    print(f"\nWrote {len(full):,} rows -> {out_path}")
    print(f"  annotated: {n_hit:,} ({100 * n_hit / len(full):.1f}%)  "
          f"[{n_swap:,} ref/alt-swapped]")
    print(f"  missing:   {len(full) - n_hit:,} "
          f"({100 * (len(full) - n_hit) / len(full):.1f}%)")
    print("Per-population non-null AF coverage:")
    for pop in POP_AF_FIELD:
        col = f"af_joint_{pop}"
        print(f"  {col}: {full[col].notna().sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
