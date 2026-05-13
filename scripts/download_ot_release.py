"""Download the latest Open Targets Platform release parquet exports.

For the all-genes pleiotropy analysis notebook we need three datasets from
the OT Platform output:

  * ``association_by_datatype_direct/``   per-(target, disease, datatype) score
  * ``disease/``                          disease metadata + therapeuticAreas
  * ``target/``                           gene metadata (symbol, biotype, ...)

Each dataset is a directory of part-*.snappy.parquet files. Total disk usage
for a typical release is roughly 1–2 GB. Files already present locally are
skipped so reruns are cheap.

Path layout changed in OT release 25.03 — both the dataset names (now
snake_case + singular) and the parent path (no more ``etl/parquet/``
intermediate dirs). This script handles both layouts transparently and
always stores data under the new names locally so downstream code doesn't
care which release it came from.

Usage::

    python scripts/download_ot_release.py                    # latest release
    python scripts/download_ot_release.py --release 26.03    # specific release
    python scripts/download_ot_release.py --release 24.09    # old-layout release
    python scripts/download_ot_release.py --data-dir data/   # custom output dir
    python scripts/download_ot_release.py --dry-run          # don't fetch, just probe
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/"

# Canonical dataset names — what we store the data under on disk, and what
# the notebook expects to find. These match the post-25.03 OT layout.
DATASETS = [
    # core pleiotropy proxy (TA / GA-score path)
    "association_by_datatype_direct",
    "disease",
    "target",
    # rigorous pleiotropy path — Gentropy fine-mapping + L2G
    "credible_set",
    "l2g_prediction",
    "study",
    # clinical / safety / tractability overlay
    "target_prioritisation",
    # 26.03+ — split out of the retired knownDrugsAggregated/known_drug
    "clinical_target",
    "clinical_indication",
    # legacy: only present in releases ≤ 25.06 (kept for backwards compat;
    # the download script silently skips a missing dataset)
    "known_drug",
]

# Some datasets only exist in older releases — listing them is allowed to
# return zero parquet files (we don't treat that as an error).
OPTIONAL_DATASETS = {"known_drug"}

# Pre-25.03 (i.e. ≤ 24.09) layout used camelCase + plural names and lived
# under output/etl/parquet/. We translate transparently. Datasets that only
# appear in a later layout don't need an entry here (they'll be skipped on
# older releases).
_OLD_LAYOUT_NAMES = {
    "association_by_datatype_direct": "associationByDatatypeDirect",
    "disease":                        "diseases",
    "target":                         "targets",
    "credible_set":                   "credibleSet",
    "l2g_prediction":                 "l2gPrediction",
    "study":                          "study",
    "target_prioritisation":          "targetPrioritisation",
    "known_drug":                     "knownDrugsAggregated",
}

# rough size hint shown in --dry-run output (MB, per dataset)
_SIZE_HINT_MB = {
    "association_by_datatype_direct": 900,
    "disease":                          40,
    "target":                          110,
    "credible_set":                    400,
    "l2g_prediction":                  600,
    "study":                            80,
    "target_prioritisation":            10,
    "clinical_target":                  40,
    "clinical_indication":              60,
    "known_drug":                      150,
}


# ---------------------------------------------------------------------------
# Path / release helpers
# ---------------------------------------------------------------------------
def _release_uses_new_layout(release: str) -> bool:
    """OT switched layouts at release 25.03.

    True for 25.03 and later (snake_case names, output/<dataset>/).
    False for 24.09 and earlier (camelCase names, output/etl/parquet/<dataset>/).
    """
    try:
        year, month = (int(p) for p in release.split("."))
    except (ValueError, AttributeError):
        # Unknown format → assume new layout (safer for current data).
        return True
    return (year, month) >= (25, 3)


def parquet_parent_url(release: str) -> str:
    """URL prefix that *contains* the per-dataset subdirectories for this release."""
    if _release_uses_new_layout(release):
        return urljoin(BASE_URL, f"{release}/output/")
    return urljoin(BASE_URL, f"{release}/output/etl/parquet/")


def remote_dataset_url(release: str, canonical_name: str) -> str | None:
    """Build the remote URL for a dataset, accounting for old/new layout.

    Returns None if the dataset doesn't exist in this layout (e.g. the new
    clinical_target / clinical_indication datasets do not exist in pre-25.03
    releases, and the old known_drug / knownDrugsAggregated has been retired
    in 26.03+).
    """
    parent = parquet_parent_url(release)
    if _release_uses_new_layout(release):
        return urljoin(parent, f"{canonical_name}/")
    legacy = _OLD_LAYOUT_NAMES.get(canonical_name)
    if legacy is None:
        return None
    return urljoin(parent, f"{legacy}/")


# ---------------------------------------------------------------------------
# Listing / downloading
# ---------------------------------------------------------------------------
def list_releases() -> list[str]:
    """Scrape the FTP release index. Returns numerically-sorted release versions."""
    r = requests.get(BASE_URL, timeout=30)
    r.raise_for_status()
    releases = re.findall(r'href="(\d+\.\d+)/"', r.text)
    return sorted(set(releases), key=lambda v: tuple(int(p) for p in v.split(".")))


def list_files(remote_dir_url: str) -> list[str]:
    """List file names inside a remote directory (parses the HTML index)."""
    r = requests.get(remote_dir_url, timeout=30)
    r.raise_for_status()
    return re.findall(r'href="([^"]+)"', r.text)


def download_dataset(remote_url: str, local_dir: Path,
                     dry_run: bool = False) -> int:
    """Download every parquet file in a remote OT dataset directory.

    Returns the number of files actually downloaded (skipped files don't count).
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = list_files(remote_url)
    except requests.HTTPError as exc:
        print(f"  ERROR listing {remote_url}: {exc}", file=sys.stderr)
        return 0
    parquets = [f for f in files if f.endswith(".parquet")]
    if not parquets:
        print(f"  (no parquet files found at {remote_url})")
        return 0
    print(f"  {len(parquets)} parquet files at {remote_url}")
    if dry_run:
        return 0

    n_new = 0
    for i, fname in enumerate(parquets, 1):
        out = local_dir / fname
        if out.exists() and out.stat().st_size > 0:
            continue
        url = urljoin(remote_url, fname)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(out, "wb") as h:
                for chunk in resp.iter_content(chunk_size=2 ** 20):
                    if chunk:
                        h.write(chunk)
        size_mb = out.stat().st_size / 1024 ** 2
        dt = time.time() - t0
        rate = size_mb / dt if dt > 0 else 0
        print(f"    [{i:>3}/{len(parquets)}] {fname}  "
              f"{size_mb:.1f} MB in {dt:.1f}s ({rate:.1f} MB/s)")
        n_new += 1
    return n_new


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--release", default=None,
        help="OT release version, e.g. '26.03' or '24.09'. Defaults to the "
             "latest release on the EBI FTP mirror.",
    )
    p.add_argument(
        "--data-dir", default="data",
        help="Local root under which a per-release subfolder is created. "
             "Defaults to ./data.",
    )
    p.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS,
        help="Which canonical datasets to fetch. Default: all three required "
             "by the pleiotropy notebook.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Only list what would be downloaded, don't fetch.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    release = args.release
    if release is None:
        try:
            releases = list_releases()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not list releases from {BASE_URL}: {exc}",
                  file=sys.stderr)
            return 2
        if not releases:
            print("No releases found at the OT FTP mirror.", file=sys.stderr)
            return 2
        release = releases[-1]
        print(f"Latest OT Platform release: {release}")

    layout = "new (snake_case)" if _release_uses_new_layout(release) else "old (camelCase, etl/parquet)"
    print(f"Path layout: {layout}")

    out_root = Path(args.data_dir) / f"ot_release_{release}"
    print(f"Output: {out_root}")

    if args.dry_run:
        approx_mb = sum(_SIZE_HINT_MB.get(d, 0) for d in args.datasets)
        print(f"Approx total download (dry run): ~{approx_mb} MB across "
              f"{len(args.datasets)} datasets")

    n_new_total = 0
    for ds in args.datasets:
        print(f"\n== {ds} ==")
        remote = remote_dataset_url(release, ds)
        if remote is None:
            print(f"  (skipping — {ds} doesn't exist in this layout)")
            continue
        # Always store locally under the canonical (new) name, even for
        # old-layout releases — keeps the notebook code path uniform.
        local = out_root / ds
        try:
            n_new_total += download_dataset(remote, local, dry_run=args.dry_run)
        except requests.HTTPError as exc:
            # Some datasets (clinical_target/indication on old releases,
            # known_drug on 26.03+) won't exist. Don't fail the whole run.
            if ds in OPTIONAL_DATASETS or exc.response is not None and exc.response.status_code == 404:
                print(f"  (not present in release {release} — skipping)")
                continue
            raise

    print(f"\nDone. {n_new_total} new file(s) downloaded.")
    print(f"To use in the notebook:\n  RELEASE_DIR = '{out_root}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
