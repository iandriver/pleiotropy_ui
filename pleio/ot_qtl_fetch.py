"""Open Targets fall-through fetchers for cis-pQTL instruments and per-disease
outcome effects.

We use these when a (target, disease) pair falls outside the hand-curated set
in :mod:`pleio.pqtl_instruments` / :mod:`pleio.disease_gwas`. The pattern is:

* try curated tables first (instant, deterministic)
* on miss, hit the Open Targets GraphQL endpoint
* parquet-cache the result on disk under ``data/`` so we don't re-fetch

The OT v25 GraphQL schema is mid-rewrite (Platform + Genetics merge), so each
fetcher tries **multiple query shapes** in order — whichever the live schema
accepts wins. If every shape fails the fetcher returns ``None`` and the
classifier falls back gracefully (no T1/T2 evidence, but T3/T4 paths still
work via the existing ``fetch_pav_evidence`` + ``genetic_association`` score).

If you find the live schema works with a shape we haven't listed, drop it
into ``_PQTL_QUERIES`` / ``_DISEASE_EFFECT_QUERIES`` — the multi-shape
fallback will pick it up automatically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from . import ot_pleiotropy as ot

# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.environ.get(
    "PLEIO_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data",
))
_INSTR_CACHE = _DATA_DIR / "ot_pqtl_instruments.parquet"
_OUT_CACHE = _DATA_DIR / "ot_disease_effects.parquet"


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001 — corrupt cache shouldn't kill the app
        return pd.DataFrame()


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    _ensure_data_dir()
    try:
        df.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 — disk failure shouldn't kill the app
        pass


# ---------------------------------------------------------------------------
# Cis-pQTL instrument fetcher
# ---------------------------------------------------------------------------
# Multiple query shapes — each represents a different way OT v25 might expose
# pQTL credible-set / evidence data. Order is most-likely-to-work first.
_PQTL_QUERIES: list[dict[str, str]] = [
    # Shape A: target.evidences filtered to pqtl-related datasources.
    {
        "name": "target.evidences[pqtl_credible_sets]",
        "query": """
        query Q($id: String!, $size: Int!) {
          target(ensemblId: $id) {
            evidences(
              datasourceIds: ["pqtl_credible_sets","ukb_ppp","decode_pqtl",
                              "pqtl_catalogue"],
              size: $size
            ) {
              count
              rows {
                datasourceId
                studyId
                variantId
                variantRsId
                beta
                betaCi95Lower
                betaCi95Upper
                pValueMantissa
                pValueExponent
                variantFunctionalConsequence { id label }
              }
            }
          }
        }
        """,
    },
    # Shape B: gentropy-style credibleSets root query filtered to pqtl + gene.
    {
        "name": "credibleSets(studyTypes:[pqtl], geneIds:[id])",
        "query": """
        query Q($id: String!, $size: Int!) {
          credibleSets(geneIds: [$id], studyTypes: ["pqtl"], size: $size) {
            rows {
              variantId
              variantRsId
              beta
              standardError
              pValueMantissa
              pValueExponent
              study { id studyType pubmedId }
            }
          }
        }
        """,
    },
    # Shape C: legacy genetics portal — variantToTarget for cis pQTLs.
    {
        "name": "target.studyLocus[pqtl]",
        "query": """
        query Q($id: String!, $size: Int!) {
          target(ensemblId: $id) {
            studyLocus(studyTypes: ["pqtl"], size: $size) {
              rows {
                variantRsId
                variantId
                beta
                standardError
                pValueExponent
                pValueMantissa
              }
            }
          }
        }
        """,
    },
]


def _parse_pqtl_response(payload: dict, shape_name: str) -> list[dict]:
    """Walk the response from one of our query shapes into our row schema."""
    rows: list[dict] = []
    # Shape A / C — under target.evidences or target.studyLocus
    target = payload.get("target")
    if target:
        block = target.get("evidences") or target.get("studyLocus")
        if block:
            for r in block.get("rows", []) or []:
                rows.append(_evidence_row_to_instrument(r, shape_name))
    # Shape B — root credibleSets
    cs = payload.get("credibleSets")
    if cs:
        for r in cs.get("rows", []) or []:
            rows.append(_evidence_row_to_instrument(r, shape_name))
    return [r for r in rows if r is not None]


def _evidence_row_to_instrument(r: dict, source_label: str) -> dict | None:
    """Coerce one OT row into our (rsid, chrom, pos, ea, oa, eaf, beta, se, p)
    schema. Returns None if essential fields are missing."""
    rsid = r.get("variantRsId") or r.get("rsId")
    if not rsid:
        # Try parsing variantId like "1_55505647_T_G".
        vid = r.get("variantId") or ""
        if not vid or vid.count("_") != 3:
            return None
        chrom, pos, oa, ea = vid.split("_")
    else:
        vid = r.get("variantId") or ""
        chrom = pos = ea = oa = None
        if vid.count("_") == 3:
            chrom, pos, oa, ea = vid.split("_")

    beta = r.get("beta")
    if beta is None:
        return None
    se = r.get("standardError")
    if se is None:
        # Derive from CI if available: SE ≈ (upper − lower) / (2 * 1.96)
        lo, hi = r.get("betaCi95Lower"), r.get("betaCi95Upper")
        if lo is not None and hi is not None:
            se = (float(hi) - float(lo)) / (2 * 1.959963984540054)
    if se is None or float(se) <= 0:
        return None

    p_mantissa = r.get("pValueMantissa")
    p_exponent = r.get("pValueExponent")
    pval: float | None = None
    if p_mantissa is not None and p_exponent is not None:
        try:
            pval = float(p_mantissa) * (10 ** float(p_exponent))
        except (TypeError, ValueError):
            pval = None

    return {
        "rsid": rsid,
        "chrom": chrom,
        "pos": int(pos) if pos and str(pos).isdigit() else None,
        "ea": ea,
        "oa": oa,
        "eaf": None,           # OT doesn't always expose EAF on credible sets
        "beta": float(beta),
        "se": float(se),
        "p": pval,
        "source": f"OT:{source_label}",
    }


def fetch_cis_pqtl_instruments(symbol_or_ensg: str,
                               size: int = 25,
                               use_cache: bool = True) -> pd.DataFrame:
    """Fetch cis-pQTL instruments for a target via OT GraphQL.

    Returns an empty DataFrame on failure — never raises. Successful fetches
    are appended to a parquet cache so subsequent calls are instant.

    The returned DataFrame matches the schema used by
    :func:`pleio.pqtl_instruments.get_instruments`:
    ``rsid, chrom, pos, ea, oa, eaf, beta, se, p, source``.
    """
    ensg = ot.resolve_target(symbol_or_ensg)
    if ensg is None:
        return pd.DataFrame()

    if use_cache:
        cache = _load_cache(_INSTR_CACHE)
        if not cache.empty and "ensg" in cache.columns:
            hit = cache[cache["ensg"] == ensg]
            if not hit.empty:
                return hit.drop(columns=["ensg"]).copy()

    rows: list[dict] = []
    for shape in _PQTL_QUERIES:
        try:
            data = ot.gql(shape["query"], {"id": ensg, "size": size})
        except Exception:  # noqa: BLE001 — try the next shape
            continue
        rows = _parse_pqtl_response(data, shape["name"])
        if rows:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["rsid"])
    # Persist to cache
    if use_cache:
        cache = _load_cache(_INSTR_CACHE)
        df_cached = df.copy()
        df_cached["ensg"] = ensg
        # Drop any stale rows for this gene before appending.
        if not cache.empty and "ensg" in cache.columns:
            cache = cache[cache["ensg"] != ensg]
        merged = pd.concat([cache, df_cached], ignore_index=True)
        _save_cache(_INSTR_CACHE, merged)
    return df


# ---------------------------------------------------------------------------
# Per-disease, per-variant outcome effect fetcher
# ---------------------------------------------------------------------------
# OT exposes per-evidence variant effects via the disease.evidences endpoint
# filtered to the GWAS credible-set datasource. We pull all evidence rows and
# filter client-side to the rsids we want — simpler than per-variant queries.
_DISEASE_EFFECT_QUERY = """
query Q($id: String!, $size: Int!) {
  disease(efoId: $id) {
    id
    name
    evidences(
      datasourceIds: ["gwas_credible_sets","ot_genetics_portal","gwas_catalog"],
      size: $size
    ) {
      count
      rows {
        datasourceId
        variantId
        variantRsId
        beta
        betaCi95Lower
        betaCi95Upper
        oddsRatio
        oddsRatioCi95Lower
        oddsRatioCi95Upper
        pValueMantissa
        pValueExponent
      }
    }
  }
}
"""


def _evidence_row_to_outcome_effect(r: dict, binary: bool) -> dict | None:
    rsid = r.get("variantRsId")
    if not rsid:
        return None
    vid = r.get("variantId") or ""
    chrom = pos = ea = oa = None
    if vid.count("_") == 3:
        chrom, pos, oa, ea = vid.split("_")

    beta = r.get("beta")
    se = None
    if beta is None and binary and r.get("oddsRatio") is not None:
        # Convert OR + CI to log-OR + SE.
        import math
        try:
            beta = math.log(float(r["oddsRatio"]))
            lo, hi = r.get("oddsRatioCi95Lower"), r.get("oddsRatioCi95Upper")
            if lo and hi:
                se = (math.log(float(hi)) - math.log(float(lo))) / (2 * 1.959963984540054)
        except (ValueError, TypeError):
            beta = None
    if beta is None:
        return None
    if se is None:
        lo, hi = r.get("betaCi95Lower"), r.get("betaCi95Upper")
        if lo is not None and hi is not None:
            se = (float(hi) - float(lo)) / (2 * 1.959963984540054)
    if se is None or float(se) <= 0:
        return None

    return {
        "rsid": rsid,
        "ea": ea,
        "oa": oa,
        "eaf": None,
        "beta": float(beta),
        "se": float(se),
    }


# Diseases with binary GWAS effect estimands (log-OR). Anything else is
# treated as continuous (SD-units β). Heuristic — falls back to "binary" for
# anything ending in -itis / -emia / etc. without strong evidence either way.
_KNOWN_CONTINUOUS_EFOS = {
    "EFO_0004458",   # CRP
    "EFO_0004611",   # LDL-C
    "EFO_0004612",   # HDL-C
    "EFO_0004530",   # TG
    "EFO_0004340",   # BMI
}


def is_efo_binary(efo_id: str) -> bool:
    """Heuristic — only the known measurement EFOs are continuous."""
    return efo_id not in _KNOWN_CONTINUOUS_EFOS


def fetch_disease_outcome_effects(efo_id: str,
                                  rsids: list[str],
                                  size: int = 500,
                                  use_cache: bool = True) -> pd.DataFrame:
    """Fetch per-variant disease GWAS effects for an arbitrary EFO.

    Returns rows in the same schema as :func:`disease_gwas.get_outcome_effects`:
    ``rsid, ea, oa, eaf, beta, se``. Empty DataFrame on failure or no overlap.
    """
    if not rsids:
        return pd.DataFrame()
    binary = is_efo_binary(efo_id)

    if use_cache:
        cache = _load_cache(_OUT_CACHE)
        if not cache.empty and "efo_id" in cache.columns:
            hit = cache[(cache["efo_id"] == efo_id)
                        & cache["rsid"].isin(rsids)]
            if not hit.empty and len(hit) >= min(3, len(rsids)):
                return hit.drop(columns=["efo_id"]).copy()

    try:
        data = ot.gql(_DISEASE_EFFECT_QUERY, {"id": efo_id, "size": size})
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    d = data.get("disease")
    if not d:
        return pd.DataFrame()

    target_rsids = set(rsids)
    rows: list[dict] = []
    for r in (d.get("evidences") or {}).get("rows", []) or []:
        if r.get("variantRsId") not in target_rsids:
            continue
        out = _evidence_row_to_outcome_effect(r, binary)
        if out is not None:
            rows.append(out)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["rsid"])

    if use_cache:
        cache = _load_cache(_OUT_CACHE)
        df_cached = df.copy()
        df_cached["efo_id"] = efo_id
        if not cache.empty and "efo_id" in cache.columns:
            cache = cache[~((cache["efo_id"] == efo_id)
                            & cache["rsid"].isin(df["rsid"]))]
        merged = pd.concat([cache, df_cached], ignore_index=True)
        _save_cache(_OUT_CACHE, merged)

    return df


# ---------------------------------------------------------------------------
# Convenience: clear OT-fetched caches (for the Streamlit "Clear cache" btn)
# ---------------------------------------------------------------------------
def clear_caches() -> None:
    for path in (_INSTR_CACHE, _OUT_CACHE):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def cache_summary() -> dict[str, Any]:
    instr_cache = _load_cache(_INSTR_CACHE)
    out_cache = _load_cache(_OUT_CACHE)
    return {
        "instruments_cached_genes":
            int(instr_cache["ensg"].nunique()) if "ensg" in instr_cache.columns else 0,
        "instruments_cached_rows": int(len(instr_cache)),
        "outcome_cached_efos":
            int(out_cache["efo_id"].nunique()) if "efo_id" in out_cache.columns else 0,
        "outcome_cached_rows": int(len(out_cache)),
    }
