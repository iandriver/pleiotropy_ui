"""Open Targets Platform GraphQL client + on-the-fly pleiotropy computation.

Recreates the *spirit* of the Gentropy pleiotropy map (Mountjoy/Ochoa et al.
2026, biorxiv 10.64898/2026.04.28.721048v1) using the public OT Platform
GraphQL API instead of the 40 GB GCS release.

Pleiotropy here is defined as the number of distinct EFO therapeutic areas
a target is associated with above a chosen genetic-association score
threshold. This is a reasonable proxy — the manuscript uses a richer
definition built on fine-mapped credible sets + L2G — but it gives directly
comparable counts and the same intermediate-vs-high distinction.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib import error, request

import pandas as pd

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# Sequence Ontology terms the manuscript treats as protein-altering.
# Matched against variantFunctionalConsequence.label (lower-snake) AND .id (SO_xxxx).
PAV_LABELS = {
    "missense_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
    "transcript_ablation",
    "transcript_amplification",
    "inframe_insertion",
    "inframe_deletion",
    "protein_altering_variant",
    "incomplete_terminal_codon_variant",
    "coding_sequence_variant",
}
PAV_SO_IDS = {
    "SO_0001583",  # missense
    "SO_0001587",  # stop_gained
    "SO_0001578",  # stop_lost
    "SO_0002012",  # start_lost
    "SO_0001589",  # frameshift
    "SO_0001574",  # splice_acceptor
    "SO_0001575",  # splice_donor
    "SO_0001630",  # splice_region
    "SO_0001893",  # transcript_ablation
    "SO_0001889",  # transcript_amplification
    "SO_0001821",  # inframe_insertion
    "SO_0001822",  # inframe_deletion
    "SO_0001818",  # protein_altering_variant
}


# ---------------------------------------------------------------------------
# Low-level GraphQL helper
# ---------------------------------------------------------------------------
class OTError(RuntimeError):
    pass


def gql(query: str, variables: dict | None = None, timeout: int = 60, retries: int = 2) -> dict:
    """POST a GraphQL query to the OT Platform endpoint.

    On HTTP errors (including 400 from a schema mismatch) the server's
    JSON response body is included in the raised exception so the caller
    sees the GraphQL validation error string, not just the bare HTTP code.
    """
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = request.Request(
                API_URL,
                data=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
            if "errors" in payload:
                raise OTError(payload["errors"])
            return payload["data"]
        except error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                detail = "<no body>"
            last_exc = OTError(f"HTTP {exc.code}: {detail[:2000]}")
            if exc.code in (429, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_exc from exc
        except (error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise OTError(f"giving up: {last_exc}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------
_SEARCH_Q = """
query S($q: String!, $e: [String!]!) {
  search(queryString: $q, entityNames: $e) {
    hits { id name entity }
  }
}
"""


@lru_cache(maxsize=4096)
def _search(query: str, entity: str) -> tuple[dict, ...]:
    data = gql(_SEARCH_Q, {"q": query, "e": [entity]})
    return tuple(data["search"]["hits"])


def resolve_target(symbol_or_id: str) -> str | None:
    """Resolve a gene symbol (or ENSG) to an Ensembl ID."""
    s = symbol_or_id.strip()
    if not s:
        return None
    if s.upper().startswith("ENSG"):
        return s
    hits = _search(s, "target")
    # prefer exact-symbol match (case-insensitive)
    for h in hits:
        if h["entity"] == "target" and h["name"].upper() == s.upper():
            return h["id"]
    for h in hits:
        if h["entity"] == "target":
            return h["id"]
    return None


def resolve_disease(name_or_id: str) -> str | None:
    """Resolve a disease name (or EFO ID) to its EFO/MONDO/etc. identifier."""
    s = name_or_id.strip()
    if not s:
        return None
    # rough heuristic for an ontology ID
    if any(s.upper().startswith(p) for p in ("EFO_", "MONDO_", "HP_", "ORPHANET_", "OTAR_", "DOID_", "MESH_")):
        return s
    hits = _search(s, "disease")
    for h in hits:
        if h["entity"] == "disease" and h["name"].lower() == s.lower():
            return h["id"]
    for h in hits:
        if h["entity"] == "disease":
            return h["id"]
    return None


# ---------------------------------------------------------------------------
# Target → associated diseases
# ---------------------------------------------------------------------------
# Core target query — metadata + associatedDiseases. Drugs are fetched
# separately because OT is mid-rewrite of the drug pipelines (March 2026)
# and `knownDrugs` was removed from the Target type in v25.
_TARGET_Q = """
query T($id: String!, $size: Int!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    associatedDiseases(page: {index: 0, size: $size}) {
      count
      rows {
        score
        datatypeScores { id score }
        disease {
          id
          name
          therapeuticAreas { id name }
        }
      }
    }
  }
}
"""

# Drugs query — tried in order; whichever shape the live schema accepts wins.
# We try the legacy knownDrugs first (still present in some releases), then
# the post-rewrite mechanismsOfAction. If neither works we return empty —
# drug data isn't on the demo critical path.
_DRUGS_QUERIES = [
    # Shape A: legacy knownDrugs (pre-v25)
    """query D($id: String!) {
      target(ensemblId: $id) {
        knownDrugs(size: 50) {
          count
          rows {
            drug { id name }
            phase status mechanismOfAction
            disease { id name }
          }
        }
      }
    }""",
    # Shape B: mechanismsOfAction (current v25 path for target → drugs)
    """query D($id: String!) {
      target(ensemblId: $id) {
        mechanismsOfAction {
          rows {
            mechanismOfAction
            actionType
            targetName
            references { source urls }
          }
        }
      }
    }""",
]


def fetch_target(ensg: str, page_size: int = 500) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    """Return (target_meta, associations_df, drugs_df).

    Drug data is best-effort: if the drugs sub-query fails (e.g. mid-release
    rewrite), we return an empty drugs DataFrame and continue. The core
    pleiotropy outputs only depend on associatedDiseases, which is stable.
    """
    data = gql(_TARGET_Q, {"id": ensg, "size": page_size})
    t = data.get("target")
    if t is None:
        return None, pd.DataFrame(), pd.DataFrame()

    rows = []
    for r in t["associatedDiseases"]["rows"]:
        d = r["disease"]
        scores = {s["id"]: s["score"] for s in (r.get("datatypeScores") or [])}
        rows.append(
            {
                "ensg": t["id"],
                "symbol": t["approvedSymbol"],
                "disease_id": d["id"],
                "disease_name": d["name"],
                "therapeutic_areas": [(ta["id"], ta["name"]) for ta in (d.get("therapeuticAreas") or [])],
                "ta_count_for_disease": len(d.get("therapeuticAreas") or []),
                "overall_score": r["score"],
                "genetic_association_score": scores.get("genetic_association", 0.0),
                "known_drug_score": scores.get("known_drug", 0.0),
                "animal_model_score": scores.get("animal_model", 0.0),
                "literature_score": scores.get("literature", 0.0),
            }
        )
    df = pd.DataFrame(rows)

    # Drugs — try shapes A and B; whichever the live schema accepts wins.
    drugs = pd.DataFrame()
    n_known_drugs = 0
    for q in _DRUGS_QUERIES:
        try:
            d_data = gql(q, {"id": ensg})
            t2 = d_data.get("target") or {}
            if "knownDrugs" in t2 and t2["knownDrugs"]:
                kd = t2["knownDrugs"]
                n_known_drugs = kd.get("count", 0)
                drugs = pd.DataFrame([
                    {
                        "drug": (r.get("drug") or {}).get("name"),
                        "phase": r.get("phase"),
                        "status": r.get("status"),
                        "mechanism": r.get("mechanismOfAction"),
                        "disease": (r.get("disease") or {}).get("name"),
                    }
                    for r in (kd.get("rows") or [])
                ])
            elif "mechanismsOfAction" in t2 and t2["mechanismsOfAction"]:
                moa = t2["mechanismsOfAction"]
                drugs = pd.DataFrame([
                    {
                        "drug": None,
                        "phase": None,
                        "status": None,
                        "mechanism": r.get("mechanismOfAction"),
                        "disease": None,
                        "action_type": r.get("actionType"),
                    }
                    for r in (moa.get("rows") or [])
                ])
                n_known_drugs = len(drugs)
            break
        except OTError:
            continue

    meta = {
        "id": t["id"],
        "symbol": t["approvedSymbol"],
        "name": t["approvedName"],
        "biotype": t.get("biotype"),
        "function": (t.get("functionDescriptions") or [None])[0],
        "n_associations_total": t["associatedDiseases"]["count"],
        "n_known_drugs_total": n_known_drugs,
    }
    return meta, df, drugs


# ---------------------------------------------------------------------------
# Disease → associated targets
# ---------------------------------------------------------------------------
_DISEASE_Q = """
query D($id: String!, $size: Int!) {
  disease(efoId: $id) {
    id
    name
    description
    therapeuticAreas { id name }
    associatedTargets(page: {index: 0, size: $size}) {
      count
      rows {
        score
        datatypeScores { id score }
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
      }
    }
  }
}
"""


def fetch_disease(efo: str, page_size: int = 200) -> tuple[dict | None, pd.DataFrame]:
    data = gql(_DISEASE_Q, {"id": efo, "size": page_size})
    d = data.get("disease")
    if d is None:
        return None, pd.DataFrame()
    rows = []
    for r in d["associatedTargets"]["rows"]:
        t = r["target"]
        scores = {s["id"]: s["score"] for s in (r.get("datatypeScores") or [])}
        rows.append(
            {
                "ensg": t["id"],
                "symbol": t["approvedSymbol"],
                "name": t["approvedName"],
                "biotype": t.get("biotype"),
                "overall_score": r["score"],
                "genetic_association_score": scores.get("genetic_association", 0.0),
                "known_drug_score": scores.get("known_drug", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    meta = {
        "id": d["id"],
        "name": d["name"],
        "description": d.get("description"),
        "therapeutic_areas": [(ta["id"], ta["name"]) for ta in (d.get("therapeuticAreas") or [])],
        "n_targets_total": d["associatedTargets"]["count"],
    }
    return meta, df


# ---------------------------------------------------------------------------
# PAV detection via target.evidences
# ---------------------------------------------------------------------------
# Datasource IDs follow OT Platform v25.x. The old "ot_genetics_portal" was
# renamed "gwas_credible_sets" when the Platform merged with Genetics; we keep
# both so we work across releases. "eva" was renamed "clinvar" in v25.
_EVIDENCE_Q = """
query Ev($id: String!, $size: Int!) {
  target(ensemblId: $id) {
    evidences(
      datasourceIds: ["gwas_credible_sets","ot_genetics_portal","gene_burden","clingen","orphanet","gene2phenotype","clinvar","eva","uniprot_literature","uniprot_variants","intogen"],
      size: $size
    ) {
      count
      rows {
        datasourceId
        disease { id name }
        variantFunctionalConsequence { id label }
        variantId
      }
    }
  }
}
"""


def _is_pav(consequence: dict | None) -> bool:
    if not consequence:
        return False
    lbl = (consequence.get("label") or "").strip().lower().replace(" ", "_")
    soid = (consequence.get("id") or "").strip()
    return lbl in PAV_LABELS or soid in PAV_SO_IDS


def fetch_pav_evidence(ensg: str, size: int = 1000) -> pd.DataFrame:
    """Return per-disease PAV evidence rows (empty if none)."""
    try:
        data = gql(_EVIDENCE_Q, {"id": ensg, "size": size})
    except OTError:
        return pd.DataFrame()
    t = data.get("target")
    if not t:
        return pd.DataFrame()
    rows = []
    for ev in (t.get("evidences") or {}).get("rows", []) or []:
        if _is_pav(ev.get("variantFunctionalConsequence")):
            rows.append(
                {
                    "disease_id": (ev.get("disease") or {}).get("id"),
                    "disease": (ev.get("disease") or {}).get("name"),
                    "consequence": (ev.get("variantFunctionalConsequence") or {}).get("label"),
                    "variant_id": ev.get("variantId"),
                    "source": ev.get("datasourceId"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pleiotropy summary
# ---------------------------------------------------------------------------
@dataclass
class PleiotropySummary:
    symbol: str
    ensg: str
    n_diseases: int
    n_therapeutic_areas: int
    therapeutic_areas: list[str]
    classification: str  # "specific" | "intermediate" | "highly_pleiotropic" | "none"
    pav_supported: bool
    pav_diseases: list[str] = field(default_factory=list)
    sweet_spot: bool = False  # intermediate AND PAV-supported

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ensg": self.ensg,
            "n_diseases": self.n_diseases,
            "n_therapeutic_areas": self.n_therapeutic_areas,
            "therapeutic_areas": "; ".join(self.therapeutic_areas),
            "classification": self.classification,
            "pav_supported": self.pav_supported,
            "sweet_spot": self.sweet_spot,
        }


def classify_pleiotropy(n_ta: int) -> str:
    """Manuscript-style buckets: 1 = specific, 2-5 = intermediate (sweet spot),
    >=6 = highly pleiotropic."""
    if n_ta == 0:
        return "none"
    if n_ta == 1:
        return "specific"
    if 2 <= n_ta <= 5:
        return "intermediate"
    return "highly_pleiotropic"


def summarize_target_pleiotropy(
    symbol_or_id: str,
    ga_threshold: float = 0.1,
    overall_threshold: float = 0.0,
    check_pav: bool = True,
) -> PleiotropySummary | None:
    """End-to-end: resolve target → fetch associations → classify → optionally check PAV."""
    ensg = resolve_target(symbol_or_id)
    if ensg is None:
        return None
    meta, df, _ = fetch_target(ensg)
    if meta is None:
        return None
    sub = df[
        (df["genetic_association_score"] >= ga_threshold)
        & (df["overall_score"] >= overall_threshold)
    ]
    tas: set[str] = set()
    for tas_list in sub["therapeutic_areas"]:
        for _id, name in tas_list:
            tas.add(name)
    n_ta = len(tas)
    cls = classify_pleiotropy(n_ta)

    pav_supported = False
    pav_diseases: list[str] = []
    if check_pav:
        pav_df = fetch_pav_evidence(ensg)
        pav_supported = not pav_df.empty
        if pav_supported:
            pav_diseases = sorted(set(pav_df["disease"].dropna().tolist()))

    return PleiotropySummary(
        symbol=meta["symbol"],
        ensg=meta["id"],
        n_diseases=int(len(sub)),
        n_therapeutic_areas=n_ta,
        therapeutic_areas=sorted(tas),
        classification=cls,
        pav_supported=pav_supported,
        pav_diseases=pav_diseases,
        sweet_spot=(cls == "intermediate" and pav_supported),
    )


def summarize_many(
    symbols: list[str],
    ga_threshold: float = 0.1,
    overall_threshold: float = 0.0,
    check_pav: bool = True,
    progress_cb=None,
) -> pd.DataFrame:
    """Batch-summarize a list of gene symbols. progress_cb(i, n, symbol) is called per gene."""
    rows = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(i, n, sym)
        try:
            s = summarize_target_pleiotropy(sym, ga_threshold, overall_threshold, check_pav)
        except Exception as exc:  # noqa: BLE001 — surface broadly in batch context
            rows.append({"symbol": sym, "ensg": None, "error": str(exc)})
            continue
        if s is None:
            rows.append({"symbol": sym, "ensg": None, "error": "not_found"})
            continue
        d = s.as_dict()
        d["error"] = None
        rows.append(d)
    if progress_cb:
        progress_cb(n, n, None)
    return pd.DataFrame(rows)
