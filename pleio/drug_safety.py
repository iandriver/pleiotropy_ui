"""Drug-safety / clinical / tractability feature table from the local OT release.

Replaces the fragile GraphQL ``knownDrugs`` path with the post-26.03 parquet
exports:

* ``target_prioritisation`` — safety event flag, tractability, genetic
  constraint, mouse KO score, etc.
* ``clinical_target``       — per (drug × target × max clinical stage).
* ``clinical_indication``   — per (drug × disease × max clinical stage).
* ``target`` / ``disease``  — symbol & name lookup.

The merged per-gene feature table is loaded from
``data/all_genes_features.parquet`` if it exists (faster cold-start), and
otherwise built from the raw OT parquets and cached to that path on first
call. This mirrors the analysis in
``notebooks/pleiotropy_drugs_safety.ipynb`` but with a clinical-stage map
that matches the 26.03 string values (``APPROVAL``, ``PHASE_3``, …) instead
of the title-case 25.x labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_FEATURES_PARQUET = _DATA_DIR / "all_genes_features.parquet"
_PLEIO_PARQUET = _DATA_DIR / "all_genes_pleiotropy_l2g.parquet"


def _latest_release_dir() -> Path:
    rels = sorted(_DATA_DIR.glob("ot_release_*"))
    if not rels:
        raise FileNotFoundError(
            f"No OT release found under {_DATA_DIR}. "
            "Run scripts/download_ot_release.py."
        )
    return rels[-1]


# ---------------------------------------------------------------------------
# Stage map — 26.03 string values from clinical_target.maxClinicalStage
# ---------------------------------------------------------------------------
STAGE_TO_NUM: dict[str, float] = {
    "APPROVAL":      4.0,
    "PREAPPROVAL":   3.5,
    "PHASE_3":       3.0,
    "PHASE_2_3":     2.5,
    "PHASE_2":       2.0,
    "PHASE_1_2":     1.5,
    "PHASE_1":       1.0,
    "EARLY_PHASE_1": 0.9,
    "IND":           0.7,
    "PRECLINICAL":   0.5,
    "UNKNOWN":       0.0,
}
STAGE_LABEL: dict[str, str] = {
    "APPROVAL":      "Approved",
    "PREAPPROVAL":   "Pre-approval",
    "PHASE_3":       "Phase 3",
    "PHASE_2_3":     "Phase 2/3",
    "PHASE_2":       "Phase 2",
    "PHASE_1_2":     "Phase 1/2",
    "PHASE_1":       "Phase 1",
    "EARLY_PHASE_1": "Early Phase 1",
    "IND":           "IND",
    "PRECLINICAL":   "Preclinical",
    "UNKNOWN":       "Unknown",
}
PHASE_NUM_TO_LABEL: dict[float, str] = {v: STAGE_LABEL[k] for k, v in STAGE_TO_NUM.items()}

BUCKET_ORDER = ["none", "specific", "intermediate", "highly_pleiotropic"]
BUCKET_COLOR = {
    "none":               "#bdbdbd",
    "specific":           "#27ae60",
    "intermediate":       "#e3b505",
    "highly_pleiotropic": "#c0392b",
}


# ---------------------------------------------------------------------------
# Build / load the merged feature table
# ---------------------------------------------------------------------------
def _stage_to_num(s: pd.Series) -> pd.Series:
    """Map a Series of OT 26.03 stage strings to numeric phase.

    Falls through to ``to_numeric`` if values already look numeric (legacy
    ``known_drug.phase`` from ≤25.06 releases).
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num.fillna(0.0)
    return s.astype(str).map(STAGE_TO_NUM).fillna(0.0)


def _build_features() -> pd.DataFrame:
    release = _latest_release_dir()

    target = pd.read_parquet(
        release / "target", columns=["id", "approvedSymbol", "biotype"]
    )
    tp = pd.read_parquet(release / "target_prioritisation")
    # OT 26.03 prioritisation columns are signed scores. Penalty-direction
    # flags (hasSafetyEvent, isCancerDriverGene) take the value -1 when the
    # event/property is present; positive-direction flags take 1 when present
    # and 0 otherwise.
    POS_BOOL_COLS = [
        "hasPocket", "hasLigand", "hasSmallMoleculeBinder",
        "hasTEP", "hasHighQualityChemicalProbes",
        "isInMembrane", "isSecreted",
    ]
    NEG_BOOL_COLS = ["hasSafetyEvent", "isCancerDriverGene"]
    feat_bool_cols = POS_BOOL_COLS + NEG_BOOL_COLS
    tp_slim = tp.rename(columns={"targetId": "geneId"})[
        ["geneId"]
        + [c for c in feat_bool_cols if c in tp.columns]
        + [c for c in ["geneticConstraint", "mouseKOScore",
                       "maxClinicalStage", "tissueSpecificity",
                       "tissueDistribution"] if c in tp.columns]
    ].copy()
    for c in POS_BOOL_COLS:
        if c in tp_slim.columns:
            tp_slim[c] = tp_slim[c].fillna(0).astype(float) > 0
    for c in NEG_BOOL_COLS:
        if c in tp_slim.columns:
            # -1 = present (penalty), 0 / NaN = absent.
            tp_slim[c] = tp_slim[c].fillna(0).astype(float) < 0
    tp_slim = tp_slim.rename(columns={
        "hasSafetyEvent":       "has_safety_event",
        "hasPocket":            "has_pocket",
        "hasLigand":            "has_ligand",
        "hasSmallMoleculeBinder": "has_small_mol_binder",
        "isCancerDriverGene":   "is_cancer_driver",
        "hasTEP":               "has_tep",
        "hasHighQualityChemicalProbes": "has_chem_probes",
        "isInMembrane":         "is_in_membrane",
        "isSecreted":           "is_secreted",
        "geneticConstraint":    "genetic_constraint",
        "mouseKOScore":         "mouse_ko_score",
        "maxClinicalStage":     "max_clinical_stage_score",
        "tissueSpecificity":    "tissue_specificity",
        "tissueDistribution":   "tissue_distribution",
    })

    # Per-target drug aggregation from clinical_target × clinical_indication.
    ct = pd.read_parquet(release / "clinical_target")
    ci_path = release / "clinical_indication"
    have_ci = ci_path.exists() and any(ci_path.iterdir())
    ct = ct.copy()
    ct["_phase"] = _stage_to_num(ct["maxClinicalStage"])
    drug_per_target = (
        ct.groupby("targetId")
          .agg(max_drug_phase=("_phase", "max"),
               n_drugs=("drugId", "nunique"))
          .reset_index()
          .rename(columns={"targetId": "geneId"})
    )
    if have_ci:
        ci = pd.read_parquet(ci_path)
        target_drug = ct[["targetId", "drugId"]].drop_duplicates()
        bridge = target_drug.merge(
            ci[["drugId", "diseaseId"]].drop_duplicates(),
            on="drugId", how="inner",
        )
        n_drugged = (
            bridge.groupby("targetId")["diseaseId"]
            .nunique()
            .reset_index()
            .rename(columns={"targetId": "geneId",
                             "diseaseId": "n_drugged_diseases"})
        )
        drug_per_target = drug_per_target.merge(
            n_drugged, on="geneId", how="left"
        )
    else:
        drug_per_target["n_drugged_diseases"] = 0

    drug_per_target["has_approved"] = drug_per_target["max_drug_phase"] >= 4
    drug_per_target["has_clinical"] = drug_per_target["max_drug_phase"] >= 1

    # Pleiotropy bucket (from the L2G notebook output).
    if _PLEIO_PARQUET.exists():
        pleio = pd.read_parquet(
            _PLEIO_PARQUET,
            columns=["geneId", "approvedSymbol", "n_loci", "n_diseases", "bucket"],
        )
    else:
        pleio = target.rename(columns={"id": "geneId"})[["geneId", "approvedSymbol"]].copy()
        pleio["n_loci"] = 0
        pleio["n_diseases"] = 0
        pleio["bucket"] = "none"

    merged = (
        pleio.merge(tp_slim, on="geneId", how="left")
             .merge(drug_per_target, on="geneId", how="left")
    )
    # fillna+astype on object columns triggers a Future-downcast warning in
    # pandas ≥2.2; cast first to avoid the chatter.
    merged["has_approved"] = merged["has_approved"].astype("boolean").fillna(False).astype(bool)
    merged["has_clinical"] = merged["has_clinical"].astype("boolean").fillna(False).astype(bool)
    merged["max_drug_phase"] = merged["max_drug_phase"].fillna(0.0)
    merged["n_drugs"] = merged["n_drugs"].fillna(0).astype(int)
    merged["n_drugged_diseases"] = merged["n_drugged_diseases"].fillna(0).astype(int)
    for c in ["has_safety_event", "has_pocket", "has_ligand",
              "has_small_mol_binder", "is_cancer_driver", "has_tep",
              "has_chem_probes", "is_in_membrane", "is_secreted"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(False).astype(bool)
    if "bucket" in merged.columns:
        merged["bucket"] = pd.Categorical(
            merged["bucket"].fillna("none"),
            categories=BUCKET_ORDER, ordered=True,
        )
    return merged


@lru_cache(maxsize=1)
def get_features() -> pd.DataFrame:
    """Return the per-gene drug/safety/tractability feature table.

    Reads from ``data/all_genes_features.parquet`` if available, otherwise
    builds the table from raw OT parquets and writes the result to that
    path so subsequent cold starts are instant.
    """
    if _FEATURES_PARQUET.exists():
        df = pd.read_parquet(_FEATURES_PARQUET)
        if "bucket" in df.columns:
            df["bucket"] = pd.Categorical(
                df["bucket"].fillna("none"),
                categories=BUCKET_ORDER, ordered=True,
            )
        return df
    df = _build_features()
    try:
        out = df.copy()
        if "bucket" in out.columns:
            out["bucket"] = out["bucket"].astype(str)
        out.to_parquet(_FEATURES_PARQUET, index=False)
    except Exception:  # noqa: BLE001
        # Caching is best-effort — never block the app if disk is read-only.
        pass
    return df


def clear_cache() -> None:
    """Drop the in-memory feature-table cache (does not delete the parquet)."""
    get_features.cache_clear()
    _target_drugs_cache.clear()


# ---------------------------------------------------------------------------
# Per-target views
# ---------------------------------------------------------------------------
def _resolve_geneid(symbol_or_id: str) -> str | None:
    """Map a gene symbol or ENSG to ENSG using the local target table."""
    if not symbol_or_id:
        return None
    s = symbol_or_id.strip()
    if s.upper().startswith("ENSG"):
        return s
    feats = get_features()
    hit = feats[feats["approvedSymbol"].str.upper() == s.upper()]
    if not hit.empty:
        return str(hit["geneId"].iloc[0])
    return None


def target_features(symbol_or_id: str) -> dict | None:
    """Return the feature row for a gene (symbol or ENSG) as a dict, or None."""
    gene = _resolve_geneid(symbol_or_id)
    if gene is None:
        return None
    feats = get_features()
    hit = feats[feats["geneId"] == gene]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


_target_drugs_cache: dict[str, pd.DataFrame] = {}
_disease_name_cache: pd.DataFrame | None = None
_disease_status_cache: dict[str, dict[str, set[str]]] = {}


def _disease_names() -> pd.DataFrame:
    global _disease_name_cache
    if _disease_name_cache is None:
        release = _latest_release_dir()
        _disease_name_cache = pd.read_parquet(
            release / "disease", columns=["id", "name"]
        ).rename(columns={"id": "diseaseId", "name": "disease_name"})
    return _disease_name_cache


@lru_cache(maxsize=1)
def all_diseases() -> pd.DataFrame:
    """Return all OT diseases as a DataFrame: ``id``, ``name``, ``label``.

    ``label`` is the human-readable string ``"<name>  (<id>)"`` used to
    populate the disease autosuggest field — keeping the id in the visible
    label makes it unambiguous when many diseases share similar names
    (e.g. ``BMI`` vs ``BMI-adjusted waist-hip ratio``).
    """
    release = _latest_release_dir()
    d = pd.read_parquet(release / "disease", columns=["id", "name"])
    d = d.dropna(subset=["id", "name"]).copy()
    d["label"] = d["name"] + "  (" + d["id"] + ")"
    return d.sort_values("name", kind="stable").reset_index(drop=True)


def diseases_by_id(disease_ids: Iterable[str]) -> pd.DataFrame:
    """Return name + label for the requested disease IDs (skips unknowns)."""
    ids = list(disease_ids)
    if not ids:
        return pd.DataFrame(columns=["id", "name", "label"])
    cat = all_diseases()
    return cat[cat["id"].isin(ids)].reset_index(drop=True)


def target_drugs(symbol_or_id: str) -> pd.DataFrame:
    """Return per-drug rows for a target.

    Columns: ``drug_id``, ``max_clinical_stage`` (label), ``phase`` (numeric),
    ``indications`` (string list of disease names from clinical_indication for
    the same drugId), ``n_indications``. Empty DataFrame if the target has no
    clinical-stage rows.
    """
    gene = _resolve_geneid(symbol_or_id)
    if gene is None:
        return pd.DataFrame()
    if gene in _target_drugs_cache:
        return _target_drugs_cache[gene]

    release = _latest_release_dir()
    ct = pd.read_parquet(release / "clinical_target")
    sub = ct[ct["targetId"] == gene].copy()
    if sub.empty:
        out = pd.DataFrame()
        _target_drugs_cache[gene] = out
        return out

    sub["phase"] = _stage_to_num(sub["maxClinicalStage"])
    sub["max_clinical_stage"] = sub["maxClinicalStage"].map(STAGE_LABEL).fillna(sub["maxClinicalStage"])

    ci_path = release / "clinical_indication"
    if ci_path.exists() and any(ci_path.iterdir()):
        ci = pd.read_parquet(ci_path)
        dn = _disease_names()
        # Map drugId → list of disease names via clinical_indication.
        drugs_in_target = sub[["drugId"]].drop_duplicates()
        joined = (
            drugs_in_target.merge(
                ci[["drugId", "diseaseId"]].drop_duplicates(),
                on="drugId", how="left",
            )
            .merge(dn, on="diseaseId", how="left")
        )
        indications = (
            joined.dropna(subset=["disease_name"])
                  .groupby("drugId")["disease_name"]
                  .apply(lambda s: sorted(set(s)))
                  .reset_index()
                  .rename(columns={"disease_name": "indications"})
        )
        sub = sub.merge(indications, on="drugId", how="left")
        sub["n_indications"] = sub["indications"].apply(
            lambda v: len(v) if isinstance(v, list) else 0
        )
    else:
        sub["indications"] = [[] for _ in range(len(sub))]
        sub["n_indications"] = 0

    out = (
        sub[["drugId", "max_clinical_stage", "phase",
             "indications", "n_indications"]]
        .rename(columns={"drugId": "drug_id"})
        .sort_values(["phase", "n_indications"], ascending=[False, False])
        .reset_index(drop=True)
    )
    _target_drugs_cache[gene] = out
    return out


# ---------------------------------------------------------------------------
# Per-disease drug-status views (used by the "Disease → targets" workflow)
# ---------------------------------------------------------------------------
def disease_drug_status(
    disease_id: str | Iterable[str],
) -> dict[str, set[str]]:
    """Return target → drug-status sets for one or more diseases.

    ``disease_id`` accepts either a single EFO/MONDO ID or an iterable of
    IDs (e.g. all BMI-related diseases). When multiple IDs are passed the
    sets are taken across the union — a target is "approved here" if it
    has an approved drug for *any* of the selected diseases.

    Result keys:
      * ``approved``       — target geneIds with ≥1 approved drug for any of
                             the selected diseases.
      * ``clinical``       — target geneIds with any drug in clinical
                             development (phase ≥ 1) for any selected disease.
      * ``approved_any``   — target geneIds with any approved drug (any
                             indication, anywhere). Independent of the
                             selected disease set.
    """
    if isinstance(disease_id, str):
        ids = (disease_id,)
    else:
        ids = tuple(disease_id)
    if not ids:
        return {"approved": set(), "clinical": set(), "approved_any": set()}

    cache_key = "|".join(sorted(ids))
    if cache_key in _disease_status_cache:
        return _disease_status_cache[cache_key]

    release = _latest_release_dir()
    ct = pd.read_parquet(release / "clinical_target")
    ci_path = release / "clinical_indication"
    if not (ci_path.exists() and any(ci_path.iterdir())):
        out = {"approved": set(), "clinical": set(),
               "approved_any": set()}
        _disease_status_cache[cache_key] = out
        return out
    ci = pd.read_parquet(ci_path)
    ci_dis = ci[ci["diseaseId"].isin(ids)].copy()
    ci_dis["_phase"] = _stage_to_num(ci_dis["maxClinicalStage"])

    drugs_approved_for_disease = set(ci_dis.loc[ci_dis["_phase"] >= 4, "drugId"])
    drugs_clinical_for_disease = set(ci_dis.loc[ci_dis["_phase"] >= 1, "drugId"])

    ct_sub_approved = ct[ct["drugId"].isin(drugs_approved_for_disease)]
    ct_sub_clinical = ct[ct["drugId"].isin(drugs_clinical_for_disease)]

    feats = get_features()
    approved_any = set(feats.loc[feats["has_approved"], "geneId"])

    out = {
        "approved":     set(ct_sub_approved["targetId"].unique()),
        "clinical":     set(ct_sub_clinical["targetId"].unique()),
        "approved_any": approved_any,
    }
    _disease_status_cache[cache_key] = out
    return out


def annotate_with_drug_status(
    df: pd.DataFrame,
    disease_id: str | Iterable[str],
    *,
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Add per-target drug-safety columns to a target-list DataFrame.

    Inputs:
      * ``df``         — must have either a ``geneId`` column or a column with
                          gene symbols (``symbol_col``).
      * ``disease_id`` — a single EFO/MONDO ID or an iterable of IDs (e.g.
                          all BMI-related diseases). When multiple IDs are
                          passed, ``approved_for_this_disease`` /
                          ``in_trials_for_this_disease`` flag a target if it
                          satisfies the condition for *any* of the diseases.

    Adds: ``geneId`` (if missing), ``bucket``, ``max_drug_phase``,
    ``max_drug_phase_label``, ``n_drugs``, ``n_drugged_diseases``,
    ``has_safety_event``, ``is_cancer_driver``, ``has_small_mol_binder``,
    ``approved_for_this_disease``, ``in_trials_for_this_disease``,
    ``approved_any_indication``, and ``novelty`` (a short string).
    """
    feats = get_features()
    out = df.copy()
    if "geneId" not in out.columns:
        sym_to_gene = (
            feats[["approvedSymbol", "geneId"]]
              .dropna(subset=["approvedSymbol"])
              .drop_duplicates(subset=["approvedSymbol"], keep="first")
              .set_index("approvedSymbol")["geneId"]
        )
        out["geneId"] = out[symbol_col].map(sym_to_gene)

    feat_cols = [
        "geneId", "bucket", "max_drug_phase", "n_drugs", "n_drugged_diseases",
        "has_safety_event", "is_cancer_driver", "has_small_mol_binder",
        "has_ligand", "is_in_membrane", "is_secreted",
        "genetic_constraint",
    ]
    feat_cols = [c for c in feat_cols if c in feats.columns]
    out = out.merge(feats[feat_cols], on="geneId", how="left")
    out["max_drug_phase"] = out["max_drug_phase"].fillna(0.0)
    out["max_drug_phase_label"] = out["max_drug_phase"].map(
        lambda v: PHASE_NUM_TO_LABEL.get(v, "—" if v == 0 else f"phase {v}")
    )
    for c in ["has_safety_event", "is_cancer_driver", "has_small_mol_binder",
              "has_ligand", "is_in_membrane", "is_secreted"]:
        if c in out.columns:
            out[c] = out[c].fillna(False).astype(bool)

    status = disease_drug_status(disease_id)
    out["approved_for_this_disease"] = out["geneId"].isin(status["approved"])
    out["in_trials_for_this_disease"] = out["geneId"].isin(status["clinical"])
    out["approved_any_indication"] = out["geneId"].isin(status["approved_any"])

    def _novelty(row: pd.Series) -> str:
        if row["approved_for_this_disease"]:
            return "approved here"
        if row["in_trials_for_this_disease"]:
            return "in trials here"
        if row["approved_any_indication"]:
            return "approved elsewhere"
        return "no drug"

    out["novelty"] = out.apply(_novelty, axis=1)
    return out


# ---------------------------------------------------------------------------
# Aggregations for the Population-stats tab
# ---------------------------------------------------------------------------
@dataclass
class BucketSummary:
    by_pav: pd.DataFrame          # bucket × pav → n_genes, n_clinical, n_approved, rates
    safety_by_pav: pd.DataFrame   # bucket × pav → n_genes, n_safety, rate_safety
    or_table: pd.DataFrame        # odds-ratio table vs reference (no-PAV, none)
    within_bucket_or: pd.DataFrame  # within-bucket PAV-vs-no-PAV ORs (the right framing)
    within_bucket_or_by_tract: pd.DataFrame  # within-bucket × tractability stratum
    safety_by_phase: pd.DataFrame  # max clinical phase → n_genes, n_safety, rate_safety
    continuous_pleiotropy: pd.DataFrame  # binned n_diseases × pav → clinical/approved rates
    tract_rates: pd.DataFrame     # bucket → tractability feature rates


def _pav_series(df: pd.DataFrame, pav_threshold: float = 0.0) -> pd.Series:
    """Reproduce the notebook's `_pav` flag from `genetic_constraint`.

    Default behaviour (``pav_threshold == 0``) matches the
    `pleiotropy_drugs_safety.ipynb` notebook: a gene is considered
    PAV-supported if its ``geneticConstraint`` score is *anything other
    than zero* (the OT 26.03 `target_prioritisation` field encodes "no
    evidence" as exactly 0 / NaN, and "some genetic evidence" as a signed
    score). NaN rows are treated as no evidence.

    For exploratory use, callers can raise the threshold above 0 to
    restrict to strongly-constrained genes only (``genetic_constraint
    > threshold``); that's the stricter "is the gene under purifying
    selection?" proxy.
    """
    if "genetic_constraint" not in df.columns:
        return pd.Series(False, index=df.index)
    gc = df["genetic_constraint"]
    if pav_threshold > 0:
        return gc.fillna(0) > pav_threshold
    # Notebook default — any non-zero geneticConstraint counts as
    # "has genetic evidence" (broad PAV-equivalent).
    return gc.fillna(0).astype(float) != 0


def bucket_summary(pav_threshold: float = 0.0) -> BucketSummary:
    """Compute the manuscript-style stratified summary used in tab 5."""
    df = get_features().copy()
    df["_pav"] = _pav_series(df, pav_threshold)

    by_pav = (
        df.groupby(["bucket", "_pav"], observed=True)
          .agg(n_genes=("geneId", "size"),
               n_clinical=("has_clinical", "sum"),
               n_approved=("has_approved", "sum"))
          .reset_index()
    )
    by_pav["rate_clinical"] = by_pav["n_clinical"] / by_pav["n_genes"]
    by_pav["rate_approved"] = by_pav["n_approved"] / by_pav["n_genes"]
    by_pav["PAV"] = by_pav["_pav"].map({True: "PAV", False: "no PAV"})
    by_pav["bucket"] = by_pav["bucket"].astype(str)

    if "has_safety_event" in df.columns:
        safety = (
            df.groupby(["bucket", "_pav"], observed=True)
              .agg(n_genes=("geneId", "size"),
                   n_safety=("has_safety_event", "sum"))
              .reset_index()
        )
        safety["rate_safety"] = safety["n_safety"] / safety["n_genes"]
        safety["PAV"] = safety["_pav"].map({True: "PAV", False: "no PAV"})
        safety["bucket"] = safety["bucket"].astype(str)
    else:
        safety = pd.DataFrame()

    # Odds-ratio table vs reference: bucket=none AND no PAV.
    ref_mask = (~df["_pav"]) & (df["bucket"] == "none")
    ref_n = int(ref_mask.sum())
    ref_event = int(df.loc[ref_mask, "has_approved"].sum())
    ref_no = ref_n - ref_event
    rows = []
    for bucket in BUCKET_ORDER:
        for pav, label in [(False, "no PAV"), (True, "PAV")]:
            if bucket == "none" and not pav:
                continue
            m = (df["_pav"] == pav) & (df["bucket"] == bucket)
            n = int(m.sum())
            e = int(df.loc[m, "has_approved"].sum())
            if n == 0:
                continue
            or_, lo, hi = _odds_ratio_2x2(e, n - e, ref_event, ref_no)
            rows.append(dict(
                bucket=bucket, PAV=label, n_genes=n, n_approved=e,
                rate=e / n if n else 0.0,
                OR=or_, OR_lcl=lo, OR_ucl=hi,
            ))
    or_table = pd.DataFrame(rows)

    # Within-bucket PAV-vs-no-PAV ORs — answers the manuscript's actual
    # claim: inside a given pleiotropy bucket, does PAV evidence raise the
    # drug-success rate? Reference is each bucket's own no-PAV cell.
    within_rows = []
    for bucket in BUCKET_ORDER:
        ref_mask_b = (df["bucket"] == bucket) & (~df["_pav"])
        exp_mask_b = (df["bucket"] == bucket) & (df["_pav"])
        n_ref = int(ref_mask_b.sum())
        n_exp = int(exp_mask_b.sum())
        if n_ref < 5 or n_exp < 5:
            continue
        for outcome_col, outcome_label in [
            ("has_clinical", "any clinical phase"),
            ("has_approved", "approved drug"),
        ]:
            e_ref = int(df.loc[ref_mask_b, outcome_col].sum())
            e_exp = int(df.loc[exp_mask_b, outcome_col].sum())
            or_, lo, hi = _odds_ratio_2x2(
                e_exp, n_exp - e_exp, e_ref, n_ref - e_ref
            )
            within_rows.append(dict(
                bucket=bucket, outcome=outcome_label,
                n_pav=n_exp, e_pav=e_exp,
                rate_pav=e_exp / n_exp if n_exp else 0.0,
                n_nopav=n_ref, e_nopav=e_ref,
                rate_nopav=e_ref / n_ref if n_ref else 0.0,
                OR=or_, OR_lcl=lo, OR_ucl=hi,
            ))
    within_bucket_or = pd.DataFrame(within_rows)

    # ------------------------------------------------------------------
    # Tractability-stratified within-bucket OR. Splits the within-bucket
    # PAV-vs-no-PAV comparison by whether the target is plausibly
    # druggable (has_small_mol_binder ∨ is_in_membrane ∨ is_secreted).
    # Directly addresses the tractability confound on the sweet-spot
    # effect.
    # ------------------------------------------------------------------
    tract_cols = [c for c in ["has_small_mol_binder",
                              "is_in_membrane", "is_secreted"]
                  if c in df.columns]
    if tract_cols:
        df["_tractable"] = df[tract_cols].any(axis=1)
        tract_rows = []
        for stratum_name, stratum_mask in [
            ("tractable",     df["_tractable"]),
            ("non-tractable", ~df["_tractable"]),
        ]:
            for bucket in BUCKET_ORDER:
                ref_b = stratum_mask & (df["bucket"] == bucket) & (~df["_pav"])
                exp_b = stratum_mask & (df["bucket"] == bucket) & (df["_pav"])
                n_ref = int(ref_b.sum())
                n_exp = int(exp_b.sum())
                if n_ref < 5 or n_exp < 5:
                    continue
                e_ref = int(df.loc[ref_b, "has_clinical"].sum())
                e_exp = int(df.loc[exp_b, "has_clinical"].sum())
                or_, lo, hi = _odds_ratio_2x2(
                    e_exp, n_exp - e_exp, e_ref, n_ref - e_ref
                )
                tract_rows.append(dict(
                    stratum=stratum_name, bucket=bucket,
                    n_pav=n_exp, e_pav=e_exp,
                    rate_pav=e_exp / n_exp if n_exp else 0.0,
                    n_nopav=n_ref, e_nopav=e_ref,
                    rate_nopav=e_ref / n_ref if n_ref else 0.0,
                    OR=or_, OR_lcl=lo, OR_ucl=hi,
                ))
        within_bucket_or_by_tract = pd.DataFrame(tract_rows)
    else:
        within_bucket_or_by_tract = pd.DataFrame()

    # ------------------------------------------------------------------
    # Safety-event rate as a function of max clinical phase reached
    # (drugged genes only). Survivorship sanity check.
    # ------------------------------------------------------------------
    if "has_safety_event" in df.columns:
        drugged = df[df["max_drug_phase"] >= 1].copy()
        if not drugged.empty:
            safety_by_phase = (
                drugged.groupby("max_drug_phase", observed=True)
                  .agg(n_genes=("geneId", "size"),
                       n_safety=("has_safety_event", "sum"))
                  .reset_index()
            )
            safety_by_phase["rate_safety"] = (
                safety_by_phase["n_safety"] / safety_by_phase["n_genes"]
            )
            safety_by_phase["phase_label"] = (
                safety_by_phase["max_drug_phase"]
                  .map(PHASE_NUM_TO_LABEL)
                  .fillna(safety_by_phase["max_drug_phase"].astype(str))
            )
            safety_by_phase = safety_by_phase.sort_values("max_drug_phase")
        else:
            safety_by_phase = pd.DataFrame()
    else:
        safety_by_phase = pd.DataFrame()

    # ------------------------------------------------------------------
    # Continuous pleiotropy — bin n_diseases on a log scale and compute
    # clinical / approved rates by PAV.
    # ------------------------------------------------------------------
    if "n_diseases" in df.columns and df["n_diseases"].max() >= 1:
        cp = df[["n_diseases", "_pav", "has_clinical", "has_approved"]].copy()
        cp = cp[cp["n_diseases"] >= 1]
        edges = np.unique(np.round(np.geomspace(1, cp["n_diseases"].max() + 1, 14)))
        cp["bin"] = pd.cut(cp["n_diseases"], bins=edges, include_lowest=True)
        cp["bin_mid"] = cp["bin"].apply(
            lambda i: float(np.sqrt(i.left * max(i.right, 1)))
            if pd.notna(i) else np.nan
        )
        binned = (
            cp.dropna(subset=["bin_mid"])
              .groupby(["bin_mid", "_pav"], observed=True)
              .agg(n_genes=("has_clinical", "size"),
                   rate_clinical=("has_clinical", "mean"),
                   rate_approved=("has_approved", "mean"))
              .reset_index()
        )
        binned["PAV"] = binned["_pav"].map({True: "PAV", False: "no PAV"})
        binned = binned[binned["n_genes"] >= 10].copy()
        continuous_pleiotropy = binned.sort_values(["PAV", "bin_mid"])
    else:
        continuous_pleiotropy = pd.DataFrame()

    tract_features = [
        c for c in ["has_small_mol_binder", "has_ligand",
                    "is_in_membrane", "is_secreted"]
        if c in df.columns
    ]
    if tract_features:
        tract_rates = (
            df.groupby("bucket", observed=True)[tract_features]
              .mean()
              .reindex(BUCKET_ORDER)
        )
    else:
        tract_rates = pd.DataFrame()

    return BucketSummary(
        by_pav=by_pav,
        safety_by_pav=safety,
        or_table=or_table,
        within_bucket_or=within_bucket_or,
        within_bucket_or_by_tract=within_bucket_or_by_tract,
        safety_by_phase=safety_by_phase,
        continuous_pleiotropy=continuous_pleiotropy,
        tract_rates=tract_rates,
    )


def _odds_ratio_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Odds ratio + 95% CI for a 2×2 table. Haldane–Anscombe +0.5 if any zero."""
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    log_or = float(np.log(or_))
    return float(or_), float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se))
