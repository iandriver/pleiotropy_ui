"""Mechanism-tier classifier for (target, disease) pairs.

Buckets follow the manuscript's evidence hierarchy:

    T1 — cis-pQTL coloc + directional consistency
    T2 — cis-eQTL coloc (or cis-pQTL coloc without directional consistency)
    T3 — protein-altering variant (PAV) only
    T4 — GWAS-only association (genetic_association datatype score over threshold)
    none — no genetic evidence above threshold

Two evidence sources feed the classifier:

* **Curated layer** — the in-package cis-pQTL instrument tables in
  ``pleio.pqtl_instruments`` paired with the disease GWAS tables in
  ``pleio.disease_gwas``. Where both exist for a (target, disease), we run
  IVW + Egger via ``pleio.mr_engine``. "Coloc" is operationalised as
  instrument-set overlap (which is true by construction for the curated
  cardiometabolic demo set). "Directional consistency" is operationalised
  as: significant IVW slope (p < α) AND a majority of per-SNP Wald ratios
  share the same sign as the IVW estimate (sign concordance ≥ 0.6).
  Egger intercept p is stored for the UI but doesn't gate the tier — with
  3–5 instruments Egger is underpowered and frequently produces false
  pleiotropy alarms even on textbook clean targets like APOC3.

* **OT Platform layer** — :func:`ot.fetch_target` gives the per-disease
  ``genetic_association`` score (used for T4), and :func:`ot.fetch_pav_evidence`
  gives PAV evidence at the variant level (used for T3).

The split lets the unit tests exercise the tier logic offline (curated
layer only) while the Streamlit UI uses the full path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import (
    disease_gwas,
    mr_engine,
    ot_pleiotropy as ot,
    ot_qtl_fetch,
    pqtl_instruments,
)


# ---------------------------------------------------------------------------
# Curated-outcome ↔ EFO/MONDO crosswalk
# ---------------------------------------------------------------------------
# Lets us connect the demo-set outcome short-codes (CAD, T2D, LDL-C…) used
# in ``disease_gwas.OUTCOMES`` with the Open Targets disease ontology IDs
# used in ``ot.fetch_target``.
OUTCOME_TO_EFO: dict[str, str] = {
    "CAD":   "EFO_0001645",   # coronary artery disease
    "T2D":   "MONDO_0005148", # type 2 diabetes mellitus
    "LDL-C": "EFO_0004611",   # LDL cholesterol measurement
    "HDL-C": "EFO_0004612",   # HDL cholesterol measurement
    "TG":    "EFO_0004530",   # triglyceride measurement
    # ----- v0.2 inflammatory expansion -----
    "CRP":   "EFO_0004458",   # C-reactive protein measurement
    "RA":    "EFO_0000685",   # rheumatoid arthritis
    "IBD":   "EFO_0003767",   # inflammatory bowel disease
    "MS":    "EFO_0003885",   # multiple sclerosis
    # ----- v0.3 BMI / obesity expansion -----
    "BMI":       "EFO_0004340",  # body mass index
    "WHRadjBMI": "EFO_0007788",  # BMI-adjusted waist-hip ratio
}
EFO_TO_OUTCOME: dict[str, str] = {v: k for k, v in OUTCOME_TO_EFO.items()}

# Loose name-based fallback for cases where OT's disease lookup returns a
# slightly different EFO/MONDO id than the canonical ones above.
_DISEASE_NAME_TO_OUTCOME: dict[str, str] = {
    "coronary artery disease": "CAD",
    "coronary heart disease": "CAD",
    "type 2 diabetes mellitus": "T2D",
    "type ii diabetes mellitus": "T2D",
    "type 2 diabetes": "T2D",
    "ldl cholesterol measurement": "LDL-C",
    "ldl cholesterol": "LDL-C",
    "low density lipoprotein cholesterol measurement": "LDL-C",
    "hdl cholesterol measurement": "HDL-C",
    "hdl cholesterol": "HDL-C",
    "high density lipoprotein cholesterol measurement": "HDL-C",
    "triglyceride measurement": "TG",
    "triglycerides measurement": "TG",
    "triglycerides": "TG",
    "serum triglyceride measurement": "TG",
    # ----- v0.2 inflammatory expansion -----
    "c-reactive protein measurement": "CRP",
    "c-reactive protein": "CRP",
    "crp": "CRP",
    "rheumatoid arthritis": "RA",
    "ra": "RA",
    "inflammatory bowel disease": "IBD",
    "ibd": "IBD",
    "ulcerative colitis": "IBD",
    "crohn's disease": "IBD",
    "multiple sclerosis": "MS",
    "ms": "MS",
    # ----- v0.3 BMI / obesity expansion -----
    "body mass index": "BMI",
    "bmi": "BMI",
    "bmi-adjusted waist-hip ratio": "WHRadjBMI",
    "waist-hip ratio (bmi-adjusted)": "WHRadjBMI",
    "bmi-adjusted waist circumference": "WHRadjBMI",
}


TIER_LABELS: dict[str, str] = {
    "T1": "T1 — cis-pQTL coloc + directional consistency",
    "T2": "T2 — cis-eQTL coloc / pQTL coloc w/o directional consistency",
    "T3": "T3 — PAV-only evidence",
    "T4": "T4 — GWAS-only association",
    "none": "No genetic evidence above threshold",
}
TIER_CONFIDENCE: dict[str, float] = {
    "T1": 1.00, "T2": 0.75, "T3": 0.55, "T4": 0.30, "none": 0.0,
}
TIER_COLOR: dict[str, str] = {
    "T1": "#1f883d",   # green
    "T2": "#9ece6a",   # yellow-green
    "T3": "#e3b505",   # amber
    "T4": "#9aa0a6",   # grey
    "none": "#cccccc",
}


def outcome_for_disease(disease_id: str | None,
                        disease_name: str | None) -> str | None:
    """Map an OT disease (EFO/MONDO + name) to a curated outcome code, if any."""
    if disease_id and disease_id in EFO_TO_OUTCOME:
        return EFO_TO_OUTCOME[disease_id]
    if disease_name:
        return _DISEASE_NAME_TO_OUTCOME.get(disease_name.strip().lower())
    return None


# ---------------------------------------------------------------------------
# Curated MR + coloc evidence
# ---------------------------------------------------------------------------
@dataclass
class CuratedColocEvidence:
    """Result of running the curated cis-QTL → outcome MR for one pair."""
    has_pqtl_coloc: bool = False
    has_eqtl_coloc: bool = False
    n_instruments_used: int = 0
    mr_beta: float | None = None
    mr_se: float | None = None
    mr_p: float | None = None
    egger_intercept_p: float | None = None
    sign_concordance: float | None = None  # fraction of per-SNP Wald ratios matching IVW sign
    significant: bool = False
    directional_consistency: str = "n/a"  # "concordant" | "inconsistent" | "n/a"
    mr_direction_text: str | None = None
    outcome_is_binary: bool | None = None
    instrument_source: str | None = None  # informational: "pQTL" / "eQTL"
    notes: list[str] = field(default_factory=list)


def _classify_instrument_kind(instr_df: pd.DataFrame) -> str:
    """Return 'pQTL' if any instrument is from a pQTL source, else 'eQTL'.

    HMGCR is the canonical mixed example — its instruments are GTEx-eQTL,
    so the whole instrument set is eQTL-only.
    """
    sources = instr_df["source"].astype(str).str.lower().tolist()
    if any("eqtl" not in s for s in sources):
        return "pQTL"
    return "eQTL"


def coloc_evidence_from_data(
    instr_df: pd.DataFrame,
    out_df: pd.DataFrame,
    *,
    instrument_kind: str,
    outcome_is_binary: bool,
    label: str = "",
    mr_significance_threshold: float = 0.05,
    sign_concordance_threshold: float = 0.6,
) -> CuratedColocEvidence:
    """Pure-data engine: given instrument + outcome dataframes, run MR and
    populate the evidence object. Reused by both the curated path and the
    OT-fall-through path in :func:`classify_mechanism`.

    ``instrument_kind`` should be ``"pQTL"`` or ``"eQTL"``. ``label`` is a
    short string (e.g. ``"PCSK9 → CAD"``) used in the human-readable note.
    """
    ev = CuratedColocEvidence()
    if instr_df is None or instr_df.empty:
        ev.notes.append("No instruments available.")
        return ev
    if out_df is None or out_df.empty:
        ev.notes.append(f"No outcome effects overlap with the instrument set"
                        + (f" for {label}" if label else "") + ".")
        return ev

    ev.instrument_source = instrument_kind
    ev.outcome_is_binary = outcome_is_binary

    harm = mr_engine.harmonize(instr_df, out_df)
    res = mr_engine.run_mr(harm, outcome_is_binary=outcome_is_binary)
    if "error" in res:
        ev.notes.append(f"MR failed: {res['error']}")
        return ev

    ivw_r = res["ivw"]
    eg_i = res["egger_intercept"]
    per_snp = res["per_snp"]
    ev.mr_beta = float(ivw_r.estimate)
    ev.mr_se = float(ivw_r.se)
    ev.mr_p = float(ivw_r.pvalue)
    ev.egger_intercept_p = float(eg_i.pvalue)
    ev.n_instruments_used = int(ivw_r.n_snps)
    ev.significant = ev.mr_p < mr_significance_threshold

    # Per-SNP sign concordance — robust at small instrument counts.
    if len(per_snp) and ev.mr_beta is not None:
        ivw_sign = 1.0 if ev.mr_beta >= 0 else -1.0
        matching = ((per_snp["ratio"] >= 0) == (ivw_sign >= 0)).sum()
        ev.sign_concordance = float(matching) / float(len(per_snp))
    else:
        ev.sign_concordance = None

    if (ev.significant and ev.sign_concordance is not None
            and ev.sign_concordance >= sign_concordance_threshold):
        ev.directional_consistency = "concordant"
    else:
        ev.directional_consistency = "inconsistent"

    ev.mr_direction_text = mr_engine.interpret_direction(
        ev.mr_beta, outcome_is_binary
    )

    if instrument_kind == "pQTL":
        ev.has_pqtl_coloc = True
    else:
        ev.has_eqtl_coloc = True

    ev.notes.append(
        f"{ev.n_instruments_used} cis-{instrument_kind} instruments"
        + (f" × {label} GWAS" if label else "")
        + f"; IVW p={ev.mr_p:.2e}, "
          f"Egger intercept p={ev.egger_intercept_p:.2f}, "
          f"sign concordance {(ev.sign_concordance or 0):.0%}."
    )
    return ev


def curated_coloc_evidence(
    target_symbol: str,
    outcome_code: str,
    *,
    mr_significance_threshold: float = 0.05,
    sign_concordance_threshold: float = 0.6,
) -> CuratedColocEvidence:
    """Compute curated cis-QTL coloc + MR for a (target, outcome) pair.

    Curated path only — no network. Returns an empty evidence object if the
    target is not in the curated instrument set or no outcome effects
    overlap. Pure pass-through to :func:`coloc_evidence_from_data` once the
    relevant dataframes have been pulled from the in-package tables.
    """
    ev = CuratedColocEvidence()
    if not pqtl_instruments.is_curated(target_symbol):
        ev.notes.append(
            f"{target_symbol} is not in the curated cis-pQTL instrument set."
        )
        return ev
    if outcome_code not in disease_gwas.OUTCOMES:
        ev.notes.append(f"{outcome_code} is not in the curated outcome set.")
        return ev

    instr = pqtl_instruments.get_instruments(
        target_symbol, allow_ot_fetch=False,
    )
    out_df = disease_gwas.get_outcome_effects(outcome_code, instr["rsid"].tolist())
    kind = _classify_instrument_kind(instr) if not instr.empty else "pQTL"
    binary = disease_gwas.OUTCOMES[outcome_code]["binary"]
    return coloc_evidence_from_data(
        instr, out_df,
        instrument_kind=kind,
        outcome_is_binary=binary,
        label=outcome_code,
        mr_significance_threshold=mr_significance_threshold,
        sign_concordance_threshold=sign_concordance_threshold,
    )


# ---------------------------------------------------------------------------
# Public mechanism-tier classifier
# ---------------------------------------------------------------------------
@dataclass
class MechanismEvidence:
    """End-to-end evidence object for a (target, disease) pair."""
    target_symbol: str
    target_ensg: str | None
    disease_name: str
    disease_id: str | None
    tier: str
    tier_label: str
    confidence: float
    # Evidence flags
    has_pqtl_coloc: bool
    has_eqtl_coloc: bool
    has_pav: bool
    has_gwas: bool
    ga_score: float
    # MR / coloc detail (curated layer)
    directional_consistency: str  # "concordant" | "inconsistent" | "n/a"
    mr_beta: float | None
    mr_se: float | None
    mr_p: float | None
    egger_intercept_p: float | None
    sign_concordance: float | None
    n_instruments_used: int
    mr_direction_text: str | None
    instrument_source: str | None
    # PAV detail
    pav_variants: list[dict]
    # Free-form notes for the UI
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        """Flat dict for batch / dataframe consumption."""
        return {
            "target": self.target_symbol,
            "target_ensg": self.target_ensg,
            "disease": self.disease_name,
            "disease_id": self.disease_id,
            "tier": self.tier,
            "confidence": self.confidence,
            "has_pqtl_coloc": self.has_pqtl_coloc,
            "has_eqtl_coloc": self.has_eqtl_coloc,
            "has_pav": self.has_pav,
            "has_gwas": self.has_gwas,
            "ga_score": self.ga_score,
            "mr_beta": self.mr_beta,
            "mr_p": self.mr_p,
            "egger_intercept_p": self.egger_intercept_p,
            "sign_concordance": self.sign_concordance,
            "n_instruments_used": self.n_instruments_used,
            "directional_consistency": self.directional_consistency,
            "mr_direction": self.mr_direction_text,
            "n_pav_variants": len(self.pav_variants),
        }


def _decide_tier(*,
                 has_pqtl_coloc: bool,
                 has_eqtl_coloc: bool,
                 has_pav: bool,
                 has_gwas: bool,
                 directional_consistency: str) -> str:
    """Pure tier-decision function — exercised by unit tests."""
    if has_pqtl_coloc and directional_consistency == "concordant":
        return "T1"
    if has_pqtl_coloc or has_eqtl_coloc:
        return "T2"
    if has_pav:
        return "T3"
    if has_gwas:
        return "T4"
    return "none"


def classify_mechanism(
    target: str,
    disease: str,
    *,
    ga_threshold: float = 0.10,
    mr_significance_threshold: float = 0.05,
    sign_concordance_threshold: float = 0.6,
    ga_score_hint: float | None = None,
    disease_name_hint: str | None = None,
) -> MechanismEvidence | None:
    """Classify a (target, disease) pair into a mechanism tier.

    `target` is a gene symbol or Ensembl ID; `disease` is a disease name or
    EFO/MONDO id. Returns ``None`` if the target can't be resolved.

    ``ga_score_hint`` is an optional pre-computed genetic-association score
    for this (target, disease) pair — useful when the caller already has it
    from the disease→target view (``ot.fetch_disease``) but the target's
    page-bounded associatedDiseases list (the lookup used here) wouldn't
    surface this disease (e.g. BMI not in a gene's top 500 disease list,
    even when the gene is in BMI's top targets). ``disease_name_hint`` gives
    a friendly label to attach when the assoc lookup misses too.

    This issues OT GraphQL calls (cached via the SQLite layer when installed),
    so it's not suitable for offline unit tests — for that, exercise
    :func:`curated_coloc_evidence` and :func:`_decide_tier` directly.
    """
    # 1. Resolve target
    ensg = ot.resolve_target(target)
    if ensg is None:
        return None
    meta, assoc_df, _ = ot.fetch_target(ensg)
    if meta is None:
        return None
    symbol = meta["symbol"]

    # 2. Resolve disease — fall back to demo-set name lookup if OT misses.
    efo = ot.resolve_disease(disease)
    disease_name_input = disease.strip()
    if efo is None:
        oc = _DISEASE_NAME_TO_OUTCOME.get(disease_name_input.lower())
        if oc is None:
            return None
        efo = OUTCOME_TO_EFO[oc]

    # 3. OT association → genetic_association score for this disease.
    # Falls back to ga_score_hint when the disease isn't in the gene's top-500
    # associatedDiseases list (common for BMI, where the gene→disease view is
    # page-bounded but the disease→gene view ranks the gene highly).
    sub = assoc_df[assoc_df["disease_id"] == efo] if not assoc_df.empty else pd.DataFrame()
    if not sub.empty:
        row = sub.iloc[0]
        ga_score = float(row["genetic_association_score"])
        disease_name = str(row["disease_name"])
    elif ga_score_hint is not None:
        ga_score = float(ga_score_hint)
        disease_name = disease_name_hint or disease_name_input
    else:
        ga_score = 0.0
        disease_name = disease_name_hint or disease_name_input
    has_gwas = ga_score >= ga_threshold

    # 4. PAV evidence (target-level, filtered to this disease)
    try:
        pav_df = ot.fetch_pav_evidence(ensg)
    except Exception:  # noqa: BLE001 — failure here shouldn't kill the pipeline
        pav_df = pd.DataFrame()
    if not pav_df.empty:
        pav_rows = pav_df[pav_df["disease_id"] == efo]
    else:
        pav_rows = pd.DataFrame()
    has_pav = not pav_rows.empty
    pav_variants = pav_rows.to_dict(orient="records") if has_pav else []

    # 5. Cis-QTL coloc + MR — prefer curated, fall through to OT.
    notes: list[str] = []
    outcome_code = outcome_for_disease(efo, disease_name)

    # 5a. Instruments: curated → OT fetch
    instr = pqtl_instruments.get_instruments(symbol, allow_ot_fetch=False)
    instrument_source_kind = _classify_instrument_kind(instr) if not instr.empty else None
    if instr.empty:
        try:
            instr = ot_qtl_fetch.fetch_cis_pqtl_instruments(symbol)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"OT pQTL fetch raised: {exc}")
            instr = pd.DataFrame()
        if not instr.empty:
            instrument_source_kind = "pQTL"  # OT pQTL credible sets are pQTL
            notes.append(
                f"Used OT-fetched cis-pQTL instruments for {symbol} "
                f"({len(instr)} variants)."
            )
        else:
            notes.append(f"No instruments available for {symbol} (curated or OT).")

    # 5b. Outcome effects: curated outcome code → OT fetch by EFO
    out_df = pd.DataFrame()
    outcome_is_binary: bool | None = None
    label = ""
    if not instr.empty:
        rsids = instr["rsid"].tolist()
        if outcome_code is not None:
            out_df = disease_gwas.get_outcome_effects(outcome_code, rsids)
            outcome_is_binary = disease_gwas.OUTCOMES[outcome_code]["binary"]
            label = outcome_code
            if out_df.empty:
                notes.append(
                    f"No curated GWAS effects for {symbol} × {outcome_code}; "
                    f"trying OT fall-through."
                )
        if out_df.empty:
            try:
                out_df = ot_qtl_fetch.fetch_disease_outcome_effects(efo, rsids)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"OT disease-effect fetch raised: {exc}")
                out_df = pd.DataFrame()
            if not out_df.empty:
                outcome_is_binary = ot_qtl_fetch.is_efo_binary(efo)
                label = disease_name
                notes.append(
                    f"Used OT-fetched disease effects for {disease_name} "
                    f"({len(out_df)} overlapping variants)."
                )

    # 5c. Run coloc evidence engine if we have both sides.
    if not instr.empty and not out_df.empty and outcome_is_binary is not None:
        curated = coloc_evidence_from_data(
            instr, out_df,
            instrument_kind=instrument_source_kind or "pQTL",
            outcome_is_binary=outcome_is_binary,
            label=label,
            mr_significance_threshold=mr_significance_threshold,
            sign_concordance_threshold=sign_concordance_threshold,
        )
        notes.extend(curated.notes)
    else:
        curated = CuratedColocEvidence()
        if outcome_code is None and out_df.empty:
            notes.append(
                "Disease has no curated outcome and no OT GWAS effects "
                "overlapping the instrument set — falling back to PAV / GA only."
            )

    tier = _decide_tier(
        has_pqtl_coloc=curated.has_pqtl_coloc,
        has_eqtl_coloc=curated.has_eqtl_coloc,
        has_pav=has_pav,
        has_gwas=has_gwas,
        directional_consistency=curated.directional_consistency,
    )

    return MechanismEvidence(
        target_symbol=symbol,
        target_ensg=ensg,
        disease_name=disease_name,
        disease_id=efo,
        tier=tier,
        tier_label=TIER_LABELS[tier],
        confidence=TIER_CONFIDENCE[tier],
        has_pqtl_coloc=curated.has_pqtl_coloc,
        has_eqtl_coloc=curated.has_eqtl_coloc,
        has_pav=has_pav,
        has_gwas=has_gwas,
        ga_score=ga_score,
        directional_consistency=curated.directional_consistency,
        mr_beta=curated.mr_beta,
        mr_se=curated.mr_se,
        mr_p=curated.mr_p,
        egger_intercept_p=curated.egger_intercept_p,
        sign_concordance=curated.sign_concordance,
        n_instruments_used=curated.n_instruments_used,
        mr_direction_text=curated.mr_direction_text,
        instrument_source=curated.instrument_source,
        pav_variants=pav_variants,
        notes=notes,
    )


def classify_many(
    pairs: list[tuple[str, str]],
    *,
    ga_threshold: float = 0.10,
    ga_score_hints: dict[tuple[str, str], float] | None = None,
    disease_name_hints: dict[str, str] | None = None,
    progress_cb: Callable[[int, int, str | None], None] | None = None,
) -> pd.DataFrame:
    """Batch over a list of (target, disease) pairs. Errors per pair are
    captured into a string column so a single bad target doesn't kill the run.

    ``ga_score_hints`` is an optional ``{(target, disease_id): ga_score}``
    map that backstops the upstream genetic-association score when a pair's
    disease isn't in the target's page-bounded associatedDiseases list.
    """
    rows = []
    n = len(pairs)
    ga_score_hints = ga_score_hints or {}
    disease_name_hints = disease_name_hints or {}
    for i, (t, d) in enumerate(pairs):
        if progress_cb is not None:
            progress_cb(i, n, t)
        try:
            ev = classify_mechanism(
                t, d,
                ga_threshold=ga_threshold,
                ga_score_hint=ga_score_hints.get((t, d)),
                disease_name_hint=disease_name_hints.get(d),
            )
        except Exception as exc:  # noqa: BLE001 — surface broadly in batch
            rows.append({"target": t, "disease": d, "tier": "error",
                         "error": str(exc)})
            continue
        if ev is None:
            rows.append({"target": t, "disease": d, "tier": "not_found",
                         "error": "could not resolve target/disease"})
            continue
        r = ev.as_row()
        r["error"] = None
        rows.append(r)
    if progress_cb is not None:
        progress_cb(n, n, None)
    return pd.DataFrame(rows)
