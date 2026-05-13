# Genetics + pQTL Target Triage Copilot

A Streamlit app that combines Open Targets disease–target associations, the
manuscript's pleiotropy framework (Mountjoy / Ochoa et al. 2026), curated
cis-pQTL instruments, and the OT 26.03 clinical / safety release into two
end-to-end workflows for finding novel therapeutic targets.

## The two workflows

**`1 · Disease → targets`** — pick one or more OT diseases (autocomplete over
~47k entries), pool them as a disease area, fetch the top GWAS-supported
targets per disease, and overlay drug-safety / clinical-stage / pleiotropy
from the local OT release. Surfaces a ranked **novel-target candidate list**
where each candidate is GWAS-supported but *not yet approved for the
selected disease*, has no OT safety-event flag, and falls in the
intermediate / specific pleiotropy bucket. Default example: BMI area
(`body mass index` + `BMI-adjusted waist-hip ratio` +
`BMI-adjusted waist circumference`).

**`2 · Gene → diseases`** — single-gene view: pleiotropy profile, therapeutic
areas, top filtered disease associations, drug-safety panel (max clinical
stage, drug count, drugged diseases, safety event, tractability chips),
known drugs from `clinical_target` × `clinical_indication`. Sub-tab
**Sweet-spot filter** scores any gene list against the manuscript's
intermediate-pleiotropy ∩ PAV-supported profile (OR ≈ 10.3). Auto-fills
with the novel candidates from the latest Disease → targets run.

Supporting deep-dive tabs:

* **`3 · Mechanism deep-dive`** — per (target, disease) tier classification.
  cis-pQTL coloc + directional consistency = T1, cis-eQTL or pQTL coloc
  without direction = T2, PAV-only = T3, GWAS-only = T4. Single-pair and
  batch (multi-disease pooled) modes, both with the OT disease autocomplete.
* **`4 · Phase 0 (MR)`** — two-sample Mendelian randomization (IVW +
  weighted median + MR-Egger, with effect-allele harmonization and
  per-instrument F-statistic) for the curated cis-pQTL set against curated
  outcomes. Default: GLP1R → BMI (semaglutide mechanism). Curated proteins:
  PCSK9, LPA, IL6R, HMGCR, APOC3, ANGPTL3/4, LDLR, CETP, APOB, IL18, TYK2,
  CRP, LEP, LEPR, GIPR, GLP1R, MC4R, FTO. Outcomes: CAD, T2D, LDL-C, HDL-C,
  TG, CRP, RA, IBD, MS, BMI, WHRadjBMI.
* **`5 · Population stats`** — manuscript-style population analysis:
  pleiotropy × PAV stacked bars, approved-drug-rate by bucket × PAV,
  safety-event rate, odds-ratio forest, full filterable per-gene table.
* **`6 · Agent`** — Claude tool-use orchestrator stub.

## Setup

```bash
git clone <your-repo-url> pleiotropy_v2
cd pleiotropy_v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data prep (required before first run)

The `data/` directory is not tracked in git. Fetch the Open Targets 26.03
parquet exports — about 4 GB total — once:

```bash
python scripts/download_ot_release.py
```

This populates `data/ot_release_26.03/` with `target`, `disease`,
`target_prioritisation`, `clinical_target`, `clinical_indication`,
`credible_set`, `l2g_prediction`, `study`, `target_prioritisation`, and
`association_by_datatype_direct`. The pre-aggregated
`data/all_genes_pleiotropy_l2g.parquet` (built by the v1 L2G notebook) is
required for the pleiotropy bucket; if you don't have it, run
`notebooks/pleiotropy_l2g_credible_sets.ipynb` once.

The per-gene drug-safety feature table
(`data/all_genes_features.parquet`) is built on demand at app startup and
cached.

### Run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # only needed for tab 6 (Agent)
streamlit run streamlit_app.py
```

## Project layout

```
pleiotropy_v2/
├── streamlit_app.py            # main entry — six tabs (see above)
├── pleio/                      # importable package
│   ├── ot_pleiotropy.py        # OT GraphQL client + pleiotropy classifier
│   ├── cache.py                # SQLite cache around the GraphQL client
│   ├── coloc.py                # mechanism-tier classifier + MR wrapper
│   ├── mr_engine.py            # IVW / weighted median / MR-Egger
│   ├── pqtl_instruments.py     # curated cis-pQTL instrument tables
│   ├── disease_gwas.py         # curated disease-outcome effect tables
│   ├── drug_safety.py          # OT 26.03 parquet loader + disease catalog
│   └── ot_qtl_fetch.py         # OT pQTL/eQTL fall-through
├── scripts/
│   └── download_ot_release.py  # idempotent fetcher for OT release parquets
├── notebooks/
│   ├── pleiotropy_l2g_credible_sets.ipynb
│   ├── pleiotropy_all_genes.ipynb
│   └── pleiotropy_drugs_safety.ipynb
├── tests/                      # MR + classifier sanity checks
├── data/                       # NOT tracked — populated by download script
└── requirements.txt
```

## Worked examples

| Workflow | Inputs | Expected result |
|---|---|---|
| Disease → targets | BMI defaults (3 EFOs), GA ≥ 0.10 | 6 novel candidates: RAB21, SPARC, PRMT2, CYTL1, GPR151, AIFM2 |
| Disease → targets | `coronary artery disease`, GA ≥ 0.10 | LRP6 highlighted (intermediate-pleiotropy, no drug, no safety flag) |
| Mechanism deep-dive (batch) | `coronary artery disease`, top 10 | T1 hits: LDLR, PCSK9 |
| Mechanism deep-dive (batch) | `body mass index`, top 10 | T1 hits: MC4R, GLP1R |
| Phase 0 MR | `GLP1R` → `BMI` (default) | IVW β = −0.113, p = 1.1e-09 — the semaglutide mechanism |
| Phase 0 MR | `PCSK9` → `CAD` | IVW OR ≈ 0.65 per SD lower plasma PCSK9 |

## Curated demo set vs. fall-through

The `4 · Phase 0 (MR)` tab and the directional-consistency check in the
Mechanism deep-dive use **curated cis-pQTL / cis-eQTL instruments** from
`pleio/pqtl_instruments.py` and **curated disease effect sizes** from
`pleio/disease_gwas.py`. Effect sizes are drawn from published lookups
(Sun 2018, UKB-PPP, deCODE 2021, CARDIoGRAMplusC4D, GLGC, DIAGRAM,
GIANT-Yengo 2018, Pulit 2019) and are illustrative — a handful of
canonical palindromic instruments (e.g. GIPR rs1800437, FTO rs9939609) are
re-encoded with a non-palindromic surrogate OA so the harmonizer keeps them;
effect directions are preserved. For non-curated targets the classifier
falls through to OT pQTL credible sets when available, and otherwise
returns T3 / T4 / none based on PAV and GA-score evidence alone.

## Tests

```bash
pip install pytest
pytest tests/
```
