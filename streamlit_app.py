"""Genetics + pQTL Target Triage Copilot — Streamlit app.

Four tabs:
  1. Pleiotropy explorer (ported from v1 ipywidgets notebook)
  2. Mechanism tier (cis-pQTL/eQTL colocalisation + PAV → T1..T4)
  3. In silico Phase 0 (two-sample MR with curated cis-pQTL instruments)
  4. Agent (Claude tool-use orchestrator with live tool-call trace)

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Make sibling package importable when running from repo root.
sys.path.insert(0, str(Path(__file__).parent))

from pleio import coloc, disease_gwas, drug_safety, mr_engine, ot_pleiotropy as ot, pqtl_instruments
from pleio.cache import get_cache

# Initialise GraphQL cache once per process.
_CACHE = get_cache()

# OT disease catalog — shared by every disease autosuggest field.
_DISEASE_CATALOG = drug_safety.all_diseases()
_LABEL_TO_ID  = dict(zip(_DISEASE_CATALOG["label"], _DISEASE_CATALOG["id"]))
_ID_TO_LABEL  = dict(zip(_DISEASE_CATALOG["id"],    _DISEASE_CATALOG["label"]))
_ID_TO_NAME   = dict(zip(_DISEASE_CATALOG["id"],    _DISEASE_CATALOG["name"]))
_DISEASE_LABELS = _DISEASE_CATALOG["label"].tolist()

st.set_page_config(
    page_title="Genetics + pQTL Target Triage Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Sidebar — global controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    ga_threshold = st.slider(
        "Genetic-association score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
        help="A target–disease association counts toward pleiotropy only if the "
             "Open Targets `genetic_association` datatype score ≥ this value.",
    )
    overall_threshold = st.slider(
        "Overall association score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.00,
        step=0.05,
    )
    check_pav = st.checkbox(
        "Compute PAV evidence (protein-altering variants)",
        value=True,
        help="Pulls target.evidences and looks for protein-altering SO terms.",
    )
    st.markdown("---")
    stats = _CACHE.stats()
    st.caption(f"GraphQL cache — hits: {stats['hits']} · misses: {stats['misses']}")
    if st.button("Clear cache"):
        _CACHE.clear()
        st.success("Cache cleared. Rerun to re-fetch.")

st.title("Genetics + pQTL Target Triage Copilot")
st.caption(
    "Pleiotropy framework after Mountjoy / Ochoa et al. 2026, extended with "
    "cis-pQTL/eQTL colocalisation and two-sample Mendelian randomization."
)

tab_disease, tab_gene, tab_mech, tab_mr, tab_pop, tab_agent = st.tabs(
    [
        "1 · Disease → targets",
        "2 · Gene → diseases",
        "3 · Mechanism deep-dive",
        "4 · Phase 0 (MR)",
        "5 · Population stats",
        "6 · Agent",
    ]
)

# =============================================================================
# Tab 1 — Disease → targets  (novel-target discovery workflow)
# =============================================================================
with tab_disease:
    st.markdown("#### Find / support novel targets for a disease area")
    st.caption(
        "Search one or more OT diseases (e.g. *BMI* surfaces *body mass index*, "
        "*BMI-adjusted waist-hip ratio*, etc.). Selecting multiple diseases "
        "pools them as a single disease area: the workflow takes the union "
        "of top GWAS-supported targets, keeps each target's best GA score "
        "across the selected diseases, and flags drugs approved or in trials "
        "for *any* of them."
    )

    # Default example: BMI-area diseases.
    _BMI_DEFAULT_IDS = [
        "EFO_0004340",  # body mass index
        "EFO_0007788",  # BMI-adjusted waist-hip ratio
        "EFO_0007789",  # BMI-adjusted waist circumference
    ]
    _bmi_default_labels = [
        _ID_TO_LABEL[d] for d in _BMI_DEFAULT_IDS if d in _ID_TO_LABEL
    ]

    selected_labels = st.multiselect(
        "Diseases (type to search — pick one or more to pool as a disease area)",
        options=_DISEASE_LABELS,
        default=_bmi_default_labels,
        key="disease_workflow_selected",
        help="All 47k Open Targets diseases. Typing 'BMI' filters to BMI-related "
             "traits; selecting multiple unions them.",
    )
    selected_disease_ids = [_LABEL_TO_ID[l] for l in selected_labels
                            if l in _LABEL_TO_ID]

    dc1, dc2, dc3 = st.columns([0.8, 0.8, 0.6])
    with dc1:
        disease_top_n = st.number_input(
            "Top N targets per disease", min_value=10, max_value=200,
            value=50, step=10, key="disease_workflow_top_n",
        )
    with dc2:
        ga_workflow_thr = st.slider(
            "Min GA score (novel)", 0.0, 1.0, 0.10, 0.05,
            key="disease_workflow_ga",
        )
    with dc3:
        st.write("")
        st.write("")
        disease_run = st.button(
            "Run", key="disease_workflow_run", use_container_width=True,
            disabled=not selected_disease_ids,
        )

    if disease_run and selected_disease_ids:
        all_disease_dfs: list[pd.DataFrame] = []
        n_total_targets = 0
        progress = st.progress(0.0, text="Fetching disease associations…")
        for i, did in enumerate(selected_disease_ids, 1):
            dname = _ID_TO_NAME.get(did, did)
            progress.progress(
                (i - 1) / len(selected_disease_ids),
                text=f"Fetching ({i}/{len(selected_disease_ids)}) {dname[:60]}…",
            )
            try:
                _meta, _df = ot.fetch_disease(did, page_size=int(disease_top_n))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"OT fetch failed for {did}: {exc}")
                continue
            if _df is None or _df.empty:
                continue
            _df = _df.copy()
            _df["_disease_id"] = did
            _df["_disease_name"] = _meta["name"]
            all_disease_dfs.append(_df)
            n_total_targets += _meta.get("n_targets_total", 0)
        progress.empty()

        if not all_disease_dfs:
            st.warning("No targets returned for any of the selected diseases.")
        else:
            combined = pd.concat(all_disease_dfs, ignore_index=True)
            agg = (
                combined.groupby(["ensg", "symbol"], as_index=False)
                  .agg(
                      name=("name", "first"),
                      biotype=("biotype", "first"),
                      overall_score=("overall_score", "max"),
                      genetic_association_score=("genetic_association_score", "max"),
                      known_drug_score=("known_drug_score", "max"),
                      n_source_diseases=("_disease_id", "nunique"),
                      source_diseases=("_disease_name",
                                       lambda s: sorted(set(s))),
                  )
                  .sort_values("genetic_association_score", ascending=False)
            )
            ann = drug_safety.annotate_with_drug_status(
                agg, selected_disease_ids,
            )
            ann["bucket"] = ann["bucket"].astype(str).fillna("none")
            # Novel-candidate filter computed at Run time so the view is
            # consistent with the GA threshold the user saw.
            ann["_eligible_novel"] = (
                (ann["genetic_association_score"].fillna(0) >= ga_workflow_thr)
                & (~ann["approved_for_this_disease"])
                & (~ann["has_safety_event"])
                & (ann["bucket"].isin(["specific", "intermediate"]))
            )
            ann["_score"] = ann["genetic_association_score"].fillna(0)

            st.session_state["disease_results"] = {
                "ann": ann,
                "combined": combined,
                "selected_disease_ids": list(selected_disease_ids),
                "n_total_targets": n_total_targets,
                "ga_threshold_at_run": ga_workflow_thr,
                "area_label": (
                    _ID_TO_NAME[selected_disease_ids[0]]
                    if len(selected_disease_ids) == 1
                    else f"{len(selected_disease_ids)} pooled diseases"
                ),
            }

    # ---- Render persisted results (survives tab switches) -----------------
    _disease_results = st.session_state.get("disease_results")
    if _disease_results:
        ann = _disease_results["ann"]
        combined = _disease_results["combined"]
        sel_ids = _disease_results["selected_disease_ids"]
        n_total_targets = _disease_results["n_total_targets"]
        ga_used = _disease_results["ga_threshold_at_run"]
        area_label = _disease_results["area_label"]
        if not disease_run:
            st.caption(
                "Showing the last completed search. Click **Run** to refresh "
                f"with the current selection (last run at GA ≥ {ga_used:.2f}, "
                f"{len(sel_ids)} disease(s))."
            )
        hm = st.columns(5)
        hm[0].metric("Disease area",
                     area_label[:24] + ("…" if len(area_label) > 24 else ""))
        hm[1].metric("OT targets (total, summed)",
                     f"{n_total_targets:,}")
        hm[2].metric("Unique targets pooled",
                     f"{len(ann):,}")
        hm[3].metric("Already approved here",
                     int(ann["approved_for_this_disease"].sum()))
        hm[4].metric("In trials here",
                     int(ann["in_trials_for_this_disease"].sum()))

        if len(sel_ids) > 1:
            with st.expander(f"Pooled diseases ({len(sel_ids)})",
                             expanded=False):
                per_disease = (
                    combined.groupby("_disease_name")
                            .size()
                            .reset_index(name="n_targets_fetched")
                            .rename(columns={"_disease_name": "Disease"})
                            .sort_values("n_targets_fetched", ascending=False)
                )
                st.dataframe(per_disease, use_container_width=True,
                             hide_index=True)

        # Novel-candidate filter was computed at Run time (column on ann).
        novel = ann[ann["_eligible_novel"]].sort_values(
            ["_score", "n_drugs"], ascending=[False, True]
        )

        st.markdown("##### Novel-target candidates")
        st.caption(
            "Filter: GA score ≥ threshold · NOT approved for this disease "
            "· no OT safety-event flag · pleiotropy bucket in "
            "{specific, intermediate}. Sorted by genetic-association score."
        )
        if novel.empty:
            st.info(
                "No candidates passed the filter. Lower the GA threshold "
                "or expand to highly pleiotropic targets."
            )
        else:
            nv_cols = [
                "symbol", "genetic_association_score", "overall_score",
                "bucket", "novelty", "max_drug_phase_label",
                "n_drugs", "n_drugged_diseases",
                "has_small_mol_binder", "is_in_membrane", "is_secreted",
            ]
            nv_cols = [c for c in nv_cols if c in novel.columns]
            nv_show = novel[nv_cols].rename(columns={
                "symbol": "Symbol",
                "genetic_association_score": "GA score",
                "overall_score": "Overall",
                "bucket": "Bucket",
                "novelty": "Drug status",
                "max_drug_phase_label": "Max stage (anywhere)",
                "n_drugs": "# drugs",
                "n_drugged_diseases": "# drugged diseases",
                "has_small_mol_binder": "SM binder",
                "is_in_membrane": "Membrane",
                "is_secreted": "Secreted",
            })
            st.dataframe(
                nv_show, use_container_width=True, hide_index=True,
                column_config={
                    "GA score":  st.column_config.NumberColumn(format="%.3f"),
                    "Overall":   st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.download_button(
                "Download novel candidates (CSV)",
                data=novel[nv_cols].to_csv(index=False).encode(),
                file_name=(
                    "novel_candidates_"
                    + (sel_ids[0]
                       if len(sel_ids) == 1
                       else f"{len(sel_ids)}_diseases")
                    + ".csv"
                ),
                mime="text/csv",
            )

        # ---- Plots ----------------------------------------------------
        st.markdown("##### Drug-safety landscape of GWAS-supported targets")
        import plotly.graph_objects as go

        p1, p2 = st.columns(2)

        NOVELTY_ORDER = ["no drug", "approved elsewhere",
                         "in trials here", "approved here"]
        NOVELTY_COLOR = {
            "no drug":            "#1f883d",
            "approved elsewhere": "#3082b8",
            "in trials here":     "#e3b505",
            "approved here":      "#9aa0a6",
        }
        with p1:
            st.markdown("**GA score × max clinical phase  ·  bubble = # drugs**")
            sc = ann.copy()
            sc["max_drug_phase"] = sc["max_drug_phase"].fillna(0.0)
            sc["size"] = (sc["n_drugs"].fillna(0).astype(float) + 1).clip(upper=20)
            fig = px.scatter(
                sc, x="genetic_association_score", y="max_drug_phase",
                color="novelty", hover_name="symbol",
                hover_data={"bucket": True, "n_drugged_diseases": True,
                            "has_safety_event": True,
                            "genetic_association_score": ":.2f",
                            "max_drug_phase": True, "size": False},
                size="size", size_max=22,
                color_discrete_map=NOVELTY_COLOR,
                category_orders={"novelty": NOVELTY_ORDER},
            )
            fig.add_vline(x=ga_used, line_dash="dash", line_color="grey",
                          annotation_text=f"GA ≥ {ga_used:.2f}")
            fig.update_yaxes(title="max clinical phase (anywhere)",
                             tickvals=[0, 1, 2, 3, 4],
                             ticktext=["—", "Ph1", "Ph2", "Ph3", "Approved"])
            fig.update_xaxes(title="OT genetic-association score")
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        with p2:
            st.markdown("**Drug-status × pleiotropy bucket**")
            grid = (
                ann.groupby(["novelty", "bucket"], observed=True)
                   .size().reset_index(name="n")
            )
            pivot = (
                grid.pivot(index="novelty", columns="bucket", values="n")
                    .reindex(NOVELTY_ORDER)
                    .reindex(columns=drug_safety.BUCKET_ORDER)
                    .fillna(0).astype(int)
            )
            fig = px.imshow(
                pivot, color_continuous_scale="Blues", text_auto=True,
                labels=dict(x="Pleiotropy bucket", y="Drug status",
                            color="# targets"),
            )
            fig.update_layout(height=440,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        p3, p4 = st.columns(2)
        with p3:
            st.markdown("**Safety / tractability profile (top-N)**")
            profile_rows = []
            for label, col in [
                ("Safety event",       "has_safety_event"),
                ("Cancer driver",      "is_cancer_driver"),
                ("Approved here",      "approved_for_this_disease"),
                ("In trials here",     "in_trials_for_this_disease"),
                ("Approved elsewhere", "approved_any_indication"),
                ("Small-mol binder",   "has_small_mol_binder"),
                ("Membrane",           "is_in_membrane"),
                ("Secreted",           "is_secreted"),
            ]:
                if col in ann.columns:
                    profile_rows.append({
                        "feature": label,
                        "rate": float(ann[col].mean()),
                        "n": int(ann[col].sum()),
                    })
            pr = pd.DataFrame(profile_rows)
            fig = px.bar(
                pr.sort_values("rate"), x="rate", y="feature",
                orientation="h", text="n",
            )
            fig.update_xaxes(tickformat=".0%",
                             range=[0, max(1.0, pr["rate"].max() * 1.1)])
            fig.update_layout(height=420,
                              margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with p4:
            st.markdown("**Pleiotropy bucket distribution (top-N)**")
            hist = (
                ann["bucket"].value_counts()
                .reindex(drug_safety.BUCKET_ORDER, fill_value=0)
                .reset_index()
            )
            hist.columns = ["bucket", "count"]
            fig = px.bar(
                hist, x="bucket", y="count", color="bucket",
                color_discrete_map=drug_safety.BUCKET_COLOR,
            )
            fig.update_layout(showlegend=False, height=420,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # ---- Full master table ---------------------------------------
        with st.expander(f"Full annotated target table  ({len(ann)} rows)"):
            show_cols = [
                "symbol", "overall_score", "genetic_association_score",
                "bucket", "novelty", "max_drug_phase_label",
                "n_drugs", "n_drugged_diseases",
                "has_safety_event", "is_cancer_driver",
                "has_small_mol_binder", "is_in_membrane", "is_secreted",
                "genetic_constraint",
            ]
            show_cols = [c for c in show_cols if c in ann.columns]
            st.dataframe(
                ann[show_cols].sort_values(
                    "genetic_association_score", ascending=False,
                ),
                use_container_width=True, hide_index=True,
                column_config={
                    "overall_score": st.column_config.NumberColumn(format="%.3f"),
                    "genetic_association_score":
                        st.column_config.NumberColumn(format="%.3f"),
                    "genetic_constraint":
                        st.column_config.NumberColumn(format="%.2f"),
                },
            )


# =============================================================================
# Tab 2 — Gene → diseases  (single gene + sweet-spot filter)
# =============================================================================
with tab_gene:
    sub_single, sub_list = st.tabs(
        ["Single gene", "Sweet-spot filter (gene list)"]
    )

    # -------------------------------------------------------------------------
    # Single gene
    # -------------------------------------------------------------------------
    with sub_single:
        st.markdown("#### Gene-centric pleiotropy")
        col1, col2 = st.columns([2, 1])
        with col1:
            gene_input = st.text_input(
                "Gene symbol or Ensembl ID",
                value="PCSK9",
                key="gene_centric_input",
            )
        with col2:
            page_size = st.number_input(
                "Max associated diseases", min_value=50, max_value=2000, value=500, step=50,
                key="gene_centric_page_size",
            )

        if gene_input:
            with st.spinner(f"Resolving {gene_input} and fetching associations..."):
                ensg = ot.resolve_target(gene_input)
                if ensg is None:
                    st.error(f"Could not resolve '{gene_input}' to an Ensembl gene.")
                else:
                    meta, df, _legacy_drugs = ot.fetch_target(ensg, page_size=int(page_size))
                    summary = ot.summarize_target_pleiotropy(
                        ensg,
                        ga_threshold=ga_threshold,
                        overall_threshold=overall_threshold,
                        check_pav=check_pav,
                    )
                    # Local parquet drugs/safety (OT 26.03 clinical_target × clinical_indication).
                    try:
                        ds_drugs = drug_safety.target_drugs(ensg)
                        ds_feat = drug_safety.target_features(ensg)
                    except FileNotFoundError as exc:
                        st.warning(f"Drug-safety release data not available: {exc}")
                        ds_drugs, ds_feat = pd.DataFrame(), None

            if meta and summary:
                # Header metrics
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Symbol", meta["symbol"])
                c2.metric("Diseases (filtered)", summary.n_diseases)
                c3.metric("Therapeutic areas", summary.n_therapeutic_areas)
                bucket_color = {
                    "specific": "🟢",
                    "intermediate": "🟡 (sweet spot)",
                    "highly_pleiotropic": "🔴",
                    "none": "⚪",
                }
                c4.metric("Pleiotropy bucket", bucket_color.get(summary.classification, summary.classification))
                c5.metric("PAV supported", "✅" if summary.pav_supported else "—")

                st.markdown(f"**{meta['name']}** · biotype: `{meta['biotype']}`")
                if meta.get("function"):
                    st.caption(meta["function"][:400] + ("…" if len(meta["function"]) > 400 else ""))

                # TA bar chart
                sub = df[
                    (df["genetic_association_score"] >= ga_threshold)
                    & (df["overall_score"] >= overall_threshold)
                ]
                ta_counts: dict[str, int] = {}
                for tas in sub["therapeutic_areas"]:
                    for _id, name in tas:
                        ta_counts[name] = ta_counts.get(name, 0) + 1
                if ta_counts:
                    ta_df = pd.DataFrame(
                        sorted(ta_counts.items(), key=lambda x: -x[1]),
                        columns=["Therapeutic area", "n diseases"],
                    )
                    fig = px.bar(
                        ta_df, x="n diseases", y="Therapeutic area",
                        orientation="h", height=max(280, 24 * len(ta_df)),
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No therapeutic areas pass the score threshold.")

                # Top diseases
                st.markdown("##### Top filtered disease associations")
                top = sub.sort_values("overall_score", ascending=False).head(20)[
                    ["disease_name", "overall_score", "genetic_association_score",
                     "known_drug_score", "literature_score"]
                ].rename(columns={
                    "disease_name": "Disease",
                    "overall_score": "Overall",
                    "genetic_association_score": "Genetic",
                    "known_drug_score": "Known drug",
                    "literature_score": "Lit.",
                })
                st.dataframe(top, use_container_width=True, hide_index=True)

                # ---------------------------------------------------------------
                # Drug / safety / tractability — from local OT 26.03 parquets
                # ---------------------------------------------------------------
                if ds_feat is not None:
                    st.markdown("##### Drug development & safety profile")
                    max_phase = float(ds_feat.get("max_drug_phase", 0.0) or 0.0)
                    max_phase_label = drug_safety.PHASE_NUM_TO_LABEL.get(
                        max_phase, "—" if max_phase == 0 else f"phase {max_phase}"
                    )
                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Max clinical stage", max_phase_label)
                    d2.metric("Drugs in dev (any phase)", int(ds_feat.get("n_drugs") or 0))
                    d3.metric("Drugged diseases",
                              int(ds_feat.get("n_drugged_diseases") or 0))
                    d4.metric("Safety event flag",
                              "⚠️ Yes" if ds_feat.get("has_safety_event") else "—")
                    d5.metric("Cancer driver",
                              "⚠️ Yes" if ds_feat.get("is_cancer_driver") else "—")

                    chip_cols = st.columns(5)
                    def _chip(col, label: str, present: bool) -> None:
                        mark = "✅" if present else "—"
                        col.markdown(
                            f"<div style='border:1px solid #ddd; border-radius:6px; padding:8px;"
                            f"text-align:center;'><div style='font-size:1.4rem;'>{mark}</div>"
                            f"<div style='font-size:0.85rem; color:#555;'>{label}</div></div>",
                            unsafe_allow_html=True,
                        )
                    _chip(chip_cols[0], "Small-mol binder",
                          bool(ds_feat.get("has_small_mol_binder")))
                    _chip(chip_cols[1], "Ligand", bool(ds_feat.get("has_ligand")))
                    _chip(chip_cols[2], "Membrane",
                          bool(ds_feat.get("is_in_membrane")))
                    _chip(chip_cols[3], "Secreted",
                          bool(ds_feat.get("is_secreted")))
                    _chip(chip_cols[4], "Chem probes",
                          bool(ds_feat.get("has_chem_probes")))

                if not ds_drugs.empty:
                    with st.expander(f"Drugs targeting {meta['symbol']} "
                                     f"({len(ds_drugs)} unique drug × stage rows)"):
                        show = ds_drugs.copy()
                        show["indications"] = show["indications"].apply(
                            lambda v: ", ".join(v[:6]) + ("…" if isinstance(v, list)
                                                          and len(v) > 6 else "")
                            if isinstance(v, list) else ""
                        )
                        st.dataframe(
                            show.rename(columns={
                                "drug_id": "ChEMBL ID",
                                "max_clinical_stage": "Max stage",
                                "phase": "Phase",
                                "indications": "Indications",
                                "n_indications": "# indications",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                        st.caption(
                            "Source: OT release 26.03 `clinical_target` × "
                            "`clinical_indication`. Drug names are ChEMBL IDs; "
                            "the OT release does not bundle a `drug` lookup."
                        )

                # PAV evidence
                if check_pav and summary.pav_supported:
                    with st.expander(f"PAV-supported diseases ({len(summary.pav_diseases)})"):
                        st.write(", ".join(summary.pav_diseases[:30]))

                # ---------------------------------------------------------------
                # Phase-0 therapeutic readout (cis-pQTL MR)
                # ---------------------------------------------------------------
                # This is the differentiator vs. OT.org: we run two-sample MR
                # using curated cis-pQTLs to predict the *direction* and *size*
                # of the therapeutic effect.
                if meta["symbol"] in pqtl_instruments.available_proteins():
                    st.markdown("---")
                    st.markdown("##### Phase-0 therapeutic readout (cis-pQTL Mendelian randomization)")
                    st.caption(
                        "What pharma actually wants to know from human genetics: "
                        "if we modulate this protein, *which direction*, *how much*, "
                        "*for which diseases*. Effects below are inverse-variance "
                        "weighted MR estimates using curated cis-pQTL instruments."
                    )
                    prot_meta = pqtl_instruments.protein_metadata(meta["symbol"])
                    exp = pqtl_instruments.get_instruments(meta["symbol"])

                    phase0_rows = []
                    for o in disease_gwas.available_outcomes():
                        out2 = disease_gwas.get_outcome_effects(o, exp["rsid"].tolist())
                        if out2.empty:
                            continue
                        h2 = mr_engine.harmonize(exp, out2)
                        r2 = mr_engine.run_mr(
                            h2, outcome_is_binary=disease_gwas.OUTCOMES[o]["binary"]
                        )
                        if "error" in r2:
                            continue
                        ivw2 = r2["ivw"]
                        binary = disease_gwas.OUTCOMES[o]["binary"]
                        if binary:
                            or_pt, or_lcl, or_ucl = ivw2.or_ci()
                            effect_str = f"OR {or_pt:.2f}  ({or_lcl:.2f}–{or_ucl:.2f})"
                        else:
                            effect_str = (f"β {ivw2.estimate:+.2f}  "
                                          f"({ivw2.estimate - 1.96*ivw2.se:+.2f}, "
                                          f"{ivw2.estimate + 1.96*ivw2.se:+.2f})")
                        # Therapeutic-direction string — short version
                        if abs(ivw2.estimate) < 0.02 or ivw2.pvalue > 0.05:
                            direction = "— (no causal signal)"
                        elif binary:
                            direction = ("↑ protein raises risk → inhibit"
                                         if ivw2.estimate > 0
                                         else "↑ protein protects → activate")
                        else:
                            direction = ("↑ protein raises trait"
                                         if ivw2.estimate > 0
                                         else "↑ protein lowers trait")
                        phase0_rows.append({
                            "Outcome": disease_gwas.OUTCOMES[o]["label"],
                            "Effect (per SD protein)": effect_str,
                            "IVW p": f"{ivw2.pvalue:.2e}",
                            "Egger int. p": f"{r2['egger_intercept'].pvalue:.2f}",
                            "n SNPs": int(ivw2.n_snps),
                            "Mean F": f"{r2['mean_F']:.0f}",
                            "Therapeutic direction": direction,
                            "_beta": ivw2.estimate,
                            "_se": ivw2.se,
                            "_p": ivw2.pvalue,
                            "_binary": binary,
                        })

                    if phase0_rows:
                        phase0_df = pd.DataFrame(phase0_rows)
                        st.dataframe(
                            phase0_df.drop(columns=[c for c in phase0_df.columns if c.startswith("_")]),
                            use_container_width=True, hide_index=True,
                        )

                        # Forest plot
                        import plotly.graph_objects as go
                        fdf = phase0_df.copy()
                        fdf["lcl"] = fdf["_beta"] - 1.96 * fdf["_se"]
                        fdf["ucl"] = fdf["_beta"] + 1.96 * fdf["_se"]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fdf["_beta"], y=fdf["Outcome"],
                            mode="markers",
                            error_x=dict(type="data", array=1.96 * fdf["_se"]),
                            marker=dict(size=12, color=[
                                "#888" if p > 0.05 else
                                ("#c0392b" if b > 0 else "#27ae60")
                                for b, p in zip(fdf["_beta"], fdf["_p"])
                            ]),
                        ))
                        fig.add_vline(x=0, line_color="grey")
                        fig.update_layout(
                            xaxis_title="IVW β per SD protein  (log-OR for binary; SD for continuous)",
                            height=max(220, 40 * len(fdf)),
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Top therapeutic strategy callout
                        sig = phase0_df[phase0_df["_p"] < 0.01].copy()
                        if not sig.empty:
                            sig["_abs"] = sig["_beta"].abs()
                            top_hit = sig.sort_values("_abs", ascending=False).iloc[0]
                            outcome_name = top_hit["Outcome"]
                            if top_hit["_binary"]:
                                or_pt = float(np.exp(top_hit["_beta"]))
                                effect = f"OR {or_pt:.2f} per SD higher plasma {meta['symbol']}"
                            else:
                                effect = f"β {top_hit['_beta']:+.2f} per SD"
                            if top_hit["_beta"] > 0:
                                rec = (f"**Inhibit / antagonize {meta['symbol']}** — "
                                       f"higher protein causally raises {outcome_name} "
                                       f"({effect}, p={top_hit['_p']:.1e}).")
                            else:
                                rec = (f"**Activate / agonize {meta['symbol']}** — "
                                       f"higher protein protects against {outcome_name} "
                                       f"({effect}, p={top_hit['_p']:.1e}).")
                            st.success(rec)
                    else:
                        st.info(
                            f"No curated outcome effects overlap with {meta['symbol']} instruments. "
                            f"(Demo set: {', '.join(disease_gwas.available_outcomes())}.)"
                        )
                else:
                    st.info(
                        f"{meta['symbol']} is not in the curated cis-pQTL instrument set "
                        f"({', '.join(pqtl_instruments.available_proteins())}). "
                        f"Phase-0 readout disabled. Switch to a curated protein "
                        f"or extend `pleio/pqtl_instruments.py`."
                    )

    # -------------------------------------------------------------------------
    # Sweet-spot filter (gene-list mode)
    # -------------------------------------------------------------------------
    with sub_list:
        st.markdown("#### Sweet-spot filter")
        st.caption(
            "Paste a candidate gene list. Returns rows matching the manuscript's "
            "intermediate-pleiotropy (2–5 TAs) ∩ PAV-supported profile (OR ≈ 10.3)."
        )
        default_list = "PCSK9\nLPA\nHMGCR\nIL6R\nLDLR\nANGPTL3\nANGPTL4\nAPOC3\nNPC1L1\nIL18\nTYK2\nIL23R\nTNF\nTP53"

        # Auto-fill from Tab 1's novel-target candidates if a Disease → targets
        # search has run. Refreshes whenever the Disease workflow runs with new
        # inputs; user edits stick across runs with the same inputs.
        _dr = st.session_state.get("disease_results")
        novel_symbols_from_disease: list[str] = []
        _dr_signature = None
        if _dr is not None:
            ann = _dr["ann"]
            if "_eligible_novel" in ann.columns:
                novel_symbols_from_disease = (
                    ann[ann["_eligible_novel"]]
                    .sort_values(["_score", "n_drugs"], ascending=[False, True])
                    ["symbol"].head(50).tolist()
                )
                _dr_signature = (
                    tuple(_dr["selected_disease_ids"]),
                    round(float(_dr["ga_threshold_at_run"]), 4),
                    tuple(novel_symbols_from_disease),
                )
        _last_sig = st.session_state.get("_sweet_spot_disease_sig")
        if (_dr_signature is not None
                and _dr_signature != _last_sig
                and novel_symbols_from_disease):
            st.session_state["sweet_spot_text"] = "\n".join(novel_symbols_from_disease)
            st.session_state["_sweet_spot_disease_sig"] = _dr_signature
            st.caption(
                f"Auto-filled with {len(novel_symbols_from_disease)} novel "
                "candidates from the latest Disease → targets run "
                f"({_dr['area_label']}). Edit freely; this overlay only "
                "refreshes when you re-Run that workflow with different inputs."
            )

        gene_text = st.text_area(
            "Gene symbols (one per line or comma-separated)",
            value=default_list, height=160, key="sweet_spot_text",
        )
        sweet_spot_run = st.button("Score candidates", key="sweet_spot_run")
        if sweet_spot_run:
            syms = [s.strip() for s in gene_text.replace(",", "\n").splitlines() if s.strip()]
            progress = st.progress(0.0, text="Scoring…")

            def _cb(i: int, n: int, sym: str | None) -> None:
                progress.progress(min(i / n, 1.0), text=f"Scoring ({i}/{n}) {sym or 'done'}")

            df = ot.summarize_many(
                syms,
                ga_threshold=ga_threshold,
                overall_threshold=overall_threshold,
                check_pav=check_pav,
                progress_cb=_cb,
            )
            progress.empty()

            if df.empty:
                st.warning("No results.")
                st.session_state["sweet_spot_results"] = None
            else:
                df = df.sort_values(
                    ["sweet_spot", "n_therapeutic_areas"], ascending=[False, True]
                )
                st.session_state["sweet_spot_results"] = {
                    "df": df,
                    "n_sweet": int(df["sweet_spot"].fillna(False).sum()),
                }

        # ---- Render persisted results (survives tab switches) -------------
        _sweet_results = st.session_state.get("sweet_spot_results")
        if _sweet_results:
            if not sweet_spot_run:
                st.caption(
                    "Showing the last scored gene list. Click **Score candidates** "
                    "to refresh."
                )
            df_sweet = _sweet_results["df"]
            st.dataframe(df_sweet, use_container_width=True, hide_index=True)
            st.success(
                f"{_sweet_results['n_sweet']}/{len(df_sweet)} candidates "
                "in the sweet spot."
            )

# =============================================================================
# Tab 3 — Mechanism deep-dive (coloc layer)
# =============================================================================
with tab_mech:
    st.markdown("#### Mechanism tier  ·  cis-pQTL / cis-eQTL coloc + PAV")
    st.info(
        "**T1**: cis-pQTL coloc + directional consistency · "
        "**T2**: cis-eQTL coloc (or pQTL coloc without directional consistency) · "
        "**T3**: PAV-only evidence · "
        "**T4**: GWAS-only association."
    )

    mech_sub_single, mech_sub_batch = st.tabs(
        ["Single pair", "Batch — top N targets for a disease"]
    )

    # -------------------------------------------------------------------------
    # Helper — render a single MechanismEvidence as a coloured tier card.
    # -------------------------------------------------------------------------
    def _render_mechanism(ev: coloc.MechanismEvidence) -> None:
        # Coloured tier banner
        color = coloc.TIER_COLOR[ev.tier]
        st.markdown(
            f"""<div style='background:{color}; color:white; padding:14px 18px;
                border-radius:8px; margin: 4px 0 14px 0;'>
                <div style='font-size:0.85rem; opacity:0.85;'>Mechanism tier</div>
                <div style='font-size:1.6rem; font-weight:700;'>{ev.tier}</div>
                <div style='font-size:0.95rem;'>{ev.tier_label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

        # Evidence chips
        chip_cols = st.columns(4)
        def _chip(col, label: str, present: bool) -> None:
            mark = "✅" if present else "—"
            col.markdown(
                f"<div style='border:1px solid #ddd; border-radius:6px; padding:8px;"
                f"text-align:center;'><div style='font-size:1.4rem;'>{mark}</div>"
                f"<div style='font-size:0.85rem; color:#555;'>{label}</div></div>",
                unsafe_allow_html=True,
            )
        _chip(chip_cols[0], "cis-pQTL coloc", ev.has_pqtl_coloc)
        _chip(chip_cols[1], "cis-eQTL coloc", ev.has_eqtl_coloc)
        _chip(chip_cols[2], "PAV evidence", ev.has_pav)
        _chip(chip_cols[3], f"GA score ≥ {ga_threshold:.2f}", ev.has_gwas)

        # Metrics row
        st.markdown("")
        m = st.columns(5)
        m[0].metric("GA score", f"{ev.ga_score:.3f}")
        if ev.mr_beta is not None:
            m[1].metric("MR IVW β", f"{ev.mr_beta:+.3f}",
                        help=f"SE {ev.mr_se:.3f}")
            m[2].metric("MR p", f"{ev.mr_p:.2e}")
            m[3].metric("Sign concordance",
                        f"{ev.sign_concordance:.0%}" if ev.sign_concordance is not None else "—",
                        help="Fraction of per-SNP Wald ratios matching the IVW sign.")
            m[4].metric("Egger intercept p",
                        f"{ev.egger_intercept_p:.2f}" if ev.egger_intercept_p is not None else "—",
                        help="Underpowered with <6 instruments; informational only.")
        else:
            m[1].metric("MR IVW β", "—")
            m[2].metric("MR p", "—")
            m[3].metric("Sign concordance", "—")
            m[4].metric("Egger intercept p", "—")

        # Plain-English direction read-out
        if ev.mr_direction_text and ev.directional_consistency == "concordant":
            st.success(
                f"**Directional consistency: concordant.** {ev.mr_direction_text} "
                f"({ev.n_instruments_used} curated cis-"
                f"{'pQTL' if ev.has_pqtl_coloc else 'eQTL'} instruments)."
            )
        elif ev.mr_beta is not None:
            st.warning(
                f"**Directional consistency: inconsistent.** IVW β {ev.mr_beta:+.3f} "
                f"(p={ev.mr_p:.2e}, sign concordance "
                f"{(ev.sign_concordance or 0):.0%}). Falling back to T2."
            )

        # PAV variants
        if ev.pav_variants:
            with st.expander(f"PAV evidence ({len(ev.pav_variants)} variants)"):
                pav_df = pd.DataFrame(ev.pav_variants)
                st.dataframe(pav_df, use_container_width=True, hide_index=True)

        # Free-form notes
        if ev.notes:
            with st.expander("Classifier notes"):
                for n in ev.notes:
                    st.write("• " + n)

    # -------------------------------------------------------------------------
    # Single (target, disease) classifier
    # -------------------------------------------------------------------------
    with mech_sub_single:
        st.markdown("##### Classify one (target, disease) pair")
        col_t, col_d, col_btn = st.columns([1.2, 1.6, 0.7])
        with col_t:
            mech_target = st.text_input(
                "Target (symbol or ENSG)", value="PCSK9", key="mech_target_input",
            )
        with col_d:
            _cad_label = _ID_TO_LABEL.get("EFO_0001645")
            _default_idx = (
                _DISEASE_LABELS.index(_cad_label)
                if _cad_label in _LABEL_TO_ID else 0
            )
            mech_disease_label = st.selectbox(
                "Disease (type to search OT catalog)",
                options=_DISEASE_LABELS,
                index=_default_idx,
                key="mech_disease_input",
                help="Autocomplete over all 47k Open Targets diseases. Start typing "
                     "(e.g. 'coronary' or 'BMI') to filter.",
            )
            mech_disease_id = _LABEL_TO_ID.get(mech_disease_label)
        with col_btn:
            st.write("")  # spacer to align button vertically
            st.write("")
            run_single = st.button("Classify", key="mech_run_single",
                                   use_container_width=True)

        st.caption(
            "Demo set with curated cis-QTL evidence: "
            f"{', '.join(pqtl_instruments.available_proteins())} × "
            f"{', '.join(disease_gwas.available_outcomes())}. "
            "Outside this set, only OT PAV and GA-score evidence is used "
            "(T3/T4 path)."
        )

        if run_single and mech_target and mech_disease_id:
            with st.spinner(f"Classifying {mech_target} → {mech_disease_label}…"):
                try:
                    ev = coloc.classify_mechanism(
                        mech_target, mech_disease_id,
                        ga_threshold=ga_threshold,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Classifier failed: {exc}")
                    ev = None
            if ev is None:
                st.error(
                    f"Could not resolve target='{mech_target}' or "
                    f"disease='{mech_disease_label}'."
                )
                st.session_state["mech_single_result"] = None
            else:
                st.session_state["mech_single_result"] = ev

        _mech_single = st.session_state.get("mech_single_result")
        if _mech_single is not None:
            if not run_single:
                st.caption(
                    "Showing the last classification. Click **Classify** to refresh."
                )
            st.markdown(
                f"**{_mech_single.target_symbol}** ({_mech_single.target_ensg})  →  "
                f"**{_mech_single.disease_name}** ({_mech_single.disease_id})"
            )
            _render_mechanism(_mech_single)

    # -------------------------------------------------------------------------
    # Batch — top N targets for a disease
    # -------------------------------------------------------------------------
    with mech_sub_batch:
        st.markdown("##### Tier the top N OT targets for a disease area")
        _batch_default_labels = [
            _ID_TO_LABEL[d] for d in ["EFO_0001645"]   # CAD as the default anchor
            if d in _ID_TO_LABEL
        ]
        batch_disease_labels = st.multiselect(
            "Diseases (type to search — pick one or more to pool as a disease area)",
            options=_DISEASE_LABELS,
            default=_batch_default_labels,
            key="mech_batch_disease",
            help="Autocomplete over all 47k OT diseases. Selecting multiple "
                 "pools their top-N target lists (highest GA score wins).",
        )
        batch_disease_ids = [_LABEL_TO_ID[l] for l in batch_disease_labels
                             if l in _LABEL_TO_ID]
        col_n, col_btn = st.columns([0.5, 0.5])
        with col_n:
            batch_n = st.number_input(
                "Top N targets per disease", min_value=3, max_value=50,
                value=10, step=1, key="mech_batch_n",
            )
        with col_btn:
            st.write("")
            st.write("")
            run_batch = st.button(
                "Run batch", key="mech_run_batch",
                use_container_width=True,
                disabled=not batch_disease_ids,
            )

        st.caption(
            "Pulls the top N targets from OT's associatedTargets ranking for "
            "each selected disease, unions them (best GA score per target), "
            "and runs the mechanism-tier classifier on each pair. "
            "For a richer demo, use a cardiometabolic anchor "
            "(CAD / T2D / LDL-C / HDL-C / TG)."
        )

        if run_batch and batch_disease_ids:
            batch_dfs: list[pd.DataFrame] = []
            n_total_targets_batch = 0
            with st.spinner(
                f"Fetching top {int(batch_n)} targets across "
                f"{len(batch_disease_ids)} disease(s)…"
            ):
                for _did in batch_disease_ids:
                    try:
                        _m, _df = ot.fetch_disease(_did, page_size=int(batch_n))
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"OT fetch failed for {_did}: {exc}")
                        continue
                    if _df is None or _df.empty:
                        continue
                    _df = _df.copy()
                    _df["_disease_id"] = _did
                    _df["_disease_name"] = _m["name"]
                    batch_dfs.append(_df)
                    n_total_targets_batch += _m.get("n_targets_total", 0)

            if not batch_dfs:
                st.warning("No targets returned for any of the selected diseases.")
            else:
                combined_batch = pd.concat(batch_dfs, ignore_index=True)
                best = (
                    combined_batch
                    .sort_values("genetic_association_score", ascending=False)
                    .drop_duplicates(subset=["symbol"], keep="first")
                )
                progress = st.progress(0.0, text="Classifying…")
                pairs = list(zip(best["symbol"].tolist(),
                                 best["_disease_id"].tolist()))
                ga_hints = {
                    (row["symbol"], row["_disease_id"]):
                        float(row["genetic_association_score"])
                    for _, row in best.iterrows()
                }
                disease_hints = {
                    row["_disease_id"]: row["_disease_name"]
                    for _, row in best.iterrows()
                }

                def _cb(i: int, n: int, sym: str | None) -> None:
                    progress.progress(min(i / n, 1.0),
                                      text=f"Classifying ({i}/{n}) {sym or 'done'}")

                tier_df = coloc.classify_many(
                    pairs,
                    ga_threshold=ga_threshold,
                    ga_score_hints=ga_hints,
                    disease_name_hints=disease_hints,
                    progress_cb=_cb,
                )
                progress.empty()

                primary_label = (
                    _ID_TO_NAME[batch_disease_ids[0]]
                    if len(batch_disease_ids) == 1
                    else f"{len(batch_disease_ids)} pooled diseases"
                )
                st.session_state["mech_batch_results"] = {
                    "tier_df": tier_df,
                    "combined_batch": combined_batch,
                    "best": best,
                    "batch_disease_ids": list(batch_disease_ids),
                    "primary_label": primary_label,
                    "n_total_targets_batch": n_total_targets_batch,
                }

        # ---- Render persisted batch results (survives tab switches) -------
        _mech_batch = st.session_state.get("mech_batch_results")
        if _mech_batch:
            tier_df = _mech_batch["tier_df"]
            combined_batch = _mech_batch["combined_batch"]
            best = _mech_batch["best"]
            mb_ids = _mech_batch["batch_disease_ids"]
            primary_label = _mech_batch["primary_label"]
            n_total_targets_batch = _mech_batch["n_total_targets_batch"]
            if not run_batch:
                st.caption("Showing the last batch. Click **Run batch** to refresh.")
            st.markdown(
                f"**{primary_label}** — {len(best)} unique targets pooled from "
                f"{n_total_targets_batch:,} OT associations"
            )
            if len(mb_ids) > 1:
                with st.expander(f"Pooled diseases ({len(mb_ids)})",
                                 expanded=False):
                    per_disease = (
                        combined_batch.groupby("_disease_name").size()
                        .reset_index(name="n_targets_fetched")
                        .rename(columns={"_disease_name": "Disease"})
                        .sort_values("n_targets_fetched", ascending=False)
                    )
                    st.dataframe(per_disease, use_container_width=True,
                                 hide_index=True)

            if tier_df.empty:
                st.warning("No results.")
            else:
                counts = (
                    tier_df["tier"].value_counts()
                    .reindex(["T1", "T2", "T3", "T4", "none"], fill_value=0)
                    .reset_index()
                )
                counts.columns = ["tier", "count"]
                fig = px.bar(
                    counts, x="tier", y="count", color="tier",
                    color_discrete_map=coloc.TIER_COLOR,
                )
                fig.update_layout(showlegend=False, height=280,
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                display = tier_df.copy()
                if "confidence" in display.columns:
                    display = display.sort_values(
                        ["confidence", "ga_score"], ascending=[False, False],
                    )
                keep = [c for c in [
                    "target", "tier", "confidence",
                    "has_pqtl_coloc", "has_eqtl_coloc",
                    "has_pav", "has_gwas",
                    "ga_score", "mr_beta", "mr_p",
                    "sign_concordance", "directional_consistency",
                    "mr_direction", "n_instruments_used",
                ] if c in display.columns]
                st.dataframe(display[keep], use_container_width=True,
                             hide_index=True)

                n_t1 = int((tier_df["tier"] == "T1").sum())
                if n_t1:
                    t1_targets = ", ".join(
                        tier_df.loc[tier_df["tier"] == "T1", "target"].tolist()
                    )
                    st.success(
                        f"T1 hits ({n_t1}): {t1_targets} — cis-pQTL coloc "
                        f"with directional consistency."
                    )

# =============================================================================
# Tab 4 — In silico Phase 0 (MR)
# =============================================================================
with tab_mr:
    st.markdown("#### In silico Phase 0 — two-sample Mendelian randomization")
    st.caption(
        "Curated cis-pQTL instruments × disease GWAS effect sizes. "
        "IVW + weighted median + MR-Egger with effect-allele harmonization and "
        "per-instrument F-statistic. Effect sizes from Sun 2018, UKB-PPP, "
        "Ferkingstad 2021, CARDIoGRAMplusC4D, GLGC, DIAGRAM."
    )

    mr_col1, mr_col2, mr_col3 = st.columns([1, 1, 1])
    _mr_proteins = pqtl_instruments.available_proteins()
    _mr_outcomes = disease_gwas.available_outcomes()
    # Default to a BMI example so the tab demonstrates the v0.3 expansion
    # out of the box (GLP1R cis-pQTL → BMI — semaglutide-style mechanism).
    _default_protein_idx = (
        _mr_proteins.index("GLP1R") if "GLP1R" in _mr_proteins else 0
    )
    _default_outcome_idx = (
        _mr_outcomes.index("BMI") if "BMI" in _mr_outcomes else 0
    )
    with mr_col1:
        protein = st.selectbox(
            "Protein (exposure)", _mr_proteins,
            index=_default_protein_idx, key="mr_protein",
        )
    with mr_col2:
        outcome = st.selectbox(
            "Outcome", _mr_outcomes,
            index=_default_outcome_idx, key="mr_outcome",
        )
    with mr_col3:
        show_forest = st.checkbox("Show forest across all outcomes", value=True, key="mr_forest")
    st.caption(
        "BMI demo set: **LEP**, **LEPR**, **GIPR**, **GLP1R**, **MC4R**, **FTO** × "
        "**BMI** / **WHRadjBMI** / **T2D**. Pick GIPR or GLP1R to reproduce the "
        "incretin-axis MR (tirzepatide / semaglutide mechanism)."
    )

    prot_meta = pqtl_instruments.protein_metadata(protein)
    out_meta = disease_gwas.outcome_metadata(outcome)
    if prot_meta and out_meta:
        st.markdown(
            f"**{protein}** ({prot_meta.description}) → **{out_meta['label']}** "
            f"·  outcome type: {'binary' if out_meta['binary'] else 'continuous'} "
            f"·  source: {out_meta['source']}"
        )

    # ---- Run MR for selected (protein, outcome) ----------------------------
    exp = pqtl_instruments.get_instruments(protein)
    out = disease_gwas.get_outcome_effects(outcome, exp["rsid"].tolist())
    if out.empty:
        st.error(f"No outcome effects available for {protein} → {outcome}.")
    else:
        harm = mr_engine.harmonize(exp, out)
        res = mr_engine.run_mr(harm, outcome_is_binary=out_meta["binary"])
        if "error" in res:
            st.warning(res["error"])
        else:
            ivw_r = res["ivw"]
            wm_r = res["weighted_median"]
            eg_s = res["egger_slope"]
            eg_i = res["egger_intercept"]

            # ---- Headline result -------------------------------------------
            hdr = st.columns(5)
            if out_meta["binary"]:
                or_lo, or_lcl, or_ucl = ivw_r.or_ci()
                hdr[0].metric("IVW OR per SD protein", f"{or_lo:.2f}",
                              help=f"95% CI {or_lcl:.2f}–{or_ucl:.2f}")
            else:
                hdr[0].metric("IVW β per SD protein", f"{ivw_r.estimate:.3f}",
                              help=f"SE {ivw_r.se:.3f}")
            hdr[1].metric("IVW p-value", f"{ivw_r.pvalue:.2e}")
            hdr[2].metric("Weighted median β", f"{wm_r.estimate:.3f}",
                          help=f"p={wm_r.pvalue:.2e}")
            hdr[3].metric("Egger intercept p",
                          f"{eg_i.pvalue:.2f}",
                          help="p<0.05 suggests horizontal pleiotropy")
            hdr[4].metric("Mean F-stat",
                          f"{res['mean_F']:.0f}",
                          help=f"min F = {res['min_F']:.0f}; rule of thumb F>10")

            # ---- Clinical interpretation -----------------------------------
            interp = mr_engine.interpret_direction(ivw_r.estimate, out_meta["binary"])
            st.markdown(f"**Clinical interpretation:** {interp}")

            # ---- Per-SNP scatter + forest of effects -----------------------
            sc_col, fr_col = st.columns([1, 1])
            harmonized_df = res["harmonized"]
            per_snp = res["per_snp"]
            with sc_col:
                st.markdown("**Per-SNP scatter (exposure vs outcome β)**")
                scatter_df = harmonized_df.copy()
                scatter_df["F"] = (scatter_df["beta_exp"] / scatter_df["se_exp"]) ** 2
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=scatter_df["beta_exp"], y=scatter_df["beta_out"],
                    mode="markers+text", text=scatter_df["rsid"],
                    textposition="top center",
                    error_x=dict(type="data", array=scatter_df["se_exp"]),
                    error_y=dict(type="data", array=scatter_df["se_out"]),
                    marker=dict(size=10, color=scatter_df["F"], colorscale="Viridis",
                                colorbar=dict(title="F-stat"), showscale=True),
                    name="instruments",
                ))
                # IVW slope line
                xs = scatter_df["beta_exp"].to_numpy()
                xr = [xs.min() * 1.1, xs.max() * 1.1]
                fig.add_trace(go.Scatter(
                    x=xr, y=[ivw_r.estimate * x for x in xr],
                    mode="lines", name="IVW slope", line=dict(dash="dash"),
                ))
                fig.update_layout(
                    xaxis_title=f"β on {protein} protein (SD)",
                    yaxis_title=f"β on {outcome}",
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            with fr_col:
                st.markdown("**Per-instrument Wald ratios**")
                fr = per_snp.copy()
                fr["lcl"] = fr["ratio"] - 1.96 * fr["se"]
                fr["ucl"] = fr["ratio"] + 1.96 * fr["se"]
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fr["ratio"], y=fr["rsid"], mode="markers",
                    error_x=dict(type="data", array=1.96 * fr["se"]),
                    marker=dict(size=10),
                    name="per-SNP",
                ))
                fig.add_vline(x=ivw_r.estimate, line_dash="dash", annotation_text="IVW")
                fig.add_vline(x=0, line_color="grey")
                fig.update_layout(
                    xaxis_title=f"Wald ratio β/β  ({outcome} per SD {protein})",
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- Instrument QC + harmonization actions ---------------------
            with st.expander("Instrument QC"):
                qc = harmonized_df[["rsid", "ea", "oa", "beta_exp", "se_exp",
                                    "beta_out", "se_out", "action"]].copy()
                qc["F"] = (qc["beta_exp"] / qc["se_exp"]) ** 2
                st.dataframe(qc, use_container_width=True, hide_index=True)

            # ---- Forest plot across outcomes -------------------------------
            if show_forest:
                st.markdown("---")
                st.markdown(f"**Effect of {protein} across outcomes (IVW)**")
                forest_rows = []
                for o in disease_gwas.available_outcomes():
                    out2 = disease_gwas.get_outcome_effects(o, exp["rsid"].tolist())
                    if out2.empty:
                        continue
                    h2 = mr_engine.harmonize(exp, out2)
                    r2 = mr_engine.run_mr(h2, outcome_is_binary=disease_gwas.OUTCOMES[o]["binary"])
                    if "error" in r2:
                        continue
                    ivw2 = r2["ivw"]
                    forest_rows.append({
                        "outcome": o, "binary": disease_gwas.OUTCOMES[o]["binary"],
                        "beta": ivw2.estimate, "se": ivw2.se,
                        "p": ivw2.pvalue, "n_snps": ivw2.n_snps,
                    })
                if forest_rows:
                    fdf = pd.DataFrame(forest_rows)
                    fdf["lcl"] = fdf["beta"] - 1.96 * fdf["se"]
                    fdf["ucl"] = fdf["beta"] + 1.96 * fdf["se"]
                    fdf["label"] = fdf.apply(
                        lambda r: f"{r['outcome']}  (n={r['n_snps']}, p={r['p']:.1e})", axis=1
                    )
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fdf["beta"], y=fdf["label"],
                        mode="markers",
                        error_x=dict(type="data", array=1.96 * fdf["se"]),
                        marker=dict(size=12, color=[
                            "#c0392b" if b > 0 else "#27ae60" for b in fdf["beta"]
                        ]),
                    ))
                    fig.add_vline(x=0, line_color="grey")
                    fig.update_layout(
                        xaxis_title="IVW β per SD protein (log-OR for binary outcomes)",
                        height=max(220, 40 * len(fdf)),
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# Tab 5 — Population stats (manuscript-style pleiotropy × PAV × clinical)
# =============================================================================
with tab_pop:
    st.markdown("#### Drugs, clinical stage & safety  ·  OT release 26.03")
    st.caption(
        "All views below come from the local OT release parquets "
        "(`target_prioritisation`, `clinical_target`, `clinical_indication`), "
        "not the GraphQL API. Pleiotropy bucket × PAV ≈ manuscript's "
        "intermediate-pleiotropy + PAV sweet spot."
    )

    try:
        feats = drug_safety.get_features()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    pav_thr = st.slider(
        "PAV proxy: geneticConstraint score threshold",
        min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
        help="OT 26.03 doesn't ship a clean per-target PAV flag. We use "
             "`geneticConstraint > threshold` from `target_prioritisation` "
             "as a proxy: higher values mean stronger purifying-selection "
             "signal, which correlates with PAV-supported genes.",
    )

    summary = drug_safety.bucket_summary(pav_threshold=pav_thr)

    ds_sub_overview, ds_sub_clinical, ds_sub_safety, ds_sub_or, ds_sub_table = st.tabs(
        ["Overview", "Clinical success", "Safety", "Odds ratios", "Full table"]
    )

    # -------------------------------------------------------------------------
    # Overview — population, PAV stack
    # -------------------------------------------------------------------------
    with ds_sub_overview:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Genes scored", f"{len(feats):,}")
        h2.metric("Genes with drug ≥ Ph1",
                  f"{int((feats['max_drug_phase'] >= 1).sum()):,}")
        h3.metric("Genes with approved drug",
                  f"{int(feats['has_approved'].sum()):,}")
        h4.metric("Genes with safety event",
                  f"{int(feats['has_safety_event'].sum()):,}")

        st.markdown("##### Pleiotropy bucket × PAV stack")
        stack = (
            summary.by_pav.pivot_table(index="bucket", columns="PAV",
                                       values="n_genes", aggfunc="sum")
                  .reindex(drug_safety.BUCKET_ORDER).fillna(0).astype(int)
        )
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stack.index.astype(str),
                             y=stack.get("no PAV", 0),
                             name="no PAV",
                             marker=dict(color="#9aa0a6")))
        fig.add_trace(go.Bar(x=stack.index.astype(str),
                             y=stack.get("PAV", 0),
                             name="PAV-supported",
                             marker=dict(color="#1f883d")))
        fig.update_layout(barmode="stack",
                          yaxis_title="Number of protein-coding genes",
                          height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if not summary.tract_rates.empty:
            st.markdown("##### Tractability rate by pleiotropy bucket")
            tr = summary.tract_rates.copy()
            tr.index = tr.index.astype(str)
            fig = px.imshow(
                tr.T, color_continuous_scale="Blues", text_auto=".1%",
                labels=dict(x="Pleiotropy bucket", y="Feature",
                            color="Fraction"),
            )
            fig.update_layout(height=320,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Clinical success rates
    # -------------------------------------------------------------------------
    with ds_sub_clinical:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Approved-drug rate by bucket × PAV**")
            fig = px.bar(
                summary.by_pav, x="bucket", y="rate_approved", color="PAV",
                barmode="group",
                color_discrete_map={"PAV": "#1f883d", "no PAV": "#9aa0a6"},
                category_orders={"bucket": drug_safety.BUCKET_ORDER,
                                 "PAV": ["no PAV", "PAV"]},
            )
            fig.update_yaxes(tickformat=".1%",
                             title="fraction with approved drug")
            fig.update_layout(height=380,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Any-clinical-phase rate by bucket × PAV**")
            fig = px.bar(
                summary.by_pav, x="bucket", y="rate_clinical", color="PAV",
                barmode="group",
                color_discrete_map={"PAV": "#1f883d", "no PAV": "#9aa0a6"},
                category_orders={"bucket": drug_safety.BUCKET_ORDER,
                                 "PAV": ["no PAV", "PAV"]},
            )
            fig.update_yaxes(tickformat=".1%",
                             title="fraction with any clinical-phase drug")
            fig.update_layout(height=380,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Max clinical phase distribution (drugged genes only)**")
        drugged = feats[feats["max_drug_phase"] >= 1].copy()
        drugged["_pav"] = drug_safety._pav_series(drugged, pav_thr)
        drugged["PAV"] = drugged["_pav"].map({True: "PAV", False: "no PAV"})
        drugged["bucket"] = drugged["bucket"].astype(str)
        fig = px.box(
            drugged, x="bucket", y="max_drug_phase", color="PAV",
            points="outliers",
            color_discrete_map={"PAV": "#1f883d", "no PAV": "#9aa0a6"},
            category_orders={"bucket": drug_safety.BUCKET_ORDER,
                             "PAV": ["no PAV", "PAV"]},
            labels={"max_drug_phase": "max clinical phase (4=approved)"},
        )
        fig.update_layout(height=380,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------
    with ds_sub_safety:
        if summary.safety_by_pav.empty:
            st.info("`has_safety_event` not available in this release.")
        else:
            st.markdown("**OT safety-event rate by bucket × PAV**")
            fig = px.bar(
                summary.safety_by_pav, x="bucket", y="rate_safety",
                color="PAV", barmode="group",
                color_discrete_map={"PAV": "#1f883d", "no PAV": "#9aa0a6"},
                category_orders={"bucket": drug_safety.BUCKET_ORDER,
                                 "PAV": ["no PAV", "PAV"]},
            )
            fig.update_yaxes(tickformat=".1%",
                             title="fraction with OT safety-event flag")
            fig.update_layout(height=380,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                summary.safety_by_pav[
                    ["bucket", "PAV", "n_genes", "n_safety", "rate_safety"]
                ],
                use_container_width=True, hide_index=True,
            )

        st.markdown("##### Highly-pleiotropic + safety-flagged + clinically advanced")
        risky = feats[
            (feats["bucket"] == "highly_pleiotropic")
            & (feats["has_safety_event"])
            & (feats["max_drug_phase"] >= 2)
        ].sort_values(["max_drug_phase", "n_drugged_diseases"], ascending=False)
        if risky.empty:
            st.info("No targets match the risky-but-progressing profile.")
        else:
            st.dataframe(
                risky[["approvedSymbol", "bucket", "n_diseases",
                       "max_drug_phase", "n_drugs", "n_drugged_diseases",
                       "is_cancer_driver", "is_in_membrane", "is_secreted"]]
                .head(50),
                use_container_width=True, hide_index=True,
            )

    # -------------------------------------------------------------------------
    # Odds ratios for approval vs reference cell
    # -------------------------------------------------------------------------
    with ds_sub_or:
        st.markdown(
            "##### Within-bucket lift from PAV evidence"
        )
        st.caption(
            "PAV-vs-no-PAV odds ratio **within** each pleiotropy bucket — "
            "each bucket's own no-PAV cell is the local reference. "
            "**OR=1** ⇒ PAV makes no difference inside this bucket · "
            "**OR>1** ⇒ PAV raises the drug-success rate · "
            "**OR<1** ⇒ PAV lowers it. This is the right framing for the "
            "manuscript's 'intermediate × PAV is the sweet spot' claim, "
            "and avoids the confusing 'no-PAV higher than PAV' visual "
            "artifact you get when everything is plotted against the "
            "no-PAV / none-bucket reference cell."
        )
        wb = summary.within_bucket_or
        if wb.empty:
            st.info("Not enough data within buckets to compute the forest.")
        else:
            import plotly.graph_objects as go
            fig = go.Figure()
            for outcome, color in [("any clinical phase", "#1f883d"),
                                   ("approved drug",     "#3082b8")]:
                sub = wb[wb["outcome"] == outcome]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["OR"], y=sub["bucket"], mode="markers",
                    name=outcome,
                    marker=dict(size=13, color=color),
                    error_x=dict(
                        type="data",
                        array=sub["OR_ucl"] - sub["OR"],
                        arrayminus=sub["OR"] - sub["OR_lcl"],
                    ),
                    text=[
                        f"{outcome}: PAV {r.rate_pav:.1%} "
                        f"({r.e_pav}/{r.n_pav}) vs no-PAV {r.rate_nopav:.1%} "
                        f"({r.e_nopav}/{r.n_nopav}) · "
                        f"OR {r.OR:.2f}  ({r.OR_lcl:.2f}–{r.OR_ucl:.2f})"
                        for r in sub.itertuples()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
            fig.add_vline(x=1.0, line_dash="dash", line_color="grey",
                          annotation_text="OR=1 — PAV no different from no-PAV",
                          annotation_position="top right")
            fig.update_layout(
                xaxis_title="Odds ratio (log scale) · OR>1 = PAV helps inside this bucket",
                xaxis_type="log",
                yaxis=dict(categoryorder="array",
                           categoryarray=drug_safety.BUCKET_ORDER,
                           autorange="reversed"),
                height=440, margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                wb[["bucket", "outcome",
                    "n_pav", "e_pav", "rate_pav",
                    "n_nopav", "e_nopav", "rate_nopav",
                    "OR", "OR_lcl", "OR_ucl"]]
                .rename(columns={
                    "n_pav": "n PAV", "e_pav": "events PAV",
                    "rate_pav": "rate PAV",
                    "n_nopav": "n no-PAV", "e_nopav": "events no-PAV",
                    "rate_nopav": "rate no-PAV",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "rate PAV":    st.column_config.NumberColumn(format="%.1f%%"),
                    "rate no-PAV": st.column_config.NumberColumn(format="%.1f%%"),
                    "OR":     st.column_config.NumberColumn(format="%.2f"),
                    "OR_lcl": st.column_config.NumberColumn(format="%.2f"),
                    "OR_ucl": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        # ---- Secondary view: ORs vs the no-PAV / none-bucket reference ----
        with st.expander("Compare to the across-bucket forest (vs. no-PAV / none reference)"):
            st.caption(
                "Older framing — each cell's OR is against `no-PAV / none-bucket`. "
                "Useful to see absolute lift over baseline-uninteresting biology, "
                "but reads counterintuitively (the no-PAV cells of impressive "
                "buckets can sit higher than their PAV counterparts even when "
                "PAV helps within-bucket — because the reference itself is "
                "no-PAV)."
            )
            ortbl = summary.or_table
            if ortbl.empty:
                st.info("Not enough data in reference cell.")
            else:
                ortbl = ortbl.assign(
                    label=lambda d: d["bucket"].astype(str) + "  ·  " + d["PAV"]
                )
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ortbl["OR"], y=ortbl["label"], mode="markers",
                    marker=dict(size=11,
                                color=["#1f883d" if p == "PAV" else "#9aa0a6"
                                       for p in ortbl["PAV"]]),
                    error_x=dict(
                        type="data",
                        array=ortbl["OR_ucl"] - ortbl["OR"],
                        arrayminus=ortbl["OR"] - ortbl["OR_lcl"],
                    ),
                    text=[f"OR {r.OR:.2f}  ({r.OR_lcl:.2f}–{r.OR_ucl:.2f})  ·  "
                          f"{r.n_approved}/{r.n_genes}"
                          for r in ortbl.itertuples()],
                    hovertemplate="%{text}<extra></extra>",
                ))
                fig.add_vline(x=1.0, line_dash="dash", line_color="grey")
                fig.update_layout(
                    xaxis_title="OR vs reference (no-PAV / none-bucket), log scale",
                    xaxis_type="log",
                    yaxis=dict(autorange="reversed"),
                    height=440, margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    ortbl[["bucket", "PAV", "n_genes", "n_approved",
                           "rate", "OR", "OR_lcl", "OR_ucl"]],
                    use_container_width=True, hide_index=True,
                )

    # -------------------------------------------------------------------------
    # Full feature table — filter + download
    # -------------------------------------------------------------------------
    with ds_sub_table:
        st.markdown("##### Per-gene feature table — filter, sort, download")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            bucket_filter = st.multiselect(
                "Pleiotropy bucket",
                options=drug_safety.BUCKET_ORDER,
                default=["intermediate"],
                key="ds_bucket_filter",
            )
        with f2:
            min_phase = st.slider("Min max-drug-phase", 0.0, 4.0,
                                  0.0, 0.5, key="ds_min_phase")
        with f3:
            require_pav = st.checkbox("PAV-supported only",
                                      value=False, key="ds_require_pav")
        with f4:
            require_safety = st.selectbox(
                "Safety event filter", ["any", "only flagged", "exclude flagged"],
                index=0, key="ds_safety_filter",
            )

        fdf = feats.copy()
        fdf["_pav"] = drug_safety._pav_series(fdf, pav_thr)
        if bucket_filter:
            fdf = fdf[fdf["bucket"].astype(str).isin(bucket_filter)]
        fdf = fdf[fdf["max_drug_phase"] >= min_phase]
        if require_pav:
            fdf = fdf[fdf["_pav"]]
        if require_safety == "only flagged":
            fdf = fdf[fdf["has_safety_event"]]
        elif require_safety == "exclude flagged":
            fdf = fdf[~fdf["has_safety_event"]]

        st.write(f"{len(fdf):,} rows match.")
        show_cols = [
            "approvedSymbol", "bucket", "n_loci", "n_diseases",
            "_pav", "max_drug_phase", "n_drugs", "n_drugged_diseases",
            "has_safety_event", "is_cancer_driver",
            "has_small_mol_binder", "is_in_membrane", "is_secreted",
            "genetic_constraint",
        ]
        show_cols = [c for c in show_cols if c in fdf.columns]
        fdf_show = (
            fdf[show_cols]
            .sort_values(["max_drug_phase", "n_drugged_diseases"],
                         ascending=False)
        )
        st.dataframe(fdf_show.head(500),
                     use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered table (CSV)",
            data=fdf_show.to_csv(index=False).encode(),
            file_name="drug_safety_features.csv",
            mime="text/csv",
        )


# =============================================================================
# Tab 6 — Agent — stub for now
# =============================================================================
with tab_agent:
    st.markdown("#### Agent (Claude tool use)")
    st.info(
        "Wraps the pleiotropy / coloc / MR functions as Anthropic tools and shows "
        "the live tool-call trace. Try the demo prompt: "
        "*'Find pQTL-supported, intermediate-pleiotropy targets for CAD with "
        "directional consistency, and run MR for the top one.'*"
    )
    st.warning("Agent layer wiring up next — see `pleio/agent.py`.")
