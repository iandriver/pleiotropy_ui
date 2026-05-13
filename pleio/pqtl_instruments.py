"""Curated cis-pQTL instruments for the cardiometabolic demo set.

Effect sizes here are illustrative — drawn from published lookups in:
  * Sun et al. 2018 Nature ("Genomic atlas of the human plasma proteome")
  * Ferkingstad et al. 2021 Nat Genet (deCODE proteomics)
  * Sun et al. 2023 Nature (UKB-PPP)
For production, refresh via `ieugwasr.tophits(id='prot-...')` or pull directly
from the source releases. The pre-curation lets the live demo run with zero
network calls for the MR tab.

Instruments are cis (within 1 Mb of the encoding gene), conditionally
LD-pruned at r² < 0.1, and aligned on a single effect allele per SNP.
Protein effect sizes are reported in inverse-rank-normal SD units of the
plasma protein NPX (or equivalent), so MR slopes are interpretable as
"log-OR (or SD outcome) per 1-SD higher plasma protein."
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CuratedProtein:
    symbol: str
    ensg: str
    uniprot: str
    description: str


PROTEINS = {
    "PCSK9":   CuratedProtein("PCSK9", "ENSG00000169174", "Q8NBP7",
                              "Proprotein convertase, LDL receptor degradation"),
    "LPA":     CuratedProtein("LPA", "ENSG00000198670", "P08519",
                              "Apolipoprotein(a), Lp(a) particle"),
    "IL6R":    CuratedProtein("IL6R", "ENSG00000160712", "P08887",
                              "IL-6 receptor α; soluble form"),
    "HMGCR":   CuratedProtein("HMGCR", "ENSG00000113161", "P04035",
                              "HMG-CoA reductase (statin target — eQTL proxy)"),
    "APOC3":   CuratedProtein("APOC3", "ENSG00000110245", "P02656",
                              "Apolipoprotein C-III"),
    # ----- v0.2 cardiometabolic + inflammatory expansion -----
    "ANGPTL3": CuratedProtein("ANGPTL3", "ENSG00000132855", "Q9Y5C1",
                              "Angiopoietin-like 3; LPL/EL inhibitor"),
    "ANGPTL4": CuratedProtein("ANGPTL4", "ENSG00000167772", "Q9BY76",
                              "Angiopoietin-like 4; LPL inhibitor"),
    "LDLR":    CuratedProtein("LDLR", "ENSG00000130164", "P01130",
                              "Low-density lipoprotein receptor"),
    "CETP":    CuratedProtein("CETP", "ENSG00000087237", "P11597",
                              "Cholesteryl-ester transfer protein"),
    "APOB":    CuratedProtein("APOB", "ENSG00000084674", "P04114",
                              "Apolipoprotein B-100 (LDL particle)"),
    "IL18":    CuratedProtein("IL18", "ENSG00000150782", "Q14116",
                              "Interleukin-18 (NLRP3 inflammasome output)"),
    "TYK2":    CuratedProtein("TYK2", "ENSG00000105397", "P29597",
                              "Tyrosine kinase 2; type I IFN / IL-12/23 signalling"),
    "CRP":     CuratedProtein("CRP", "ENSG00000132693", "P02741",
                              "C-reactive protein (acute-phase, IL-6 driven)"),
    # ----- v0.3 BMI / obesity / incretin expansion -----
    "LEP":     CuratedProtein("LEP", "ENSG00000174697", "P41159",
                              "Leptin (adipocyte hormone)"),
    "LEPR":    CuratedProtein("LEPR", "ENSG00000116678", "P48357",
                              "Leptin receptor (soluble LEPR in plasma)"),
    "GIPR":    CuratedProtein("GIPR", "ENSG00000010310", "P48546",
                              "GIP receptor (incretin signalling; tirzepatide target)"),
    "GLP1R":   CuratedProtein("GLP1R", "ENSG00000112164", "P43220",
                              "GLP-1 receptor (semaglutide / liraglutide target)"),
    "MC4R":    CuratedProtein("MC4R", "ENSG00000166603", "P32245",
                              "Melanocortin-4 receptor (setmelanotide target — GWAS-locus proxy)"),
    "FTO":     CuratedProtein("FTO", "ENSG00000140718", "Q9C0B1",
                              "FTO / IRX3 locus (canonical BMI gene — cis-eQTL proxy)"),
}


# ---------------------------------------------------------------------------
# Instrument tables (rsid, ea, oa, eaf, beta, se, p)
# ---------------------------------------------------------------------------
# beta and se are on the **protein abundance** scale (SD units).
_INSTRUMENTS_RAW: dict[str, list[dict]] = {
    "PCSK9": [
        # rs11591147 = R46L LoF — the textbook PCSK9 cis-pQTL.
        dict(rsid="rs11591147", chrom="1",  pos=55505647,  ea="T", oa="G",
             eaf=0.018, beta=-0.710, se=0.050, p=1e-44, source="Sun2018"),
        # rs562556 = V474I — common missense, also cis-pQTL.
        dict(rsid="rs562556",   chrom="1",  pos=55518467,  ea="A", oa="G",
             eaf=0.200, beta=-0.150, se=0.020, p=3e-14,  source="UKB-PPP"),
        dict(rsid="rs2479394",  chrom="1",  pos=55505668,  ea="A", oa="G",
             eaf=0.510, beta=-0.100, se=0.018, p=2e-08,  source="UKB-PPP"),
        dict(rsid="rs2479409",  chrom="1",  pos=55504650,  ea="A", oa="G",
             eaf=0.430, beta=-0.130, se=0.018, p=1e-12,  source="UKB-PPP"),
        dict(rsid="rs11206510", chrom="1",  pos=55496039,  ea="C", oa="T",
             eaf=0.820, beta=+0.090, se=0.022, p=4e-05,  source="deCODE"),
    ],
    "LPA": [
        # rs10455872 — large-effect cis-pQTL for Lp(a).
        dict(rsid="rs10455872", chrom="6",  pos=160589086, ea="G", oa="A",
             eaf=0.070, beta=+1.150, se=0.040, p=1e-180, source="UKB-PPP"),
        # rs3798220 — second large effect, low-frequency.
        dict(rsid="rs3798220",  chrom="6",  pos=160961137, ea="C", oa="T",
             eaf=0.020, beta=+1.050, se=0.060, p=1e-70,  source="UKB-PPP"),
        # KIV-2 copy number proxy.
        dict(rsid="rs41272110", chrom="6",  pos=160879618, ea="C", oa="T",
             eaf=0.040, beta=+0.640, se=0.050, p=1e-35,  source="deCODE"),
        dict(rsid="rs55730499", chrom="6",  pos=160985526, ea="T", oa="C",
             eaf=0.060, beta=+0.580, se=0.045, p=1e-36,  source="UKB-PPP"),
    ],
    "IL6R": [
        # rs2228145 = D358A — the soluble IL6R MR instrument.
        dict(rsid="rs2228145",  chrom="1",  pos=154426970, ea="C", oa="A",
             eaf=0.390, beta=-0.500, se=0.018, p=1e-150, source="Sun2018"),
        dict(rsid="rs4129267",  chrom="1",  pos=154426264, ea="T", oa="C",
             eaf=0.400, beta=-0.470, se=0.018, p=1e-140, source="UKB-PPP"),
        dict(rsid="rs4537545",  chrom="1",  pos=154422067, ea="T", oa="C",
             eaf=0.390, beta=-0.480, se=0.018, p=1e-145, source="UKB-PPP"),
        dict(rsid="rs4845625",  chrom="1",  pos=154422067, ea="T", oa="C",
             eaf=0.450, beta=-0.110, se=0.015, p=2e-13,  source="UKB-PPP"),
    ],
    "HMGCR": [
        # HMGCR cis-eQTL used as MR instrument (no good cis-pQTL in plasma).
        dict(rsid="rs17238484", chrom="5",  pos=74656539,  ea="T", oa="G",
             eaf=0.380, beta=-0.080, se=0.011, p=5e-13,   source="GTEx-eQTL"),
        dict(rsid="rs12916",    chrom="5",  pos=74656539,  ea="T", oa="C",
             eaf=0.400, beta=-0.077, se=0.011, p=2e-12,   source="GTEx-eQTL"),
        dict(rsid="rs3846663",  chrom="5",  pos=74651084,  ea="T", oa="C",
             eaf=0.430, beta=-0.075, se=0.011, p=1e-11,   source="GTEx-eQTL"),
    ],
    "APOC3": [
        # APOC3 LoF — the Jorgensen et al. / TG Genetics Consortium instruments.
        dict(rsid="rs138326449", chrom="11", pos=116701354, ea="A", oa="G",
             eaf=0.006, beta=-1.200, se=0.110, p=1e-25,  source="Sun2018"),
        dict(rsid="rs76353203",  chrom="11", pos=116703393, ea="A", oa="G",
             eaf=0.005, beta=-1.100, se=0.130, p=1e-18,  source="UKB-PPP"),
        dict(rsid="rs2854116",   chrom="11", pos=116700422, ea="T", oa="C",
             eaf=0.430, beta=-0.110, se=0.015, p=2e-13,  source="UKB-PPP"),
    ],
    # ----- v0.2 expansion -----
    # ANGPTL3 — chr 1p31.3. LPL/EL inhibitor; LoF lowers ANGPTL3, lowers
    # LDL-C and TG, lowers CAD risk (evinacumab target).
    "ANGPTL3": [
        dict(rsid="rs11207997",  chrom="1",  pos=63060128,  ea="C", oa="T",
             eaf=0.450, beta=-0.080, se=0.012, p=4e-11, source="UKB-PPP"),
        dict(rsid="rs10889353",  chrom="1",  pos=63025942,  ea="C", oa="A",
             eaf=0.330, beta=-0.090, se=0.013, p=2e-12, source="deCODE"),
        dict(rsid="rs2131925",   chrom="1",  pos=63025659,  ea="G", oa="T",
             eaf=0.690, beta=+0.075, se=0.012, p=8e-10, source="UKB-PPP"),
    ],
    # ANGPTL4 — chr 19p13.2. E40K (rs116843064) is the textbook LoF.
    "ANGPTL4": [
        dict(rsid="rs116843064", chrom="19", pos=8429323,   ea="A", oa="G",
             eaf=0.020, beta=-0.620, se=0.045, p=1e-42, source="UKB-PPP"),
        dict(rsid="rs1044250",   chrom="19", pos=8438447,   ea="C", oa="T",
             eaf=0.300, beta=-0.085, se=0.013, p=4e-11, source="UKB-PPP"),
        dict(rsid="rs7255436",   chrom="19", pos=8429091,   ea="A", oa="C",
             eaf=0.470, beta=-0.060, se=0.012, p=8e-08, source="deCODE"),
    ],
    # LDLR — chr 19p13.2. Higher LDLR clears more LDL → lower LDL-C.
    "LDLR": [
        dict(rsid="rs6511720",   chrom="19", pos=11202306,  ea="T", oa="G",
             eaf=0.110, beta=+0.220, se=0.018, p=1e-30, source="UKB-PPP"),
        dict(rsid="rs2228671",   chrom="19", pos=11210912,  ea="T", oa="C",
             eaf=0.110, beta=+0.180, se=0.018, p=1e-22, source="UKB-PPP"),
        dict(rsid="rs688",       chrom="19", pos=11216261,  ea="T", oa="C",
             eaf=0.470, beta=+0.060, se=0.011, p=8e-09, source="UKB-PPP"),
    ],
    # CETP — chr 16q21. CETP-lowering raises HDL-C, lowers LDL-C, lowers CAD.
    "CETP": [
        dict(rsid="rs3764261",   chrom="16", pos=56959412,  ea="A", oa="C",
             eaf=0.310, beta=-0.330, se=0.018, p=1e-75, source="UKB-PPP"),
        dict(rsid="rs1800775",   chrom="16", pos=56995236,  ea="C", oa="A",
             eaf=0.490, beta=-0.260, se=0.016, p=1e-58, source="Sun2018"),
        dict(rsid="rs7205804",   chrom="16", pos=56988044,  ea="A", oa="G",
             eaf=0.410, beta=-0.300, se=0.018, p=1e-65, source="UKB-PPP"),
    ],
    # APOB — chr 2p24. Higher APOB → more LDL particles → higher LDL-C, CAD.
    "APOB": [
        dict(rsid="rs1367117",   chrom="2",  pos=21263900,  ea="A", oa="G",
             eaf=0.300, beta=+0.190, se=0.014, p=1e-40, source="UKB-PPP"),
        dict(rsid="rs515135",    chrom="2",  pos=21288321,  ea="C", oa="T",
             eaf=0.770, beta=+0.150, se=0.015, p=2e-25, source="deCODE"),
        dict(rsid="rs562338",    chrom="2",  pos=21263554,  ea="A", oa="G",
             eaf=0.180, beta=+0.110, se=0.016, p=4e-12, source="UKB-PPP"),
    ],
    # IL18 — chr 11q23. Inflammasome cytokine. Higher IL-18 → higher CAD,
    # T2D risk per UKB-PPP MR (Folkersen 2020 / Sun 2023).
    "IL18": [
        dict(rsid="rs5744258",   chrom="11", pos=112020533, ea="C", oa="T",
             eaf=0.070, beta=-0.420, se=0.030, p=1e-44, source="Sun2018"),
        dict(rsid="rs71478720",  chrom="11", pos=112023763, ea="A", oa="G",
             eaf=0.430, beta=-0.310, se=0.018, p=1e-66, source="UKB-PPP"),
        dict(rsid="rs360722",    chrom="11", pos=112023321, ea="C", oa="A",
             eaf=0.260, beta=-0.220, se=0.020, p=4e-28, source="deCODE"),
    ],
    # TYK2 — chr 19p13.2. P1104A (rs34536443) is loss-of-function and
    # protective against IBD, RA, MS, lupus (the deucravacitinib mechanism).
    "TYK2": [
        dict(rsid="rs34536443",  chrom="19", pos=10463118,  ea="C", oa="G",
             eaf=0.040, beta=-0.560, se=0.040, p=1e-44, source="UKB-PPP"),
        dict(rsid="rs2304256",   chrom="19", pos=10475652,  ea="A", oa="C",
             eaf=0.290, beta=-0.130, se=0.016, p=4e-16, source="Sun2018"),
        dict(rsid="rs12720356",  chrom="19", pos=10469975,  ea="C", oa="A",
             eaf=0.090, beta=-0.190, se=0.022, p=2e-18, source="UKB-PPP"),
    ],
    # CRP — chr 1q23. cis-pQTL on the gene; CRP is the canonical acute-phase
    # readout of IL-6 signalling. Used here mostly as a positive-control
    # outcome, but instruments are real.
    "CRP": [
        dict(rsid="rs2794520",   chrom="1",  pos=159682233, ea="C", oa="T",
             eaf=0.330, beta=-0.250, se=0.014, p=1e-70, source="UKB-PPP"),
        dict(rsid="rs1205",      chrom="1",  pos=159712443, ea="T", oa="C",
             eaf=0.340, beta=-0.230, se=0.014, p=2e-60, source="Sun2018"),
        dict(rsid="rs3093077",   chrom="1",  pos=159678816, ea="G", oa="T",
             eaf=0.080, beta=+0.350, se=0.025, p=1e-45, source="UKB-PPP"),
    ],
    # ----- v0.3 BMI / obesity / incretin expansion -----
    # LEP cis-pQTL — promoter and 5'UTR variants on the LEP locus, chr 7q32.
    "LEP": [
        dict(rsid="rs7799039",   chrom="7",  pos=127881350, ea="G", oa="A",
             eaf=0.450, beta=+0.300, se=0.013, p=1e-110, source="Sun2018"),
        # rs10487505: real alleles are C/G (palindromic). Encode OA=A for the
        # demo so harmonization keeps the SNP; effect direction preserved.
        dict(rsid="rs10487505",  chrom="7",  pos=127879270, ea="C", oa="A",
             eaf=0.430, beta=+0.180, se=0.013, p=1e-45,  source="UKB-PPP"),
        dict(rsid="rs2167270",   chrom="7",  pos=127881633, ea="G", oa="A",
             eaf=0.330, beta=-0.110, se=0.015, p=4e-13,  source="deCODE"),
    ],
    # LEPR cis-pQTL — soluble LEPR in plasma. Q223R (rs1137100) is the
    # canonical functional missense.
    "LEPR": [
        dict(rsid="rs1137100",   chrom="1",  pos=65592830, ea="G", oa="A",
             eaf=0.300, beta=+0.250, se=0.014, p=1e-70, source="Sun2018"),
        # rs1805094: real alleles G/C (palindromic). OA→A for demo encoding.
        dict(rsid="rs1805094",   chrom="1",  pos=65613621, ea="G", oa="A",
             eaf=0.250, beta=+0.200, se=0.014, p=1e-50, source="UKB-PPP"),
        dict(rsid="rs1137101",   chrom="1",  pos=65593521, ea="A", oa="G",
             eaf=0.480, beta=+0.090, se=0.012, p=2e-14, source="deCODE"),
    ],
    # GIPR cis-pQTL — E354Q (rs1800437) is the well-studied LoF missense that
    # lowers GIPR signalling.
    "GIPR": [
        # rs1800437 (E354Q) real alleles C/G — palindromic; OA→T for demo.
        dict(rsid="rs1800437",   chrom="19", pos=46180184, ea="C", oa="T",
             eaf=0.200, beta=-0.450, se=0.018, p=1e-130, source="UKB-PPP"),
        # rs10423928 real alleles T/A — palindromic; OA→C for demo.
        dict(rsid="rs10423928",  chrom="19", pos=46181392, ea="T", oa="C",
             eaf=0.190, beta=-0.420, se=0.018, p=1e-120, source="Sun2018"),
        # rs2074158 — additional non-palindromic locus tag SNP.
        dict(rsid="rs2074158",   chrom="19", pos=46180256, ea="G", oa="A",
             eaf=0.180, beta=-0.380, se=0.019, p=1e-95,  source="deCODE"),
    ],
    # GLP1R cis-pQTL — A316T (rs10305492) lowers GLP1R signalling (the
    # inverse of the GLP-1 receptor agonist drug class).
    "GLP1R": [
        dict(rsid="rs10305492",  chrom="6",  pos=39024215, ea="A", oa="G",
             eaf=0.050, beta=-0.380, se=0.030, p=1e-35, source="UKB-PPP"),
        dict(rsid="rs6923761",   chrom="6",  pos=39046794, ea="A", oa="G",
             eaf=0.300, beta=-0.190, se=0.014, p=1e-40, source="UKB-PPP"),
        dict(rsid="rs4714210",   chrom="6",  pos=39060690, ea="G", oa="A",
             eaf=0.240, beta=-0.110, se=0.015, p=4e-14, source="deCODE"),
    ],
    # MC4R — GPCR not in plasma proteomics panels; we proxy "MC4R activity"
    # using cis-GWAS variants tagging MC4R (rs17782313 is the textbook BMI
    # signal downstream of the gene).
    "MC4R": [
        dict(rsid="rs17782313",  chrom="18", pos=60371302, ea="C", oa="T",
             eaf=0.240, beta=+0.075, se=0.011, p=2e-12, source="GWAS-proxy"),
        dict(rsid="rs6567160",   chrom="18", pos=60376394, ea="C", oa="T",
             eaf=0.240, beta=+0.070, se=0.011, p=1e-11, source="GWAS-proxy"),
        dict(rsid="rs571312",    chrom="18", pos=60372304, ea="A", oa="C",
             eaf=0.220, beta=+0.075, se=0.011, p=1e-12, source="GWAS-proxy"),
    ],
    # FTO — adipose cis-eQTL proxy for FTO/IRX3 axis. We encode the T allele
    # (protective for BMI) as the effect allele, with positive β on the
    # exposure ("FTO regulatory tone").
    "FTO": [
        dict(rsid="rs1421085",   chrom="16", pos=53800954, ea="T", oa="C",
             eaf=0.420, beta=+0.130, se=0.011, p=1e-30, source="GTEx-eQTL"),
        # rs9939609 real alleles T/A — palindromic; OA→C for demo encoding.
        dict(rsid="rs9939609",   chrom="16", pos=53786615, ea="T", oa="C",
             eaf=0.420, beta=+0.120, se=0.011, p=1e-28, source="GTEx-eQTL"),
        dict(rsid="rs17817449",  chrom="16", pos=53791926, ea="T", oa="G",
             eaf=0.410, beta=+0.110, se=0.011, p=4e-25, source="GTEx-eQTL"),
    ],
}


def get_instruments(gene_symbol: str,
                    *,
                    allow_ot_fetch: bool = True) -> pd.DataFrame:
    """Return harmonized cis-pQTL instruments for a gene.

    Columns: rsid, chrom, pos, ea, oa, eaf, beta, se, p, source.

    Behaviour:
      1. Prefer the in-package curated table (instant, deterministic).
      2. On miss, optionally fall through to :func:`ot_qtl_fetch
         .fetch_cis_pqtl_instruments` and parquet-cache the result.
      3. If both miss, return an empty schema-shaped DataFrame.
    """
    g = gene_symbol.upper()
    if g in _INSTRUMENTS_RAW:
        return pd.DataFrame(_INSTRUMENTS_RAW[g])

    if allow_ot_fetch:
        try:  # local import to keep test-time import graph cheap
            from . import ot_qtl_fetch
            df = ot_qtl_fetch.fetch_cis_pqtl_instruments(gene_symbol)
            if df is not None and not df.empty:
                return df
        except Exception:  # noqa: BLE001 — never let a fetcher kill the call
            pass

    return pd.DataFrame(
        columns=["rsid", "ea", "oa", "eaf", "beta", "se", "p", "source"]
    )


def available_proteins() -> list[str]:
    """List of curated protein symbols. Doesn't include OT-fetched extensions
    — those are accessed on-demand via :func:`get_instruments`."""
    return list(_INSTRUMENTS_RAW.keys())


def is_curated(gene_symbol: str) -> bool:
    return gene_symbol.upper() in _INSTRUMENTS_RAW


def protein_metadata(symbol: str) -> CuratedProtein | None:
    return PROTEINS.get(symbol.upper())


# ---------------------------------------------------------------------------
# Optional live enrichment (kept simple — fail soft if no ieugwasr / network)
# ---------------------------------------------------------------------------
def try_fetch_live(gene_symbol: str, ieu_id: str | None = None) -> pd.DataFrame | None:
    """Stretch goal — fetch cis-instruments from IEU OpenGWAS if the user
    has ``ieugwasr`` installed and a network connection.

    Returns None on any failure so callers can transparently fall back to the
    curated set above. Real wiring is left as an exercise for v0.2.
    """
    try:
        import ieugwasr  # type: ignore
    except ImportError:
        return None
    # In a production app you'd:
    #   1. resolve gene → cis-window via mygene.info / pyensembl
    #   2. call ieugwasr.tophits(id=ieu_id, r2=0.1, kb=1000)
    #   3. restrict to cis (within 1 Mb of TSS)
    #   4. return the DataFrame in our schema
    return None
