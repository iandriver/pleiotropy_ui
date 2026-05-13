"""Curated outcome (disease) GWAS effects for the cardiometabolic demo set.

For each instrument SNP in ``pqtl_instruments.py`` we look up the published
effect on a handful of outcomes:

  * CAD     — CARDIoGRAMplusC4D 1000G (Nikpay 2015) + UKB extension
  * T2D     — DIAGRAM 2018 (Mahajan)
  * LDL-C   — GLGC 2021 (Graham), in SD units
  * HDL-C   — GLGC 2021, in SD units
  * TG      — GLGC 2021, in SD units (log-transformed)

Effect sizes are illustrative and rounded — the production version would
pull harmonized summary stats from IEU OpenGWAS / GWAS Catalog directly.
"""
from __future__ import annotations

import pandas as pd

OUTCOMES = {
    "CAD":   dict(label="Coronary artery disease",      binary=True,  source="Nikpay2015+UKB"),
    "T2D":   dict(label="Type 2 diabetes",              binary=True,  source="DIAGRAM2018"),
    "LDL-C": dict(label="LDL cholesterol",              binary=False, source="GLGC2021"),
    "HDL-C": dict(label="HDL cholesterol",              binary=False, source="GLGC2021"),
    "TG":    dict(label="Triglycerides (log)",          binary=False, source="GLGC2021"),
    # ----- v0.2 inflammatory expansion -----
    "CRP":   dict(label="C-reactive protein (log)",     binary=False, source="UKB+CHARGE"),
    "RA":    dict(label="Rheumatoid arthritis",         binary=True,  source="Okada2014"),
    "IBD":   dict(label="Inflammatory bowel disease",   binary=True,  source="deLange2017"),
    "MS":    dict(label="Multiple sclerosis",           binary=True,  source="IMSGC2019"),
    # ----- v0.3 BMI / obesity / adiposity expansion -----
    "BMI":        dict(label="Body mass index (SD)",          binary=False, source="GIANT-Yengo2018"),
    "WHRadjBMI":  dict(label="Waist-hip ratio (BMI-adj, SD)", binary=False, source="Pulit2019"),
}

# Effects are stored as (effect_allele, oa, eaf, beta, se).
# CAD/T2D betas are ln(OR); LDL/HDL/TG are SD-unit beta.
_OUTCOME_EFFECTS: dict[str, dict[str, dict]] = {
    # ---------- CAD ----------
    "CAD": {
        "rs11591147": dict(ea="T", oa="G", eaf=0.018, beta=-0.300, se=0.040),
        "rs562556":   dict(ea="A", oa="G", eaf=0.200, beta=-0.050, se=0.012),
        "rs2479394":  dict(ea="A", oa="G", eaf=0.510, beta=-0.040, se=0.010),
        "rs2479409":  dict(ea="A", oa="G", eaf=0.430, beta=-0.050, se=0.010),
        "rs11206510": dict(ea="C", oa="T", eaf=0.820, beta=+0.080, se=0.013),
        # LPA
        "rs10455872": dict(ea="G", oa="A", eaf=0.070, beta=+0.420, se=0.024),
        "rs3798220":  dict(ea="C", oa="T", eaf=0.020, beta=+0.380, se=0.035),
        "rs41272110": dict(ea="C", oa="T", eaf=0.040, beta=+0.220, se=0.028),
        "rs55730499": dict(ea="T", oa="C", eaf=0.060, beta=+0.250, se=0.026),
        # IL6R
        "rs2228145":  dict(ea="C", oa="A", eaf=0.390, beta=-0.039, se=0.009),
        "rs4129267":  dict(ea="T", oa="C", eaf=0.400, beta=-0.038, se=0.009),
        "rs4537545":  dict(ea="T", oa="C", eaf=0.390, beta=-0.036, se=0.009),
        "rs4845625":  dict(ea="T", oa="C", eaf=0.450, beta=-0.009, se=0.008),
        # HMGCR
        "rs17238484": dict(ea="T", oa="G", eaf=0.380, beta=-0.031, se=0.008),
        "rs12916":    dict(ea="T", oa="C", eaf=0.400, beta=-0.030, se=0.008),
        "rs3846663":  dict(ea="T", oa="C", eaf=0.430, beta=-0.028, se=0.008),
        # APOC3
        "rs138326449": dict(ea="A", oa="G", eaf=0.006, beta=-0.380, se=0.090),
        "rs76353203":  dict(ea="A", oa="G", eaf=0.005, beta=-0.330, se=0.110),
        "rs2854116":   dict(ea="T", oa="C", eaf=0.430, beta=-0.025, se=0.010),
        # ----- v0.2 expansion: ANGPTL3 (lower → lower CAD) -----
        "rs11207997":  dict(ea="C", oa="T", eaf=0.450, beta=-0.045, se=0.011),
        "rs10889353":  dict(ea="C", oa="A", eaf=0.330, beta=-0.050, se=0.011),
        "rs2131925":   dict(ea="G", oa="T", eaf=0.690, beta=+0.040, se=0.011),
        # ANGPTL4 (E40K LoF lowers TG, lowers CAD) -----
        "rs116843064": dict(ea="A", oa="G", eaf=0.020, beta=-0.140, se=0.030),
        "rs1044250":   dict(ea="C", oa="T", eaf=0.300, beta=-0.025, se=0.012),
        # LDLR (LDLR-raising → LDL-C-lowering → CAD-lowering) -----
        "rs6511720":   dict(ea="T", oa="G", eaf=0.110, beta=-0.075, se=0.015),
        "rs2228671":   dict(ea="T", oa="C", eaf=0.110, beta=-0.065, se=0.015),
        "rs688":       dict(ea="T", oa="C", eaf=0.470, beta=-0.020, se=0.012),
        # CETP (CETP-lowering → CAD-lowering) -----
        "rs3764261":   dict(ea="A", oa="C", eaf=0.310, beta=-0.030, se=0.010),
        "rs1800775":   dict(ea="C", oa="A", eaf=0.490, beta=-0.025, se=0.010),
        "rs7205804":   dict(ea="A", oa="G", eaf=0.410, beta=-0.028, se=0.010),
        # APOB (APOB-raising → CAD-raising) -----
        "rs1367117":   dict(ea="A", oa="G", eaf=0.300, beta=+0.080, se=0.012),
        "rs515135":    dict(ea="C", oa="T", eaf=0.770, beta=+0.060, se=0.013),
        "rs562338":    dict(ea="A", oa="G", eaf=0.180, beta=+0.045, se=0.014),
        # IL18 (lower → lower CAD modestly) -----
        "rs5744258":   dict(ea="C", oa="T", eaf=0.070, beta=-0.025, se=0.020),
        "rs71478720":  dict(ea="A", oa="G", eaf=0.430, beta=-0.020, se=0.011),
        "rs360722":    dict(ea="C", oa="A", eaf=0.260, beta=-0.018, se=0.013),
        # CRP (CRP itself is largely non-causal for CAD per MR — small/zero) -----
        "rs2794520":   dict(ea="C", oa="T", eaf=0.330, beta=-0.005, se=0.011),
        "rs1205":      dict(ea="T", oa="C", eaf=0.340, beta=-0.005, se=0.011),
        "rs3093077":   dict(ea="G", oa="T", eaf=0.080, beta=+0.008, se=0.020),
    },
    # ---------- T2D ----------
    "T2D": {
        "rs11591147":  dict(ea="T", oa="G", eaf=0.018, beta=+0.190, se=0.045),
        "rs562556":    dict(ea="A", oa="G", eaf=0.200, beta=+0.020, se=0.011),
        "rs2479394":   dict(ea="A", oa="G", eaf=0.510, beta=+0.015, se=0.009),
        "rs2479409":   dict(ea="A", oa="G", eaf=0.430, beta=+0.022, se=0.009),
        "rs10455872":  dict(ea="G", oa="A", eaf=0.070, beta=-0.030, se=0.017),
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=+0.012, se=0.008),
        "rs4129267":   dict(ea="T", oa="C", eaf=0.400, beta=+0.011, se=0.008),
        "rs4537545":   dict(ea="T", oa="C", eaf=0.390, beta=+0.013, se=0.008),
        "rs17238484":  dict(ea="T", oa="G", eaf=0.380, beta=+0.010, se=0.008),
        "rs12916":     dict(ea="T", oa="C", eaf=0.400, beta=+0.011, se=0.008),
        "rs3846663":   dict(ea="T", oa="C", eaf=0.430, beta=+0.012, se=0.008),
        "rs138326449": dict(ea="A", oa="G", eaf=0.006, beta=-0.050, se=0.080),
        "rs2854116":   dict(ea="T", oa="C", eaf=0.430, beta=-0.010, se=0.009),
        # ----- v0.2 expansion -----
        # ANGPTL4 E40K — modest T2D risk per UKB (Dewey 2016).
        "rs116843064": dict(ea="A", oa="G", eaf=0.020, beta=+0.060, se=0.025),
        # IL18 — higher IL-18 raises T2D risk modestly.
        "rs5744258":   dict(ea="C", oa="T", eaf=0.070, beta=+0.030, se=0.022),
        "rs71478720":  dict(ea="A", oa="G", eaf=0.430, beta=+0.018, se=0.011),
    },
    # ---------- LDL-C ----------
    "LDL-C": {
        "rs11591147":  dict(ea="T", oa="G", eaf=0.018, beta=-0.490, se=0.020),
        "rs562556":    dict(ea="A", oa="G", eaf=0.200, beta=-0.100, se=0.008),
        "rs2479394":   dict(ea="A", oa="G", eaf=0.510, beta=-0.080, se=0.007),
        "rs2479409":   dict(ea="A", oa="G", eaf=0.430, beta=-0.100, se=0.007),
        "rs11206510":  dict(ea="C", oa="T", eaf=0.820, beta=+0.060, se=0.009),
        "rs10455872":  dict(ea="G", oa="A", eaf=0.070, beta=+0.150, se=0.013),
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=+0.005, se=0.005),
        "rs17238484":  dict(ea="T", oa="G", eaf=0.380, beta=-0.130, se=0.006),
        "rs12916":     dict(ea="T", oa="C", eaf=0.400, beta=-0.125, se=0.006),
        "rs3846663":   dict(ea="T", oa="C", eaf=0.430, beta=-0.120, se=0.006),
        "rs138326449": dict(ea="A", oa="G", eaf=0.006, beta=-0.110, se=0.035),
        "rs2854116":   dict(ea="T", oa="C", eaf=0.430, beta=-0.012, se=0.005),
        # ----- v0.2 expansion -----
        # ANGPTL3-lowering → lower LDL-C.
        "rs11207997":  dict(ea="C", oa="T", eaf=0.450, beta=-0.090, se=0.005),
        "rs10889353":  dict(ea="C", oa="A", eaf=0.330, beta=-0.085, se=0.005),
        "rs2131925":   dict(ea="G", oa="T", eaf=0.690, beta=+0.080, se=0.005),
        # ANGPTL4 — minimal LDL effect (mostly TG axis).
        "rs116843064": dict(ea="A", oa="G", eaf=0.020, beta=-0.030, se=0.018),
        # LDLR — strongest cis effect on LDL-C.
        "rs6511720":   dict(ea="T", oa="G", eaf=0.110, beta=-0.190, se=0.005),
        "rs2228671":   dict(ea="T", oa="C", eaf=0.110, beta=-0.170, se=0.005),
        "rs688":       dict(ea="T", oa="C", eaf=0.470, beta=-0.050, se=0.004),
        # CETP — small LDL-lowering as side effect.
        "rs3764261":   dict(ea="A", oa="C", eaf=0.310, beta=-0.030, se=0.005),
        "rs1800775":   dict(ea="C", oa="A", eaf=0.490, beta=-0.020, se=0.005),
        "rs7205804":   dict(ea="A", oa="G", eaf=0.410, beta=-0.025, se=0.005),
        # APOB — strong LDL-raising.
        "rs1367117":   dict(ea="A", oa="G", eaf=0.300, beta=+0.220, se=0.005),
        "rs515135":    dict(ea="C", oa="T", eaf=0.770, beta=+0.180, se=0.006),
        "rs562338":    dict(ea="A", oa="G", eaf=0.180, beta=+0.140, se=0.007),
    },
    # ---------- HDL-C ----------
    "HDL-C": {
        "rs11591147":  dict(ea="T", oa="G", eaf=0.018, beta=+0.030, se=0.020),
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=-0.012, se=0.005),
        "rs10455872":  dict(ea="G", oa="A", eaf=0.070, beta=-0.020, se=0.012),
        "rs138326449": dict(ea="A", oa="G", eaf=0.006, beta=+0.380, se=0.035),
        "rs76353203":  dict(ea="A", oa="G", eaf=0.005, beta=+0.330, se=0.040),
        "rs2854116":   dict(ea="T", oa="C", eaf=0.430, beta=+0.020, se=0.005),
        # ----- v0.2 expansion -----
        # CETP-lowering → strongly raises HDL-C (canonical biology).
        "rs3764261":   dict(ea="A", oa="C", eaf=0.310, beta=+0.230, se=0.005),
        "rs1800775":   dict(ea="C", oa="A", eaf=0.490, beta=+0.180, se=0.005),
        "rs7205804":   dict(ea="A", oa="G", eaf=0.410, beta=+0.200, se=0.005),
        # ANGPTL3-lowering — small HDL effect.
        "rs11207997":  dict(ea="C", oa="T", eaf=0.450, beta=-0.020, se=0.005),
        # LDLR — minimal HDL effect.
        "rs6511720":   dict(ea="T", oa="G", eaf=0.110, beta=-0.005, se=0.006),
    },
    # ---------- TG ----------
    "TG": {
        "rs11591147":  dict(ea="T", oa="G", eaf=0.018, beta=-0.040, se=0.020),
        "rs10455872":  dict(ea="G", oa="A", eaf=0.070, beta=+0.030, se=0.012),
        "rs17238484":  dict(ea="T", oa="G", eaf=0.380, beta=-0.022, se=0.006),
        "rs138326449": dict(ea="A", oa="G", eaf=0.006, beta=-1.450, se=0.110),
        "rs76353203":  dict(ea="A", oa="G", eaf=0.005, beta=-1.380, se=0.130),
        "rs2854116":   dict(ea="T", oa="C", eaf=0.430, beta=-0.105, se=0.006),
        # ----- v0.2 expansion -----
        # ANGPTL3-lowering → lower TG.
        "rs11207997":  dict(ea="C", oa="T", eaf=0.450, beta=-0.075, se=0.005),
        "rs10889353":  dict(ea="C", oa="A", eaf=0.330, beta=-0.080, se=0.005),
        "rs2131925":   dict(ea="G", oa="T", eaf=0.690, beta=+0.070, se=0.005),
        # ANGPTL4 E40K — large TG-lowering effect (canonical).
        "rs116843064": dict(ea="A", oa="G", eaf=0.020, beta=-0.350, se=0.015),
        "rs1044250":   dict(ea="C", oa="T", eaf=0.300, beta=-0.060, se=0.005),
        "rs7255436":   dict(ea="A", oa="C", eaf=0.470, beta=-0.045, se=0.005),
        # CETP-lowering — modest TG-lowering.
        "rs3764261":   dict(ea="A", oa="C", eaf=0.310, beta=-0.045, se=0.005),
        "rs1800775":   dict(ea="C", oa="A", eaf=0.490, beta=-0.035, se=0.005),
        "rs7205804":   dict(ea="A", oa="G", eaf=0.410, beta=-0.040, se=0.005),
        # APOB — TG-raising (modest).
        "rs1367117":   dict(ea="A", oa="G", eaf=0.300, beta=+0.040, se=0.005),
    },
    # ---------- CRP (continuous, log-transformed) ----------
    # CRP-lowering instruments mostly come from the CRP gene itself.
    "CRP": {
        "rs2794520":   dict(ea="C", oa="T", eaf=0.330, beta=-0.300, se=0.008),
        "rs1205":      dict(ea="T", oa="C", eaf=0.340, beta=-0.280, se=0.008),
        "rs3093077":   dict(ea="G", oa="T", eaf=0.080, beta=+0.450, se=0.018),
        # IL-6R cis-pQTLs — soluble IL-6R-lowering modestly lowers CRP
        # downstream (because IL-6 → CRP).
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=-0.045, se=0.006),
        "rs4129267":   dict(ea="T", oa="C", eaf=0.400, beta=-0.040, se=0.006),
        # IL-18 — mild CRP elevation downstream of inflammasome activity.
        "rs5744258":   dict(ea="C", oa="T", eaf=0.070, beta=+0.020, se=0.020),
        "rs71478720":  dict(ea="A", oa="G", eaf=0.430, beta=+0.012, se=0.007),
    },
    # ---------- Rheumatoid arthritis (binary, ln(OR)) ----------
    "RA": {
        # IL-6R-lowering protects against RA (tocilizumab/sarilumab mechanism).
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=-0.080, se=0.018),
        "rs4129267":   dict(ea="T", oa="C", eaf=0.400, beta=-0.075, se=0.018),
        "rs4537545":   dict(ea="T", oa="C", eaf=0.390, beta=-0.078, se=0.018),
        # TYK2 LoF strongly protects against RA.
        "rs34536443":  dict(ea="C", oa="G", eaf=0.040, beta=-0.310, se=0.040),
        "rs2304256":   dict(ea="A", oa="C", eaf=0.290, beta=-0.045, se=0.020),
        "rs12720356":  dict(ea="C", oa="A", eaf=0.090, beta=-0.080, se=0.030),
    },
    # ---------- Inflammatory bowel disease (binary) ----------
    "IBD": {
        # TYK2 LoF protects against IBD (deucravacitinib mechanism).
        "rs34536443":  dict(ea="C", oa="G", eaf=0.040, beta=-0.220, se=0.035),
        "rs2304256":   dict(ea="A", oa="C", eaf=0.290, beta=-0.030, se=0.018),
        "rs12720356":  dict(ea="C", oa="A", eaf=0.090, beta=-0.060, se=0.025),
        # IL-6R has weak / no effect in IBD per MR — null instruments.
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=-0.005, se=0.018),
    },
    # ---------- Multiple sclerosis (binary) ----------
    "MS": {
        # TYK2 LoF protective against MS.
        "rs34536443":  dict(ea="C", oa="G", eaf=0.040, beta=-0.180, se=0.040),
        "rs2304256":   dict(ea="A", oa="C", eaf=0.290, beta=-0.020, se=0.020),
        "rs12720356":  dict(ea="C", oa="A", eaf=0.090, beta=-0.040, se=0.028),
    },
    # ---------- BMI (continuous, SD-units, GIANT 2018) ----------
    "BMI": {
        # LEP cis-pQTL — high circulating leptin slightly tracks higher BMI
        # (leptin resistance signature; not the leptin-deficiency case).
        "rs7799039":   dict(ea="G", oa="A", eaf=0.450, beta=+0.005, se=0.003),
        "rs10487505":  dict(ea="C", oa="A", eaf=0.430, beta=+0.004, se=0.003),
        "rs2167270":   dict(ea="G", oa="A", eaf=0.330, beta=-0.002, se=0.003),
        # LEPR Q223R — reduced LEPR signalling → mild BMI increase.
        "rs1137100":   dict(ea="G", oa="A", eaf=0.300, beta=+0.012, se=0.003),
        "rs1805094":   dict(ea="G", oa="A", eaf=0.250, beta=+0.010, se=0.003),
        "rs1137101":   dict(ea="A", oa="G", eaf=0.480, beta=+0.004, se=0.003),
        # GIPR E354Q — loss-of-function lowers BMI per UKB MR.
        "rs1800437":   dict(ea="C", oa="T", eaf=0.200, beta=-0.020, se=0.004),
        "rs10423928":  dict(ea="T", oa="C", eaf=0.190, beta=-0.018, se=0.004),
        # GLP1R A316T — reduced GLP1R signalling → raised BMI
        # (the inverse of the GLP-1 RA drug class effect).
        "rs10305492":  dict(ea="A", oa="G", eaf=0.050, beta=+0.045, se=0.012),
        "rs6923761":   dict(ea="A", oa="G", eaf=0.300, beta=+0.022, se=0.005),
        "rs4714210":   dict(ea="G", oa="A", eaf=0.240, beta=+0.010, se=0.005),
        # MC4R — canonical BMI signal.
        "rs17782313":  dict(ea="C", oa="T", eaf=0.240, beta=+0.060, se=0.004),
        "rs6567160":   dict(ea="C", oa="T", eaf=0.240, beta=+0.058, se=0.004),
        "rs571312":    dict(ea="A", oa="C", eaf=0.220, beta=+0.060, se=0.004),
        # FTO — encoded with T as effect allele (protective for BMI per
        # Yengo 2018). Positive β on exposure × negative β on BMI = MR
        # interpretation "higher FTO regulatory tone → lower BMI".
        "rs1421085":   dict(ea="T", oa="C", eaf=0.420, beta=-0.090, se=0.003),
        "rs9939609":   dict(ea="T", oa="C", eaf=0.420, beta=-0.085, se=0.003),
        "rs17817449":  dict(ea="T", oa="G", eaf=0.410, beta=-0.080, se=0.003),
        # PCSK9 / LPA / IL6R / lipid proteins have negligible BMI effects.
        "rs11591147":  dict(ea="T", oa="G", eaf=0.018, beta=-0.002, se=0.010),
        "rs2228145":   dict(ea="C", oa="A", eaf=0.390, beta=+0.005, se=0.003),
        "rs10455872":  dict(ea="G", oa="A", eaf=0.070, beta=-0.001, se=0.007),
    },
    # ---------- WHR-adjusted-for-BMI (continuous, SD, Pulit 2019) ----------
    "WHRadjBMI": {
        # LEPR — modest fat-distribution signal.
        "rs1137100":   dict(ea="G", oa="A", eaf=0.300, beta=+0.008, se=0.003),
        "rs1805094":   dict(ea="G", oa="A", eaf=0.250, beta=+0.006, se=0.003),
        "rs1137101":   dict(ea="A", oa="G", eaf=0.480, beta=+0.003, se=0.003),
        # GIPR — LoF associated with lower WHR (favourable fat distribution).
        "rs1800437":   dict(ea="C", oa="T", eaf=0.200, beta=-0.015, se=0.004),
        "rs10423928":  dict(ea="T", oa="C", eaf=0.190, beta=-0.014, se=0.004),
        # rs2074158 also covered below in the late update block.
        # MC4R — central adiposity signal.
        "rs17782313":  dict(ea="C", oa="T", eaf=0.240, beta=+0.025, se=0.004),
        "rs6567160":   dict(ea="C", oa="T", eaf=0.240, beta=+0.022, se=0.004),
        "rs571312":    dict(ea="A", oa="C", eaf=0.220, beta=+0.024, se=0.004),
        # FTO — weaker than BMI (most signal is overall adiposity, not
        # distribution), but still nonzero.
        "rs1421085":   dict(ea="T", oa="C", eaf=0.420, beta=-0.020, se=0.003),
        "rs9939609":   dict(ea="T", oa="C", eaf=0.420, beta=-0.018, se=0.003),
        "rs17817449":  dict(ea="T", oa="G", eaf=0.410, beta=-0.017, se=0.003),
        # ANGPTL3 / ANGPTL4 — TG-axis, modest WHR signal.
        "rs11207997":  dict(ea="C", oa="T", eaf=0.450, beta=-0.012, se=0.003),
        "rs10889353":  dict(ea="C", oa="A", eaf=0.330, beta=-0.013, se=0.003),
        "rs2131925":   dict(ea="G", oa="T", eaf=0.690, beta=+0.011, se=0.003),
        "rs116843064": dict(ea="A", oa="G", eaf=0.020, beta=-0.020, se=0.012),
        "rs1044250":   dict(ea="C", oa="T", eaf=0.300, beta=-0.008, se=0.004),
        "rs7255436":   dict(ea="A", oa="C", eaf=0.470, beta=-0.006, se=0.004),
    },
}

# ---------------------------------------------------------------------------
# Extend existing outcomes with v0.3 BMI-protein effects where biologically
# meaningful. Defined here (not inline) to keep the canonical block above
# readable.
# ---------------------------------------------------------------------------
_OUTCOME_EFFECTS["T2D"].update({
    # GIPR E354Q — protective against T2D (lower BMI + incretin-axis).
    "rs1800437":   dict(ea="C", oa="T", eaf=0.200, beta=-0.060, se=0.013),
    "rs10423928":  dict(ea="T", oa="C", eaf=0.190, beta=-0.055, se=0.013),
    "rs2074158":   dict(ea="G", oa="A", eaf=0.180, beta=-0.050, se=0.014),
    # GLP1R A316T — raises T2D (inverse of semaglutide).
    "rs10305492":  dict(ea="A", oa="G", eaf=0.050, beta=+0.070, se=0.020),
    "rs6923761":   dict(ea="A", oa="G", eaf=0.300, beta=+0.030, se=0.010),
    "rs4714210":   dict(ea="G", oa="A", eaf=0.240, beta=+0.018, se=0.010),
    # MC4R — modest T2D risk via BMI mediator.
    "rs17782313":  dict(ea="C", oa="T", eaf=0.240, beta=+0.040, se=0.010),
    "rs6567160":   dict(ea="C", oa="T", eaf=0.240, beta=+0.038, se=0.010),
    "rs571312":    dict(ea="A", oa="C", eaf=0.220, beta=+0.040, se=0.010),
    # FTO — modest T2D risk via BMI mediator.
    "rs1421085":   dict(ea="T", oa="C", eaf=0.420, beta=-0.045, se=0.008),
    "rs9939609":   dict(ea="T", oa="C", eaf=0.420, beta=-0.042, se=0.008),
    "rs17817449":  dict(ea="T", oa="G", eaf=0.410, beta=-0.040, se=0.008),
})
_OUTCOME_EFFECTS["CAD"].update({
    # GIPR — small protective effect via T2D / BMI mediation.
    "rs1800437":   dict(ea="C", oa="T", eaf=0.200, beta=-0.018, se=0.010),
    # MC4R / FTO — small CAD effects via BMI mediation.
    "rs17782313":  dict(ea="C", oa="T", eaf=0.240, beta=+0.020, se=0.010),
    "rs1421085":   dict(ea="T", oa="C", eaf=0.420, beta=-0.025, se=0.008),
    "rs9939609":   dict(ea="T", oa="C", eaf=0.420, beta=-0.022, se=0.008),
})
_OUTCOME_EFFECTS["BMI"].update({
    # Match the new rs2074158 GIPR proxy.
    "rs2074158":   dict(ea="G", oa="A", eaf=0.180, beta=-0.017, se=0.004),
})
_OUTCOME_EFFECTS["WHRadjBMI"].update({
    "rs2074158":   dict(ea="G", oa="A", eaf=0.180, beta=-0.012, se=0.004),
})


def get_outcome_effects(outcome: str, rsids: list[str]) -> pd.DataFrame:
    """Lookup outcome effect-allele-aligned betas/SEs for a list of rsids.

    Returns a DataFrame with columns: rsid, ea, oa, eaf, beta, se.
    SNPs without an entry for this outcome are returned with NaN — the
    harmonizer will then drop them.
    """
    if outcome not in _OUTCOME_EFFECTS:
        raise KeyError(f"Unknown outcome '{outcome}'. Available: {list(_OUTCOME_EFFECTS)}")
    rows = []
    table = _OUTCOME_EFFECTS[outcome]
    for r in rsids:
        e = table.get(r)
        if e is None:
            continue
        rows.append({"rsid": r, **e})
    return pd.DataFrame(rows)


def available_outcomes() -> list[str]:
    return list(OUTCOMES.keys())


def outcome_metadata(name: str) -> dict | None:
    return OUTCOMES.get(name)
