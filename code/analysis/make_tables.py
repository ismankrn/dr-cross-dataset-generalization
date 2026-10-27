#!/usr/bin/env python3
"""
make_tables.py — bangkitkan tabel LaTeX (tables/T1.tex .. T7.tex) dari hasil terhitung:
rekap_out/*.csv (keluaran compute_rekap.py) dan predictions/*/pred_*.csv.
Jalankan dari folder manuscript/:  python make_tables.py
Setiap kali compute_rekap.py dijalankan ulang, jalankan skrip ini lagi; angka di naskah mengikuti.
"""
import os, sys, numpy as np, pandas as pd

ROOT = os.path.abspath(os.environ.get("DR_ROOT", "."))  # akar paket
OUT = os.path.join(ROOT, "outputs", "tables"); os.makedirs(OUT, exist_ok=True)
R = lambda n: pd.read_csv(os.path.join(ROOT, os.environ.get("DR_RESULTS", "results"), n))
NAME = {"R50-P-RAW": "ResNet50", "D121-P-RAW": "DenseNet121", "EB0-P-RAW": "EfficientNet-B0", "VIT-P-RAW": "ViT-B/16"}
MAIN = list(NAME)
DS = ["Val-APTOS", "IDRID", "Messidor"]; DSN = {"Val-APTOS": "APTOS (val)", "IDRID": "IDRiD", "Messidor": "Messidor"}
PREP_LABEL = {"RAW": "None (baseline)", "1A": "Circular crop", "2A": "CLAHE", "3A": "Graham norm.", "4A": "Crop + CLAHE", "5A": "Crop + Graham",
              "6A": "Crop + CLAHE + Graham", "1B": "Box crop", "4B": "Box + CLAHE", "5B": "Box + Graham", "6B": "Box + CLAHE + Graham", "5A-AUG": "Crop + Graham + augmentation"}

def f(x, d=3): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
def sg(x): return "0.000" if round(x, 3) == 0 else f"{x:+.3f}".replace("-", "$-$")
def ci(lo, hi): return f"[{f(lo)}, {f(hi)}]"
def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh: fh.write(body)
    print("ditulis", name)

# ------------------------------------------------------------------ T0 experimental design (statis, mengikuti sheet Models)
t = ["\\begin{tabular}{>{\\raggedright\\arraybackslash}p{27mm}>{\\raggedright\\arraybackslash}p{31mm}>{\\raggedright\\arraybackslash}p{30mm}l>{\\raggedright\\arraybackslash}p{40mm}lc}", "\\toprule",
     "Group & Experiment & Backbone & Initialization & Preprocessing & Augmentation & Models \\\\", "\\midrule",
     "1. Main comparison & Backbone comparison & All four & ImageNet & None & No & 4 \\\\", "\\midrule",
     "\\multirow[t]{3}{27mm}{\\raggedright 2. Preprocessing and augmentation} & Preprocessing ablation & ResNet50 & ImageNet & Ten configurations (Section~\\ref{sec:preproc}) & No & 10 \\\\",
     " & Cross-architecture replication & DenseNet121, EfficientNet-B0, ViT-B/16 & ImageNet & Circular crop, and circular crop with CLAHE & No & 6 \\\\",
     " & Augmentation (exploratory) & ResNet50 & ImageNet & Circular crop with Graham normalization & Yes & 1 \\\\", "\\midrule",
     "3. Pretraining control & Training from scratch & All four & Random & None & No & 4 \\\\", "\\midrule",
     "Total & & & & & & 25 \\\\", "\\bottomrule", "\\end{tabular}"]
write("T0.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T_models backbones and shared training configuration (statis, dari notebook)
t = ["\\begin{tabular}{lllcll>{\\raggedright\\arraybackslash}p{42mm}}", "\\toprule",
     "Backbone & Family & Parameters & Pretrained weights & Output & Loss & Input normalization \\\\", "\\midrule",
     "ResNet50 & Residual CNN & 23.6\\,M & ImageNet-1k & 2-unit softmax & Sparse categorical CE & RGB to BGR, ImageNet mean subtracted \\\\",
     "DenseNet121 & Densely connected CNN & 7.0\\,M & ImageNet-1k & 1-unit sigmoid & Binary CE & Scaled to [0, 1], ImageNet mean and s.d. \\\\",
     "EfficientNet-B0 & Compound-scaled CNN & 4.1\\,M & ImageNet-1k & 1-unit sigmoid & Binary CE & Rescaling and normalization inside the network \\\\",
     "ViT-B/16 & Vision transformer & 85.8\\,M & ImageNet-1k & 2-unit softmax & Sparse categorical CE & Image converter of the KerasHub preset \\\\",
     "\\midrule",
     "\\multicolumn{7}{l}{\\textit{Settings shared by all 25 models}} \\\\",
     "Input & \\multicolumn{6}{l}{224 $\\times$ 224 pixels, bilinear resizing} \\\\",
     "Representation & \\multicolumn{6}{l}{Global average pooling (CNN) or class token (ViT), dropout 0.3, dense output layer} \\\\",
     "Fine-tuning & \\multicolumn{6}{l}{All layers trainable, no frozen blocks} \\\\",
     "Optimizer & \\multicolumn{6}{l}{SGD with Nesterov momentum 0.9, initial learning rate $10^{-4}$, batch size 32} \\\\",
     "Learning-rate schedule & \\multicolumn{6}{l}{Halved after 4 epochs without improvement of validation loss, minimum $10^{-6}$} \\\\",
     "Stopping & \\multicolumn{6}{l}{At most 100 epochs, early stopping after 10 epochs without improvement, best weights restored} \\\\",
     "Class weighting & \\multicolumn{6}{l}{None} \\\\",
     "Random seed & \\multicolumn{6}{l}{42 for data partition, shuffling, and weight initialization} \\\\",
     "\\bottomrule", "\\end{tabular}"]
write("T_models.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T_prep preprocessing configurations (statis, terverifikasi piksel terhadap folder Drive)
t = ["\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{36mm}>{\\raggedright\\arraybackslash}p{30mm}>{\\raggedright\\arraybackslash}p{36mm}ll}", "\\toprule",
     "Configuration & Localization & CLAHE & Graham normalization & Background & Stored size \\\\", "\\midrule",
     "None & None & None & None & Original & Original \\\\",
     "Circular crop & Circular mask fitted to the fundus & None & None & Black & Original canvas \\\\",
     "CLAHE & None & Clip 2.0 & None & Original & Original \\\\",
     "Graham & Crop to pixels above 10$^{a}$ & None & $\\sigma = 15$, reflective padding & Mid-gray & 512 $\\times$ 512 \\\\",
     "Circular crop + CLAHE & Circular mask fitted to the fundus & Clip 2.5, before resampling & None & Black & Original canvas \\\\",
     "Circular crop + Graham & Circular mask, then crop to pixels above 10 & None & $\\sigma = 15$, reflective padding & Mid-gray & 512 $\\times$ 512 \\\\",
     "Circular crop + CLAHE + Graham & Circular mask, then crop to pixels above 10 & Clip 2.5, before resampling & $\\sigma = 15$, reflective padding & Mid-gray & 512 $\\times$ 512 \\\\",
     "Box crop & Bounding box, 2\\% margin & None & None & Dark corners & Cropped \\\\",
     "Box crop + CLAHE & Bounding box, 2\\% margin & Clip 2.0 & None & Dark corners & Cropped \\\\",
     "Box crop + Graham$^{a}$ & Bounding box, 2\\% margin & None & $\\sigma = 15$, reflective padding & Mid-gray & 512 $\\times$ 512 \\\\",
     "Box crop + CLAHE + Graham & Bounding box, 2\\% margin & Clip 2.0 & $\\sigma = 15$, reflective padding & Mid-gray & 512 $\\times$ 512 \\\\",
     "\\midrule",
     "\\multicolumn{6}{l}{$^{a}$Graham and box crop + Graham produce identical images (Section~\\ref{sec:preproc}).} \\\\",
     "\\bottomrule", "\\end{tabular}"]
write("T_prep.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T1 datasets
val = pd.read_csv(os.path.join(ROOT, os.environ.get("DR_PRED", "predictions_full"), "D121-P-RAW", "pred_val.csv"))
vg = val.diagnosis.value_counts().reindex(range(5), fill_value=0).tolist()
rows = [("APTOS 2019 (train split)", "Source, training", [1805-vg[0], 370-vg[1], 999-vg[2], 193-vg[3], 295-vg[4]], "ICDR 0--4", "Various cameras, rural India"),
        ("APTOS 2019 (validation split)", "Source, in-domain test", vg, "ICDR 0--4", "Various cameras, rural India"),
        ("IDRiD", "External", [129, 22, 156, 84, 64], "ICDR 0--4", "Kowa VX-10, 50$^\\circ$ FOV, 4288$\\times$2848"),
        ("Messidor", "External", [546, 153, 247, 254, 0], "Messidor 0--3", "Topcon TRC NW6, 45$^\\circ$ FOV, TIFF")]
t = ["\\begin{tabular}{llrrrrrrrrl}", "\\toprule", "Dataset & Role & N & G0 & G1 & G2 & G3 & G4 & Prev.\\ any-DR & Prev.\\ rDR & Acquisition \\\\", "\\midrule"]
for n, role, g, scale, acq in rows:
    N = sum(g); t.append(f"{n} & {role} & {N} & {g[0]} & {g[1]} & {g[2]} & {g[3]} & {g[4]} & {f(sum(g[1:])/N)} & {f(sum(g[2:])/N)} & {acq} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T1.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T2 main results
t2 = R("T2.csv"); t2 = t2[t2.model.isin(MAIN)].copy()
t2["sens"] = t2.TP / (t2.TP + t2.FN); t2["spec"] = t2.TN / (t2.TN + t2.FP); t2["bal"] = (t2.sens + t2.spec) / 2; t2["acc"] = (t2.TP + t2.TN) / (t2.TP + t2.TN + t2.FP + t2.FN)
t = ["\\begin{tabular}{ll" + "cccc" * 3 + "}", "\\toprule",
     "& & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DSN[d]}}}" for d in DS) + " \\\\",
     "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}",
     "Backbone & & " + " & ".join("AUROC [95\\% CI] & Sens. & Spec. & Bal.\\ acc." for _ in DS) + " \\\\", "\\midrule"]
for m in MAIN:
    cells = []
    for d in DS:
        r = t2[(t2.model == m) & (t2.dataset == d)]
        if len(r) == 0: cells += ["--"] * 4; continue
        r = r.iloc[0]; cells += [f"{f(r.AUROC)} {ci(r.CI_low, r.CI_high)}", f(r.sens), f(r.spec), f(r.bal)]
    t.append(f"{NAME[m]} & & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T2.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T3 ablation
t3 = R("T3.csv"); t3["prep"] = t3.prep.astype(str).str.split(" ").str[0]
t3.loc[t3.model.str.endswith("-AUG"), "prep"] = "5A-AUG"
def raw_auc(bb, d):
    r = t3[(t3.model == f"{bb}-P-RAW") & (t3.dataset == d)]; return float(r.AUROC.iloc[0]) if len(r) else np.nan
order = ["RAW", "1A", "2A", "3A", "4A", "5A", "6A", "1B", "4B", "5B", "6B", "5A-AUG"]
t = ["\\begin{tabular}{llcccccc}", "\\toprule", "Backbone & Preprocessing & \\multicolumn{3}{c}{IDRiD} & \\multicolumn{3}{c}{Messidor} \\\\",
     "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}", "& & AUROC & 95\\% CI & $\\Delta$ vs none & AUROC & 95\\% CI & $\\Delta$ vs none \\\\", "\\midrule"]
for bb, bbn in (("R50", "ResNet50"), ("D121", "DenseNet121"), ("EB0", "EfficientNet-B0"), ("VIT", "ViT-B/16")):
    sub = t3[t3.model.str.startswith(f"{bb}-P-")]
    first = True
    for p in order:
        rr = sub[sub.prep == p]
        if len(rr) == 0: continue
        cells = []
        for d in ("IDRID", "Messidor"):
            r = rr[rr.dataset == d]
            if len(r) == 0: cells += ["--"] * 3; continue
            r = r.iloc[0]; delta = "--" if p == "RAW" else sg(r.AUROC - raw_auc(bb, d))
            cells += [f(r.AUROC), ci(r.CI_low, r.CI_high), delta]
        t.append(f"{bbn if first else ''} & {PREP_LABEL.get(p, p)} & " + " & ".join(cells) + " \\\\"); first = False
    t.append("\\midrule")
t[-1] = "\\bottomrule"; t.append("\\end{tabular}")
write("T3.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T4 decomposition + T4b calibration (baris dikelompokkan per target)
t4 = R("T4.csv"); t4 = t4[t4.model.isin(MAIN)]
cm = t2.set_index(["model", "dataset"])
t = ["\\begin{tabular}{llcccccccc}", "\\toprule",
     "Target & Backbone & Acc. & $\\Delta_{\\mathrm{prev}}$ & BA (0.5) & BA (Youden) & BA ($t^{*}$) & $\\Delta_{\\mathrm{thr}}$ & AUROC & $\\Delta_{\\mathrm{rep}}$ \\\\", "\\midrule"]
tc = ["\\begin{tabular}{llcccc}", "\\toprule",
      "Target & Backbone & Brier & Brier (corrected) & ECE & ECE (corrected) \\\\", "\\midrule"]
for i, d in enumerate(("IDRID", "Messidor")):
    if i: t.append("\\midrule"); tc.append("\\midrule")
    for m in MAIN:
        r = t4[(t4.model == m) & (t4.dataset == d)]
        if len(r) == 0: continue
        r = r.iloc[0]; c = cm.loc[(m, d)]; src = cm.loc[(m, "Val-APTOS")]
        acc_src = r.prev_src * c.sens + (1 - r.prev_src) * c.spec
        t.append(f"{DSN[d]} & {NAME[m]} & {f(c.acc)} & {sg(c.acc - acc_src)} & {f(r.ba_05)} & {f(r.ba_val)} & {f(r.ba_oracle)} & {sg(r.ba_oracle - r.ba_05)} & {f(c.AUROC)} & {src.AUROC - c.AUROC:.3f} \\\\")
        tc.append(f"{DSN[d]} & {NAME[m]} & {f(r.brier)} & {f(r.brier_corr)} & {f(r.ece)} & {f(r.ece_corr)} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]; tc += ["\\bottomrule", "\\end{tabular}"]
write("T4.tex", "\n".join(t) + "\n"); write("T4b.tex", "\n".join(tc) + "\n")

# ------------------------------------------------------------------ T5 per grade
t5 = R("T5.csv"); t5 = t5[t5.model.isin(MAIN)]
t = ["\\begin{tabular}{llccccc}", "\\toprule", "Backbone & Dataset & Spec.\\ grade 0 & Sens.\\ grade 1 & Sens.\\ grade 2 & Sens.\\ grade 3 & Sens.\\ grade 4 \\\\", "\\midrule"]
for m in MAIN:
    for d in DS:
        r = t5[(t5.model == m) & (t5.dataset == d)]
        if len(r) == 0: continue
        r = r.iloc[0]
        cells = [f"{f(r[f'rate_g{g}'])} ({int(r[f'n_g{g}'])})" if r[f"n_g{g}"] > 0 else "--" for g in range(5)]
        t.append(f"{NAME[m]} & {DSN[d]} & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T5.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T6 label definition
t6 = R("T6.csv"); t6 = t6[t6.model.isin(MAIN)]
def parse_ci(s):
    try: lo, hi = eval(s); return lo, hi
    except Exception: return (np.nan, np.nan)
t = ["\\begin{tabular}{llcccc}", "\\toprule", "Backbone & Dataset & AUROC any-DR [95\\% CI] & AUROC rDR [95\\% CI] & $\\Delta$ & AUPRC any / rDR \\\\", "\\midrule"]
for m in MAIN:
    for d in DS:
        r = t6[(t6.model == m) & (t6.dataset == d)]
        if len(r) == 0: continue
        r = r.iloc[0]; a = parse_ci(r.CI_any); b = parse_ci(r.CI_rDR)
        t.append(f"{NAME[m]} & {DSN[d]} & {f(r.AUROC_any)} {ci(*a)} & {f(r.AUROC_rDR)} {ci(*b)} & {sg(r.AUROC_rDR - r.AUROC_any)} & {f(r.AUPRC_any)} / {f(r.AUPRC_rDR)} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T6.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T_backbone ringkasan perilaku backbone lintas analisis
PARAMS = {"R50-P-RAW": "23.6\\,M", "D121-P-RAW": "7.0\\,M", "EB0-P-RAW": "4.1\\,M", "VIT-P-RAW": "85.8\\,M"}
BBN = {"R50-P-RAW": "ResNet50", "D121-P-RAW": "DenseNet121", "EB0-P-RAW": "EfficientNet-B0", "VIT-P-RAW": "ViT-B/16"}
t2i = t2.set_index(["model", "dataset"]); t4i = R("T4.csv").set_index(["model", "dataset"])
t5i = R("T5.csv").set_index(["model", "dataset"]); t6i = R("T6.csv").set_index(["model", "dataset"])
t7i = R("T7.csv").set_index(["backbone", "dataset"])
rows = [("Parameters", lambda m: PARAMS[m]),
        ("AUROC, APTOS (val)", lambda m: f(t2i.loc[(m, "Val-APTOS")].AUROC)),
        ("AUROC, IDRiD", lambda m: f(t2i.loc[(m, "IDRID")].AUROC)),
        ("AUROC, Messidor (any-DR)", lambda m: f(t2i.loc[(m, "Messidor")].AUROC)),
        ("AUROC, Messidor (rDR)", lambda m: f(t6i.loc[(m, "Messidor")].AUROC_rDR)),
        ("BA (0.5), Messidor", lambda m: f(t4i.loc[(m, "Messidor")].ba_05)),
        ("ECE, IDRiD", lambda m: f(t4i.loc[(m, "IDRID")].ece)),
        ("ECE, Messidor", lambda m: f(t4i.loc[(m, "Messidor")].ece)),
        ("Sens.\\ grade 3, Messidor", lambda m: f(t5i.loc[(m, "Messidor")].rate_g3)),
        ("$\\Delta$AUROC from pretraining, IDRiD", lambda m: sg(t7i.loc[(BBN[m], "IDRID")].auroc_P - t7i.loc[(BBN[m], "IDRID")].auroc_NP)),
        ("$\\Delta$AUROC from pretraining, Messidor", lambda m: sg(t7i.loc[(BBN[m], "Messidor")].auroc_P - t7i.loc[(BBN[m], "Messidor")].auroc_NP))]
t = ["\\begin{tabular}{lcccc}", "\\toprule", "Measure & " + " & ".join(NAME[m] for m in MAIN) + " \\\\", "\\midrule"]
for lab, fn in rows: t.append(lab + " & " + " & ".join(fn(m) for m in MAIN) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T_backbone.tex", "\n".join(t) + "\n")

# ------------------------------------------------------------------ T7 pretrained vs scratch (AUROC + sens/spec pada threshold 0.5)
t7 = R("T7.csv")
BBID = {"ResNet50": "R50", "DenseNet121": "D121", "EfficientNet-B0": "EB0", "ViT-B/16": "VIT"}
PF = {"Val-APTOS": "pred_val.csv", "IDRID": "pred_idrid.csv", "Messidor": "pred_messidor.csv"}
def sens_spec_file(mid, d):
    p = pd.read_csv(os.path.join(ROOT, os.environ.get("DR_PRED", "predictions_full"), mid, PF[d])); yhat = (p.y_prob >= 0.5).astype(int)
    se = ((yhat == 1) & (p.y_true == 1)).sum() / (p.y_true == 1).sum(); sp = ((yhat == 0) & (p.y_true == 0)).sum() / (p.y_true == 0).sum()
    return se, sp
t = ["\\begin{tabular}{llccccc}", "\\toprule",
     "& & \\multicolumn{3}{c}{AUROC} & \\multicolumn{2}{c}{Sens.\\ / Spec.\\ at 0.5} \\\\", "\\cmidrule(lr){3-5} \\cmidrule(lr){6-7}",
     "Backbone & Dataset & Pretrained & Scratch & $\\Delta$ & Pretrained & Scratch \\\\", "\\midrule"]
for bb in ["ResNet50", "DenseNet121", "EfficientNet-B0", "ViT-B/16"]:
    for d in DS:
        r = t7[(t7.backbone == bb) & (t7.dataset == d)]
        if len(r) == 0: continue
        r = r.iloc[0]; sp_ = sens_spec_file(f"{BBID[bb]}-P-RAW", d); sn_ = sens_spec_file(f"{BBID[bb]}-NP-RAW", d)
        t.append(f"{bb} & {DSN[d]} & {f(r.auroc_P)} & {f(r.auroc_NP)} & {sg(r.auroc_P - r.auroc_NP)} & {f(sp_[0])} / {f(sp_[1])} & {f(sn_[0])} / {f(sn_[1])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
write("T7.tex", "\n".join(t) + "\n")
print("selesai:", OUT)
