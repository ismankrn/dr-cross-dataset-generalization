#!/usr/bin/env python3
"""
make_figures.py — gambar F3 (ROC), F4 (ablasi), F5 (dekomposisi), F6 (per grade + rDR)
dari predictions/ dan rekap_out/. Keluaran PDF vektor di figures/.
Jalankan dari folder manuscript/:  python make_figures.py
Warna kategorikal memakai palet tervalidasi (urutan tetap, tidak diputar).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT = os.path.abspath(os.environ.get("DR_ROOT", "."))  # akar paket
FIG = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG, exist_ok=True)
R = lambda n: pd.read_csv(os.path.join(ROOT, os.environ.get("DR_RESULTS", "results"), n))
P = lambda m, f: pd.read_csv(os.path.join(ROOT, os.environ.get("DR_PRED", "predictions_full"), m, f))

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # slot 1-4: biru, oranye, aqua, kuning (urutan tetap)
MAIN = ["R50-P-RAW", "D121-P-RAW", "EB0-P-RAW", "VIT-P-RAW"]
NAME = {"R50-P-RAW": "ResNet50", "D121-P-RAW": "DenseNet121", "EB0-P-RAW": "EfficientNet-B0", "VIT-P-RAW": "ViT-B/16"}
COL = dict(zip(MAIN, PAL))
DS = ["Val-APTOS", "IDRID", "Messidor"]; DSN = {"Val-APTOS": "APTOS (val)", "IDRID": "IDRiD", "Messidor": "Messidor"}
DSCOL = dict(zip(DS, PAL[:3])); FILE = {"Val-APTOS": "pred_val.csv", "IDRID": "pred_idrid.csv", "Messidor": "pred_messidor.csv"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a",
                     "axes.linewidth": 0.6, "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a", "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})

# ------------------------------------------------------------------ F3 ROC
t2 = R("T2.csv").set_index(["model", "dataset"])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, d in zip(axes, DS):
    for m in MAIN:
        p = P(m, FILE[d]); fpr, tpr, _ = roc_curve(p.y_true, p.y_prob); r = t2.loc[(m, d)]
        ax.plot(fpr, tpr, color=COL[m], lw=1.2, label=f"{NAME[m]}  {r.AUROC:.3f} [{r.CI_low:.3f}, {r.CI_high:.3f}]")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="#9a9a9a"); ax.set_title(DSN[d], fontsize=9); ax.set_xlabel("1 $-$ specificity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True); ax.legend(loc="lower right", fontsize=6, title="AUROC [95% CI]", title_fontsize=6)
axes[0].set_ylabel("Sensitivity"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "F3_roc.pdf")); plt.close(fig); print("F3_roc.pdf")

# ------------------------------------------------------------------ F4 ablation (ResNet50)
t3 = R("T3.csv"); t3["prep"] = t3.prep.astype(str).str.split(" ").str[0]; t3.loc[t3.model.str.endswith("-AUG"), "prep"] = "5A-AUG"
r50 = t3[t3.model.str.startswith("R50-P-")]
order = ["RAW", "1A", "2A", "3A", "4A", "5A", "6A", "1B", "4B", "5B", "6B", "5A-AUG"]
LAB = {"RAW": "None", "1A": "Circ. crop", "2A": "CLAHE", "3A": "Graham", "4A": "Crop+CLAHE", "5A": "Crop+Graham", "6A": "Crop+CLAHE+Graham",
       "1B": "Box crop", "4B": "Box+CLAHE", "5B": "Box+Graham", "6B": "Box+CLAHE+Graham", "5A-AUG": "Crop+Graham+aug."}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, d in zip(axes, ["IDRID", "Messidor"]):
    sub = r50[r50.dataset == d].set_index("prep").reindex(order); y = np.arange(len(order))[::-1]
    raw = sub.loc["RAW", "AUROC"]; ax.axvline(raw, ls="--", lw=0.7, color="#9a9a9a")
    ax.errorbar(sub.AUROC, y, xerr=[sub.AUROC - sub.CI_low, sub.CI_high - sub.AUROC], fmt="o", ms=4, color=DSCOL[d], ecolor=DSCOL[d], elinewidth=1, capsize=0)
    ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in order]); ax.set_title(DSN[d], fontsize=9); ax.set_xlabel("AUROC (95% CI)"); ax.grid(True, axis="x")
    ax.text(raw, len(order) - 0.5, "none", fontsize=6, color="#6a6a6a", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "F4_ablation.pdf")); plt.close(fig); print("F4_ablation.pdf")

# ------------------------------------------------------------------ F5 decomposition of balanced-accuracy drop
t4 = R("T4.csv").set_index(["model", "dataset"])
def ba(m, d):
    r = t2.loc[(m, d)]; return (r.TP / (r.TP + r.FN) + r.TN / (r.TN + r.FP)) / 2
fig, ax = plt.subplots(figsize=(7.2, 3.1))
labels, y = [], []
k = 0
for d in ["IDRID", "Messidor"]:
    for m in MAIN:
        ba_val, ba_t = ba(m, "Val-APTOS"), ba(m, d); orc = t4.loc[(m, d), "ba_oracle"]
        thr = max(orc - ba_t, 0); rep = max(ba_val - orc, 0)
        ax.barh(k, thr, color=PAL[0], height=0.6, label="Recoverable by threshold" if k == 0 else None)
        ax.barh(k, rep, left=thr + 0.004, color=PAL[1], height=0.6, label="Remaining (representation)" if k == 0 else None)
        ax.text(thr + rep + 0.012, k, f"{ba_val - ba_t:.2f}", va="center", fontsize=6.5, color="#333")
        labels.append(f"{NAME[m]} · {DSN[d]}"); y.append(k); k += 1
    k += 0.6
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("Drop in balanced accuracy from APTOS (val) to target")
ax.grid(True, axis="x"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "F5_decomposition.pdf")); plt.close(fig); print("F5_decomposition.pdf")

# ------------------------------------------------------------------ F6 per-grade sensitivity (ViT) + any-DR vs rDR
t5 = R("T5.csv").set_index(["model", "dataset"]); t6 = R("T6.csv").set_index(["model", "dataset"])
best = "VIT-P-RAW"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
grades = [1, 2, 3, 4]; w = 0.26
for i, d in enumerate(DS):
    r = t5.loc[(best, d)]; vals = [r[f"rate_g{g}"] if r[f"n_g{g}"] > 0 else np.nan for g in grades]
    a1.bar(np.arange(len(grades)) + (i - 1) * w, vals, width=w - 0.02, color=DSCOL[d], label=DSN[d])
a1.set_xticks(range(len(grades))); a1.set_xticklabels([f"grade {g}" for g in grades]); a1.set_ylim(0, 1.05); a1.set_ylabel("Sensitivity at threshold 0.5")
a1.set_title(f"Per-grade sensitivity, {NAME[best]}", fontsize=9); a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3); a1.grid(True, axis="y")
k = 0; yt, yl = [], []
for d in DS:
    for m in MAIN:
        r = t6.loc[(m, d)]
        a2.plot([r.AUROC_any, r.AUROC_rDR], [k, k], color="#bdbdbd", lw=1.2, zorder=1)
        a2.scatter(r.AUROC_any, k, color=COL[m], s=18, zorder=2, marker="o"); a2.scatter(r.AUROC_rDR, k, color=COL[m], s=18, zorder=2, marker="s")
        yt.append(k); yl.append(f"{NAME[m]} · {DSN[d]}"); k += 1
    k += 0.6
a2.set_yticks(yt); a2.set_yticklabels(yl, fontsize=6.5); a2.invert_yaxis(); a2.set_xlabel("AUROC"); a2.grid(True, axis="x")
a2.scatter([], [], color="#555", marker="o", s=18, label="any-DR (grade $\\geq$1)"); a2.scatter([], [], color="#555", marker="s", s=18, label="referable DR (grade $\\geq$2)")
a2.legend(fontsize=6.5, loc="lower right"); a2.set_title("Label definition", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "F6_grade_rdr.pdf")); plt.close(fig); print("F6_grade_rdr.pdf")
print("selesai:", FIG)
