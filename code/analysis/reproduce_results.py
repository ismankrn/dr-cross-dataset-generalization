#!/usr/bin/env python3
"""
reproduce_results.py -- recompute every result table of the paper from the per-image
predictions with labels (output of merge_labels.py).

  python code/analysis/reproduce_results.py --pred predictions_full --out results_reproduced

Writes T2 (main results), T3 (preprocessing ablation), T4 (decomposition and calibration),
T5 (per-grade detection rate), T6 (any-DR vs referable DR), and T7 (pretraining control)
as CSV files. The metric functions are identical to those used for the paper.
Bootstrap: 1,000 resamples with seed 0, so the CIs are reproduced exactly.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

N_BOOT, BOOT_SEED, N_BINS, THR = 1000, 0, 15, 0.5
FILES = {"Val-APTOS": "pred_val.csv", "IDRID": "pred_idrid.csv", "Messidor": "pred_messidor.csv"}
MAIN = ["R50-P-RAW", "D121-P-RAW", "EB0-P-RAW", "VIT-P-RAW"]
BACKBONE = {"R50": "ResNet50", "D121": "DenseNet121", "EB0": "EfficientNet-B0", "VIT": "ViT-B/16"}

def cm(y, p, thr=THR):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn

def sens_spec(y, p, thr=THR):
    tp, tn, fp, fn = cm(y, p, thr)
    s = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return s, sp

def bal_acc(y, p, thr=THR):
    s, sp = sens_spec(y, p, thr); return (s + sp) / 2

def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def boot_ci(y, p, fn=safe_auc, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p); j = tpr - fpr
    t = thr[np.argmax(j)]; return float(min(t, 1.0))

def oracle_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p); ba = (tpr + (1 - fpr)) / 2
    t = thr[np.argmax(ba)]; return float(min(t, 1.0))

def brier(y, p): return float(np.mean((p - y) ** 2))

def ece(y, p, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1); e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def prior_correct(p, prev_src, prev_tgt):
    """Saerens et al. 2002: sesuaikan posterior ke prevalensi target."""
    w1 = prev_tgt / prev_src; w0 = (1 - prev_tgt) / (1 - prev_src)
    return p * w1 / (p * w1 + (1 - p) * w0)


def load(pred_root, model, d):
    f = os.path.join(pred_root, model, FILES[d])
    if not os.path.exists(f): return None
    df = pd.read_csv(f)
    return pd.DataFrame({"y_prob": df.y_prob.astype(float), "y_true": df.y_true.astype(int), "grade": df.diagnosis.astype(int)})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", default="predictions_full"); ap.add_argument("--out", default="results_reproduced")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    models = sorted(os.listdir(a.pred))
    T2, T3, T4, T5, T6, T7 = [], [], [], [], [], []
    raw = {}
    for m in models:
        bb, kind, prep = m.split("-")[0], m.split("-")[1], "-".join(m.split("-")[2:])
        preds = {d: load(a.pred, m, d) for d in FILES}; preds = {d: v for d, v in preds.items() if v is not None}
        val = preds.get("Val-APTOS")
        for d, df in preds.items():
            y, p = df.y_true.values, df.y_prob.values
            tp, tn, fp, fn = cm(y, p); auc = safe_auc(y, p); lo, hi = boot_ci(y, p)
            aupr = float(average_precision_score(y, p)); s, sp = sens_spec(y, p); acc = (tp + tn) / len(y); ba = (s + sp) / 2
            if prep == "RAW": raw.setdefault((BACKBONE[bb], d), {})[kind] = (auc, acc)
            if m in MAIN:
                T2.append(dict(model=m, dataset=d, TP=tp, TN=tn, FP=fp, FN=fn, AUROC=auc, CI_low=lo, CI_high=hi, AUPRC=aupr))
            if kind == "P" and m != "R50-P-1A-ORIG" and (bb == "R50" or prep in ("RAW", "1A", "4A")):
                T3.append(dict(model=m, prep=prep, dataset=d, TP=tp, TN=tn, FP=fp, FN=fn, accuracy=acc, bal_acc=ba, AUROC=auc, CI_low=lo, CI_high=hi))
            if m in MAIN:
                row = dict(model=m, dataset=d)
                for g in range(5):
                    sub = df[df.grade == g]; n = len(sub)
                    row[f"n_g{g}"] = n; row[f"rate_g{g}"] = float(((sub.y_prob >= THR).astype(int) == (1 if g > 0 else 0)).mean()) if n else np.nan
                T5.append(row)
                y_r = (df.grade >= 2).astype(int).values; auc_r = safe_auc(y_r, p); lo_r, hi_r = boot_ci(y_r, p)
                T6.append(dict(model=m, dataset=d, AUROC_any=auc, CI_any=(lo, hi), AUROC_rDR=auc_r, CI_rDR=(lo_r, hi_r), AUPRC_any=aupr, AUPRC_rDR=float(average_precision_score(y_r, p))))
                if d != "Val-APTOS" and val is not None:
                    yv, pv = val.y_true.values, val.y_prob.values
                    ps, pt = float(yv.mean()), float(y.mean()); tv, to = youden_thr(yv, pv), oracle_thr(y, p); pc = prior_correct(p, ps, pt)
                    T4.append(dict(model=m, dataset=d, prev_src=ps, prev_tgt=pt, thr_val=tv, thr_oracle=to, ba_05=bal_acc(y, p, THR), ba_val=bal_acc(y, p, tv),
                                   ba_oracle=bal_acc(y, p, to), brier=brier(y, p), ece=ece(y, p), brier_corr=brier(y, pc), ece_corr=ece(y, pc)))
    for (bb, d), v in sorted(raw.items()):
        T7.append(dict(backbone=bb, dataset=d, auroc_P=v.get("P", (np.nan,))[0], auroc_NP=v.get("NP", (np.nan,))[0], acc_P=v.get("P", (0, np.nan))[1], acc_NP=v.get("NP", (0, np.nan))[1]))
    for name, rows in (("T2", T2), ("T3", T3), ("T4", T4), ("T5", T5), ("T6", T6), ("T7", T7)):
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    print("written to", a.out)

if __name__ == "__main__":
    main()
