#!/usr/bin/env python3
"""
merge_labels.py -- attach the official dataset labels to the released predictions.

The released files in predictions/<model>/pred_{val,idrid,messidor}.csv contain only the
image identifier (id_code) and the predicted probability of DR (y_prob), because the
dataset licenses do not allow redistribution of the annotations. Download the datasets
from their official sources and run, for example:

  python code/analysis/merge_labels.py \
      --aptos path/to/aptos2019/train.csv \
      --idrid-train "path/to/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv" \
      --idrid-test  "path/to/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv" \
      --messidor path/to/messidor/   (folder with Annotation_Base*.xls, or one CSV) \
      --pred predictions --out predictions_full

Output: predictions_full/<model>/pred_*.csv with columns id_code, diagnosis, y_true, y_prob,
where diagnosis is the original grade and y_true = 1 if diagnosis >= 1 (any-DR).
IDRiD test images are identified with the suffix "test" (e.g., IDRiD_001test), since the
official training and testing sets reuse the same file names.
"""
import argparse, glob, os
import pandas as pd

def col(df, *names):
    norm = {c.strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in norm: return norm[n.lower()]
    raise KeyError(f"none of {names} in {list(df.columns)}")

def load_aptos(path):
    d = pd.read_csv(path)
    return pd.DataFrame({"id_code": d[col(d, "id_code")].astype(str), "diagnosis": d[col(d, "diagnosis")].astype(int)})

def load_idrid(train, test):
    out = []
    for path, suffix in ((train, ""), (test, "test")):
        d = pd.read_csv(path).dropna(how="all", axis=1)
        out.append(pd.DataFrame({"id_code": d[col(d, "Image name", "id_code")].astype(str).str.replace(r"\.jpg$", "", regex=True) + suffix,
                                 "diagnosis": d[col(d, "Retinopathy grade", "diagnosis")].astype(int)}))
    return pd.concat(out, ignore_index=True)

def load_messidor(path):
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.xls")) + glob.glob(os.path.join(path, "*.xlsx")) + glob.glob(os.path.join(path, "*.csv")))
    frames = [pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f) for f in files]
    d = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({"id_code": d[col(d, "Image name")].astype(str), "diagnosis": d[col(d, "Retinopathy grade")].astype(int)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aptos", required=True); ap.add_argument("--idrid-train", required=True); ap.add_argument("--idrid-test", required=True)
    ap.add_argument("--messidor", required=True); ap.add_argument("--pred", default="predictions"); ap.add_argument("--out", default="predictions_full")
    a = ap.parse_args()
    labels = {"pred_val.csv": load_aptos(a.aptos), "pred_idrid.csv": load_idrid(a.idrid_train, a.idrid_test), "pred_messidor.csv": load_messidor(a.messidor)}
    n = 0
    for mdir in sorted(glob.glob(os.path.join(a.pred, "*"))):
        if not os.path.isdir(mdir): continue
        os.makedirs(os.path.join(a.out, os.path.basename(mdir)), exist_ok=True)
        for fname, lab in labels.items():
            f = os.path.join(mdir, fname)
            if not os.path.exists(f): continue
            p = pd.read_csv(f, dtype={"id_code": str})
            m = p.merge(lab, on="id_code", how="left")
            missing = m.diagnosis.isna().sum()
            if missing: raise SystemExit(f"{f}: {missing} images without a label, check the label files")
            m["diagnosis"] = m.diagnosis.astype(int); m["y_true"] = (m.diagnosis >= 1).astype(int)
            m[["id_code", "diagnosis", "y_true", "y_prob"]].to_csv(os.path.join(a.out, os.path.basename(mdir), fname), index=False); n += 1
    print(f"{n} files written to {a.out}/")

if __name__ == "__main__":
    main()
