#!/usr/bin/env python3
"""
predict_all.py — inference semua model di models_manifest.csv pada Val-APTOS,
IDRID, dan Messidor; hasilnya predictions/<Model ID>/pred_{val,idrid,messidor}.csv
(format yang dibaca compute_rekap.py).

Pipeline per gambar DIBUAT SAMA dengan notebook training V2:
  ResNet50 / DenseNet121 / EfficientNet-B0 : decode -> resize 224 -> preprocess_input backbone
  ViT                                       : decode -> keras_hub ViTImageConverter (preset vit_base_patch16_224_imagenet)
Val-APTOS = train_test_split(aptos_csv, test_size=0.2, random_state=42, stratify=label_num),
persis seperti notebook training, sehingga confusion matrix val dapat dibandingkan dengan log Keras.

Pemakaian (di server):
  python predict_all.py --root /data/dr --manifest models_manifest.csv --check     # cek file & folder saja
  python predict_all.py --root /data/dr --manifest models_manifest.csv             # jalankan semua
  python predict_all.py --root /data/dr --only R50-P-RAW,VIT-P-RAW                 # sebagian
  python predict_all.py --root /data/dr --manifest manifest_messidor_missing.csv --datasets Messidor   # hanya Messidor
--root adalah folder yang berisi aptos_dataset/, idrid_dataset/, messidor_dataset/, "Preprocessed image"/, V2/Model/.
Model yang sudah punya tiga file prediksi dilewati (idempoten); pakai --force untuk mengulang.
"""
import argparse, os, sys, time, json
import numpy as np
import pandas as pd

IMG_SIZE, BATCH = 224, 32
VIT_PRESET = "vit_base_patch16_224_imagenet"

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# ----------------------------------------------------------------------------- label & path
def load_labels(root):
    aptos = pd.read_csv(os.path.join(root, "aptos_dataset", "train.csv"))
    aptos["label_num"] = (aptos["diagnosis"] > 0).astype(int); aptos["grade"] = aptos["diagnosis"]
    aptos["stem"] = aptos["id_code"]
    idrid = pd.read_csv(os.path.join(root, "idrid_dataset", "idrid_labels.csv"))[["id_code", "diagnosis"]]
    idrid["label_num"] = (idrid["diagnosis"] > 0).astype(int); idrid["grade"] = idrid["diagnosis"]; idrid["stem"] = idrid["id_code"]
    mess = pd.read_csv(os.path.join(root, "messidor_dataset", "messidor_labels.csv"))
    mess["id_code"] = mess["Image name"]; mess["label_num"] = (mess["Retinopathy grade"] > 0).astype(int)
    mess["grade"] = mess["Retinopathy grade"]; mess["stem"] = mess["Image name"]     # file preprocessed: "<name>.tif.png"
    return {"APTOS": aptos, "IDRID": idrid, "Messidor": mess}

def val_split(aptos):
    from sklearn.model_selection import train_test_split
    _, val = train_test_split(aptos, test_size=0.2, random_state=42, stratify=aptos["label_num"])
    return val.reset_index(drop=True)

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
def resolve_paths(df, folder, ext):
    """Cari file per stem tanpa bergantung ekstensi: coba stem+ext, stem apa adanya,
    stem tanpa ekstensi asli (Messidor: 'x.tif' -> 'x.png'), dan stem+ekstensi apa pun."""
    cache = {}
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            cache.setdefault(f, True); cache.setdefault(os.path.splitext(f)[0], f)
    paths, ok = [], []
    for s in df["stem"]:
        base = os.path.splitext(s)[0] if s.lower().endswith(EXTS) else s
        found = None
        for cand in ([f"{s}{ext}"] if ext else []) + [s] + [f"{s}{e}" for e in EXTS] + [f"{base}{e}" for e in EXTS]:
            if cand in cache and cache[cand] is True: found = cand; break
        if found is None and base in cache and cache[base] is not True: found = cache[base]
        paths.append(os.path.join(folder, found or f"{s}{ext}")); ok.append(found is not None)
    return paths, np.array(ok)

# ----------------------------------------------------------------------------- loader
def make_loader(backbone):
    import tensorflow as tf
    from PIL import Image
    if backbone == "vit":
        import keras_hub
        conv = keras_hub.layers.ViTImageConverter.from_preset(VIT_PRESET)
        prep = lambda img: conv(img)
        resize = False
    else:
        mod = {"resnet50": "resnet50", "densenet121": "densenet", "efficientnetb0": "efficientnet"}[backbone]
        prep_fn = __import__(f"tensorflow.keras.applications.{mod}", fromlist=["preprocess_input"]).preprocess_input
        prep = lambda img: prep_fn(tf.image.resize(img, (IMG_SIZE, IMG_SIZE)))
        resize = True

    def _pil_read(path):
        path = path.item() if hasattr(path, "item") else path
        path = path.decode() if isinstance(path, bytes) else str(path)
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def load(path):
        ext = tf.strings.lower(tf.strings.substr(path, -4, 4))
        is_tif = tf.logical_or(tf.strings.regex_full_match(ext, ".*tif"), tf.strings.regex_full_match(ext, ".*iff"))
        def _tf_decode():
            return tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        def _pil_decode():
            img = tf.numpy_function(_pil_read, [path], tf.uint8); img.set_shape((None, None, 3)); return img
        img = tf.cond(is_tif, _pil_decode, _tf_decode)
        img.set_shape((None, None, 3))
        return prep(img)
    return load

def decode_any(path):
    """decode PNG/JPEG via TF, TIFF via PIL (untuk folder 3a dan Messidor mentah); hasil uint8 HxWx3."""
    import tensorflow as tf
    from PIL import Image
    def _pil_read(p):
        p = p.item() if hasattr(p, "item") else p
        p = p.decode() if isinstance(p, bytes) else str(p)
        return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
    ext = tf.strings.lower(tf.strings.substr(path, -4, 4))
    is_tif = tf.logical_or(tf.strings.regex_full_match(ext, ".*tif"), tf.strings.regex_full_match(ext, ".*iff"))
    def _tf(): return tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    def _pil():
        img = tf.numpy_function(_pil_read, [path], tf.uint8); img.set_shape((None, None, 3)); return img
    img = tf.cond(is_tif, _pil, _tf); img.set_shape((None, None, 3)); return img

def predict(model, paths, backbone):
    import tensorflow as tf
    load = make_loader(backbone)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    prob = model.predict(ds, verbose=0)
    return prob[:, 1] if prob.shape[1] == 2 else prob[:, 0]      # softmax -> P(DR) kolom 1; sigmoid -> kolom 0

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--manifest", default="models_manifest.csv")
    ap.add_argument("--out", default="predictions"); ap.add_argument("--only", default="")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", default="Val-APTOS,IDRID,Messidor", help="subset dataset, mis. --datasets Messidor")
    a = ap.parse_args()
    DSEL = set(a.datasets.split(","))
    man = pd.read_csv(a.manifest).fillna("")
    if a.only: man = man[man.model_id.isin(a.only.split(","))]
    labels = load_labels(a.root); labels["Val-APTOS"] = val_split(labels["APTOS"])
    log(f"label: val {len(labels['Val-APTOS'])}, IDRID {len(labels['IDRID'])}, Messidor {len(labels['Messidor'])}")

    # ---- cek
    problems = []
    for _, m in man.iterrows():
        mp = os.path.join(a.root, m.model_file)
        if not os.path.exists(mp): problems.append(f"{m.model_id}: model tidak ada: {mp}")
        for ds, fcol, ecol in (("Val-APTOS", "folder_aptos", "ext_aptos"), ("IDRID", "folder_idrid", "ext_idrid"), ("Messidor", "folder_messidor", "ext_messidor")):
            if ds not in DSEL: continue
            folder = os.path.join(a.root, m[fcol]); ext = m[ecol] if m[ecol] else ""
            if not os.path.isdir(folder): problems.append(f"{m.model_id}: folder tidak ada: {folder}"); continue
            _, ok = resolve_paths(labels[ds], folder, ext)
            if ok.sum() < len(ok): problems.append(f"{m.model_id}: {ds}: {len(ok)-ok.sum()} dari {len(ok)} gambar tidak ditemukan di {folder}")
    print("\n".join(problems) if problems else "CEK OK: semua model dan gambar ditemukan")
    if a.check: sys.exit(1 if problems else 0)

    # ---- inference
    import tensorflow as tf
    log("TF", tf.__version__, "GPU:", tf.config.list_physical_devices("GPU"))
    summary = []
    for _, m in man.iterrows():
        outdir = os.path.join(a.out, m.model_id); os.makedirs(outdir, exist_ok=True)
        targets = {k: v for k, v in {"Val-APTOS": "pred_val.csv", "IDRID": "pred_idrid.csv", "Messidor": "pred_messidor.csv"}.items() if k in DSEL}
        if not a.force and all(os.path.exists(os.path.join(outdir, f)) for f in targets.values()):
            log(f"{m.model_id}: sudah ada, lewati"); continue
        mp = os.path.join(a.root, m.model_file)
        if not os.path.exists(mp): log(f"{m.model_id}: model tidak ada, lewati"); continue
        t0 = time.time()
        try:
            if m.backbone == "vit": import keras_hub  # noqa: registrasi kelas ViTBackbone untuk load_model
            model = tf.keras.models.load_model(mp, compile=False)
        except Exception as e:
            log(f"{m.model_id}: GAGAL load model: {e}"); continue
        for ds, fname in targets.items():
            fcol, ecol = {"Val-APTOS": ("folder_aptos", "ext_aptos"), "IDRID": ("folder_idrid", "ext_idrid"), "Messidor": ("folder_messidor", "ext_messidor")}[ds]
            folder = os.path.join(a.root, m[fcol]); ext = m[ecol] if m[ecol] else ""
            df = labels[ds].copy(); paths, ok = resolve_paths(df, folder, ext); df = df[ok].reset_index(drop=True)
            paths = [p for p, k in zip(paths, ok) if k]
            if not paths: log(f"{m.model_id}: {ds}: tidak ada gambar, lewati"); continue
            prob = predict(model, paths, m.backbone)
            out = pd.DataFrame({"id_code": df["id_code"], "diagnosis": df["grade"], "y_true": df["label_num"], "y_prob": prob})
            out.to_csv(os.path.join(outdir, fname), index=False)
            pred = (prob >= 0.5).astype(int); acc = float((pred == df["label_num"].values).mean())
            try:
                from sklearn.metrics import roc_auc_score; auc = float(roc_auc_score(df["label_num"], prob))
            except Exception: auc = float("nan")
            summary.append(dict(model=m.model_id, dataset=ds, n=len(df), acc=acc, auroc=auc))
            log(f"{m.model_id:14s} {ds:9s} n={len(df):4d} acc={acc:.4f} auroc={auc:.4f}")
        del model; tf.keras.backend.clear_session()
        log(f"{m.model_id}: selesai {time.time()-t0:.0f}s")
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "summary_inference.csv"), index=False)
    log("SELESAI. Salin folder", a.out, "ke komputer lokal lalu jalankan compute_rekap.py")

if __name__ == "__main__":
    main()
