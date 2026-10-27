#!/usr/bin/env python3
"""
train_r50_prep.py — latih ulang ResNet50 softmax pretrained pada satu folder preprocessing,
dengan konfigurasi IDENTIK dengan notebook V2 (lihat train_r50_5a_aug.py), lalu simpan
model, log, dan prediksi per gambar.

Dipakai karena model ResNet50 2A-6B di Drive tidak lagi cocok dengan folder gambar mana pun
(uji silang match_grid.py), sehingga hasil logbooknya tidak dapat direproduksi.

Pemakaian:  python train_r50_prep.py --root /data/dr --prep 2A [--epochs 100] [--aug]
Output:     V2/Model/(retrain_<prep>)resnet50_p_softmax.keras, V2/Logs/(retrain_<prep>)resnet50_p_softmax.csv,
            predictions_retrain/R50-P-<prep>[-AUG]/pred_{val,idrid,messidor}.csv
"""
import argparse, os, shutil, time, sys
import numpy as np, pandas as pd

IMG_SIZE, BATCH, SEED = 224, 32, 42
FOLDER = {"1A": "Circular Crop/1a", "4A": "Circular Crop/4a", "5A": "Circular Crop/5a", "6A": "Circular Crop/6a",
          "1B": "Box Crop/1b", "4B": "Box Crop/4b", "5B": "Box Crop/5b", "6B": "Box Crop/6b", "2A": "2a", "3A": "3a",
          # 6A/6B baru: Graham yang sama dengan 5A/5B, diterapkan pada 4A/4B (make_graham_6x.py)
          "6A_G": "Circular Crop/6a_g", "6B_G": "Box Crop/6b_g"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--prep", required=True, choices=sorted(FOLDER))
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--aug", action="store_true")
    ap.add_argument("--out", default="predictions_retrain")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import predict_all as pa
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    tf.keras.utils.set_random_seed(SEED)

    tag = f"retrain_{a.prep}" + ("_aug" if a.aug else ""); model_id = f"R50-P-{a.prep}" + ("-AUG" if a.aug else "")
    P = lambda *x: os.path.join(a.root, *x)
    base = P("Preprocessed image", FOLDER[a.prep])
    labels = pa.load_labels(a.root)
    dfs = {}
    for ds, sub in (("APTOS", "APTOS"), ("IDRID", "IDRID"), ("Messidor", "MESSIDOR")):
        df = labels[ds].copy(); paths, ok = pa.resolve_paths(df, os.path.join(base, sub), ".png")
        assert ok.all(), f"{ds}: {(~ok).sum()} gambar tidak ditemukan di {base}/{sub}"
        df["path"] = paths; dfs[ds] = df
    train_df, val_df = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        dfs["APTOS"], test_size=0.2, random_state=SEED, stratify=dfs["APTOS"].label_num)
    print(f"[{model_id}] folder={base} train={len(train_df)} val={len(val_df)}", flush=True)

    augment = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0), tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)), tf.keras.layers.RandomContrast(0.2)], name="augment")
    # decode + resize sekali lalu di-cache di RAM (2929 x 224 x 224 x 3 float32 = 1,8 GB);
    # augmentasi dan preprocess_input diterapkan setelah cache sehingga hasilnya identik dengan notebook.
    def decode_resize(path, label):
        return tf.image.resize(pa.decode_any(path), (IMG_SIZE, IMG_SIZE)), label
    def finish_eval(img, label): return preprocess_input(img), label
    def finish_train(img, label):
        if a.aug: img = augment(img, training=True)
        return preprocess_input(img), label
    def ds_of(df, train=False):
        ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label_num.values)).map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE).cache()
        if train: ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True).map(finish_train, num_parallel_calls=tf.data.AUTOTUNE)
        else: ds = ds.map(finish_eval, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    backbone = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)); backbone.trainable = True
    model = tf.keras.Sequential([backbone, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    save_path = P("V2", "Model", f"({tag})resnet50_p_softmax.keras"); log_path = P("V2", "Logs", f"({tag})resnet50_p_softmax.csv")
    backup_dir = P("V2", "Backup", f"resnet50_{tag}"); shutil.rmtree(backup_dir, ignore_errors=True)
    for p in (save_path, log_path): os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(log_path): os.remove(log_path)
    cbs = [tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=0),
           tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_delta=1e-4, min_lr=1e-6, verbose=0),
           tf.keras.callbacks.BackupAndRestore(backup_dir), tf.keras.callbacks.CSVLogger(log_path, append=True)]
    t0 = time.time()
    model.fit(ds_of(train_df, train=True), validation_data=ds_of(val_df), epochs=a.epochs, callbacks=cbs, verbose=2)
    log = pd.read_csv(log_path); b = int(log.val_loss.idxmin())
    print(f"[{model_id}] training {time.time()-t0:.0f}s, epoch {len(log)}, best {b+1}: val_loss {log.val_loss[b]:.4f} val_acc {log.val_accuracy[b]:.4f}", flush=True)

    outdir = os.path.join(a.out, model_id); os.makedirs(outdir, exist_ok=True)
    from sklearn.metrics import roc_auc_score
    for name, df, fname in (("Val-APTOS", val_df, "pred_val.csv"), ("IDRID", dfs["IDRID"], "pred_idrid.csv"), ("Messidor", dfs["Messidor"], "pred_messidor.csv")):
        prob = model.predict(ds_of(df), verbose=0)[:, 1]
        pd.DataFrame({"id_code": df.id_code.values, "diagnosis": df.grade.values, "y_true": df.label_num.values, "y_prob": prob}).to_csv(os.path.join(outdir, fname), index=False)
        print(f"[{model_id}] {name:9s} n={len(df)} acc={((prob>=0.5).astype(int)==df.label_num.values).mean():.4f} auroc={roc_auc_score(df.label_num, prob):.4f}", flush=True)
    print(f"[{model_id}] SELESAI", flush=True)

if __name__ == "__main__":
    main()
