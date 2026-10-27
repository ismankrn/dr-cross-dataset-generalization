#!/usr/bin/env python3
"""
train_r50_5a_aug.py — eksperimen tambahan R50-P-5A-AUG.
Konfigurasi IDENTIK dengan notebook V2/ResNet50/(5a)Resnet50 Softmax Pretrained.ipynb
(ResNet50 ImageNet, full fine-tuning, Dense(2, softmax), SGD 1e-4 momentum 0.9 nesterov,
batch 32, 224 px, early stopping patience 10 pada val_loss, ReduceLROnPlateau 0.5/4,
seed 42, split 80:20 random_state 42) + augmentasi HANYA pada train_dataset.

Pemakaian:
  python train_r50_5a_aug.py --root /data/dr [--epochs 100] [--no-aug]
--no-aug melatih konfigurasi 5A tanpa augmentasi (untuk memverifikasi skrip ini
mereproduksi hasil notebook 5A: val acc ~0.9686, best epoch ~13).
Output: V2/Model/(5a_aug)resnet50_p_softmax.keras, V2/Logs/(5a_aug)resnet50_p_softmax.csv,
        predictions/R50-P-5A-AUG/pred_{val,idrid,messidor}.csv
"""
import argparse, os, shutil, time
import numpy as np, pandas as pd

IMG_SIZE, BATCH, SEED = 224, 32, 42

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--no-aug", action="store_true"); ap.add_argument("--out", default="predictions")
    a = ap.parse_args()
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from sklearn.model_selection import train_test_split
    tf.keras.utils.set_random_seed(SEED)

    tag = "5a_noaug_check" if a.no_aug else "5a_aug"
    model_id = "R50-P-5A-NOAUG-CHECK" if a.no_aug else "R50-P-5A-AUG"
    P = lambda *x: os.path.join(a.root, *x)
    prep = {"APTOS": P("Preprocessed image", "Circular Crop", "5a", "APTOS"),
            "IDRID": P("Preprocessed image", "Circular Crop", "5a", "IDRID"),
            "Messidor": P("Preprocessed image", "Circular Crop", "5a", "MESSIDOR")}

    # ---- label (sama dengan notebook)
    aptos = pd.read_csv(P("aptos_dataset", "train.csv")); aptos["label_num"] = (aptos.diagnosis > 0).astype(int)
    aptos["path"] = [os.path.join(prep["APTOS"], f"{i}.png") for i in aptos.id_code]
    idrid = pd.read_csv(P("idrid_dataset", "idrid_labels.csv"))[["id_code", "diagnosis"]]; idrid["label_num"] = (idrid.diagnosis > 0).astype(int)
    idrid["path"] = [os.path.join(prep["IDRID"], f"{i}.png") for i in idrid.id_code]
    mess = pd.read_csv(P("messidor_dataset", "messidor_labels.csv")); mess["id_code"] = mess["Image name"]
    mess["diagnosis"] = mess["Retinopathy grade"]; mess["label_num"] = (mess.diagnosis > 0).astype(int)
    mess["path"] = [os.path.join(prep["Messidor"], f"{i}.png") for i in mess["Image name"]]
    for name, df in (("APTOS", aptos), ("IDRID", idrid), ("Messidor", mess)):
        miss = (~df.path.apply(os.path.exists)).sum(); assert miss == 0, f"{name}: {miss} gambar tidak ditemukan di folder 5a"
    train_df, val_df = train_test_split(aptos, test_size=0.2, random_state=SEED, stratify=aptos.label_num)
    print("train", len(train_df), "val", len(val_df))

    # ---- pipeline
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augment")

    def decode(path):
        img = tf.image.decode_png(tf.io.read_file(path), channels=3)
        return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    def load_eval(path, label): return preprocess_input(decode(path)), label
    def load_train(path, label):
        img = decode(path)
        if not a.no_aug: img = augment(img, training=True)   # skala 0-255, sebelum preprocess_input
        return preprocess_input(img), label
    def ds_of(df, fn, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label_num.values)).map(fn, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle: ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    train_ds, val_ds = ds_of(train_df, load_train, shuffle=True), ds_of(val_df, load_eval)

    # ---- model (identik dengan notebook 5A)
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)); base.trainable = True
    model = tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(2, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    save_path = P("V2", "Model", f"({tag})resnet50_p_softmax.keras"); log_path = P("V2", "Logs", f"({tag})resnet50_p_softmax.csv")
    backup_dir = P("V2", "Backup", f"resnet50_p_softmax_{tag}")
    shutil.rmtree(backup_dir, ignore_errors=True)                    # jangan lanjut dari run lama
    for p in (save_path, log_path): os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(log_path): os.remove(log_path)
    cbs = [tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1),
           tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, restore_best_weights=True, verbose=1),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_delta=1e-4, min_lr=1e-6, verbose=1),
           tf.keras.callbacks.BackupAndRestore(backup_dir), tf.keras.callbacks.CSVLogger(log_path, append=True)]
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs)
    print(f"training selesai {time.time()-t0:.0f}s")

    # ---- evaluasi + prediksi per gambar
    log = pd.read_csv(log_path); b = log.val_loss.idxmin()
    print(f"best epoch {b+1}: val_loss {log.val_loss[b]:.4f} val_acc {log.val_accuracy[b]:.4f}  (isi ke sheet Models)")
    outdir = os.path.join(a.out, model_id); os.makedirs(outdir, exist_ok=True)
    for name, df, fname in (("Val-APTOS", val_df, "pred_val.csv"), ("IDRID", idrid, "pred_idrid.csv"), ("Messidor", mess, "pred_messidor.csv")):
        prob = model.predict(ds_of(df, load_eval), verbose=0)[:, 1]
        pd.DataFrame({"id_code": df.id_code.values, "diagnosis": df.diagnosis.values, "y_true": df.label_num.values, "y_prob": prob}).to_csv(os.path.join(outdir, fname), index=False)
        from sklearn.metrics import roc_auc_score
        print(f"{name:9s} n={len(df)} acc={((prob>=0.5).astype(int)==df.label_num.values).mean():.4f} auroc={roc_auc_score(df.label_num, prob):.4f}")

if __name__ == "__main__":
    main()
