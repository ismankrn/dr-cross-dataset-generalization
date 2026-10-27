#!/usr/bin/env python3
"""
gradcam_f7.py — bahan gambar F7: Grad-CAM untuk model terbaik pada tiga dataset.
- CNN (DenseNet121 / EfficientNet-B0 / ResNet50): Grad-CAM klasik pada peta fitur konv terakhir.
- ViT-B/16: Grad-CAM pada token keluaran blok encoder ke-11 (blok terakhir tidak dipakai karena
  prediksi hanya membaca token CLS, sehingga gradien ke token patch di keluaran akhir nol);
  bobot per kanal = rata-rata gradien atas token patch, CAM = ReLU(sum_c w_c * token_c) -> 14x14.
Untuk tiap dataset dipilih 3 TP, 3 FP, 3 FN berkeyakinan tertinggi (dari pred_*.csv),
disimpan overlay per gambar + satu lembar grid 3x3 per dataset, plus CSV daftar gambar.

Pemakaian: python gradcam_f7.py --root /data/dr --model VIT-P-RAW [--model D121-P-RAW] --out gradcam_out
"""
import argparse, os, sys, numpy as np, pandas as pd

IMG = 224

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--pred-root", default="predictions"); ap.add_argument("--out", default="gradcam_out"); ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import predict_all as pa
    import tensorflow as tf, cv2
    man = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_manifest.csv")).fillna("")
    labels = pa.load_labels(a.root); labels["Val-APTOS"] = pa.val_split(labels["APTOS"])
    DS = {"Val-APTOS": ("folder_aptos", "ext_aptos", "pred_val.csv"), "IDRID": ("folder_idrid", "ext_idrid", "pred_idrid.csv"), "Messidor": ("folder_messidor", "ext_messidor", "pred_messidor.csv")}

    for mid in a.model:
        m = man[man.model_id == mid].iloc[0]
        model = tf.keras.models.load_model(os.path.join(a.root, m.model_file), compile=False)
        if m.backbone == "vit": import keras_hub  # noqa
        head = model.layers[2:]          # [Dropout, Dense] untuk CNN Sequential; ViT: [dropout, dense] juga di indeks 2..3
        if m.backbone == "vit":
            bb = model.layers[1]; pe = bb.get_layer("vit_patching_and_embedding"); enc = bb.get_layer("vit_encoder")
            dense = model.layers[-1]
            def cam_and_prob(x):
                with tf.GradientTape() as tape:
                    t = pe(x)
                    for blk in enc.encoder_layers[:-1]: t = blk(t)
                    tape.watch(t)
                    z = enc.encoder_layers[-1](t); z = enc.layer_norm(z)
                    out = dense(z[:, 0, :]); prob_dr = out[:, 1] if out.shape[-1] == 2 else out[:, 0]
                g = tape.gradient(prob_dr, t)[:, 1:, :]           # gradien token patch (B,196,768)
                tok = t[:, 1:, :]; w = tf.reduce_mean(g, axis=1, keepdims=True)
                cam = tf.nn.relu(tf.reduce_sum(w * tok, axis=-1))  # (B,196)
                return tf.reshape(cam, (-1, 14, 14)).numpy(), prob_dr.numpy()
        else:
            base = model.layers[0]; gap, drop, dense = model.layers[1], model.layers[2], model.layers[3]
            def cam_and_prob(x):
                with tf.GradientTape() as tape:
                    f = base(x, training=False); tape.watch(f)
                    out = dense(drop(gap(f), training=False)); prob_dr = out[:, 1] if out.shape[-1] == 2 else out[:, 0]
                g = tape.gradient(prob_dr, f); w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
                cam = tf.nn.relu(tf.reduce_sum(w * f, axis=-1))
                return cam.numpy(), prob_dr.numpy()
        load = pa.make_loader(m.backbone)

        rows = []
        for ds, (fcol, ecol, pfile) in DS.items():
            pred = pd.read_csv(os.path.join(a.pred_root, mid, pfile)); pred["y_pred"] = (pred.y_prob >= 0.5).astype(int)
            pred["conf"] = np.abs(pred.y_prob - 0.5)
            cats = {"TP": (1, 1), "FP": (0, 1), "FN": (1, 0), "TN": (0, 0)}
            df = labels[ds]; folder = os.path.join(a.root, m[fcol]); ext = m[ecol] if m[ecol] else ""
            paths, ok = pa.resolve_paths(df, folder, ext); pmap = dict(zip(df.id_code.astype(str), paths))
            tiles = {}
            for cat, (yt, yp) in cats.items():
                sel = pred[(pred.y_true == yt) & (pred.y_pred == yp)].sort_values("conf", ascending=False).head(a.k)
                for i, r in enumerate(sel.itertuples()):
                    path = pmap.get(str(r.id_code));
                    if not path: continue
                    x = tf.expand_dims(load(tf.constant(path)), 0)
                    cam, prob = cam_and_prob(x); cam = cam[0]; cam = cam / (cam.max() + 1e-8)
                    heat = cv2.resize(cam, (IMG, IMG)); heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]
                    raw = pa.decode_any(tf.constant(path)).numpy(); raw = cv2.resize(raw, (IMG, IMG))
                    over = np.uint8(0.6 * raw + 0.4 * heat)
                    cv2.putText(over, f"{cat} p={float(prob[0]):.2f} g{int(r.diagnosis)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    od = os.path.join(a.out, mid, ds); os.makedirs(od, exist_ok=True)
                    fn = f"{cat}_{i+1}_{r.id_code}.png".replace("/", "_"); cv2.imwrite(os.path.join(od, fn), np.hstack([raw, over])[:, :, ::-1])
                    tiles.setdefault(cat, []).append(over); rows.append(dict(model=mid, dataset=ds, cat=cat, id_code=r.id_code, grade=int(r.diagnosis), y_prob=float(r.y_prob), file=fn))
            # grid 3x3: baris TP, FP, FN
            grid = []
            for cat in ("TP", "FP", "FN"):
                row = tiles.get(cat, [])[:a.k]
                while len(row) < a.k: row.append(np.zeros((IMG, IMG, 3), np.uint8))
                grid.append(np.hstack(row))
            cv2.imwrite(os.path.join(a.out, mid, f"F7_grid_{ds}.png"), np.vstack(grid)[:, :, ::-1])
            print(f"[{mid}] {ds}: grid {len(rows)} tile tersimpan", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, mid, "gradcam_index.csv"), index=False)
        del model; tf.keras.backend.clear_session()
    print("GRADCAM SELESAI", flush=True)

if __name__ == "__main__":
    main()
