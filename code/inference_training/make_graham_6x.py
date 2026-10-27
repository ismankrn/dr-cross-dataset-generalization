#!/usr/bin/env python3
"""
make_graham_6x.py — buat ulang 6A dan 6B dengan implementasi Graham yang SAMA dengan 3A, 5A, 5B
(fungsi ben_graham_normalize dari notebook "contrast normalization.ipynb", disalin apa adanya).

  6A baru = Graham(4A)   (4A = circular crop + CLAHE)      sejajar dengan 5A = Graham(1A)
  6B baru = Graham(4B)   (4B = box crop + CLAHE)           sejajar dengan 5B = Graham(1B)

Pemakaian:
  # 1) verifikasi: Graham(1a) harus identik piksel demi piksel dengan 5a yang tersimpan
  python make_graham_6x.py --verify "<root>/Preprocessed image/Circular Crop/1a/APTOS" "<root>/Preprocessed image/Circular Crop/5a/APTOS"
  # 2) buat folder baru
  python make_graham_6x.py --src "<root>/Preprocessed image/Circular Crop/4a" --dst "<root>/Preprocessed image/Circular Crop/6a_g"
  python make_graham_6x.py --src "<root>/Preprocessed image/Box Crop/4b"     --dst "<root>/Preprocessed image/Box Crop/6b_g"
"""
import argparse, os, sys
import numpy as np, cv2

VALID = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def graham(image, target_size=512, crop_thresh=10, sigmaX_ratio=0.03):
    # ---- core logic, identik dengan ben_graham_normalize di notebook ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > crop_thresh
    if np.any(mask):
        x_idx, y_idx = np.where(mask)
        image = image[np.min(x_idx):np.max(x_idx), np.min(y_idx):np.max(y_idx)]
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    pad_len = int(target_size * 0.1)
    padded = cv2.copyMakeBorder(resized, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_REFLECT)
    sigmaX = max(1, int(target_size * sigmaX_ratio))
    blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX)
    normalized_padded = cv2.addWeighted(padded, 4, blurred, -4, 128)
    normalized = normalized_padded[pad_len:-pad_len, pad_len:-pad_len]
    return np.clip(normalized, 0, 255).astype(np.uint8)

def run_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(VALID)); fail = []
    for i, f in enumerate(files):
        try:
            img = cv2.imread(os.path.join(src, f))
            if img is None: raise ValueError("gagal dibaca")
            cv2.imwrite(os.path.join(dst, f), graham(img))
        except Exception as e:
            fail.append((f, str(e)))
        if (i + 1) % 500 == 0: print(f"  {dst}: {i+1}/{len(files)}", flush=True)
    print(f"{dst}: {len(files) - len(fail)}/{len(files)} OK", flush=True)
    return fail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", nargs=2, metavar=("SRC_1A", "REF_5A"))
    ap.add_argument("--src"); ap.add_argument("--dst")
    a = ap.parse_args()
    if a.verify:
        src, ref = a.verify; n = same = 0; worst = 0
        for f in sorted(os.listdir(ref)):
            if not os.path.exists(os.path.join(src, f)): continue
            out = graham(cv2.imread(os.path.join(src, f))); r = cv2.imread(os.path.join(ref, f))
            n += 1; d = int(np.abs(out.astype(int) - r.astype(int)).max()) if out.shape == r.shape else 999
            same += d == 0; worst = max(worst, d)
        print(f"VERIFIKASI: {same}/{n} identik, selisih piksel maksimum {worst}")
        sys.exit(0 if n and same == n else 1)
    fails = []
    for sub in ("APTOS", "IDRID", "MESSIDOR"):
        fails += run_dir(os.path.join(a.src, sub), os.path.join(a.dst, sub))
    print("GAGAL:", fails[:5] if fails else "tidak ada")

if __name__ == "__main__":
    main()
