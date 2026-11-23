"""
Step 8: quick evaluation. PSNR and SSIM on the test set, per exposure, for the
blurry input and every model that has predictions in preds/.

Ground truth is synth/.../sharp (it has the same gain as the blurry image),
never the original TartanAir frame. A border of BORDER px is ignored on every
side, because the warp's border padding makes that band unreliable.

Run:  python evaluate.py            (after: python train.py --model <m> --predict)
Writes: outputs/eval_<split>.csv     (one row per image and model)
        prints a table: model x exposure (mean PSNR / SSIM)

Needs scikit-image for SSIM:  pip install scikit-image
"""
import csv
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_ROOT = os.path.join(PROJECT_ROOT, "synth")
PREDS_ROOT = os.path.join(PROJECT_ROOT, "preds")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

SPLIT = "test"
MODELS = ["blurry", "ours", "nafnet"]   # "blurry" = the input itself; missing models are skipped
BORDER = 32


def load(path):
    """PNG -> float64 (H, W, 3) in [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def crop_border(img, b=BORDER):
    return img[b:-b, b:-b]


def psnr(pred, gt):
    """Peak signal-to-noise ratio in dB, images in [0, 1]. Capped at 100 dB."""
    mse = max(np.mean((pred - gt) ** 2), 1e-10)
    return 10 * np.log10(1.0 / mse)


def ssim(pred, gt):
    """Structural similarity, colour images in [0, 1]."""
    return structural_similarity(pred, gt, channel_axis=-1, data_range=1.0)


def pred_path(model, row):
    """Where the image for this model and test row lives."""
    if model == "blurry":
        return os.path.join(SYNTH_ROOT, row["blur"])
    return os.path.join(PREDS_ROOT, model, row["blur"])


def evaluate():
    """PSNR and SSIM for every test image and model. Saves outputs/eval_<split>.csv."""
    index_path = os.path.join(SYNTH_ROOT, SPLIT, "index.csv")
    with open(index_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    models = [m for m in MODELS
              if m == "blurry" or os.path.isdir(os.path.join(PREDS_ROOT, m))]
    print(f"evaluating {models} on {len(rows)} {SPLIT} images", flush=True)

    results = []
    for i, row in enumerate(rows, start=1):
        gt = crop_border(load(os.path.join(SYNTH_ROOT, row["sharp"])))
        for model in models:
            pred = crop_border(load(pred_path(model, row)))
            results.append({
                "model": model,
                "env": row["env"],
                "traj": row["traj"],
                "frame": row["frame"],
                "k": row["k"],
                "T": float(row["T"]),
                "extent_p95": float(row["extent_p95"]),
                "psnr": psnr(pred, gt),
                "ssim": ssim(pred, gt),
            })
        if i % 100 == 0:
            print(f"Evaluated {i}/{len(rows)} rows", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    output_path = os.path.join(OUT_DIR, f"eval_{SPLIT}.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved evaluation results to {output_path}")
    return results


def summarize(results):
    """
    Table of mean PSNR / SSIM: rows = models, columns = each exposure T and "all".
    Then the gain of every model over the blurry input.
    """
    models = [m for m in MODELS if any(r["model"] == m for r in results)]
    Ts = sorted({r["T"] for r in results})
    cols = [f"T={T:g}" for T in Ts] + ["all"]

    def mean_of(model, T=None):
        sel = [r for r in results if r["model"] == model and (T is None or r["T"] == T)]
        return np.mean([r["psnr"] for r in sel]), np.mean([r["ssim"] for r in sel])

    table = {m: [mean_of(m, T) for T in Ts] + [mean_of(m)] for m in models}
    n_img = sum(1 for r in results if r["model"] == models[0])

    print(f"\nPSNR (dB) / SSIM, {n_img} {SPLIT} images, {len(Ts)} exposures")
    print(f"{'model':10s}" + "".join(f"{c:18s}" for c in cols))
    for m in models:
        print(f"{m:10s}" + "".join(f"{p:6.2f} / {s:.3f}    " for p, s in table[m]))

    if "blurry" in table and len(models) > 1:
        print("\ngain over blurry")
        for m in models:
            if m == "blurry":
                continue
            gains = [(p - pb, s - sb) for (p, s), (pb, sb) in zip(table[m], table["blurry"])]
            print(f"{m:10s}" + "".join(f"{dp:+6.2f} / {ds:+.3f}   " for dp, ds in gains))


if __name__ == "__main__":
    summarize(evaluate())