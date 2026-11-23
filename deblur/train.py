"""
Train ours or NAFNet on synth/, then write test predictions.

  python train.py --model ours --quick      # few-minute sanity run: fits in memory? loss drops?
  python train.py --model ours              # full run (resumes from runs/ours/last.pt if present)
  python train.py --model ours --predict    # best checkpoint -> preds/ours/<split>/...

Outputs:
  runs/<model>/best.pt, last.pt, log.csv
  preds/<model>/<index "blur" path>        (see HANDOFF.md section 6)
  preds/blurry/<index "blur" path>         (the input itself, as the no-processing reference)
"""

import argparse
import csv
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import build_model, BlurOperator, crop_K

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_ROOT = os.path.join(PROJECT_ROOT, "synth")
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
PREDS_ROOT = os.path.join(PROJECT_ROOT, "preds")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CFG = dict(
    crop=256,
    margin=32,          # crops stay this far from the image border (border-padding band)
    batch=4,
    lr=1e-3, betas=(0.9, 0.9), weight_decay=0.0,
    lr_min=1e-7,        # cosine schedule end
    iters=30000,        # set from the s/it the --quick run reports
    log_every=100,
    val_every=2000,
    max_val=None,       # None = whole val split
    val_border=32,      # PSNR ignores this many px at each border
    time_limit_h=10,    # stops cleanly before this and saves last.pt
    num_workers=2,      # Windows: set 0 if the DataLoader errors
    amp=True,           # mixed precision (CUDA only)
    grad_clip=1.0,
    seed=0,
)
QUICK = dict(iters=300, log_every=20, val_every=100, max_val=20)


# ---------------------------------------------------------------------------
# Colour (torch versions of blur.py's)
# ---------------------------------------------------------------------------
def srgb_to_linear_t(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_t(x):
    x = x.clamp(1e-8, 1.0)            # pow on negatives would give NaN gradients
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_png(path):
    """PNG -> float32 tensor (3, H, W) in [0, 1]."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class SynthDataset(Dataset):
    """Rows of synth/<split>/index.csv.

    Returns a dict:
      blur, sharp  (3, h, w) sRGB in [0, 1]
      depth        (h, w)
      v, omega     (3,)
      T            scalar
      K            (3, 3) intrinsics of THIS crop
      idx          row index (to find the output path in predict)
    train: random crop, top-left at least `margin` from the border.
    eval:  full frame. No flips: a flip would also need v, omega and K mirrored.
    """

    def __init__(self, split, train, crop=256, margin=32, max_items=None):
        path = os.path.join(SYNTH_ROOT, split, "index.csv")
        with open(path, newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        if max_items:
            self.rows = self.rows[:max_items]
        self.train, self.crop, self.margin = train, crop, margin

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        blur = load_png(os.path.join(SYNTH_ROOT, row["blur"]))
        sharp = load_png(os.path.join(SYNTH_ROOT, row["sharp"]))
        m = np.load(os.path.join(SYNTH_ROOT, row["meta"]))
        depth = torch.from_numpy(m["depth"].astype(np.float32))
        K = torch.from_numpy(m["K"].astype(np.float32))

        if self.train:
            H, W = depth.shape
            c, g = self.crop, self.margin
            x0 = int(np.random.randint(g, W - g - c + 1))
            y0 = int(np.random.randint(g, H - g - c + 1))
            blur = blur[:, y0:y0 + c, x0:x0 + c]
            sharp = sharp[:, y0:y0 + c, x0:x0 + c]
            depth = depth[y0:y0 + c, x0:x0 + c]
            K = crop_K(K, x0, y0)

        return dict(blur=blur, sharp=sharp, depth=depth, K=K,
                    v=torch.from_numpy(m["v"].astype(np.float32)),
                    omega=torch.from_numpy(m["omega"].astype(np.float32)),
                    T=torch.tensor(float(m["T"]), dtype=torch.float32),
                    idx=i)


def worker_seed(worker_id):
    """Different random crops in every DataLoader worker."""
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


# ---------------------------------------------------------------------------
# Forward pass shared by train / val / predict
# ---------------------------------------------------------------------------
def run_model(model, name, batch, device, amp):
    """Returns the prediction in sRGB, (B, 3, h, w), float32."""
    blur = batch["blur"].to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        if name == "ours":
            with torch.autocast(device.type, enabled=False):     # geometry in fp32
                op = BlurOperator(batch["depth"].to(device), batch["v"].to(device),
                                  batch["omega"].to(device), batch["T"].to(device),
                                  batch["K"].to(device))
            y_lin = srgb_to_linear_t(blur)
            out = linear_to_srgb_t(model(y_lin, op).float())
        else:
            out = model(blur)
    return out.float()


def psnr(pred, target, border):
    """Mean of per-image PSNRs, ignoring `border` px on each side."""
    if border:
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    mse = ((pred.clamp(0, 1) - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()


@torch.no_grad()
def validate(model, name, loader, device, cfg):
    """Returns (model PSNR, blurry-input PSNR) over the loader."""
    model.eval()
    p_model, p_blur, n = 0.0, 0.0, 0
    for batch in loader:
        sharp = batch["sharp"].to(device)
        out = run_model(model, name, batch, device, cfg["amp"])
        b = sharp.shape[0]
        p_model += psnr(out, sharp, cfg["val_border"]) * b
        p_blur += psnr(batch["blur"].to(device), sharp, cfg["val_border"]) * b
        n += b
    model.train()
    return p_model / n, p_blur / n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def cosine_lr(it, cfg):
    t = min(it / cfg["iters"], 1.0)
    return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * t))


def train(name, quick):
    cfg = dict(CFG, **(QUICK if quick else {}))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["amp"] = cfg["amp"] and device.type == "cuda"
    run_dir = os.path.join(RUNS_ROOT, name + ("_quick" if quick else ""))
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {name}: {n_params / 1e6:.3f}M params, device {device}, amp {cfg['amp']}")

    train_ds = SynthDataset("train", True, cfg["crop"], cfg["margin"])
    val_ds = SynthDataset("val", False, max_items=cfg["max_val"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True, drop_last=True,
                          num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                          worker_init_fn=worker_seed, persistent_workers=cfg["num_workers"] > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=cfg["betas"],
                            weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"])

    # Resume
    it, best = 0, -float("inf")
    last_path, best_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "best.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); it, best = ck["it"], ck["best"]
        print(f"resumed from {last_path} at iteration {it}")

    def save(path):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        it=it, best=best, cfg=cfg, name=name), path)

    log_path = os.path.join(run_dir, "log.csv")
    log_new = not os.path.exists(log_path)
    log_fh = open(log_path, "a", newline="")
    log = csv.writer(log_fh)
    if log_new:
        log.writerow(["it", "loss", "lr", "val_psnr", "blur_psnr", "alphas", "s_per_it", "max_mem_gb"])

    model.train()
    t_start, t_last, loss_sum, loss_n = time.time(), time.time(), 0.0, 0
    blur_psnr = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while it < cfg["iters"]:
        for batch in train_dl:
            if it >= cfg["iters"]:
                break
            lr = cosine_lr(it, cfg)
            for g in opt.param_groups:
                g["lr"] = lr

            sharp = batch["sharp"].to(device, non_blocking=True)
            out = run_model(model, name, batch, device, cfg["amp"])
            loss = F.l1_loss(out, sharp)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at iteration {it}; "
                                   "check python model.py passes, then try a lower lr")

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            it += 1
            loss_sum += loss.item(); loss_n += 1

            if it % cfg["log_every"] == 0:
                s_it = (time.time() - t_last) / cfg["log_every"]
                t_last = time.time()
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                alphas = (" ".join(f"{a:.3f}" for a in model.alphas().tolist())
                          if name == "ours" else "")
                print(f"it {it:6d}  loss {loss_sum / loss_n:.4f}  lr {lr:.2e}  "
                      f"{s_it:.3f} s/it  mem {mem:.2f} GB" + (f"  alphas {alphas}" if alphas else ""),
                      flush=True)
                log.writerow([it, loss_sum / loss_n, lr, "", "", alphas, s_it, mem])
                log_fh.flush()
                loss_sum, loss_n = 0.0, 0

            out_of_time = (time.time() - t_start) / 3600 > cfg["time_limit_h"]
            if it % cfg["val_every"] == 0 or it == cfg["iters"] or out_of_time:
                val_psnr, blur_psnr = validate(model, name, val_dl, device, cfg)
                improved = val_psnr > best
                best = max(best, val_psnr)
                print(f"  val PSNR {val_psnr:.2f} dB  (blurry input {blur_psnr:.2f} dB)"
                      f"{'  * best' if improved else ''}", flush=True)
                log.writerow([it, "", lr, val_psnr, blur_psnr, "", "", ""])
                log_fh.flush()
                if improved:
                    save(best_path)
                save(last_path)
            if out_of_time:
                print("time limit reached; saved last.pt (rerun the same command to resume)")
                log_fh.close()
                return

    log_fh.close()
    print(f"done: best val PSNR {best:.2f} dB -> {best_path}")


@torch.no_grad()
def predict(name, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = os.path.join(RUNS_ROOT, name, "best.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_model(name).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    amp = ck["cfg"]["amp"] and device.type == "cuda"
    print(f"loaded {ck_path} (iteration {ck['it']}, val PSNR {ck['best']:.2f} dB)")

    ds = SynthDataset(split, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=CFG["num_workers"])
    for n, batch in enumerate(dl):
        row = ds.rows[int(batch["idx"][0])]
        out = run_model(model, name, batch, device, amp)[0].clamp(0, 1)
        out8 = (out.cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        dst = os.path.join(PREDS_ROOT, name, row["blur"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(out8).save(dst)

        ref = os.path.join(PREDS_ROOT, "blurry", row["blur"])          # input, for reference
        if not os.path.exists(ref):
            os.makedirs(os.path.dirname(ref), exist_ok=True)
            shutil.copyfile(os.path.join(SYNTH_ROOT, row["blur"]), ref)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(ds)}", flush=True)
    print(f"wrote {len(ds)} predictions to {os.path.join(PREDS_ROOT, name, split)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ours", "nafnet"], required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    if args.predict:
        predict(args.model, args.split)
    else:
        train(args.model, args.quick)