"""
Dataset synthesis: sharp TartanAir frames -> blurry/sharp pairs + metadata,
written straight into train / val / test folders.

Reuses:
  flow_test.py   geometry (backproject, transform_points, project)  -- validated
  velocities.py  pose loading, NED -> optical, velocities
  blur.py        colour conversion, warping, averaging, noise

Adds:
  - splits by trajectory / environment (never by random frames)
  - per-sample randomization: exposure T, shot/read noise, gain
  - motion along the TRUE camera path (poses t-1, t, t+1) instead of
    constant velocity, so the synthesis differs from the network's operator

Output:
  synth/<split>/<env>_<diff>_<traj>/blur/<frame>_<k>.png
  synth/<split>/<env>_<diff>_<traj>/sharp/<frame>_<k>.png   (same gain as blur)
  synth/<split>/<env>_<diff>_<traj>/meta/<frame>_<k>.npz   (v, omega, T, depth, K, ...)
  synth/<split>/index.csv   (paths relative to SYNTH_ROOT, so the folder can be moved)

Run:  python synthesize.py
Safe to stop and re-run: finished samples are skipped.
"""

import csv
import glob
import os
import time
import zlib

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from flow_test import K, H, W, DEPTH_CLAMP, backproject, transform_points, project, pixel_grid
from velocities import load_poses, ned_to_optical, velocities
from blur import (srgb_to_linear, linear_to_srgb, to_sampling_grid, blur_linear,
                  saturate_and_noise, load_rgb)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All paths are relative to the project folder (the parent of deblur/),
# so the whole GroupProj folder can be copied to another machine as is.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SYNTH_ROOT = os.path.join(PROJECT_ROOT, "synth")

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_ENVS = ["office", "hospital", "carwelding", "abandonedfactory"]
TEST_ENVS = ["japanesealley"]
DIFFICULTIES = ["Hard"]

TRAIN_TRAJ_PER_ENV = 5          # first 5 trajectories of each env -> train (carwelding has only 3 + val)
                                # last trajectory of each env      -> val

SPLITS = {
    "train": {"stride": 8, "exposures": "random", "n_per_frame": 1},
    "val":   {"stride": 8, "exposures": [0.5]},                  # checkpoint selection only
    "test":  {"stride": 8, "exposures": [0.25, 0.5, 1.0]},
}

# ---------------------------------------------------------------------------
# Randomization ranges
# ---------------------------------------------------------------------------
T_RANGE = (0.1, 1.0)            # exposure in frames (train)
MIN_BLUR_PX = 6.0               # train only: raise T (up to T_RANGE[1]) until the blur
                                # reaches this many px (99th pct displacement from
                                # mid-exposure, so the streak is about twice as long)
SHOT_RANGE = (1e-5, 1e-3)       # log-uniform
READ_RANGE = (5e-4, 3e-3)       # log-uniform
GAIN_RANGE = (1.0, 1.3)         # >1 lets highlights saturate

N_SUB = 64                      # sub-exposures
DEPTH_MIN = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def list_trajectories(envs):
    """All (env, diff, traj, seq_dir) on disk. Missing envs are skipped."""
    out = []
    for env in envs:
        for diff in DIFFICULTIES:
            for seq_dir in sorted(glob.glob(f"{DATA_ROOT}/{env}/{diff}/P*")):
                if os.path.exists(f"{seq_dir}/pose_left.txt"):
                    out.append((env, diff, os.path.basename(seq_dir), seq_dir))
    return out


def assign_splits():
    """Train/val from TRAIN_ENVS (by trajectory), test = TEST_ENVS entirely."""
    splits = {"train": [], "val": [], "test": list_trajectories(TEST_ENVS)}
    groups = {}
    for item in list_trajectories(TRAIN_ENVS):
        groups.setdefault((item[0], item[1]), []).append(item)
    for group in groups.values():
        if len(group) == 1:
            splits["train"] += group
            continue
        splits["val"].append(group[-1])
        splits["train"] += group[:-1][:TRAIN_TRAJ_PER_ENV]
    return splits


# ---------------------------------------------------------------------------
# Motion along the true camera path
# ---------------------------------------------------------------------------
def neighbor_relative_poses(T_opt, t):
    """Poses of cameras t-1 and t+1 expressed in camera t's frame."""
    inv_t = np.linalg.inv(T_opt[t])
    return inv_t @ T_opt[t - 1], inv_t @ T_opt[t + 1]


def partial_pose(P, f):
    """Fraction f (0..1) of the way from the identity to pose P."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(f * Rotation.from_matrix(P[:3, :3]).as_rotvec()).as_matrix()
    out[:3, 3] = f * P[:3, 3]
    return out


def interpolated_pose(P_prev, P_next, tau):
    """Camera pose at time tau (frames) relative to camera t, on the true path."""
    return partial_pose(P_next, tau) if tau >= 0 else partial_pose(P_prev, -tau)


def displacement_fields_traj(Z, P_prev, P_next, T, N):
    """Per-pixel displacement at N instants in [-T/2, T/2]. Returns (N, H, W, 2)."""
    X = backproject(Z, K)
    u, v_pix = pixel_grid()
    p = np.stack([u, v_pix], axis=-1)
    d = np.zeros((N, H, W, 2), dtype=np.float32)
    for n, tau in enumerate(np.linspace(-T / 2, T / 2, N)):
        d[n] = project(transform_points(X, interpolated_pose(P_prev, P_next, tau)), K) - p
    return d


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------
def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def blur_at_T1(Z, P_prev, P_next):
    """Blur size (px, 99th pct displacement from mid-exposure) if T were 1.0.
    Blur grows about linearly with T, so the T needed for a target blur is target / this."""
    d = displacement_fields_traj(Z, P_prev, P_next, 1.0, 2)      # tau = -0.5 and +0.5
    return float(np.nanpercentile(np.linalg.norm(d, axis=-1), 99))


def sample_params(rng, cfg, k, b1=None):
    if cfg["exposures"] == "random":
        T = float(rng.uniform(*T_RANGE))
        if b1 is not None and b1 > 0:                             # minimum blur, train only
            T = min(max(T, MIN_BLUR_PX / b1), T_RANGE[1])
    else:
        T = float(cfg["exposures"][k])
    return dict(T=T,
                shot=log_uniform(rng, *SHOT_RANGE),
                read=log_uniform(rng, *READ_RANGE),
                gain=float(rng.uniform(*GAIN_RANGE)))


def sample_seed(env, diff, traj, frame, k):
    return zlib.crc32(f"{env}/{diff}/{traj}/{frame}/{k}".encode())


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------
def make_sample(img, Z, P_prev, P_next, params, seed):
    """Returns (blur sRGB (H,W,3), sharp sRGB (H,W,3), displacements)."""
    linear = srgb_to_linear(img) * params["gain"]

    d = displacement_fields_traj(Z, P_prev, P_next, params["T"], N_SUB)
    grids = to_sampling_grid(d).to(DEVICE)
    lin_t = torch.from_numpy(linear.transpose(2, 0, 1).astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        B_lin = blur_linear(lin_t, grids).cpu().numpy()

    B_lin = saturate_and_noise(B_lin, np.random.default_rng(seed),
                               shot=params["shot"], read=params["read"])
    blur = linear_to_srgb(B_lin).transpose(1, 2, 0)
    sharp = linear_to_srgb(np.clip(linear, 0.0, 1.0))      # same gain + saturation, no noise
    return blur.astype(np.float32), sharp.astype(np.float32), d


def save_sample(out_dir, name, blur, sharp, meta):
    for sub in ("blur", "sharp", "meta"):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
    to8 = lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(to8(blur)).save(f"{out_dir}/blur/{name}.png")
    Image.fromarray(to8(sharp)).save(f"{out_dir}/sharp/{name}.png")
    np.savez_compressed(f"{out_dir}/meta/{name}.npz", **meta)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
INDEX_FIELDS = ["split", "env", "diff", "traj", "frame", "k", "blur", "sharp", "meta",
                "T", "shot", "read", "gain", "extent_max", "extent_p95"]


def run_split(split, trajs):
    cfg = SPLITS[split]
    n_k = cfg.get("n_per_frame") or len(cfg["exposures"])
    os.makedirs(f"{SYNTH_ROOT}/{split}", exist_ok=True)
    index_path = f"{SYNTH_ROOT}/{split}/index.csv"
    new_file = not os.path.exists(index_path)

    with open(index_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        if new_file:
            writer.writeheader()

        for env, diff, traj, seq_dir in trajs:
            T_opt = ned_to_optical(load_poses(f"{seq_dir}/pose_left.txt"))
            v_all, w_all = velocities(T_opt)
            rel_dir = f"{split}/{env}_{diff}_{traj}"
            out_dir = f"{SYNTH_ROOT}/{rel_dir}"
            frames = list(range(1, len(T_opt) - 1, cfg["stride"]))
            print(f"[{split}] {env}/{diff}/{traj}: {len(frames)} frames x {n_k}", flush=True)
            t0, done = time.time(), 0

            for frame in frames:
                img = None
                for k in range(n_k):
                    name = f"{frame:06d}_{k}"
                    if os.path.exists(f"{out_dir}/meta/{name}.npz"):
                        continue
                    if img is None:
                        img = load_rgb(f"{seq_dir}/image_left/{frame:06d}_left.png")
                        depth_path = f"{seq_dir}/depth_left/{frame:06d}_left_depth.npy"
                        Z = np.clip(np.load(depth_path), DEPTH_MIN, DEPTH_CLAMP)
                        P_prev, P_next = neighbor_relative_poses(T_opt, frame)
                        b1 = blur_at_T1(Z, P_prev, P_next) if cfg["exposures"] == "random" else None

                    seed = sample_seed(env, diff, traj, frame, k)
                    params = sample_params(np.random.default_rng(seed), cfg, k, b1)
                    blur, sharp, d = make_sample(img, Z, P_prev, P_next, params, seed)

                    ext = np.linalg.norm(d, axis=-1)
                    meta = dict(v=v_all[frame], omega=w_all[frame], P_prev=P_prev, P_next=P_next,
                                depth=Z.astype(np.float16),   # clipped z-depth (m), so synth/ is self-contained
                                K=K,
                                depth_path=os.path.relpath(depth_path, DATA_ROOT).replace("\\", "/"),
                                seed=seed,
                                extent_max=float(np.nanmax(ext)),
                                extent_p95=float(np.nanpercentile(ext, 95)), **params)
                    save_sample(out_dir, name, blur, sharp, meta)
                    writer.writerow(dict(
                        split=split, env=env, diff=diff, traj=traj, frame=frame, k=k,
                        blur=f"{rel_dir}/blur/{name}.png", sharp=f"{rel_dir}/sharp/{name}.png",
                        meta=f"{rel_dir}/meta/{name}.npz",
                        T=params["T"], shot=params["shot"], read=params["read"], gain=params["gain"],
                        extent_max=meta["extent_max"], extent_p95=meta["extent_p95"]))
                    done += 1
                fh.flush()
            rate = (time.time() - t0) / max(done, 1)
            print(f"    {done} new samples, {rate:.2f} s/sample", flush=True)


if __name__ == "__main__":
    print("device:", DEVICE)
    splits = assign_splits()
    for s in ("train", "val", "test"):
        print(f"{s}: {[f'{e}/{d}/{t}' for e, d, t, _ in splits[s]]}")
    for s in ("train", "val", "test"):
        run_split(s, splits[s])
    print("done")