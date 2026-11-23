"""
Step 3: flow test.

Predict the optical flow from frame t to t+1 using ONLY depth, poses and
intrinsics, then compare with TartanAir's ground-truth flow.

If this passes (median endpoint error well under 1 px), the pose handling,
NED -> optical conversion and projection are all correct, and the same
geometry can be trusted for the blur operator.

Run:  python flow_test.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from velocities import load_poses, ned_to_optical

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_DIR = os.path.join(PROJECT_ROOT, "data", "office", "Hard", "P000")
POSE_FILE = f"{SEQ_DIR}/pose_left.txt"
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def depth_path(t):
    return f"{SEQ_DIR}/depth_left/{t:06d}_left_depth.npy"


def flow_path(t):
    return f"{SEQ_DIR}/flow/{t:06d}_{t + 1:06d}_flow.npy"


def mask_path(t):
    return f"{SEQ_DIR}/flow/{t:06d}_{t + 1:06d}_mask.npy"


# ---------------------------------------------------------------------------
# Camera (tartanair_tools/data_type.md)
# ---------------------------------------------------------------------------
FX, FY, CX, CY = 320.0, 320.0, 320.0, 240.0
H, W = 480, 640
K = np.array([[FX, 0.0, CX],
              [0.0, FY, CY],
              [0.0, 0.0, 1.0]])
DEPTH_CLAMP = 100.0


def pixel_grid():
    """
    Returns u, v arrays of shape (H, W) with the integer pixel coordinates.
    u = column index (x), v = row index (y).
    """
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64), indexing="xy")
    return u, v


def backproject(Z, K):
    """
    Pixels -> 3D points in the camera frame.
        X = Z * K^{-1} [u, v, 1]^T

    Z : (H, W) depth.
    Returns X : (H, W, 3).
    """
    u, v = pixel_grid()
    p_h = np.stack([u, v, np.ones_like(u)], axis=-1)      # (H, W, 3) homogeneous pixels
    rays = p_h @ np.linalg.inv(K).T                        # K^{-1} p for every pixel
    return rays * Z[..., None]


def transform_points(X, T_rel):
    """
    Express points from camera t's frame in camera t+1's frame.

    T_rel = inv(T_t) @ T_{t+1}  (camera t+1 pose in camera t's frame)
    X'    = R_rel^T (X - t_rel)

    X : (H, W, 3). Returns X' : (H, W, 3).
    """
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    # R^T (X - t) for every point; row-vector form: (X - t) @ R
    return (X - t_rel) @ R_rel


def project(X, K):
    """
    3D points -> pixels:  p = K X / z

    X : (H, W, 3). Returns (u', v') as an (H, W, 2) array.
    Watch for z <= 0 (points behind the camera): set those to NaN.
    """
    p = X @ K.T                                            # K X for every point
    z = p[..., 2:3]
    uv = p[..., :2] / np.where(z > 1e-6, z, np.nan)        # NaN where behind camera
    return uv


def predict_flow(t, T_opt):
    """
    Predicted flow from frame t to t+1, shape (H, W, 2), channels (du, dv).
    """
    Z = np.minimum(np.load(depth_path(t)), DEPTH_CLAMP)

    T_rel = np.linalg.inv(T_opt[t]) @ T_opt[t + 1]
    X = backproject(Z, K)
    X2 = transform_points(X, T_rel)
    p2 = project(X2, K)
    u, v = pixel_grid()
    return p2 - np.stack([u, v], axis=-1)


def evaluate(t, T_opt, save_map=False):
    flow_pred = predict_flow(t, T_opt)
    flow_gt = np.load(flow_path(t))
    mask = np.load(mask_path(t))

    if t == 0:
        print("flow_gt shape:", flow_gt.shape, flow_gt.dtype)
        vals, counts = np.unique(mask, return_counts=True)
        print("mask values / pixel counts:", dict(zip(vals.tolist(), counts.tolist())))

    # ASSUMPTION: valid pixels have mask == 0. Check the printed mask values:
    # the valid value should cover most of the image.
    valid = (mask == 0) & np.isfinite(flow_pred).all(axis=-1)

    epe = np.linalg.norm(flow_pred - flow_gt, axis=-1)
    e = epe[valid]
    print(f"pair {t:4d}->{t + 1:<4d}  valid {valid.mean():5.1%}   "
          f"EPE median {np.median(e):.4f}  mean {e.mean():.4f}  "
          f"95th {np.percentile(e, 95):.4f} px   "
          f"|gt flow| median {np.median(np.linalg.norm(flow_gt, axis=-1)[valid]):.2f} px")

    if save_map:
        import os
        os.makedirs(OUT_DIR, exist_ok=True)
        m = np.where(valid, epe, np.nan)
        plt.figure(figsize=(8, 5))
        plt.imshow(m, vmin=0, vmax=1, cmap="magma")
        plt.colorbar(label="endpoint error (px)")
        plt.title(f"Flow EPE, pair {t}->{t + 1}")
        plt.savefig(f"{OUT_DIR}/flow_epe_{t:06d}.png", dpi=120, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    T_opt = ned_to_optical(load_poses(POSE_FILE))

    for t in [0, 100, 300, 500, 700]:
        evaluate(t, T_opt, save_map=(t == 100))