"""
Step 4: the blur operator.

Physics: a blurry image is the average of what the camera saw during the
exposure (Eq. 1 of the proposal). We split the exposure T into N instants,
warp the sharp image to the camera pose at each instant (using depth +
motion), and average.

The SAME function is used for:
  - synthesis   (N = 64, linear light, saturation, noise)   -> the dataset
  - the network (N = 8)                                     -> operator A

Geometry is reused from flow_test.py, which was verified against TartanAir's
ground-truth optical flow (EPE ~ 0).

Units: v in metres/frame, omega in radians/frame, T in frames
(e.g. T = 0.5 means the shutter is open for half the frame interval).

Run:  python blur.py      (blurs one test frame at a few exposures)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

from flow_test import K, H, W, DEPTH_CLAMP, backproject, transform_points, project, pixel_grid

# ---------------------------------------------------------------------------
# Paths for the self-test
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT= os.path.join (PROJECT_ROOT,"data")
SEQ_DIR = os.path.join(DATA_ROOT, "office", "Hard", "P000")
OUT_DIR =  os.path.join(PROJECT_ROOT, "outputs")
TEST_FRAME = 100


# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------
def srgb_to_linear(x):
    """x in [0, 1], sRGB-encoded -> linear intensity (standard sRGB formula)."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)     


def linear_to_srgb(x):
    """Inverse of srgb_to_linear. Clip x to [0, 1] first."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Motion during the exposure
# ---------------------------------------------------------------------------
def subexposure_times(T, N):
    """
    N instants evenly spaced and CENTRED on the frame time:
        tau_n in [-T/2, +T/2]
    Centred so the sharp ground truth corresponds to mid-exposure.
    Returns (N,) array.
    """
    return np.linspace(-T / 2.0, T / 2.0, N)


def subexposure_pose(v, omega, tau):
    """
    Camera pose at time tau, relative to the mid-exposure camera,
    assuming constant velocity:
        R(tau) = exp(tau [omega]_x)     (Rotation.from_rotvec)
        t(tau) = tau * v
    Returns a 4x4 matrix in the same convention as T_rel in flow_test.py,
    so transform_points(X, pose) gives the points as seen at time tau.
    """

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(tau * omega).as_matrix()
    pose[:3, 3] = tau * v
    return pose


def displacement_fields(Z, v, omega, T, N):
    """
    For each sub-exposure instant n, where does each pixel move?
        d_n(p) = project(transform_points(backproject(Z), pose_n)) - p

    Z : (H, W) depth (already clamped).
    Returns d : (N, H, W, 2), channels (du, dv), in pixels.
    """
    # Backproject depth into 3D points (computed only once)
    X = backproject(Z, K)

    # Original pixel coordinates
    u, v_pix = pixel_grid()
    p = np.stack([u, v_pix], axis=-1)

    # Allocate displacement fields
    d = np.zeros((N, H, W, 2), dtype=np.float32)

    # Compute displacement at each sub-exposure instant
    for n, tau in enumerate(subexposure_times(T, N)):
        pose = subexposure_pose(v, omega, tau)

        # Transform 3D points to the camera pose at time tau
        X_t = transform_points(X, pose)

        # Project transformed points into image coordinates
        uv = project(X_t, K)

        # Displacement relative to original pixel coordinates
        d[n] = uv - p

    return d


# ---------------------------------------------------------------------------
# Warping and averaging
# ---------------------------------------------------------------------------
def to_sampling_grid(d):
    """
    Build grid_sample grids for BACKWARD warping:
        warped_n(p) = I(p - d_n(p))        (first-order, reference depth)

    grid_sample wants coordinates normalised to [-1, 1]. With
    align_corners=True:  x_norm = 2*u/(W-1) - 1,  y_norm = 2*v/(H-1) - 1.

    d : (N, H, W, 2) numpy. Returns torch tensor (N, H, W, 2), float32.
    Points that were NaN (behind camera) -> put them far outside [-1, 1]
    so they sample the border.
    """
    u, v_pix = pixel_grid()

    # Backward warp: sample the sharp image at p - d_n(p)
    src_u = u[None] - d[..., 0]
    src_v = v_pix[None] - d[..., 1]

    # Normalise pixel coordinates to [-1, 1] (align_corners=True convention)
    x = 2.0 * src_u / (W - 1) - 1.0
    y = 2.0 * src_v / (H - 1) - 1.0

    grid = np.stack([x, y], axis=-1)            # (N, H, W, 2), x first
    grid = np.nan_to_num(grid, nan=2.0)         # NaN -> outside -> border
    return torch.from_numpy(grid).float()


def blur_linear(I_lin, grids):
    """
    Average of warped copies.

    I_lin : torch (3, H, W) linear-intensity image.
    grids : torch (N, H, W, 2) from to_sampling_grid.
    Returns torch (3, H, W).

    Use F.grid_sample(..., mode="bilinear", padding_mode="border",
    align_corners=True). Use the SAME three settings everywhere, including
    later inside the network's operator.
    """
    N = grids.shape[0]

    # N copies of the image as a batch: (N, 3, H, W)
    batch = I_lin.unsqueeze(0).expand(N, -1, -1, -1)

    warped = F.grid_sample(batch, grids, mode="bilinear",
                           padding_mode="border", align_corners=True)

    # Average over the sub-exposure instants
    return warped.mean(dim=0)



# ---------------------------------------------------------------------------
# Camera effects (synthesis only, NOT inside the network)
# ---------------------------------------------------------------------------
def saturate_and_noise(B_lin, rng, gain=1.0, shot=1e-4, read=1e-3):
    """
    Apply sensor saturation, shot noise and read noise in linear space.

    B_lin : (3, H, W) torch tensor or numpy array.
    rng    : np.random.Generator.
    Returns noisy, clipped linear-intensity image.
    """
    # Sensor saturation
    B = np.clip(B_lin, 0.0, 1.0)

    # Shot noise (brightness-dependent)
    shot_std = np.sqrt(shot * B)
    shot_noise = rng.normal(0.0, shot_std, size=B.shape)

    read_noise = rng.normal(0.0, read, size=B.shape)

    B_noisy = B + shot_noise + read_noise
    B_noisy = np.clip(B_noisy, 0.0, 1.0)

    return B_noisy


# ---------------------------------------------------------------------------
# One frame end to end
# ---------------------------------------------------------------------------
def load_rgb(path):
    """PNG -> float32 numpy (H, W, 3) in [0, 1], RGB order (PIL, not cv2)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0



def synthesize_frame(img, Z, v, omega, T, N=64, gain=1.0, seed=0):
    """
    img : (H, W, 3) sRGB in [0, 1].   Z : (H, W) depth.
    Returns (blurry sRGB (H, W, 3) in [0, 1], max blur extent in px).
    """
    # Convert sRGB to linear intensity and apply gain
    linear = srgb_to_linear(img) * gain

    # Clamp depth and compute displacement fields
    Z = np.clip(Z, 0.05, DEPTH_CLAMP)
    d = displacement_fields(Z, v, omega, T, N)

    # Convert displacement fields to sampling grids
    grids = to_sampling_grid(d)

    # Convert image from (H, W, 3) to (3, H, W)
    linear_torch = torch.from_numpy(
        linear.transpose(2, 0, 1).astype(np.float32)
    )

    # Perform backward warping and temporal averaging
    B_lin = blur_linear(linear_torch, grids)

    # Apply sensor saturation and noise
    B_lin = saturate_and_noise(
        B_lin.numpy(),
        np.random.default_rng(seed)
    )

    # Convert linear intensity back to sRGB
    B_srgb = linear_to_srgb(B_lin)

    # Convert from (3, H, W) to (H, W, 3)
    B_srgb = B_srgb.transpose(1, 2, 0).astype(np.float32)

    # Calculate maximum displacement magnitude in pixels
    blur_extent = np.nanmax(np.linalg.norm(d, axis=-1))

    return B_srgb, blur_extent



# ---------------------------------------------------------------------------
# Self-test: blur one frame at several exposures and save the results
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from velocities import load_poses, ned_to_optical, velocities

    os.makedirs(OUT_DIR, exist_ok=True)

    T_opt = ned_to_optical(load_poses(f"{SEQ_DIR}/pose_left.txt"))
    v_all, w_all = velocities(T_opt)
    v, omega = v_all[TEST_FRAME], w_all[TEST_FRAME]
    print("v =", v.round(4), " omega =", omega.round(4))

    img = load_rgb(f"{SEQ_DIR}/image_left/{TEST_FRAME:06d}_left.png")
    Z = np.load(f"{SEQ_DIR}/depth_left/{TEST_FRAME:06d}_left_depth.npy")

    panels = [img]
    for T in [0.25, 0.5, 1.0]:
        B, extent = synthesize_frame(img, Z, v, omega, T)
        print(f"T = {T:4.2f} frames  ->  max blur extent {extent:6.1f} px")
        panels.append(B)

    strip = np.concatenate(panels, axis=1)
    Image.fromarray((strip * 255).round().astype(np.uint8)).save(
        f"{OUT_DIR}/blur_test_{TEST_FRAME:06d}.png")
    print("saved", f"{OUT_DIR}/blur_test_{TEST_FRAME:06d}.png",
          "(sharp | T=0.25 | T=0.5 | T=1.0)")