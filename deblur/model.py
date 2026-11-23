"""
Step 6: our model. Physics blur operator A + unrolled deblurring network.
Also builds the NAFNet baseline, so both models share one file.

Operator A (torch, batched, GPU):
    A(x) = mean over N sub-exposures of warp(x, grid_n)
    grid_n comes from (v, omega, Z, K, T), assuming constant velocity.
    Simpler than the synthesis on purpose: N = 8 (not 64), constant velocity
    (not the true trajectory), no noise.

Unrolled network:
    x_0 = y
    for k = 1..K:
        x = x - alpha_k * A^T(A(x) - y) / A^T(1)     data step (physics, SART-preconditioned)
        x = D(x)                                     denoiser (NAFNet U-Net), shared weights

    Dividing by the column sums A^T(1) matters: with border padding, every sample that
    leaves the frame lands on the edge pixels, so A^T(1) reaches ~40 there. The plain
    step x - alpha A^T(Ax - y) then blows up at the borders (x ~1e5 after 5 steps,
    NaN under fp16). With the division (SART / Cimmino preconditioning, as in CT
    reconstruction) the iteration is stable for 0 < alpha < 2.

Shapes (B = batch, N = sub-exposures, h, w = crop size):
    x, y      (B, 3, h, w)       linear light
    Z         (B, h, w)          depth in metres, already clipped
    v, omega  (B, 3)             per frame, optical frame
    T         (B,)               exposure, fraction of the frame interval
    K         (B, 3, 3)          intrinsics of the crop (see crop_K)
    grids     (B, N, h, w, 2)    sampling coordinates for grid_sample

Run:  python model.py     (operator vs blur.py, adjoint, forward/backward)
"""
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nafnet import NAFNet

N_SUB = 8


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def rodrigues(rotvec):
    """
    Axis-angle (B, 3) -> rotation matrices (B, 3, 3).
        R = I + sin(theta) [k]_x + (1 - cos(theta)) [k]_x^2,   k = rotvec / theta
    Same as scipy's Rotation.from_rotvec. theta = 0 gives the identity.
    """
    theta = rotvec.norm(dim=1, keepdim=True)                 # (B, 1)
    k = rotvec / theta.clamp_min(1e-12)
    kx, ky, kz = k.unbind(dim=1)
    zero = torch.zeros_like(kx)
    Kx = torch.stack([zero, -kz, ky,
                      kz, zero, -kx,
                      -ky, kx, zero], dim=1).reshape(-1, 3, 3)
    s = torch.sin(theta)[..., None]
    c = (1 - torch.cos(theta))[..., None]
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + s * Kx + c * (Kx @ Kx)


def crop_K(K, x0, y0):
    """
    Intrinsics of a crop whose top-left corner is (x0, y0) in the full image:
    same focal lengths, principal point shifted. K: (3, 3) or (B, 3, 3).
    """
    K = K.clone()
    K[..., 0, 2] -= x0
    K[..., 1, 2] -= y0
    return K


def displacement_grids(Z, v, omega, T, K, N=N_SUB):
    """
    Torch version of blur.displacement_fields + blur.to_sampling_grid.

    For each sub-exposure time tau in [-T/2, T/2]:
        pose  = (R = rodrigues(tau * omega), t = tau * v)     constant velocity
        X'    = (X - t) @ R                                    as in flow_test.transform_points
        d     = project(X') - p
        grid  = normalised (p - d)                             backward warp
    Points that end up behind the camera are sent outside the image (2.0).
    Returns (B, N, h, w, 2), float32.
    """
    B, h, w = Z.shape
    dev, dt = Z.device, Z.dtype

    # Pixel grid and back-projection, once
    vv, uu = torch.meshgrid(torch.arange(h, device=dev, dtype=dt),
                            torch.arange(w, device=dev, dtype=dt), indexing="ij")
    p_h = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)                   # (h, w, 3)
    rays = torch.einsum("bij,hwj->bhwi", torch.linalg.inv(K), p_h)
    X = rays * Z[..., None]                                                     # (B, h, w, 3)

    # Sub-exposure poses
    taus = torch.linspace(-0.5, 0.5, N, device=dev, dtype=dt)[None] * T[:, None]   # (B, N)
    R = rodrigues((taus[..., None] * omega[:, None]).reshape(-1, 3)).reshape(B, N, 3, 3)
    t = taus[..., None] * v[:, None]                                                # (B, N, 3)

    # Transform and project
    Xt = torch.einsum("bnhwi,bnij->bnhwj", X[:, None] - t[:, :, None, None], R)
    q = torch.einsum("bij,bnhwj->bnhwi", K, Xt)
    z = q[..., 2:3]
    uv = q[..., :2] / torch.where(z > 1e-6, z, torch.full_like(z, float("nan")))

    # Backward-warp sampling grid, normalised for align_corners=True
    p = p_h[..., :2]
    src = p - (uv - p)
    grid = torch.stack([2 * src[..., 0] / (w - 1) - 1,
                        2 * src[..., 1] / (h - 1) - 1], dim=-1)
    return torch.nan_to_num(grid, nan=2.0).float()


# ---------------------------------------------------------------------------
# Operator A and its adjoint
# ---------------------------------------------------------------------------
def apply_A(x, grids):
    """
    Warp x with each of the N grids and average.
    Same grid_sample settings as blur.blur_linear.
    """
    B, N, h, w, _ = grids.shape
    C = x.shape[1]
    xb = x[:, None].expand(B, N, C, h, w).reshape(B * N, C, h, w)
    warped = F.grid_sample(xb, grids.reshape(B * N, h, w, 2), mode="bilinear",
                           padding_mode="border", align_corners=True)
    return warped.reshape(B, N, C, h, w).mean(dim=1)


def apply_AT(y, grids):
    """
    A^T y, from autograd: the gradient of <A(z), y> with respect to z is A^T y.
    No graph is kept; AdjointFn supplies the backward pass.
    """
    with torch.enable_grad():
        z = torch.zeros_like(y, requires_grad=True)
        (g,) = torch.autograd.grad(apply_A(z, grids), z, grad_outputs=y)
    return g


class AdjointFn(torch.autograd.Function):
    """
    A^T with an explicit backward pass. A^T is linear and the grids are fixed,
    so backpropagating through A^T is just A. Training never needs a second
    derivative through grid_sample.
    """

    @staticmethod
    def forward(ctx, y, grids):
        ctx.save_for_backward(grids)
        return apply_AT(y, grids)

    @staticmethod
    def backward(ctx, g):
        (grids,) = ctx.saved_tensors
        return apply_A(g, grids), None


class BlurOperator:
    """
    Builds the grids (and the column sums A^T(1)) for one batch once; A and AT reuse
    them in every unrolled step.
    Always runs in fp32: under mixed precision the denoiser output is fp16, and
    fp16 sampling coordinates are not accurate enough (they produced NaNs in testing).
    """

    def __init__(self, Z, v, omega, T, K, N=N_SUB):
        with torch.no_grad():
            self.grids = displacement_grids(Z.float(), v.float(), omega.float(),
                                            T.float(), K.float(), N)
            # column sums A^T(1), the SART preconditioner; ~1 inside, large at the borders
            ones = torch.ones(Z.shape[0], 1, *Z.shape[1:], device=Z.device)
            self.colsum = apply_AT(ones, self.grids).clamp_min(1e-3)

    def A(self, x):
        return apply_A(x.float(), self.grids)

    def AT(self, y):
        return AdjointFn.apply(y.float(), self.grids)


# ---------------------------------------------------------------------------
# Unrolled network
# ---------------------------------------------------------------------------
NAFNET_2M = dict(img_channel=3, width=16, middle_blk_num=2,
                 enc_blk_nums=[1, 1, 2, 4], dec_blk_nums=[1, 1, 1, 1])   # 1.99M params


class UnrolledDeblur(nn.Module):
    """
    n_steps data + denoiser steps with one shared NAFNet.
    Step sizes alpha_k = softplus(alpha_raw_k) are learned and stay positive.
    With the SART preconditioning the data step is stable for 0 < alpha < 2,
    so alpha = 1 is a safe start.
    """

    def __init__(self, n_steps=5, alpha_init=1.0, use_checkpoint=True):
        super().__init__()
        self.denoiser = NAFNet(**NAFNET_2M)
        alpha_raw = math.log(math.expm1(alpha_init))            # inverse softplus
        self.alpha_raw = nn.Parameter(torch.full((n_steps,), alpha_raw))
        self.n_steps = n_steps
        self.use_checkpoint = use_checkpoint

    def alphas(self):
        return F.softplus(self.alpha_raw)

    def step(self, x, y, op, k):
        x = x.float()
        x = x - self.alphas()[k] * op.AT(op.A(x) - y.float()) / op.colsum
        return self.denoiser(x)

    def forward(self, y, op):
        """y: blurry image in linear light; op: BlurOperator for this batch."""
        x = y
        for k in range(self.n_steps):
            if self.use_checkpoint and self.training:
                # recompute this step in the backward pass instead of storing it:
                # memory of about one step instead of n_steps
                x = checkpoint(self.step, x, y, op, k, use_reentrant=False)
            else:
                x = self.step(x, y, op, k)
        return x


def build_model(name):
    """'ours' -> unrolled network, called as model(y, op).
    'nafnet' -> image-only baseline with the same denoiser, called as model(y)."""
    if name == "ours":
        return UnrolledDeblur()
    if name == "nafnet":
        return NAFNet(**NAFNET_2M)
    raise ValueError(f"unknown model {name!r}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def test_adjoint(op, shape):
    """<A u, w> must equal <u, A^T w>."""
    u = torch.randn(shape, device=op.grids.device)
    w = torch.randn(shape, device=op.grids.device)
    lhs = (op.A(u) * w).sum().item()
    rhs = (u * op.AT(w)).sum().item()
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    print(f"adjoint:   <Au,w> = {lhs:.6f}   <u,ATw> = {rhs:.6f}   relative error {rel:.1e}")
    assert rel < 1e-4, "adjoint test failed"


def test_against_blur_py(meta_path, sharp_path, device):
    """
    Torch A must match blur.py's NumPy version on a real sample (N = 8, no noise).
    Also checks crop_K: A on a crop with the crop's K must equal the same crop of
    the full-frame result, away from the crop borders.
    """
    import blur

    m = np.load(meta_path)
    Z = m["depth"].astype(np.float32)
    v, omega, T = m["v"], m["omega"], float(m["T"])
    x_np = blur.srgb_to_linear(blur.load_rgb(sharp_path)).transpose(2, 0, 1).astype(np.float32)

    # NumPy reference (float64 geometry)
    grids_np = blur.to_sampling_grid(blur.displacement_fields(Z, v, omega, T, N_SUB))
    ref = blur.blur_linear(torch.from_numpy(x_np.copy()), grids_np).numpy()

    # Torch, full frame
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)[None]
    x = t(x_np)
    op = BlurOperator(t(Z), t(v), t(omega), t(T), t(m["K"]), N_SUB)
    out = op.A(x)
    err = np.abs(out[0].cpu().numpy() - ref).max()
    print(f"blur.py:   full frame max difference {err:.1e}   (blur up to {float(m['extent_max']):.1f} px)")
    assert err < 1e-4, "torch A does not match blur.py"

    # Torch, 256 x 256 crop with its own K
    x0, y0, c, margin = 200, 120, 256, 64       # margin must exceed the blur size
    Kc = crop_K(t(m["K"]), x0, y0)
    op_c = BlurOperator(t(Z[y0:y0 + c, x0:x0 + c]), t(v), t(omega), t(T), Kc, N_SUB)
    out_c = op_c.A(x[..., y0:y0 + c, x0:x0 + c])
    inner = (slice(None), slice(None), slice(margin, c - margin), slice(margin, c - margin))
    err_c = (out_c[inner] - out[..., y0:y0 + c, x0:x0 + c][inner]).abs().max().item()
    print(f"crop K:    crop vs full frame max difference {err_c:.1e}")
    assert err_c < 1e-4, "crop_K is wrong"
    return op


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SYNTH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth")

    # 1. Operator against blur.py, on the first val sample
    with open(os.path.join(SYNTH_ROOT, "val", "index.csv"), newline="") as fh:
        row = next(csv.DictReader(fh))
    op = test_against_blur_py(os.path.join(SYNTH_ROOT, row["meta"]),
                              os.path.join(SYNTH_ROOT, row["sharp"]), device)

    # 2. Adjoint, on the same real operator
    test_adjoint(op, (1, 3, 480, 640))

    # 3. Both models: size, forward and backward on a 256 x 256 batch
    B, h = 2, 256
    op = BlurOperator(torch.full((B, h, h), 3.0, device=device),
                      torch.tensor([[0.1, 0.0, 0.05], [0.0, 0.05, 0.1]], device=device),
                      torch.tensor([[0.0, 0.01, 0.0], [0.005, 0.0, 0.002]], device=device),
                      torch.full((B,), 0.2, device=device),
                      crop_K(torch.tensor([[320.0, 0, 320], [0, 320, 240], [0, 0, 1]],
                                          device=device), 192, 112).expand(B, 3, 3))
    for name in ("ours", "nafnet"):
        model = build_model(name).to(device).train()
        y = torch.rand(B, 3, h, h, device=device)
        out = model(y, op) if name == "ours" else model(y)
        out.mean().backward()
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:7s}    {n / 1e6:.3f}M params, output {tuple(out.shape)}, backward ok")
    print("all checks passed")