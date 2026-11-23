"""
Step 2: camera velocities from TartanAir poses.

Goal: for each frame t, compute the camera's linear velocity v[t] and
angular velocity omega[t] in the OPTICAL camera frame
(x right, y down, z forward), in units of metres/frame and radians/frame.

Run:  python velocities.py
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Paths (edit if your folder layout differs)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
POSE_FILE = os.path.join(DATA_PATH, "office", "Hard","P000", "pose_left.txt")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Optical -> NED axis map:  x_ned = M @ x_opt
#   NED x (forward) = optical z
#   NED y (right)   = optical x
#   NED z (down)    = optical y
M = np.array([[0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]])


def load_poses(path):
    """
    Read a TartanAir pose file.

    Each row: tx ty tz qx qy qz qw  (camera-to-world, NED frame,
    quaternion in xyzw order, same as scipy).

    Returns
    -------
    T : (N, 4, 4) array of camera-to-world poses (NED).
    """
    poses = np.loadtxt(path)
    
    N = poses.shape[0]

    # Initialize N 4x4 identity matrices
    T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()

    # Top-left 3x3: rotation matrices
    T[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()

    # Last column (rows 0-2): translation vectors
    T[:, :3, 3] = poses[:, :3]

    return T


def ned_to_optical(T):
    """
    Convert camera-to-world poses from the NED camera convention to the
    optical camera convention.

    For each pose: R_opt = R_ned @ M, translation unchanged.

    Returns
    -------
    T_opt : (N, 4, 4) array.
    """
    M = np.array([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    T_opt = T.copy()
    
    # Apply R_opt = R_ned @ M across all N matrices (translation stays unchanged)
    T_opt[:, :3, :3] = T[:, :3, :3] @ M
    
    return T_opt


def velocities(T):
    """
    Central-difference velocities in the camera frame.

    For t = 1 .. N-2:
        dT       = inv(T[t-1]) @ T[t+1]
        v[t]     = translation(dT) / 2          (metres / frame)
        omega[t] = rotvec(rotation(dT)) / 2     (radians / frame)

    First and last rows have no neighbour on one side: set them to NaN.

    Returns
    -------
    v, omega : (N, 3) arrays each.
    """

    N = T.shape[0]
    v = np.full((N, 3), np.nan, dtype=float)
    omega = np.full((N, 3), np.nan, dtype=float)

    if N < 3:
        return v, omega

    # Relative transforms from t-1 to t+1: dT = inv(T[t-1]) @ T[t+1]
    dT = np.linalg.inv(T[:-2]) @ T[2:]

    # Linear velocity (metres / frame)
    v[1:-1] = dT[:, :3, 3] / 2.0

    # Angular velocity via axis-angle rotation vector (radians / frame)
    omega[1:-1] = Rotation.from_matrix(dT[:, :3, :3]).as_rotvec() / 2.0

    return v, omega
    


def report(v, omega):
    """Print the sanity checks. Expected values are for office/Hard/P000."""
    valid = ~np.isnan(v).any(axis=1)
    v, omega = v[valid], omega[valid]

    v_norm = np.linalg.norm(v, axis=1)
    w_norm = np.linalg.norm(omega, axis=1)

    print(f"frames with velocities: {len(v)}")

    print("\n1. |v| percentiles 50/90/99 ")
    print("  ", np.percentile(v_norm, [50, 90, 99]).round(4))

    print("\n2. |omega| percentiles 50/90/99 ")
    print("  ", np.percentile(w_norm, [50, 90, 99]).round(4))

    print("\n3. mean unit direction of v ")
    print("  ", (v / v_norm[:, None]).mean(axis=0).round(3))

    print("\n4. mean |omega| per axis ")
    print("  ", np.abs(omega).mean(axis=0).round(4))


if __name__ == "__main__":
    T_ned = load_poses(POSE_FILE)
    print("poses:", T_ned.shape)         
    T_opt = ned_to_optical(T_ned)
    v, omega = velocities(T_opt)

    report(v, omega)

