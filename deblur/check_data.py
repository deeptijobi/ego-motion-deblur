import os, glob
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for seq in sorted(glob.glob(os.path.join(ROOT, "*", "Hard", "P*"))):
    poses = sum(1 for _ in open(os.path.join(seq, "pose_left.txt")))
    imgs = len(glob.glob(os.path.join(seq, "image_left", "*.png")))
    deps = len(glob.glob(os.path.join(seq, "depth_left", "*.npy")))
    flag = "" if poses == imgs == deps else "   <-- MISMATCH"
    print(f"{os.path.relpath(seq, ROOT):35s} poses {poses:5d}  img {imgs:5d}  depth {deps:5d}{flag}")