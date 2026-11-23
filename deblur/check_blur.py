import csv, os
import numpy as np

SYNTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "synth")
for s in ["train", "val", "test"]:
    rows = list(csv.DictReader(open(os.path.join(SYNTH, s, "index.csv"))))
    p95 = np.array([float(r["extent_p95"]) for r in rows])
    mx = np.array([float(r["extent_max"]) for r in rows])
    print(f"{s:5s} n={len(rows):4d}   blur p95 (px): median {np.median(p95):5.1f}, "
          f"90th pct {np.percentile(p95, 90):5.1f}   largest blur: median {np.median(mx):5.1f}")