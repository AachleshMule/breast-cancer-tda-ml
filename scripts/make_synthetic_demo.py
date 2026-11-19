"""
SYNTHETIC DEMO DATA - ONLY for checking that the code runs.
Results obtained on this data must NEVER be reported in the paper, report or presentation.

Creates demo_data/benign and demo_data/malignant with simple generated images
(class 1 contains ring-like structures, i.e. loops) and prints the command to run the pipeline.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def make_image(rng, malignant, size=128):
    base = gaussian_filter(rng.random((size, size)), 6)
    base = (base - base.min()) / (base.max() - base.min())
    if malignant:
        yy, xx = np.mgrid[:size, :size]
        for _ in range(int(rng.integers(2, 5))):
            cy, cx = rng.integers(20, size - 20, 2)
            r = rng.integers(6, 14)
            ring = np.exp(-((np.hypot(yy - cy, xx - cx) - r) ** 2) / 6.0)
            base = base + 0.8 * ring
    base = base + 0.05 * rng.standard_normal(base.shape)
    base = (base - base.min()) / (base.max() - base.min())
    return (255 * base).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_data")
    ap.add_argument("--n", type=int, default=40, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls, mal in (("benign", False), ("malignant", True)):
        d = Path(a.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            Image.fromarray(make_image(rng, mal)).save(d / f"p{i // 2:03d}_{cls}_{i}.png")
    print(f"SYNTHETIC images written to {a.out}")
    print(f"python scripts/build_manifest.py folder --root {a.out} --out manifests/demo.csv --patient-regex \"(p\\d+)\"")
    print(f"python scripts/run_experiments.py --manifest manifests/demo.csv --root {a.out} --name demo --fast")


if __name__ == "__main__":
    main()
