#!/usr/bin/env python
"""Compare two .zst detection files over the same slide: coordinate frame + class agreement.

Written to check this arm's whole-slide output against the platform's own file for the same slide
(e.g. the KongNet reference), which is the only end-to-end check of the frame (extent, y flip,
bounds offset) that does not depend on our own writer.

  python det/scripts/compare_zst.py --a <ours>.zst --b <reference>.zst --radius 12
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

# categoryId -> name is NOT fixed across files: the platform numbers tumour 1 in 'tnt'/'tnt-lymph' files
# and 2 in 'tnt-lymph-nt1' files (see ZST_CONVENTIONS in det/wsi_infer.py). The `nt` property is the
# only field that means the same thing everywhere (nt=False <=> tumour), so names are read from it.
def class_name(cat, nt):
    if not nt:
        return "tumor"
    return "Lymphocyte" if str(cat) == "3" else "non_tumor"


def load(path):
    import zstandard
    import mapbox_vector_tile as mvt
    blob = zstandard.ZstdDecompressor().decompress(open(path, "rb").read(), max_output_size=8_000_000_000)
    layer = mvt.decode(blob)["default"]
    feats = layer["features"]
    xy = np.array([f["geometry"]["coordinates"] for f in feats], dtype=np.float64)
    cat = np.array([class_name(f["properties"]["categoryId"], f["properties"].get("nt", True)) for f in feats],
                   dtype=object)
    return layer["extent"], xy, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="this arm's .zst")
    ap.add_argument("--b", required=True, help="reference .zst")
    ap.add_argument("--radius", type=float, default=12.0, help="level-0 px for a match")
    args = ap.parse_args()

    ea, xa, ca = load(args.a)
    eb, xb, cb = load(args.b)
    print(f"A {args.a}\n   extent={ea}  n={len(xa):,}  cats={dict(sorted(collections.Counter(ca).items()))}")
    print(f"B {args.b}\n   extent={eb}  n={len(xb):,}  cats={dict(sorted(collections.Counter(cb).items()))}")
    if ea != eb:
        print(f"\n!! extents differ ({ea} vs {eb}) - these are different canvases, coordinates are "
              f"NOT comparable; only the class mix is.")
        return

    from scipy.spatial import cKDTree
    tb = cKDTree(xb)
    d, j = tb.query(xa, k=1, workers=-1)
    print(f"\nnearest reference detection for each of our {len(xa):,}:")
    for q in (50, 75, 90, 99):
        print(f"   p{q:<2d} = {np.percentile(d, q):8.2f} px")
    for r in (2, 5, args.radius, 30):
        print(f"   within {r:5.1f} px: {100 * (d <= r).mean():5.1f} %")
    # a frame error (y flip / offset) shows up as a large systematic median offset
    m = d <= args.radius
    if m.sum():
        dxy = xa[m] - xb[j[m]]
        print(f"   median offset of matched pairs: dx={np.median(dxy[:, 0]):+.2f}  dy={np.median(dxy[:, 1]):+.2f} px")
        conf = collections.Counter(zip(cb[j[m]], ca[m]))
        cats_a = sorted(set(ca.tolist())); cats_b = sorted(set(cb.tolist()))
        print(f"\nclass agreement on {int(m.sum()):,} matched pairs (rows = reference, cols = ours):")
        print("            " + "".join(f"{c:>16s}" for c in cats_a))
        agree = 0
        for rb in cats_b:
            row = [conf.get((rb, cx), 0) for cx in cats_a]
            print(f"{rb:>12s}" + "".join(f"{v:>16,d}" for v in row))
            if rb in cats_a:
                agree += conf.get((rb, rb), 0)
        print(f"   overall class agreement: {100 * agree / m.sum():.1f} %")


if __name__ == "__main__":
    main()
