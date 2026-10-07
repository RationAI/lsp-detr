#!/usr/bin/env python
"""Convert a det/inference.py predictions.json into the TILs leaderboard's PUMA Track-1 format.

The leaderboard fixes the label ids as 0=other / 1=lymphocyte / 2=tumor; this arm's head order is
0=non_tumor / 1=tumor / 2=Lymphocyte, so the ids are permuted (they are NOT the same numbers).
Everything else in inference.py's output already matches: bbox is [x, y, w, h] in the *input image's*
pixels, image ids are 1..N and `file_name` keeps the stem.

  python det/scripts/to_puma_leaderboard.py -i <out>/predictions.json -o lsp_detr_det_<tag>_puma_t1.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os

# leaderboard id -> name
LB = [(0, "other"), (1, "lymphocyte"), (2, "tumor")]
# how a source category *name* maps onto the leaderboard ids (aliases the spec allows)
NAME_TO_LB = {
    "other": 0, "non_tumor": 0, "non-tumor": 0, "nontumor": 0, "nt": 0,
    "lymphocyte": 1, "lymph": 1, "til": 1, "lymphocytes": 1,
    "tumor": 2, "tumour": 2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="det/inference.py predictions.json")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--keep-meta", action="store_true", default=True)
    ap.add_argument("--min-score", type=float, default=0.05,
                    help="drop anything below this (the server thresholds itself; this only trims the file)")
    ap.add_argument("--model-name", default=None, help="written into meta.model_name")
    args = ap.parse_args()

    src = json.load(open(args.input))
    # map source category id -> leaderboard id, by NAME (never by position)
    remap = {}
    for c in src["categories"]:
        key = str(c["name"]).strip().lower()
        if key not in NAME_TO_LB:
            raise SystemExit(f"category {c['name']!r} is not one of {sorted(set(NAME_TO_LB))}")
        remap[int(c["id"])] = NAME_TO_LB[key]
    if len(set(remap.values())) != len(remap):
        raise SystemExit(f"two source categories map to the same leaderboard class: {remap}")
    print("[map] " + ", ".join(f"{c['id']}:{c['name']} -> {remap[int(c['id'])]}:{dict(LB)[remap[int(c['id'])]]}"
                               for c in src["categories"]))

    images = [{"id": int(im["id"]), "file_name": os.path.basename(im["file_name"])} for im in src["images"]]
    stems = {os.path.splitext(im["file_name"])[0] for im in images}
    if len(stems) != len(images):
        raise SystemExit("duplicate image stems")

    anns, dropped, bad = [], 0, 0
    counts = collections.Counter()
    for a in src["annotations"]:
        s = float(a["score"])
        if s < args.min_score:
            dropped += 1
            continue
        b = [float(v) for v in a["bbox"]]
        if not (math.isfinite(s) and all(math.isfinite(v) for v in b)):
            bad += 1
            continue
        if b[2] <= 0 or b[3] <= 0:
            bad += 1
            continue
        cid = remap[int(a["category_id"])]
        counts[cid] += 1
        anns.append({"image_id": int(a["image_id"]), "category_id": cid,
                     "bbox": [round(v, 3) for v in b], "score": round(s, 5)})
    if bad:
        print(f"[warn] dropped {bad} annotations with non-finite / degenerate bbox or score")

    out = {"categories": [{"id": i, "name": n} for i, n in LB], "images": images, "annotations": anns}
    if args.keep_meta:
        m = dict(src.get("meta", {}))
        out["meta"] = {"model_name": args.model_name, "checkpoint": m.get("checkpoint"),
                       "config": m.get("config"), "weights": m.get("weights"), "last_epoch": m.get("last_epoch"),
                       "arm": (m.get("arm") or {}).get("center_mode"), "num_classes": m.get("num_classes"),
                       "conf_threshold": m.get("conf_threshold"), "tiling": m.get("tiling"),
                       "patch_size": m.get("patch_size"), "step_size": m.get("step_size"),
                       "filter_size": m.get("filter_size"),
                       "postprocessor": m.get("postprocessor"), "source_cli": m.get("cli"),
                       "source_categories": src["categories"]}
    with open(args.output, "w") as f:
        json.dump(out, f)
    sz = os.path.getsize(args.output)
    print(f"[out] {args.output}  ({sz/2**20:.1f} MB)  images={len(images)}  annotations={len(anns):,}"
          + (f"  (dropped {dropped:,} below {args.min_score})" if dropped else ""))
    print("[cnt] " + "  ".join(f"{dict(LB)[i]}={counts[i]:,}" for i, _ in LB))
    print(f"[score] min={min(a['score'] for a in anns):.5f}  max={max(a['score'] for a in anns):.5f}")


if __name__ == "__main__":
    main()
