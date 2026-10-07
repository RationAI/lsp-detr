#!/usr/bin/env python
"""Build the 3-class (non_tumor / tumor / Lymphocyte) TILs fine-tuning bundle from the two
per-patch label files in ``/workspace/TILs-images``.

Geometry is the recipe's own: ``crop_size=1536`` crops whose *centre* ``center_gt_size=672`` window
carries the GT (Dome ``RandomCropWithGrid._handle_large_image``).  A 4096 image therefore has a
6x6 grid of non-overlapping centre cells, cell ``(ix, iy)`` owning ``[ix*672, ix*672+672)`` and being
cropped as ``[ix*672-432, ix*672+1104)``; the linear grid index used by the transform is
``iy*num_tiles_x + ix``.

Split (see ``--val-cells``): a few interior cells become fixed 1536x1536 validation tiles written to
disk; training crops are restricted to the cells that are neither a val cell nor a 4-neighbour of one.
That guarantees

  * **no GT leakage** - train and val GT come from disjoint centre cells, and
  * **< 50 % pixel overlap** - the closest allowed train crop is a diagonal neighbour of a val tile,
    i.e. (1536-672)^2 / 1536^2 = 31.6 % of its area (asserted below).

Labels: ``0 non_tumor``, ``1 tumor``, ``2 Lymphocyte`` are kept; ``3 Exclusions`` and
``4 Others_unmatched`` are dropped entirely (they become background supervision).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

KEEP = {0: "non_tumor", 1: "tumor", 2: "Lymphocyte"}
DROP_NAMES = {3: "Exclusions", 4: "Others_unmatched"}


def parse_cells(spec: str):
    out = []
    for pair in spec.split():
        ix, iy = pair.split(",")
        out.append((int(ix), int(iy)))
    return out


def grid_dims(img_w: int, img_h: int, center_gt: int):
    """Mirror of Dome's _handle_large_image tile counting (stride == center_gt_size)."""
    nx = max(0, (img_w - center_gt) // center_gt) + 1
    ny = max(0, (img_h - center_gt) // center_gt) + 1
    return nx, ny


def cell_crop_origin(ix: int, iy: int, crop: int, center_gt: int):
    margin = (crop - center_gt) // 2
    return ix * center_gt - margin, iy * center_gt - margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="the *_ki67.json label files")
    ap.add_argument("--img-dir", default="/workspace/TILs-images/images",
                    help="TILs-images patch PNGs (external input; the labelled patches are not in this repo)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tils3"),
                    help="bundle directory (default det/data/tils3, git-excluded)")
    ap.add_argument("--crop-size", type=int, default=1536)
    ap.add_argument("--center-gt", type=int, default=672)
    ap.add_argument("--val-cells", default="1,1 4,1 1,4 4,4", help="'ix,iy ...' centre cells held out for validation")
    args = ap.parse_args()

    crop, cgt = args.crop_size, args.center_gt
    val_cells = parse_cells(args.val_cells)
    os.makedirs(os.path.join(args.out, "val_tiles"), exist_ok=True)

    cats = [{"id": i, "name": n} for i, n in sorted(KEEP.items())]
    train = {"info": {"description": "TILs-images 3-class detection fine-tune - TRAIN (full patches, grid-restricted crops)"},
             "images": [], "annotations": [], "categories": cats}
    val = {"info": {"description": "TILs-images 3-class detection fine-tune - VAL (fixed 1536 tiles, centre-672 GT)"},
           "images": [], "annotations": [], "categories": cats}
    allowed_grids = {}
    report = {"crop_size": crop, "center_gt_size": cgt, "val_cells": val_cells, "images": {}}

    next_val_img_id, next_val_ann_id, next_train_ann_id = 0, 1, 1

    for si, src in enumerate(args.src):
        d = json.load(open(src))
        assert len(d["images"]) == 1, f"{src}: expected exactly one image entry"
        info = d["images"][0]
        fn = info["file_name"]
        path = os.path.join(args.img_dir, fn)
        with Image.open(path) as im:
            W, H = im.size
        nx, ny = grid_dims(W, H, cgt)

        # ---- cell bookkeeping -------------------------------------------------
        all_cells = {(ix, iy) for iy in range(ny) for ix in range(nx)}
        assert set(val_cells) <= all_cells, f"val cells outside the {nx}x{ny} grid"
        for ix, iy in val_cells:  # every val tile must fit without reflect padding
            ox, oy = cell_crop_origin(ix, iy, crop, cgt)
            assert 0 <= ox and ox + crop <= W and 0 <= oy and oy + crop <= H, \
                f"val cell ({ix},{iy}) crop [{ox},{ox+crop}) x [{oy},{oy+crop}) leaves the {W}x{H} image"
        banned = set(val_cells)
        for ix, iy in val_cells:
            banned |= {(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)}
        allowed = sorted(all_cells - banned, key=lambda c: (c[1], c[0]))
        # < 50 % pixel overlap between any allowed train crop and any val tile
        worst = 0.0
        for ax, ay in allowed:
            aox, aoy = cell_crop_origin(ax, ay, crop, cgt)
            for vx, vy in val_cells:
                vox, voy = cell_crop_origin(vx, vy, crop, cgt)
                ow = max(0, min(aox + crop, vox + crop) - max(aox, vox))
                oh = max(0, min(aoy + crop, voy + crop) - max(aoy, voy))
                worst = max(worst, ow * oh / float(crop * crop))
        assert worst < 0.5, f"{fn}: a train crop overlaps a val tile by {worst:.1%}"

        # ---- annotations ------------------------------------------------------
        anns = [a for a in d["annotations"] if a["category_id"] in KEEP]
        dropped = Counter(DROP_NAMES.get(a["category_id"], a["category_id"])
                          for a in d["annotations"] if a["category_id"] not in KEEP)

        train_img_id = si
        train["images"].append({"id": train_img_id, "file_name": fn, "width": W, "height": H})
        allowed_grids[fn] = [iy * nx + ix for ix, iy in allowed]

        def cell_of(a):
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2.0, y + h / 2.0
            return int(cx // cgt), int(cy // cgt)

        per_cell = Counter()
        train_cls, val_cls = Counter(), Counter()
        val_set = set(val_cells)
        allowed_set = set(allowed)

        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            ix, iy = cell_of(a)
            per_cell[(ix, iy)] += 1
            # every annotation stays in the train json; the wrapper only ever selects allowed cells,
            # and RandomCropWithGrid drops anything outside the crop's own centre cell.
            train["annotations"].append({"id": next_train_ann_id, "image_id": train_img_id,
                                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                                         "category_id": a["category_id"]})
            next_train_ann_id += 1
            if (ix, iy) in allowed_set:
                train_cls[a["category_id"]] += 1

        # ---- val tiles --------------------------------------------------------
        with Image.open(path) as im:
            im = im.convert("RGB")
            for ix, iy in val_cells:
                ox, oy = cell_crop_origin(ix, iy, crop, cgt)
                tile_name = f"{os.path.splitext(fn)[0]}_ix{ix}_iy{iy}.png"
                im.crop((ox, oy, ox + crop, oy + crop)).save(os.path.join(args.out, "val_tiles", tile_name))
                vid = next_val_img_id
                next_val_img_id += 1
                val["images"].append({"id": vid, "file_name": tile_name, "width": crop, "height": crop,
                                      "source_image": fn, "cell": [ix, iy], "crop_origin": [ox, oy]})
                cl, ct = ix * cgt, iy * cgt          # centre-672 window in image coords
                n = 0
                for a in anns:
                    x, y, w, h = a["bbox"]
                    if w <= 0 or h <= 0:
                        continue
                    cx, cy = x + w / 2.0, y + h / 2.0
                    if not (cl <= cx < cl + cgt and ct <= cy < ct + cgt):
                        continue
                    tx, ty = x - ox, y - oy          # tile frame
                    x1, y1 = max(0.0, tx), max(0.0, ty)
                    x2, y2 = min(float(crop), tx + w), min(float(crop), ty + h)
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    val["annotations"].append({"id": next_val_ann_id, "image_id": vid,
                                               "bbox": [x1, y1, x2 - x1, y2 - y1],
                                               "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                               "category_id": a["category_id"]})
                    next_val_ann_id += 1
                    val_cls[a["category_id"]] += 1
                    n += 1
                assert n == per_cell[(ix, iy)], (tile_name, n, per_cell[(ix, iy)])

        report["images"][fn] = {
            "size": [W, H], "grid": [nx, ny],
            "val_cells": val_cells, "n_allowed_train_cells": len(allowed),
            "allowed_train_cells": allowed, "banned_cells": sorted(banned, key=lambda c: (c[1], c[0])),
            "max_train_val_pixel_overlap": round(worst, 4),
            "kept_annotations": len(anns), "dropped": dict(dropped),
            "train_gt_in_allowed_cells": {KEEP[k]: v for k, v in sorted(train_cls.items())},
            "val_gt": {KEEP[k]: v for k, v in sorted(val_cls.items())},
        }

    for name, obj in (("train_coco.json", train), ("val_coco.json", val)):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f)
    with open(os.path.join(args.out, "allowed_grids.json"), "w") as f:
        json.dump(allowed_grids, f, indent=1)
    with open(os.path.join(args.out, "build_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))
    print(f"\ntrain: {len(train['images'])} images, {len(train['annotations'])} anns "
          f"(usable in allowed cells: {sum(sum(v['train_gt_in_allowed_cells'].values()) for v in report['images'].values())})")
    print(f"val:   {len(val['images'])} tiles, {len(val['annotations'])} anns")
    print(f"out:   {args.out}")


if __name__ == "__main__":
    main()
