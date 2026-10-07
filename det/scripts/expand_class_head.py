#!/usr/bin/env python
"""Expand a trained ``LSPDetrDetection`` checkpoint from K to K' classes (K' > K).

``decoder.class_head`` is a single ``Linear(dim, num_classes)`` shared by all six decoder layers
(sigmoid VFL, no explicit background row), so growing the label set is exactly "append rows".
Every existing row keeps its meaning - here ``0 = non_tumor``, ``1 = tumor`` stay warm and
``2 = Lymphocyte`` is added - and the new rows are initialised the way the port initialises a fresh
class head: zero weight + bias ``-log(99)`` (focal prior p = 0.01), so at step 0 the new class fires
at 1 % everywhere and cannot disturb the two trained rows.

Both ``model`` and ``ema.module`` are expanded; the optimizer / scheduler state is dropped because
the result is meant for ``train.py --tuning`` (a fresh optimizer over the new label set), not for
``--resume``.  Every other tensor is asserted to keep its exact shape.

  python det/scripts/expand_class_head.py -i ckpt/checkpoint0022.pth -o ckpt/checkpoint0022_3class.pth \
      --num-classes 3 --names non_tumor tumor Lymphocyte
"""

from __future__ import annotations

import argparse
import math
import os

import torch

HEAD_W = "decoder.class_head.weight"
HEAD_B = "decoder.class_head.bias"


def expand(sd, num_classes: int, mode: str, prior: float):
    w, b = sd[HEAD_W], sd[HEAD_B]
    old = w.shape[0]
    assert w.shape[0] == b.shape[0] == old, (w.shape, b.shape)
    assert num_classes > old, f"num_classes {num_classes} must exceed the checkpoint's {old}"
    new_w = torch.zeros((num_classes, w.shape[1]), dtype=w.dtype)
    new_b = torch.full((num_classes,), -math.log((1.0 - prior) / prior), dtype=b.dtype)
    new_w[:old] = w
    new_b[:old] = b
    if mode == "copy0":  # optional: seed each new row from row 0 (non-tumour) instead of zeros
        for k in range(old, num_classes):
            new_w[k] = w[0]
    sd = dict(sd)
    sd[HEAD_W], sd[HEAD_B] = new_w, new_b
    return sd, old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--names", nargs="*", default=None, help="class names, for the metadata block")
    ap.add_argument("--new-row-init", choices=["prior", "copy0"], default="prior")
    ap.add_argument("--prior", type=float, default=0.01, help="focal prior for the new rows' bias")
    args = ap.parse_args()

    ck = torch.load(args.input, map_location="cpu", weights_only=False)
    assert "model" in ck, list(ck)
    out = {"expanded_from": os.path.abspath(args.input),
           "expanded_from_last_epoch": ck.get("last_epoch"),
           "class_expansion": {"num_classes": args.num_classes, "new_row_init": args.new_row_init,
                               "prior": args.prior, "names": args.names}}

    for key in ("model", "ema"):
        if key not in ck:
            continue
        sd = ck[key]["module"] if key == "ema" else ck[key]
        new_sd, old = expand(sd, args.num_classes, args.new_row_init, args.prior)
        # nothing but the class head may change shape
        bad = [(k, tuple(sd[k].shape), tuple(new_sd[k].shape)) for k in sd
               if k not in (HEAD_W, HEAD_B) and tuple(sd[k].shape) != tuple(new_sd[k].shape)]
        assert not bad, bad
        assert set(sd) == set(new_sd)
        if key == "ema":
            out["ema"] = {"module": new_sd, "updates": ck["ema"].get("updates", 0)}
        else:
            out["model"] = new_sd
        print(f"[{key}] class_head {old} -> {args.num_classes} classes "
              f"({len(new_sd)} tensors, {sum(v.numel() for v in new_sd.values()):,} params)")
        print(f"       new bias row(s): {new_sd[HEAD_B][old:].tolist()}  "
              f"kept: {new_sd[HEAD_B][:old].tolist()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(out, args.output)
    print(f"wrote {args.output} ({os.path.getsize(args.output)/1e6:.0f} MB); "
          f"keys = {sorted(k for k in out)}")


if __name__ == "__main__":
    main()
