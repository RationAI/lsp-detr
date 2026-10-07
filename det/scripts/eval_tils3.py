#!/usr/bin/env python
"""Per-class evaluation of an LSP-DETR detection checkpoint on the TILs-images 3-class val tiles.

The trainer's ``log.txt`` only carries the class-averaged COCO numbers, which is not enough to say
whether the *lymphocyte* row was learned.  This script runs the same model / postprocessor as
``det/train.py --test-only`` (single process, so the multi-GPU ``coco_eval_aitod.merge()`` area-subset
bug cannot apply) and reports

  * COCO AP / AP50 overall and **per category**, and
  * centroid-matched precision / recall / F1 per class at a score threshold - the metric the TILs
    work is actually graded on (a prediction matches the nearest unused GT centre within
    ``--match-radius`` px; class is compared only on matched pairs).

  python det/scripts/eval_tils3.py -c det/configs/LSP-T-tils3.yml -r <run>/best_stg1.pth \
      --weights ema --device cuda:0 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # det/

import lsp_det  # noqa: F401,E402  (bootstraps Dome + registers the LSP components)
from src.core import YAMLConfig  # noqa: E402


def build(cfg_path: str, resume: str, weights: str, device: str, updates):
    from src.core import yaml_utils
    upd = yaml_utils.parse_cli(updates) if updates else {}
    cfg = YAMLConfig(cfg_path, **upd)
    model = cfg.model
    post = cfg.postprocessor
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        sd = ck["ema"]["module"] if (weights == "ema" and "ema" in ck) else ck["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        print(f"[eval] loaded {weights} weights from {resume} (last_epoch={ck.get('last_epoch')})")
    return cfg, model.to(device).eval(), post.to(device).eval()


@torch.no_grad()
def predict(cfg, model, post, device, batch_size: int):
    loader = cfg.val_dataloader
    ds = loader.dataset
    preds = {}
    for images, targets in torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=loader.collate_fn):
        images = images.to(device)
        sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        out = post(model(images, targets=None), sizes)
        for t, r in zip(targets, out):
            preds[int(t["image_id"].item())] = {k: v.cpu() for k, v in r.items()
                                                if isinstance(v, torch.Tensor)}
        print(f"  ...{len(preds)}/{len(ds)} tiles", flush=True)
    return preds


def coco_results(preds):
    res = []
    for img_id, p in preds.items():
        b = p["boxes"].numpy()
        for (x1, y1, x2, y2), s, l in zip(b, p["scores"].tolist(), p["labels"].tolist()):
            res.append({"image_id": img_id, "category_id": int(l),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(s)})
    return res


def coco_eval(gt_json, results, cat_ids=None, max_dets=(500, 1000, 2000)):
    from faster_coco_eval import COCO, COCOeval_faster
    coco_gt = COCO(gt_json)
    if not results:
        return None
    coco_dt = coco_gt.loadRes(list(results))
    e = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    e.params.maxDets = list(max_dets)
    if cat_ids is not None:
        e.params.catIds = list(cat_ids)
    e.evaluate(); e.accumulate(); e.summarize()
    return list(e.stats)


def centroid_match(gt_json, preds, score_thr: float, radius: float, num_classes: int):
    """Class-agnostic greedy nearest-centroid matching (score-descending), then per-class P/R/F1."""
    gt = json.load(open(gt_json))
    per_img = defaultdict(list)
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        per_img[a["image_id"]].append((x + w / 2.0, y + h / 2.0, a["category_id"]))

    tp = np.zeros(num_classes); fp = np.zeros(num_classes); fn = np.zeros(num_classes)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)   # rows GT, cols pred
    det_tp = det_fp = det_fn = 0
    for img_id, items in per_img.items():
        g = np.array([[c[0], c[1]] for c in items], dtype=np.float64).reshape(-1, 2)
        gcls = np.array([c[2] for c in items], dtype=np.int64)
        p = preds.get(img_id)
        if p is None or len(p["scores"]) == 0:
            for c in gcls:
                fn[c] += 1
            det_fn += len(gcls)
            continue
        keep = p["scores"].numpy() >= score_thr
        b = p["boxes"].numpy()[keep]
        pcls = p["labels"].numpy()[keep]
        order = np.argsort(-p["scores"].numpy()[keep])
        b, pcls = b[order], pcls[order]
        pc = np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1) if len(b) else np.zeros((0, 2))

        used = np.zeros(len(g), dtype=bool)
        for k in range(len(pc)):
            if len(g) == 0:
                fp[pcls[k]] += 1; det_fp += 1; continue
            d = np.hypot(g[:, 0] - pc[k, 0], g[:, 1] - pc[k, 1])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= radius:
                used[j] = True
                det_tp += 1
                conf[gcls[j], pcls[k]] += 1
                if gcls[j] == pcls[k]:
                    tp[pcls[k]] += 1
                else:
                    fp[pcls[k]] += 1
                    fn[gcls[j]] += 1
            else:
                fp[pcls[k]] += 1; det_fp += 1
        miss = int((~used).sum())
        det_fn += miss
        for c in gcls[~used]:
            fn[c] += 1
    return tp, fp, fn, conf, (det_tp, det_fp, det_fn)


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "configs", "LSP-T-tils3.yml"))
    ap.add_argument("-r", "--resume", required=True)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--match-radius", type=float, default=12.0, help="px; the label file's own Ki-67 match threshold")
    ap.add_argument("--score-sweep", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7",
                    help="comma separated score thresholds to sweep (empty string disables)")
    ap.add_argument("--out", default=None)
    ap.add_argument("-u", "--update", nargs="+", default=None)
    args = ap.parse_args()

    cfg, model, post = build(args.config, args.resume, args.weights, args.device, args.update)
    names = {c["id"]: c["name"] for c in json.load(open(cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]))["categories"]}
    gt_json = cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]
    num_classes = int(cfg.yaml_cfg["num_classes"])

    preds = predict(cfg, model, post, args.device, args.batch_size)
    results = coco_results(preds)
    print(f"[eval] {len(results)} detections over {len(preds)} tiles (score >= {post.nms_score_threshold})")

    report = {"checkpoint": os.path.abspath(args.resume), "weights": args.weights, "gt": gt_json,
              "score_thr": args.score_thr, "match_radius": args.match_radius, "coco": {}, "matched": {}}

    print("\n================ COCO (all classes) ================")
    report["coco"]["all"] = coco_eval(gt_json, results)
    for cid in sorted(names):
        print(f"\n================ COCO (class {cid} = {names[cid]}) ================")
        sub = [r for r in results if r["category_id"] == cid]
        report["coco"][names[cid]] = coco_eval(gt_json, sub, cat_ids=[cid]) if sub else None

    tp, fp, fn, conf, (dtp, dfp, dfn) = centroid_match(gt_json, preds, args.score_thr, args.match_radius, num_classes)
    dp, dr, df = prf(dtp, dfp, dfn)
    print(f"\n===== centroid matching @ score>={args.score_thr}, radius {args.match_radius}px =====")
    print(f"detection (class-agnostic): P {dp:.4f}  R {dr:.4f}  F1 {df:.4f}   (TP {dtp} FP {dfp} FN {dfn})")
    report["matched"]["detection"] = {"precision": dp, "recall": dr, "f1": df, "tp": dtp, "fp": dfp, "fn": dfn}
    macro = []
    for c in range(num_classes):
        p, r, f = prf(tp[c], fp[c], fn[c])
        macro.append(f)
        print(f"  {names.get(c, c):12s} P {p:.4f}  R {r:.4f}  F1 {f:.4f}   (TP {int(tp[c])} FP {int(fp[c])} FN {int(fn[c])})")
        report["matched"][names.get(c, str(c))] = {"precision": p, "recall": r, "f1": f,
                                                   "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c])}
    print(f"  macro-F1 {np.mean(macro):.4f}")
    report["matched"]["macro_f1"] = float(np.mean(macro))
    print("\nconfusion (rows = GT, cols = pred, matched pairs only):")
    print("            " + "".join(f"{names.get(c, c):>14s}" for c in range(num_classes)))
    for c in range(num_classes):
        print(f"{names.get(c, c):>12s}" + "".join(f"{int(conf[c, k]):>14d}" for k in range(num_classes)))
    report["matched"]["confusion"] = conf.tolist()

    # ---- operating-point sweep: the per-class score threshold is not a free lunch, and the class
    # that decides this arm (Lymphocyte) is strongly asymmetric, so report the whole curve.
    if args.score_sweep:
        thrs = [float(t) for t in args.score_sweep.split(",") if t.strip()]
        print(f"\n===== score-threshold sweep (radius {args.match_radius}px) =====")
        hdr = f"{'thr':>6s} {'det_P':>7s} {'det_R':>7s} {'det_F1':>7s} " + \
              "".join(f"{names.get(c, c)[:10]+'_F1':>14s}" for c in range(num_classes)) + f"{'macroF1':>9s}"
        print(hdr)
        sweep = []
        for t in thrs:
            tp_, fp_, fn_, _, (a, b_, c_) = centroid_match(gt_json, preds, t, args.match_radius, num_classes)
            dP, dR, dF = prf(a, b_, c_)
            f1s = [prf(tp_[c], fp_[c], fn_[c])[2] for c in range(num_classes)]
            row = {"thr": t, "det_p": dP, "det_r": dR, "det_f1": dF,
                   "per_class_f1": {names.get(c, str(c)): f1s[c] for c in range(num_classes)},
                   "macro_f1": float(np.mean(f1s))}
            sweep.append(row)
            print(f"{t:6.2f} {dP:7.4f} {dR:7.4f} {dF:7.4f} " +
                  "".join(f"{v:14.4f}" for v in f1s) + f"{np.mean(f1s):9.4f}")
        best_macro = max(sweep, key=lambda r: r["macro_f1"])
        best_lym = max(sweep, key=lambda r: r["per_class_f1"].get("Lymphocyte", 0.0))
        print(f"  best macro-F1      @ thr {best_macro['thr']:.2f}  = {best_macro['macro_f1']:.4f}")
        print(f"  best Lymphocyte-F1 @ thr {best_lym['thr']:.2f}  = {best_lym['per_class_f1'].get('Lymphocyte', 0.0):.4f}")
        report["sweep"] = sweep
        report["best_macro_thr"] = best_macro["thr"]
        report["best_lymphocyte_thr"] = best_lym["thr"]

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "predictions.json"), "w") as f:
            json.dump({"categories": [{"id": k, "name": v} for k, v in sorted(names.items())],
                       "annotations": results}, f)
        with open(os.path.join(args.out, "eval_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.out}/eval_report.json")


if __name__ == "__main__":
    main()
