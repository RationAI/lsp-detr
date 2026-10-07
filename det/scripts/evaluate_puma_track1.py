#!/usr/bin/env python
"""Score the 3-class LSP-DETR-det arm on the PUMA Track 1 training ROIs.

Scoring is a port of ``KongNet_Inference_Main/evaluate_puma_track1.py`` -- same GT parsing
(area-weighted polygon centroid), same Track-1 class folding, same 6 um matching radius, same
greedy nearest-first one-to-one assignment, same two matchings and the same printed tables -- so the
numbers are directly comparable to the KongNet runs in ``/mnt/14T-2/tksong/PUMA/eval_logs/``.
Only how the *predictions* are produced differs.

  joint        a prediction counts for class c only if it lands on a class-c nucleus
               (detection and classification folded together)
  class-blind   everything matched ignoring class, matched pairs form a confusion matrix
               (isolates the classifier)

Prediction path (the part that is ours):
  * PUMA ROIs are 1024x1024 at **0.23 um/px**; this arm runs at **0.5 um/px**, so each ROI is
    resampled by ``model_mpp / roi_mpp`` (2.174x down) before the forward and every predicted
    centre is mapped straight back to ROI pixels. Feeding 0.23 um/px pixels to a 0.5 um/px model
    is a 2.2x scale error and is the single biggest thing that can silently ruin this number
    (cf. the TILs-images resolution mismatch), hence ``--scale-sweep`` to show the curve.
  * the resampled ROI (471x471) is reflect-padded to the model's 1536 input and centred, exactly
    like the training small-image path; the model only fires in the centre 672 window, which the
    ROI sits well inside. Detections whose centre falls outside the real-image rectangle (i.e. in
    the reflected margin) are dropped.
  * class map: this arm's ``1 tumor`` -> PUMA ``0 tumour``, ``2 Lymphocyte`` -> ``1 TIL``,
    ``0 non_tumor`` -> ``2 other``.

  python det/scripts/evaluate_puma_track1.py -r <run>/best_stg1.pth --conf-threshold 0.4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

DET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DET_ROOT)

Image.MAX_IMAGE_PIXELS = None

# ---- verbatim from KongNet_Inference_Main/evaluate_puma_track1.py (keep identical) -------------
NAMES = ["tumour", "TIL", "other"]
TUMOUR, TIL, OTHER = 0, 1, 2
TRACK1 = {
    "nuclei_tumor": TUMOUR,
    "nuclei_lymphocyte": TIL,
    "nuclei_plasma_cell": TIL,
    "nuclei_histiocyte": OTHER,
    "nuclei_melanophage": OTHER,
    "nuclei_neutrophil": OTHER,
    "nuclei_stroma": OTHER,
    "nuclei_epithelium": OTHER,
    "nuclei_endothelium": OTHER,
    "nuclei_apoptosis": OTHER,
}
# this arm's label ids -> PUMA Track 1 ids
LSP_TO_TRACK1 = {0: OTHER, 1: TUMOUR, 2: TIL}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def polygon_centroid(ring):
    """Area-weighted centroid; falls back to the vertex mean for a degenerate ring."""
    a = np.asarray(ring, np.float64)
    if len(a) < 3:
        return a.mean(0) if len(a) else np.zeros(2)
    x, y = a[:, 0], a[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return a.mean(0)
    return np.array([((x + x2) * cross).sum(), ((y + y2) * cross).sum()]) / (6.0 * area)


def load_gt(path):
    data = json.load(open(path))
    feats = data["features"] if isinstance(data, dict) else data
    pts, lab = [], []
    for f in feats:
        cls = (f.get("properties") or {}).get("classification") or {}
        c = TRACK1.get(cls.get("name") if isinstance(cls, dict) else None)
        if c is None:
            continue
        geom = f["geometry"]
        rings = geom["coordinates"]
        if geom["type"] == "MultiPolygon":
            rings = max(rings, key=lambda r: len(r[0]))
        pts.append(polygon_centroid(rings[0]))
        lab.append(c)
    return np.asarray(pts, np.float64).reshape(-1, 2), np.asarray(lab, np.int64)


def greedy_pairs(pred, gt, radius):
    """Nearest-first one-to-one assignment within `radius`; returns [(p, g), ...]."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    edges = []
    for p, cands in enumerate(cKDTree(gt).query_ball_point(pred, r=radius)):
        for g in cands:
            edges.append((float(np.linalg.norm(pred[p] - gt[g])), p, g))
    edges.sort()
    out, tp, tg = [], set(), set()
    for _, p, g in edges:
        if p in tp or g in tg:
            continue
        tp.add(p); tg.add(g); out.append((p, g))
    return out
# ---- end verbatim ------------------------------------------------------------------------------


def mutual_pairs(pred, gt, radius):
    """Reciprocal (mutual) nearest-neighbour assignment within `radius`.

    A pair (p, g) is kept only when g is p's nearest GT *and* p is g's nearest prediction. That is
    one-to-one by construction and, unlike the greedy variant, does not let a prediction claim a
    nucleus that is somebody else's obvious partner - it is the stricter, order-independent option.
    """
    if len(pred) == 0 or len(gt) == 0:
        return []
    dp, jg = cKDTree(gt).query(pred, k=1, workers=-1)     # each pred -> nearest gt
    _, jp = cKDTree(pred).query(gt, k=1, workers=-1)      # each gt   -> nearest pred
    return [(int(p), int(jg[p])) for p in range(len(pred))
            if dp[p] <= radius and jp[jg[p]] == p]


MATCHERS = {"greedy": greedy_pairs, "mutual": mutual_pairs}


def roi_mpp_from_tiff(path, default):
    """PUMA ROIs carry XResolution in px/cm (unit 3); the released docs round it to 0.23."""
    try:
        with Image.open(path) as im:
            t = getattr(im, "tag_v2", None)
            if t is None:
                return default
            xr, unit = t.get(282), t.get(296)
            if xr and unit == 3:
                return 10000.0 / float(xr)
            if xr and unit == 2:
                return 25400.0 / float(xr)
    except Exception:
        pass
    return default


def build(args):
    """LSPDetrDetection + DomePostProcessor through det/inference.py's own builder."""
    import torch
    import inference as det_inference

    build_args = types.SimpleNamespace(
        config=args.config, resume=args.resume, weights=args.weights, update=list(args.update),
        nms_iou_threshold=args.nms_iou_threshold, nms_score_threshold=args.nms_score_threshold,
        num_top_queries=args.num_top_queries, use_nms=None, class_agnostic_nms=None,
        input_size=args.input_size, allow_arm_mismatch=args.allow_arm_mismatch)
    model, pp, cfg, input_size, meta = det_inference.build_model_and_postprocessor(build_args)
    det_inference.apply_tf32_policy(args.tf32)
    device = torch.device(args.device)
    return model.eval().to(device), pp.to(device), input_size, meta, device, det_inference


@np.errstate(all="ignore")
def run_rois(args, rois, gt_dir, model, pp, input_size, device, det_inference, model_mpp):
    """Predict on every ROI resampled to `model_mpp`.

    Returns {stem: (centres in ROI px, labels, scores, n_raw, scale)} where
    ``scale = roi_px per model_px``; it is per file because each PUMA ROI carries its own
    XResolution (0.2262-0.2274 um/px), so "resample to 0.5 mpp" is not one global factor.
    """
    import torch
    import torch.nn.functional as F

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = {}
    batch_imgs, batch_keys, batch_geo = [], [], []

    def flush():
        if not batch_imgs:
            return
        n = len(batch_imgs)
        x = torch.cat(batch_imgs, 0)
        if n < args.batch_size:                      # flex_attention is compiled dynamic=False
            x = torch.cat([x, x[-1:].expand(args.batch_size - n, -1, -1, -1)], 0)
        x = (x - mean) / std
        sizes = torch.full((x.shape[0], 2), float(input_size), device=device)
        with torch.no_grad():
            o = model(x)
            labels, boxes, scores, _ = pp(o, sizes)
        for i in range(n):
            lab = labels[i].cpu().numpy()
            box = boxes[i].cpu().numpy()             # xyxy in the 1536 canvas
            sc = scores[i].cpu().numpy()
            pl, pt, w_m, h_m, scale = batch_geo[i]
            cx = (box[:, 0] + box[:, 2]) * 0.5 - pl  # -> resampled-ROI pixels
            cy = (box[:, 1] + box[:, 3]) * 0.5 - pt
            keep = (cx >= 0) & (cx < w_m) & (cy >= 0) & (cy < h_m)   # drop the reflected margin
            out[batch_keys[i]] = (np.stack([cx[keep] * scale, cy[keep] * scale], 1),
                                  np.array([LSP_TO_TRACK1[int(v)] for v in lab[keep]], np.int64),
                                  sc[keep],
                                  int(len(sc)),      # raw count, to catch top-k truncation
                                  scale)
        batch_imgs.clear(); batch_keys.clear(); batch_geo.clear()

    for i, roi in enumerate(rois):
        stem = os.path.splitext(os.path.basename(roi))[0]
        mpp_i = roi_mpp_from_tiff(roi, args.mpp) if args.mpp_mode == "auto" else args.mpp
        scale = model_mpp / mpp_i
        img = np.asarray(Image.open(roi).convert("RGB"))     # PUMA ROIs are RGBA on disk
        h, w = img.shape[:2]
        w_m, h_m = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
        t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(255.0).to(device)
        if (w_m, h_m) != (w, h):
            t = F.interpolate(t, size=(h_m, w_m), mode="bilinear", align_corners=False, antialias=True)
        if w_m > input_size or h_m > input_size:
            raise SystemExit(f"{stem}: resampled ROI {w_m}x{h_m} exceeds the {input_size} model input")
        pl, pt = (input_size - w_m) // 2, (input_size - h_m) // 2
        t = det_inference.reflect_pad(t, pl, input_size - w_m - pl, pt, input_size - h_m - pt)
        batch_imgs.append(t); batch_keys.append(stem); batch_geo.append((pl, pt, w_m, h_m, scale))
        if len(batch_imgs) == args.batch_size:
            flush()
        if args.log_every and (i + 1) % args.log_every == 0:
            print(f"  {i+1}/{len(rois)} ROI", flush=True)
    flush()
    return out


def score(preds, rois, gt_dir, radius, thr, matcher="greedy", frame="roi"):
    """Thresholding lives here, not in run_rois, so one forward pass can serve a --conf-sweep.

    `thr` is a per-PUMA-class array. `frame='model'` matches in the resampled 0.5 mpp grid (radius in
    model px); `frame='roi'` matches in ROI pixels (radius in ROI px).
    """
    match = MATCHERS[matcher]
    joint = {c: [0, 0, 0] for c in range(3)}
    det = [0, 0, 0]
    conf = np.zeros((3, 3), np.int64)
    n_gt = np.zeros(3, np.int64)
    n_pred = np.zeros(3, np.int64)
    truncated = 0
    for roi in rois:
        stem = os.path.splitext(os.path.basename(roi))[0]
        gt_path = os.path.join(gt_dir, f"{stem}_nuclei.geojson")
        if not os.path.exists(gt_path):
            print(f"  주석 없음, 건너뜀: {stem}")
            continue
        gt_pts, gt_lab = load_gt(gt_path)
        p_pts, p_lab, p_sc, n_raw, scale = preds[stem]
        m = p_sc >= np.asarray(thr, np.float32)[p_lab] if len(p_lab) else np.zeros(0, bool)
        p_pts, p_lab = p_pts[m], p_lab[m]
        truncated += int(n_raw >= 0.98 * TOPK[0])
        if frame == "model":                      # match in the 0.5 mpp grid the model saw
            p_pts, gt_pts = p_pts / scale, gt_pts / scale
        for c in range(3):
            g = gt_pts[gt_lab == c]
            p = p_pts[p_lab == c]
            n_gt[c] += len(g); n_pred[c] += len(p)
            tp = len(match(p, g, radius))
            joint[c][0] += tp; joint[c][1] += len(p) - tp; joint[c][2] += len(g) - tp
        pairs = match(p_pts, gt_pts, radius)
        det[0] += len(pairs); det[1] += len(p_pts) - len(pairs); det[2] += len(gt_pts) - len(pairs)
        for p, g in pairs:
            conf[gt_lab[g], p_lab[p]] += 1
    return joint, det, conf, n_gt, n_pred, truncated


TOPK = [2000]


def report(label, joint, det, conf, n_gt, n_pred, n_roi, radius, radius_desc, meta, extra):
    matched = conf.sum()
    result = {"label": label, "n_roi": n_roi, "radius_px": round(radius, 2),
              "confusion_gt_pred": conf.tolist(), "n_gt": n_gt.tolist(), "n_pred": n_pred.tolist(),
              "classes": NAMES, "model": meta, **extra}
    print(f"\n{'='*94}\n{label}   ROI {n_roi}장 · 매칭 반경 {radius_desc}\n{'='*94}")
    print("\n[검출 + 분류]  탐지 위치와 클래스가 모두 맞아야 TP")
    print(f"{'클래스':<10}{'GT':>9}{'예측':>9}{'precision':>11}{'sensitivity':>13}{'F1':>9}")
    print("-" * 61)
    f1s = []
    result["joint"] = {}
    for c in range(3):
        p, r, f = prf(*joint[c])
        f1s.append(f)
        result["joint"][NAMES[c]] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4),
                                     "tp": joint[c][0], "fp": joint[c][1], "fn": joint[c][2]}
        print(f"{NAMES[c]:<10}{n_gt[c]:>9,}{n_pred[c]:>9,}{p:>11.4f}{r:>13.4f}{f:>9.4f}")
    dp, dr, df = prf(*det)
    result["joint"]["macro_F1"] = round(float(np.mean(f1s)), 4)
    result["detection"] = {"P": round(dp, 4), "R": round(dr, 4), "F1": round(df, 4)}
    print("-" * 61)
    print(f"{'macro':<10}{n_gt.sum():>9,}{n_pred.sum():>9,}{'':>11}{'':>13}{np.mean(f1s):>9.4f}")
    print(f"{'검출(무관)':<10}{'':>9}{'':>9}{dp:>11.4f}{dr:>13.4f}{df:>9.4f}")

    print(f"\n[분류만]  클래스 무관 매칭된 {matched:,}개 세포에 대한 라벨 정확도")
    print(f"{'클래스':<10}{'accuracy':>10}{'precision':>11}{'sensitivity':>13}{'F1':>9}{'support':>10}")
    print("-" * 63)
    result["classification"] = {}
    for c in range(3):
        tp = conf[c, c]; fn = conf[c].sum() - tp; fp = conf[:, c].sum() - tp
        tn = matched - tp - fp - fn
        acc = (tp + tn) / matched if matched else 0.0
        p, r, f = prf(tp, fp, fn)
        result["classification"][NAMES[c]] = {"accuracy": round(float(acc), 4), "P": round(p, 4),
                                              "R": round(r, 4), "F1": round(f, 4),
                                              "support": int(conf[c].sum())}
        print(f"{NAMES[c]:<10}{acc:>10.4f}{p:>11.4f}{r:>13.4f}{f:>9.4f}{conf[c].sum():>10,}")
    overall = np.trace(conf) / matched if matched else 0.0
    result["classification"]["overall_accuracy"] = round(float(overall), 4)
    print("-" * 63)
    print(f"{'전체':<10}{overall:>10.4f}")
    print("\n혼동행렬 (행 = 정답, 열 = 예측)")
    print(f"{'':<10}" + "".join(f"{n:>10}" for n in NAMES))
    for c in range(3):
        print(f"{NAMES[c]:<10}" + "".join(f"{v:>10,}" for v in conf[c]))
    return result


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(DET_ROOT, "configs", "LSP-T-tils3.yml"))
    ap.add_argument("-r", "--resume", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("-u", "--update", nargs="+", default=[])
    ap.add_argument("--label", default=None)
    ap.add_argument("--roi_dir", default="/mnt/14T-2/tksong/PUMA/roi_tif/01_training_dataset_tif_ROIs")
    ap.add_argument("--gt_dir", default="/mnt/14T-2/tksong/PUMA/geojson_nuclei/01_training_dataset_geojson_nuclei")
    ap.add_argument("--exts", default=".tif,.tiff",
                    help="ROI extensions; '.tif' alone selects the 205 released as .tif "
                         "(v5 re-added metastatic_roi_103 as .tiff, making 206)")
    ap.add_argument("--radius_um", type=float, default=6.0,
                    help="matching radius in microns (ignored when --radius-px is given)")
    ap.add_argument("--radius-px", type=float, default=None,
                    help="matching radius in pixels of the MODEL grid (--model-mpp); overrides --radius_um")
    ap.add_argument("--matching", default="greedy", choices=sorted(MATCHERS),
                    help="greedy = nearest-first one-to-one (KongNet protocol); "
                         "mutual = reciprocal nearest neighbour")
    ap.add_argument("--mpp", type=float, default=0.23, help="PUMA ROI mpp fallback (40x)")
    ap.add_argument("--mpp-mode", default="auto", choices=["auto", "fixed"],
                    help="auto = per-file XResolution tag (0.2262-0.2274); fixed = --mpp for every ROI")
    ap.add_argument("--model-mpp", type=float, default=0.5, help="the mpp this arm is run at")
    ap.add_argument("--scale-sweep", default=None,
                    help="comma separated model-mpp values to try (diagnostic; detection F1 only)")
    ap.add_argument("--conf-threshold", type=float, default=0.4, help="default score threshold")
    ap.add_argument("--class-conf", default=None,
                    help="per-class overrides, e.g. 'tumour=0.6' (names: tumour,TIL,other)")
    ap.add_argument("--conf-sweep", default=None, help="comma separated score thresholds (diagnostic)")
    ap.add_argument("--nms-iou-threshold", type=float, default=None)
    ap.add_argument("--nms-score-threshold", type=float, default=None)
    ap.add_argument("--num-top-queries", type=int, default=8000,
                    help="raised from the yml's 2000: an ROI padded into 1536 also yields detections in "
                         "the reflected margin, which would otherwise eat the budget")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--allow-arm-mismatch", action="store_true")
    ap.add_argument("-d", "--device", default="cuda")
    ap.add_argument("-b", "--batch-size", type=int, default=4)
    ap.add_argument("--tf32", default="keep", choices=["keep", "off", "on"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    TOPK[0] = args.num_top_queries

    rois = sorted(sum((glob.glob(os.path.join(args.roi_dir, "*" + e.strip()))
                       for e in args.exts.split(",") if e.strip()), []))
    if args.limit:
        rois = rois[: args.limit]

    thr = np.full(3, float(args.conf_threshold), np.float32)
    if args.class_conf:
        for tok in args.class_conf.split(","):
            k, v = tok.split("=")
            if k.strip() not in NAMES:
                raise SystemExit(f"--class-conf: unknown class {k!r} (have {NAMES})")
            thr[NAMES.index(k.strip())] = float(v)

    if args.radius_px is not None:                 # radius given in the model (0.5 mpp) grid
        frame, radius = "model", float(args.radius_px)
        radius_desc = f"{radius:.1f}px @ {args.model_mpp}mpp = {radius * args.model_mpp:.2f}um"
    else:                                          # KongNet protocol: microns -> ROI pixels
        frame, radius = "roi", args.radius_um / args.mpp
        radius_desc = f"{radius:.2f}px @ {args.mpp}mpp = {args.radius_um}um"
    label = args.label or f"LSP-DETR-det 3class ({os.path.basename(os.path.dirname(args.resume))})"
    print(f"[data] {len(rois)} ROI ({args.exts})  ·  matching {args.matching} in the {frame} frame, "
          f"radius {radius_desc}")
    print(f"[thr ] " + "  ".join(f"{NAMES[c]}>={thr[c]:.2f}" for c in range(3)))

    model, pp, input_size, meta, device, det_inference = build(args)

    if args.scale_sweep:
        print("\n===== scale sweep (diagnostic; class-agnostic detection only) =====")
        print(f"{'model_mpp':>10}{'resample':>10}{'ROI px':>9}{'pred':>10}{'det_P':>8}{'det_R':>8}{'det_F1':>8}")
        for mm in [float(v) for v in args.scale_sweep.split(",")]:
            preds = run_rois(args, rois, args.gt_dir, model, pp, input_size, device, det_inference, mm)
            _, det, _, _, n_pred, _ = score(preds, rois, args.gt_dir, radius, thr, args.matching, frame)
            p, r, f = prf(*det)
            sc = mm / args.mpp
            print(f"{mm:>10.3f}{sc:>10.3f}{int(round(1024/sc)):>9}{int(n_pred.sum()):>10,}"
                  f"{p:>8.4f}{r:>8.4f}{f:>8.4f}", flush=True)
        return

    t0 = time.time()
    preds = run_rois(args, rois, args.gt_dir, model, pp, input_size, device, det_inference, args.model_mpp)
    scales = np.array([v[4] for v in preds.values()])
    print(f"[predict] {len(preds)} ROI in {time.time()-t0:.1f}s (resample "
          f"{scales.min():.3f}-{scales.max():.3f}x -> ~{int(round(1024/scales.mean()))}px, centred in {input_size})")

    sweeps = [[float(v)] * 3 for v in args.conf_sweep.split(",")] if args.conf_sweep else [thr]
    result = None
    for t in sweeps:
        t = np.asarray(t, np.float32)
        joint, det, conf, n_gt, n_pred, trunc = score(preds, rois, args.gt_dir, radius, t,
                                                      args.matching, frame)
        if trunc:
            print(f"[warn] {trunc} ROI hit >=98% of the top-k budget ({args.num_top_queries}); raise --num-top-queries")
        tag = "/".join(f"{NAMES[c]}>={t[c]:g}" for c in range(3))
        result = report(f"{label}  [{tag}]", joint, det, conf, n_gt, n_pred, len(rois), radius,
                        radius_desc, meta,
                        {"thresholds": {NAMES[c]: float(t[c]) for c in range(3)},
                         "matching": args.matching, "match_frame": frame,
                         "radius": radius, "radius_desc": radius_desc,
                         "model_mpp": args.model_mpp, "mpp_mode": args.mpp_mode,
                         "roi_mpp_range": [float(args.model_mpp / scales.max()),
                                           float(args.model_mpp / scales.min())],
                         "exts": args.exts, "checkpoint": os.path.abspath(args.resume)})
    if args.out_json and result:
        json.dump(result, open(args.out_json, "w"), indent=1, ensure_ascii=False)
        print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
