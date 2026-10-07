#!/usr/bin/env python
"""Local replica of the TILs leaderboard's PUMA Track-1 scorer, for use before submitting.

Rules as specified for the server: bbox-centre **mutual nearest neighbour** within **10 px**,
score thresholds **base 0.3 / tumor 0.6**, micro-summed over the 205 ROIs.
This is a replica, not the server: once `check_puma_pred.py` from the kit is available, that is the
authority and this file is only a pre-flight.

  python det/scripts/score_puma_leaderboard.py --pred <pred>.json --gt <puma_t1>.json
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
from scipy.spatial import cKDTree

NAMES = {0: "other", 1: "lymphocyte", 2: "tumor"}


def centres(anns):
    by = collections.defaultdict(list)
    for a in anns:
        x, y, w, h = a["bbox"]
        by[int(a["image_id"])].append((x + w / 2.0, y + h / 2.0, int(a["category_id"]),
                                       float(a.get("score", 1.0))))
    return by


def mutual(pred, gt, radius):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    dp, jg = cKDTree(gt).query(pred, k=1, workers=-1)
    _, jp = cKDTree(pred).query(gt, k=1, workers=-1)
    p = np.array([i for i in range(len(pred)) if dp[i] <= radius and jp[jg[i]] == i], np.int64)
    return p, jg[p].astype(np.int64) if len(p) else np.zeros(0, np.int64)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--thr-base", type=float, default=0.3)
    ap.add_argument("--thr-tumor", type=float, default=0.6)
    args = ap.parse_args()

    P, G = json.load(open(args.pred)), json.load(open(args.gt))
    pid = {int(i["id"]): i["file_name"] for i in P["images"]}
    gid = {int(i["id"]): i["file_name"] for i in G["images"]}
    # match images by stem, not by id: the two files number them independently
    stem = lambda f: f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    g_by_stem = {stem(v): k for k, v in gid.items()}
    missing = [stem(v) for v in pid.values() if stem(v) not in g_by_stem]
    if missing:
        raise SystemExit(f"{len(missing)} predicted images are not in the GT (e.g. {missing[:3]})")
    print(f"[data] pred images {len(pid)} / GT images {len(gid)}  radius {args.radius}px  "
          f"thr base {args.thr_base} tumor {args.thr_tumor}")

    pc, gc = centres(P["annotations"]), centres(G["annotations"])
    thr = {0: args.thr_base, 1: args.thr_base, 2: args.thr_tumor}
    joint = {c: [0, 0, 0] for c in NAMES}
    det = [0, 0, 0]
    conf = np.zeros((3, 3), np.int64)

    for p_img, fn in pid.items():
        g_img = g_by_stem[stem(fn)]
        pr = [t for t in pc.get(p_img, []) if t[3] >= thr[t[2]]]
        gt = gc.get(g_img, [])
        pxy = np.array([[t[0], t[1]] for t in pr], np.float64).reshape(-1, 2)
        plb = np.array([t[2] for t in pr], np.int64)
        gxy = np.array([[t[0], t[1]] for t in gt], np.float64).reshape(-1, 2)
        glb = np.array([t[2] for t in gt], np.int64)
        for c in NAMES:                                   # joint: match within the class
            pi, _ = mutual(pxy[plb == c], gxy[glb == c], args.radius)
            tp = len(pi)
            joint[c][0] += tp
            joint[c][1] += int((plb == c).sum()) - tp
            joint[c][2] += int((glb == c).sum()) - tp
        pi, gj = mutual(pxy, gxy, args.radius)            # class-agnostic detection
        det[0] += len(pi); det[1] += len(pxy) - len(pi); det[2] += len(gxy) - len(pi)
        for a, b in zip(gj, pi):
            conf[glb[a], plb[b]] += 1

    print(f"\n{'class':<12}{'GT':>9}{'pred':>9}{'TP':>9}{'precision':>11}{'sensitivity':>13}{'F1':>9}")
    print("-" * 72)
    f1 = {}
    out = {"per_class": {}}
    for c in NAMES:
        p, r, f = prf(*joint[c])
        f1[c] = f
        out["per_class"][NAMES[c]] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4),
                                      "tp": joint[c][0], "fp": joint[c][1], "fn": joint[c][2]}
        print(f"{NAMES[c]:<12}{joint[c][0]+joint[c][2]:>9,}{joint[c][0]+joint[c][1]:>9,}"
              f"{joint[c][0]:>9,}{p:>11.4f}{r:>13.4f}{f:>9.4f}")
    dp, dr, df = prf(*det)
    matched = int(conf.sum())
    acc = float(np.trace(conf) / matched) if matched else 0.0
    macro = float(np.mean(list(f1.values())))
    out.update({"detection": {"P": round(dp, 4), "R": round(dr, 4), "F1": round(df, 4)},
                "macro_F1": round(macro, 4), "accuracy_3class": round(acc, 4),
                "matched": matched, "confusion_gt_pred": conf.tolist()})
    print("-" * 72)
    print(f"{'detection':<12}{det[0]+det[2]:>9,}{det[0]+det[1]:>9,}{det[0]:>9,}{dp:>11.4f}{dr:>13.4f}{df:>9.4f}")
    print(f"\n  tumor F1      {f1[2]:.4f}")
    print(f"  lymph F1      {f1[1]:.4f}")
    print(f"  other F1      {f1[0]:.4f}")
    print(f"  macro-F1      {macro:.4f}   (mean of the three)")
    print(f"  detection F1  {df:.4f}")
    print(f"  3-class acc   {acc:.4f}   (over {matched:,} class-agnostically matched cells)")
    print("\n혼동행렬 (행=GT, 열=예측)   " + "".join(f"{NAMES[c]:>12}" for c in NAMES))
    for c in NAMES:
        print(f"{NAMES[c]:>26}" + "".join(f"{v:>12,}" for v in conf[c]))
    print("\n" + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
