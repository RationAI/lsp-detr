#!/usr/bin/env python
"""Summarise a det/scripts/run_tils3_wsi.sh output directory: per-slide counts, class mix, geometry,
and a re-verification that each .zst really carries the convention its meta.json claims.

  python det/scripts/summarize_tils3_wsi.py results/tils3-wsi
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys


def check_zst(path, expect):
    """Decode the finished file and return counts / extent / per-category term+nt actually stored."""
    import zstandard
    import mapbox_vector_tile as mvt
    raw = open(path, "rb").read()
    blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=8_000_000_000)
    layer = mvt.decode(blob)["default"]
    feats = layer["features"]
    cats = collections.Counter(f["properties"]["categoryId"] for f in feats)
    terms = {}
    for f in feats:
        p = f["properties"]
        terms.setdefault(p["categoryId"], (p["termId"], bool(p["nt"])))
    bad = [c for c, (t, nt) in terms.items()
           if c not in expect or expect[c][0] != t or expect[c][1] != nt]
    return {"n": len(feats), "extent": layer["extent"], "cats": cats, "terms": terms,
            "mismatched_categories": bad, "size_mb": len(raw) / 2**20}


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "tils3-wsi")
    metas = sorted(glob.glob(os.path.join(out_dir, "*.meta.json")))
    if not metas:
        sys.exit(f"no *.meta.json in {out_dir}")

    print(f"{'slide':32s} {'W x H':>17s} {'mpp':>7s} {'tiles':>7s} {'total':>11s} "
          f"{'non_tumor':>11s} {'tumor':>10s} {'Lymphocyte':>11s}  {'lym%':>5s} {'min':>6s}")
    grand = collections.Counter()
    rows = []
    for mp in metas:
        m = json.load(open(mp))
        t = m["class_totals"]
        n = m["num_detections"]
        lym = t.get("Lymphocyte", 0)
        grand.update(t)
        rows.append((m, mp))
        print(f"{m['slide']:32s} {m['width']:8d}x{m['height']:<8d} "
              f"{(m['mpp'] or 0):7.4f} {m['tiles']:7,d} {n:11,d} "
              f"{t.get('non_tumor', 0):11,d} {t.get('tumor', 0):10,d} {lym:11,d}  "
              f"{100 * lym / max(n, 1):5.1f} {m['elapsed_sec'] / 60:6.1f}")
    tot = sum(grand.values())
    print("-" * 122)
    print(f"{'TOTAL':32s} {'':17s} {'':7s} {'':7s} {tot:11,d} "
          f"{grand['non_tumor']:11,d} {grand['tumor']:10,d} {grand['Lymphocyte']:11,d}  "
          f"{100 * grand['Lymphocyte'] / max(tot, 1):5.1f}")

    print("\n--- .zst re-verification (decode the finished file, not the in-memory arrays) ---")
    for m, mp in rows:
        zp = mp[: -len(".meta.json")] + ".zst"
        expect = {c["categoryId"]: (c["termId"], c["nt"]) for c in m["zst"]["classes"]}
        try:
            r = check_zst(zp, expect)
        except Exception as e:
            print(f"  {os.path.basename(zp):48s} FAILED: {type(e).__name__}: {e}")
            continue
        ok_n = r["n"] == m["num_detections"]
        ok_ext = r["extent"] == m["zst"]["extent"] == m["height"]
        ok_cls = not r["mismatched_categories"]
        by_cat = {c["categoryId"]: m["class_totals"][c["name"]] for c in m["zst"]["classes"]}
        ok_cnt = all(r["cats"].get(c, 0) == v for c, v in by_cat.items())
        flag = "OK " if (ok_n and ok_ext and ok_cls and ok_cnt) else "BAD"
        print(f"  [{flag}] {os.path.basename(zp):50s} {r['n']:10,d} features  extent={r['extent']:<8d}"
              f" {r['size_mb']:6.1f} MB  cats={dict(sorted(r['cats'].items()))}")
        if flag == "BAD":
            print(f"          n_match={ok_n} extent_match={ok_ext} class_encoding={ok_cls} counts={ok_cnt}"
                  f" mismatched={r['mismatched_categories']}")


if __name__ == "__main__":
    main()
