#!/usr/bin/env python3
"""LSP-DETR (bbox port) whole-slide inference  ->  MVT/zstd (.zst) + COCO json.

Counterpart of ``det/inference.py`` for inputs that do not fit in memory: a WSI is streamed
tile by tile through the same model/postprocessor and the detections are written in the
container the AIVIS viewer reads (zstd-compressed Mapbox Vector Tile), i.e. the same output
``AIVIS-Dome-DETR/integrated_wsi_infer.py`` + ``lsp-detr/wsi/normalize_domedetr_zst.py``
produce for the 2-class tumour/non-tumour arm.

Geometry (identical to the Dome-DETR tumour/non-tumour WSI recipe, cf.
``lsp-detr/wsi/run_domedetr_wsi.sh``: ``--patch-size 1536 --step-size 672 --filter-size 672``):
  * the slide is read at the pyramid level closest to ``--target-mpp`` (0.5, the mpp the arm
    is run at); a level within ``--mpp-tolerance`` is used as is, otherwise tiles are read
    larger/smaller and resized so the model always sees ``--target-mpp`` pixels;
  * detections are kept only when their centre falls in the tile's central
    ``--filter-size`` window - the recipe only has GT there - and the window is also the
    stride, so every detection is emitted exactly once;
  * tiles at the slide border are reflect-padded, like the training small-image path.

Outputs (in ``--output-dir``; base name = the slide's file name *including its extension* plus
``_<--tag>``, e.g. ``129S.tif_LSPDETR-TNT``, like the platform's own
``SSMH_BRS_HE_051.i2syntax_LSPDETR_512.zst``. Keeping the extension is what lets a consumer resolve
the base back to the slide file on disk (normalize_domedetr_zst.py does exactly that); those
consumers recover the slide name by stripping the whole suffix they were told to expect, so a
hyphenated one-underscore tag is safe. ``--output-name`` overrides the base name wholesale):
  * ``<name>.zst``       MVT layer "default", extent = level-0 slide height, one Point per
                         detection at the bbox centre, stored y-flipped (y_mvt = h_wsi - y_img).
                         Viewer convention picked by --zst-convention (default: by num_classes) -
                         see ZST_CONVENTIONS: 'tnt' (2-class) tumour 1 / non-tumour 5;
                         'tnt-lymph' (3-class) tumour 1 / non-tumour 2 / lymphocyte 3.
  * ``<name>.json``      COCO predictions in level-0 pixels (``--no-json`` to skip)
  * ``<name>.meta.json`` run metadata (model, geometry, mpp, per-class counts, timings)

Multi-GPU: ``--gpus 0,1,2,3`` shards the tile list over one worker process per GPU (each
worker re-runs this file with ``--worker-index``) and the parent merges the shards.

Usage: see det/run_inference.sh (WSI=... or a .tif/.svs/.ndpi/.mrxs/.i2syntax input).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import numpy as np

DET_ROOT = os.path.dirname(os.path.abspath(__file__))

# Philips .isyntax/.i2syntax: OpenSlide cannot open these at all, so they are read through the
# pixel-engine wrapper vendored in TILs-Inference (utils.read_isyntax + philips_pixel_engine).
# $TILS_ROOT relocates that checkout (default: a sibling clone of AIVIS-inc/TILs-Inference next to
# this repo); TILs-Inference/env.sh puts the same two directories on PYTHONPATH, so a caller that
# already sourced it needs nothing here. Only .isyntax/.i2syntax input touches it.
ISYNTAX_EXT = (".isyntax", ".i2syntax")
TILS_ROOT = os.environ.get("TILS_ROOT") or os.path.abspath(os.path.join(DET_ROOT, "..", "..", "TILs-Inference"))
if DET_ROOT not in sys.path:
    sys.path.insert(0, DET_ROOT)

# Viewer conventions: model label id -> (categoryId, termId, nt, name). `nt` is the viewer's
# "non-tumour" flag, i.e. everything that is not the tumour class (so the T/NT display mode still
# reads a lymphocyte as non-tumour).
#   'tnt'        the 2-class tumour/non-tumour arm, cf. lsp-detr/wsi/normalize_domedetr_zst.py
#   'tnt-lymph'  the platform's 3-class encoding - identical to what det/combine_tnt_lymph.py writes
#                and to the KongNet reference file SSMH_BRS_HE_106.i2syntax_KongNet.zst
#                (verified by decoding it: categoryId 1/2/3 with these termIds, plus imageId and
#                positivity_rank, which write_zst already emits).
#                NOTE this remaps non-tumour from the 'tnt' arm's 5 / 67bc099c... to 2 / 66d54d2a...
ZST_CONVENTIONS = {
    "tnt": [("5", "67bc099c32a0394aeaccc959", True,  "Non-tumor"),
            ("1", "66d54cd789181badfeac2d69", False, "Tumor")],
    "tnt-lymph": [("2", "66d54d2a89181badfeac2d75", True,  "non_tumor"),
                  ("1", "66d54cd789181badfeac2d69", False, "tumor"),
                  ("3", "66d54cee89181badfeac2d6d", True,  "Lymphocyte")],
    # 'tnt-lymph-nt1' is 'tnt-lymph' with the tumour/non-tumour *numbers* swapped, on request
    # (2026-08-27). Each class keeps its own termId and nt flag - only which number carries which
    # term changes - so a consumer that resolves classes by termId sees no difference, while one
    # that reads categoryId sees non-tumour at 1 and tumour at 2. Note every other .zst on this box
    # (Dome, KongNet, LSPDETR, AD1) puts tumour at 1, so do not mix the two in one viewer session.
    "tnt-lymph-nt1": [("1", "66d54d2a89181badfeac2d75", True,  "non_tumor"),
                      ("2", "66d54cd789181badfeac2d69", False, "tumor"),
                      ("3", "66d54cee89181badfeac2d6d", True,  "Lymphocyte")],
}
# num_classes -> convention used when --zst-convention is 'auto'
AUTO_CONVENTION = {2: "tnt", 3: "tnt-lymph"}


def resolve_convention(name: str, num_classes: int, class_names=None):
    """(convention_name, [(categoryId, termId, nt, name)] indexed by model label)."""
    if name == "auto":
        if num_classes not in AUTO_CONVENTION:
            raise SystemExit(f"no default .zst convention for num_classes={num_classes}; "
                             f"pass --zst-convention {'|'.join(ZST_CONVENTIONS)}")
        name = AUTO_CONVENTION[num_classes]
    if name not in ZST_CONVENTIONS:
        raise SystemExit(f"unknown --zst-convention {name!r} (have {sorted(ZST_CONVENTIONS)})")
    table = list(ZST_CONVENTIONS[name])
    if len(table) != num_classes:
        raise SystemExit(f"--zst-convention {name!r} encodes {len(table)} classes but the model has "
                         f"{num_classes}; the label ids would be mis-mapped")
    if class_names:
        if len(class_names) != num_classes:
            raise SystemExit(f"--class-names has {len(class_names)} entries for {num_classes} classes")
        table = [(c, t, nt, n) for (c, t, nt, _), n in zip(table, class_names)]
    return name, table
# The Dome-DETR tumour/non-tumour arm is run at 0.5 mpp (integrated_wsi_infer.py --target-mpp).
TARGET_MPP = 0.5

DET_DTYPE = np.dtype([("x1", np.float32), ("y1", np.float32), ("x2", np.float32), ("y2", np.float32),
                      ("score", np.float32), ("label", np.int8)])


def _bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "on")


# --------------------------------------------------------------------------------------- slide
def _isyntax_reader():
    """(initial_pixelengine, get_view_wsi, get_level_dimensions, read_region) from TILs-Inference.

    Imported lazily and only for Philips slides: the pixel engine is a native SDK, and every
    other format must keep working on a box that does not have it.
    """
    try:
        import utils.read_isyntax as r
    except ImportError:
        for sub in ("data/readers", "data"):        # -> utils.*, philips_pixel_engine.*
            d = os.path.join(TILS_ROOT, sub)
            if os.path.isdir(d) and d not in sys.path:
                sys.path.insert(0, d)
        import utils.read_isyntax as r
    return r.initial_pixelengine, r.get_view_wsi, r.get_level_dimensions, r.read_region


class Slide:
    """OpenSlide-backed reader exposing the bounds rectangle when the format declares one.

    ``read`` takes bounds-relative level-0 coordinates; every other coordinate in this script
    (tile origins, boxes, the MVT extent) is bounds-relative too, which is what the Dome-DETR
    writer does for .mrxs (integrated_wsi_infer.py: bound_x/bound_y).
    """

    def __init__(self, path: str, source_mpp: Optional[float] = None):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.is_isyntax = os.path.splitext(self.path)[1].lower() in ISYNTAX_EXT
        self.osr = self.view = None
        self.off_x = self.off_y = 0
        self.mpp_source = "--source-mpp" if source_mpp else None

        if self.is_isyntax:
            self._init_isyntax()
        else:
            import openslide

            self.osr = openslide.OpenSlide(self.path)
            self.level_dims = list(self.osr.level_dimensions)
            self.level_downsamples = [float(d) for d in self.osr.level_downsamples]
            self._apply_bounds()
            if not source_mpp:
                self.mpp = self._detect_mpp()
        self.width, self.height = self.level_dims[0]
        if source_mpp:
            self.mpp = float(source_mpp)

    def _init_isyntax(self) -> None:
        """Philips pixel engine instead of OpenSlide.

        The engine declares no bounds rectangle, so off_x/off_y stay 0 and this class's frame is
        already the level-0 absolute one. ``samplesPerPixel`` is samples per micron, hence mpp is
        its reciprocal. Reads are thread-safe - 24 regions read serially and through an 8-thread
        pool come back byte-identical - so ``--reader-threads`` needs no clamping here.
        """
        initial_pixelengine, get_view_wsi, get_level_dimensions, _ = _isyntax_reader()
        facade = initial_pixelengine(self.path)
        if not facade:
            raise RuntimeError(f"pixel engine could not open {self.path}")
        self._facade = facade                       # the view borrows from it: keep it alive
        self.view = get_view_wsi(facade)
        dims, downs = get_level_dimensions(self.view, return_downsmaple_ratio=True)
        self.level_dims = [(int(w), int(h)) for w, h in dims]
        self.level_downsamples = [float(d[0]) for d in downs]
        if self.mpp_source is None:
            self.mpp = 1.0 / float(self.view.samplesPerPixel())
            self.mpp_source = "philips samplesPerPixel"

    def _apply_bounds(self) -> None:
        import openslide

        p = self.osr.properties
        try:
            bx, by = int(p[openslide.PROPERTY_NAME_BOUNDS_X]), int(p[openslide.PROPERTY_NAME_BOUNDS_Y])
            bw, bh = int(p[openslide.PROPERTY_NAME_BOUNDS_WIDTH]), int(p[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
        except (KeyError, TypeError, ValueError):
            return
        if bw <= 0 or bh <= 0:
            return
        cw, ch = self.level_dims[0]
        self.off_x, self.off_y = bx, by
        self.level_dims = [(max(1, min(fw, math.ceil(bw / ds))), max(1, min(fh, math.ceil(bh / ds))))
                           for (fw, fh), ds in zip(self.level_dims, self.level_downsamples)]
        print(f"[bounds] {bw}x{bh} at (+{bx}, +{by}) of the {cw}x{ch} canvas", flush=True)

    def _detect_mpp(self) -> Optional[float]:
        """openslide.mpp-x, else the TIFF resolution tags (generic-tiff exposes no mpp property)."""
        import openslide

        mpp = self.osr.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp:
            self.mpp_source = "openslide.mpp-x"
            return float(mpp)
        try:
            import tifffile

            with tifffile.TiffFile(self.path) as tf:
                page = tf.pages[0]
                num, den = page.tags["XResolution"].value
                unit = int(page.tags["ResolutionUnit"].value)      # 2 = INCH, 3 = CENTIMETER
                px_per_unit = float(num) / float(den)
                if unit == 3 and px_per_unit > 0:                  # px per cm -> µm per px
                    self.mpp_source = "tiff XResolution (cm)"
                    return 1e4 / px_per_unit
                if unit == 2 and px_per_unit > 0:                  # px per inch
                    self.mpp_source = "tiff XResolution (inch)"
                    return 25400.0 / px_per_unit
        except Exception as e:
            print(f"[warn] cannot read TIFF resolution tags: {type(e).__name__}: {e}", flush=True)
        return None

    def read(self, x0: int, y0: int, level: int, w: int, h: int) -> np.ndarray:
        """RGB uint8 [h, w, 3]; (x0, y0) are bounds-relative level-0 coordinates."""
        if self.is_isyntax:
            read_region = _isyntax_reader()[3]
            img = read_region(self.view, (x0, y0), level,
                              int(round(self.level_downsamples[level])), w, h)
            arr = (np.zeros((h, w, 3), np.uint8) if img is None
                   else np.asarray(img, dtype=np.uint8))
        else:
            img = self.osr.read_region((x0 + self.off_x, y0 + self.off_y), level, (w, h)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
        if arr.shape[0] != h or arr.shape[1] != w:                 # short reads at the canvas edge
            out = np.zeros((h, w, 3), np.uint8)
            out[: arr.shape[0], : arr.shape[1]] = arr[:h, :w]
            arr = out
        return arr

    def best_level_for(self, downsample: float) -> int:
        return int(np.argmin([abs(d - downsample) for d in self.level_downsamples]))


def resolve_scale(slide: Slide, target_mpp: float, tol: float, patch: int) -> dict:
    """Pick the pyramid level to read and the read/model pixel ratio (integrated_wsi_infer.py rule).

    Returns {level, level_mpp, effective_mpp, read_ratio, eff_ds}, where ``read_ratio`` is
    read-level px per model px and ``eff_ds`` is level-0 px per model px.
    """
    if not slide.mpp:
        print(f"[warn] slide declares no mpp: assuming level 0 is the target {target_mpp} mpp "
              f"(pass --source-mpp to override)", flush=True)
        return {"level": 0, "level_mpp": target_mpp, "effective_mpp": target_mpp,
                "read_ratio": 1.0, "eff_ds": 1.0, "level_patch": patch}
    level_mpps = [slide.mpp * ds for ds in slide.level_downsamples]
    level = int(np.argmin([abs(m - target_mpp) for m in level_mpps]))
    level_mpp = level_mpps[level]
    if abs(level_mpp - target_mpp) <= tol:      # close enough: read the level as it is, no resize
        ratio, eff_mpp = 1.0, level_mpp
    else:                                       # read a differently sized window and resize to `patch`
        ratio, eff_mpp = target_mpp / level_mpp, target_mpp
    return {"level": level, "level_mpp": level_mpp, "effective_mpp": eff_mpp, "read_ratio": ratio,
            "eff_ds": slide.level_downsamples[level] * ratio, "level_patch": int(round(patch * ratio))}


# --------------------------------------------------------------------------------------- tissue
def tissue_mask(slide: Slide, target_downsample: int, sat_min: int) -> Tuple[np.ndarray, float]:
    """Otsu-on-saturation tissue mask at a coarse level -> (mask[H, W] bool, level-0 px per mask px).

    Same recipe as lsp-detr/wsi/lspdetr_wsi_infer.py, implemented with numpy/scipy so the det
    port needs no OpenCV.
    """
    from PIL import Image
    from scipy import ndimage

    level = slide.best_level_for(target_downsample)
    ds = slide.level_downsamples[level]
    w, h = slide.level_dims[level]
    print(f"[tissue] level {level} ({w}x{h}, downsample {ds:.0f})", flush=True)

    thumb = slide.read(0, 0, level, w, h)
    sat = np.asarray(Image.fromarray(thumb).convert("HSV"), dtype=np.uint8)[:, :, 1]

    hist = np.bincount(sat.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu[-1] * omega - mu) ** 2 / denom, 0.0)
    thr = int(np.nanargmax(sigma_b))

    mask = (sat > thr) & (sat >= sat_min) & (thumb.mean(axis=2) < 235)
    k = np.ones((5, 5), bool)
    mask = ndimage.binary_closing(mask, structure=k, iterations=2)
    mask = ndimage.binary_opening(mask, structure=k, iterations=1)
    print(f"[tissue] otsu(saturation) > {thr}  ->  {100.0 * mask.mean():.1f}% of the slide area", flush=True)
    return mask, ds


def enumerate_tiles(mask: np.ndarray, mask_ds: float, eff_ds: float, step: int,
                    w_model: int, h_model: int) -> np.ndarray:
    """Model-space origins (ox, oy) of the `step`-sized cells that contain tissue."""
    mh, mw = mask.shape
    scale = eff_ds / mask_ds                       # mask px per model px
    tiles = []
    for oy in range(0, h_model, step):
        my0 = int(oy * scale)
        my1 = max(my0 + 1, int((oy + step) * scale))
        if my0 >= mh:
            continue
        row = mask[my0: min(my1, mh)]
        for ox in range(0, w_model, step):
            mx0 = int(ox * scale)
            mx1 = max(mx0 + 1, int((ox + step) * scale))
            if mx0 >= mw:
                continue
            if row[:, mx0: min(mx1, mw)].any():
                tiles.append((ox, oy))
    return np.asarray(tiles, dtype=np.int64).reshape(-1, 2)


def read_tile(slide: Slide, ox: int, oy: int, geo: dict) -> np.ndarray:
    """The `patch`-sized model-space tile centred on cell (ox, oy), reflect-padded at the border."""
    from PIL import Image

    patch, margin, eff_ds = geo["patch"], geo["margin"], geo["eff_ds"]
    ratio, level = geo["read_ratio"], geo["level"]
    tx0, ty0 = ox - margin, oy - margin
    vx0, vy0 = max(0, tx0), max(0, ty0)
    vx1, vy1 = min(geo["w_model"], tx0 + patch), min(geo["h_model"], ty0 + patch)
    w_m, h_m = vx1 - vx0, vy1 - vy0
    if w_m <= 0 or h_m <= 0:
        return np.zeros((patch, patch, 3), np.uint8)

    arr = slide.read(int(round(vx0 * eff_ds)), int(round(vy0 * eff_ds)), level,
                     max(1, int(round(w_m * ratio))), max(1, int(round(h_m * ratio))))
    if arr.shape[1] != w_m or arr.shape[0] != h_m:   # --target-mpp resize (BILINEAR == val Resize)
        arr = np.asarray(Image.fromarray(arr).resize((w_m, h_m), Image.BILINEAR), dtype=np.uint8)

    pad_l, pad_t = vx0 - tx0, vy0 - ty0
    pad_r, pad_b = patch - w_m - pad_l, patch - h_m - pad_t
    if pad_l or pad_t or pad_r or pad_b:
        # np.pad("reflect") needs every pad strictly smaller than the axis it mirrors
        mode = "reflect" if (max(pad_l, pad_r) < w_m and max(pad_t, pad_b) < h_m) else "symmetric"
        arr = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)
    return arr


# --------------------------------------------------------------------------------------- worker
def run_shard(args, geo: dict, tiles: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Run one shard of tiles through the model; returns level-0 detections (DET_DTYPE)."""
    import argparse as _argparse
    import queue
    import threading

    import torch

    import inference as det_inference          # det/inference.py: model build + forward parity

    slide = Slide(args.wsi, args.source_mpp)
    build_args = _argparse.Namespace(
        config=args.config, resume=args.resume, weights=args.weights, update=list(args.update),
        nms_iou_threshold=args.nms_iou_threshold, nms_score_threshold=args.nms_score_threshold,
        num_top_queries=args.num_top_queries, use_nms=args.use_nms,
        class_agnostic_nms=args.class_agnostic_nms, input_size=args.patch_size,
        allow_arm_mismatch=args.allow_arm_mismatch)
    model, postprocessor, _cfg, input_size, meta = det_inference.build_model_and_postprocessor(build_args)
    det_inference.apply_tf32_policy(args.tf32)

    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    runner_args = _argparse.Namespace(
        batch_size=args.batch_size, amp=args.amp, patch_size=args.patch_size,
        step_size=args.step_size, filter_size=args.filter_size, tiling="auto", tile_merge_nms=None)
    runner = det_inference.Runner(runner_args, model, postprocessor, input_size, device)
    if not args.no_warmup:
        runner.warmup()

    patch, margin, filt = geo["patch"], geo["margin"], geo["filter"]
    eff_ds = geo["eff_ds"]
    sizes = torch.tensor([[patch, patch]] * args.batch_size, device=device, dtype=torch.float32)

    # Tile decoding is the CPU-side bottleneck (JPEG-compressed WSI tiles), so read ahead of the GPU.
    batches = [tiles[i: i + args.batch_size] for i in range(0, len(tiles), args.batch_size)]
    q: "queue.Queue" = queue.Queue(maxsize=max(2, args.prefetch))

    def _reader():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.reader_threads) as ex:
            for chunk in batches:
                arrs = list(ex.map(lambda c: read_tile(slide, int(c[0]), int(c[1]), geo), list(chunk)))
                q.put((chunk, np.stack(arrs)))
        q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    out_x1, out_y1, out_x2, out_y2, out_sc, out_lb = [], [], [], [], [], []
    done, n_det, t0 = 0, 0, time.time()
    while True:
        item = q.get()
        if item is None:
            break
        chunk, arrs = item
        x = torch.from_numpy(arrs).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        labels, boxes, scores, _cls = runner._forward(x, sizes[: len(chunk)])
        for (ox, oy), lb, bx, sc in zip(chunk, labels, boxes, scores):
            if len(bx) == 0:
                continue
            cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
            keep = (sc >= args.conf_threshold)
            # one owner per detection: the centre must fall in this tile's central `filter` window
            keep &= (cx >= margin) & (cx < margin + filt) & (cy >= margin) & (cy < margin + filt)
            if not args.include_non_tumor:
                keep &= lb == 1
            if not keep.any():
                continue
            # tile px -> model space -> level-0 px
            gx, gy = int(ox) - margin, int(oy) - margin
            b = (bx[keep] + torch.tensor([gx, gy, gx, gy], device=bx.device, dtype=bx.dtype)) * eff_ds
            bcx, bcy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            inside = (bcx >= 0) & (bcx < slide.width) & (bcy >= 0) & (bcy < slide.height)
            b = b[inside]
            if not len(b):
                continue
            b[:, 0::2] = b[:, 0::2].clamp(0, slide.width)
            b[:, 1::2] = b[:, 1::2].clamp(0, slide.height)
            b = b.cpu().numpy()
            out_x1.append(b[:, 0]); out_y1.append(b[:, 1]); out_x2.append(b[:, 2]); out_y2.append(b[:, 3])
            out_sc.append(sc[keep][inside].cpu().numpy()); out_lb.append(lb[keep][inside].cpu().numpy())
            n_det += len(b)
        done += len(chunk)
        if done % (args.batch_size * args.log_every) < args.batch_size:
            el = time.time() - t0
            rate = done / max(el, 1e-6)
            print(f"  [{args.worker_index}] {done}/{len(tiles)} tiles  {rate:.2f} tiles/s  {n_det:,} det  "
                  f"eta={(len(tiles) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    n = sum(len(a) for a in out_lb)
    det = np.empty(n, dtype=DET_DTYPE)
    if n:
        det["x1"] = np.concatenate(out_x1); det["y1"] = np.concatenate(out_y1)
        det["x2"] = np.concatenate(out_x2); det["y2"] = np.concatenate(out_y2)
        det["score"] = np.concatenate(out_sc); det["label"] = np.concatenate(out_lb)
    print(f"  [{args.worker_index}] done: {len(tiles)} tiles, {n:,} detections in "
          f"{time.time() - t0:.1f}s", flush=True)
    return det, meta


# --------------------------------------------------------------------------------------- outputs
def write_zst(det: np.ndarray, slide: Slide, path: str, table, verify: bool = True) -> None:
    """MVT layer "default" + zstd, in the viewer's frame; ``table`` is a resolve_convention() table.

    extent = level-0 slide height, and the value *stored in the tile* for a detection whose bbox
    centre is (cx, cy) in image pixels is ``(cx, h_wsi - cy)`` - the convention of every writer on
    the platform (TILs-Inference/engine/wsi_infer.py, common/postprocess/mvt_encode.py,
    AIVIS-Dome-DETR/integrated_wsi_infer.py, lsp-detr/wsi/lspdetr_wsi_infer.py).

    Getting there needs care, because mapbox_vector_tile transforms y on the way in *and* on the
    way out (raw-protobuf check on mapbox_vector_tile with extent=1000, input y=200):
        encode(y_coord_down=True)   stores the input as is         ->  stored 200
        encode(y_coord_down=False)  stores `extent - input`        ->  stored 800
        decode()                    always returns `extent - stored` (y-up), whichever was used
    So the pre-flipped ``h_wsi - cy`` must go in with **y_coord_down=True**; passing it with
    y_coord_down=False flips it a second time and the slide renders upside down.
    (normalize_domedetr_zst.py legitimately uses y_coord_down=False because it feeds back the
    *decoded* - i.e. already y-up - coordinate cy, not the flipped one.)
    A finished file therefore decodes back to the image-space cy, which is what `verify` checks.
    """
    import mapbox_vector_tile as mvt
    import zstandard as zstd

    cx = (det["x1"].astype(np.float64) + det["x2"]) * 0.5
    cy = (det["y1"].astype(np.float64) + det["y2"]) * 0.5
    extent = int(math.ceil(slide.height))
    mvt_y = float(slide.height) - cy                      # what must end up stored in the tile
    labels = det["label"].astype(np.int64)
    if len(labels) and (labels.min() < 0 or labels.max() >= len(table)):
        raise RuntimeError(f"[zst] label {labels.min()}..{labels.max()} outside the "
                           f"{len(table)}-class convention table")

    features = []
    for i in range(len(det)):
        cid, term, nt, _ = table[labels[i]]
        features.append({
            "geometry": {"type": "Point", "coordinates": [float(cx[i]), float(mvt_y[i])]},
            "properties": {"imageId": 0, "categoryId": cid, "termId": term,
                           "nt": bool(nt), "positivity_rank": 0},
            "id": int(i),
        })
    blob = mvt.encode({"name": "default", "features": features},
                      default_options={"quantize_bounds": None, "y_coord_down": True,
                                       "extents": extent})
    packed = zstd.ZstdCompressor().compress(blob)
    with open(path, "wb") as fp:
        fp.write(packed)
    print(f"[zst] {path} ({len(packed) / 2**20:.1f} MB, {len(det):,} features, extent={extent})", flush=True)
    if verify and len(det):
        verify_zst(path, cx, cy, labels, table)


def verify_zst(path: str, cx: np.ndarray, cy: np.ndarray, labels, table, sample: int = 2000) -> None:
    """Decode the file we just wrote and assert it comes back in image space (catches a y flip)."""
    import mapbox_vector_tile as mvt
    import zstandard as zstd

    layer = mvt.decode(zstd.ZstdDecompressor().decompress(open(path, "rb").read()))["default"]
    feats = layer["features"]
    if len(feats) != len(cx):
        raise RuntimeError(f"[zst] verify: {len(feats)} features decoded, {len(cx)} written")
    idx = np.unique(np.linspace(0, len(feats) - 1, min(sample, len(feats))).astype(np.int64))
    dx = np.array([feats[i]["geometry"]["coordinates"][0] for i in idx], dtype=np.float64) - cx[idx]
    dy = np.array([feats[i]["geometry"]["coordinates"][1] for i in idx], dtype=np.float64) - cy[idx]
    if np.abs(dx).max() > 1.5 or np.abs(dy).max() > 1.5:
        flipped = np.abs(np.array([feats[i]["geometry"]["coordinates"][1] for i in idx], dtype=np.float64)
                         - (layer["extent"] - cy[idx])).max() <= 1.5
        raise RuntimeError(f"[zst] verify FAILED: max |dx|={np.abs(dx).max():.1f} max |dy|={np.abs(dy).max():.1f}"
                           + (" - the y axis is flipped" if flipped else ""))
    # geometry is sound; now check the class encoding round-trips (a mis-mapped label would render
    # the right dots in the wrong colour, which no coordinate check can catch).
    def _wrong(i):
        p, t = feats[i]["properties"], table[int(labels[i])]
        return (p.get("categoryId") != t[0] or p.get("termId") != t[1]
                or bool(p.get("nt")) != bool(t[2]))
    bad_cls = [i for i in idx if _wrong(i)]
    if bad_cls:
        i = bad_cls[0]
        raise RuntimeError(f"[zst] verify: {len(bad_cls)} sampled features carry the wrong "
                           f"categoryId/termId/nt for their label - e.g. feature {i} has "
                           f"{feats[i]['properties'].get('categoryId')!r}/"
                           f"{feats[i]['properties'].get('termId')!r}/{feats[i]['properties'].get('nt')!r} "
                           f"but label {int(labels[i])} maps to {table[int(labels[i])][:3]}")
    print(f"[zst] verify: {len(idx)} sampled features decode back to image coordinates and carry the "
          f"convention's categoryId/termId/nt (max |dx|={np.abs(dx).max():.2f}, "
          f"max |dy|={np.abs(dy).max():.2f})", flush=True)


def write_json(det: np.ndarray, slide: Slide, path: str, class_names) -> None:
    """COCO predictions in level-0 pixels, streamed (millions of dicts would cost several GB)."""
    head = {"images": [{"id": 0, "file_name": slide.name, "width": int(slide.width),
                        "height": int(slide.height), "mpp": slide.mpp}],
            "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]}
    x = np.rint(det["x1"]).astype(np.int64)
    y = np.rint(det["y1"]).astype(np.int64)
    w = np.maximum(1, np.rint(det["x2"] - det["x1"])).astype(np.int64)
    h = np.maximum(1, np.rint(det["y2"] - det["y1"])).astype(np.int64)
    lab, sc = det["label"].tolist(), np.round(det["score"], 4).tolist()
    xl, yl, wl, hl = x.tolist(), y.tolist(), w.tolist(), h.tolist()
    with open(path, "w") as fp:
        fp.write(json.dumps(head)[:-1])
        fp.write(', "annotations": [')
        for i in range(len(det)):
            if i:
                fp.write(",")
            fp.write(f'{{"id":{i},"image_id":0,"category_id":{lab[i]},'
                     f'"bbox":[{xl[i]},{yl[i]},{wl[i]},{hl[i]}],"score":{sc[i]},'
                     f'"area":{wl[i] * hl[i]},"iscrowd":0}}')
        fp.write("]}")
    print(f"[json] {path} ({os.path.getsize(path) / 2**20:.1f} MB, {len(det):,} annotations)", flush=True)


# --------------------------------------------------------------------------------------- driver
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--wsi", required=True,
                   help="whole-slide image (anything OpenSlide reads, or Philips .isyntax/.i2syntax)")
    p.add_argument("-c", "--config", default=os.path.join(DET_ROOT, "configs", "LSP-T-combined.yml"))
    p.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("-u", "--update", nargs="+", default=[], help="extra yaml overrides")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--tag", default="LSPDETR-TNT",
                   help="output suffix appended to the slide's full file name (default LSPDETR-TNT, "
                        "giving <slide>.<ext>_LSPDETR-TNT.zst); must not contain '_'")
    p.add_argument("--output-name", default=None,
                   help="output base name, overriding <slide file name>_<tag> entirely")
    p.add_argument("--no-json", dest="save_json", action="store_false", default=True,
                   help="write only the .zst (the COCO json runs to hundreds of MB)")
    # geometry / thresholds - defaults are the Dome-DETR tumour/non-tumour WSI recipe
    p.add_argument("--patch-size", type=int, default=1536)
    p.add_argument("--step-size", type=int, default=672)
    p.add_argument("--filter-size", type=int, default=672)
    p.add_argument("--target-mpp", type=float, default=TARGET_MPP, help="mpp the arm is run at (default 0.5)")
    p.add_argument("--source-mpp", type=float, default=None, help="override the slide's level-0 mpp")
    p.add_argument("--mpp-tolerance", type=float, default=0.05,
                   help="read a pyramid level as is when it is this close to --target-mpp (default 0.05)")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--nms-iou-threshold", type=float, default=None)
    p.add_argument("--nms-score-threshold", type=float, default=None)
    p.add_argument("--num-top-queries", type=int, default=None)
    p.add_argument("--use-nms", type=_bool, default=None)
    p.add_argument("--class-agnostic-nms", type=_bool, default=None)
    p.add_argument("--include-non-tumor", type=_bool, default=True,
                   help="False -> drop label 0 (the non-tumour class in every convention)")
    p.add_argument("--zst-convention", default="auto", choices=["auto"] + sorted(ZST_CONVENTIONS),
                   help="viewer categoryId/termId/nt encoding; 'auto' picks by the model's num_classes "
                        "(2 -> tnt, 3 -> tnt-lymph)")
    p.add_argument("--class-names", default=None,
                   help="comma separated, overrides the convention's names in the .json/.meta.json")
    p.add_argument("--seg-downsample", type=int, default=32, help="downsample of the tissue mask")
    p.add_argument("--sat-min", type=int, default=8, help="minimum HSV saturation for tissue")
    p.add_argument("--max-tiles", type=int, default=0, help="cap the tile count (smoke tests)")
    # runtime
    p.add_argument("--gpus", default="0", help="comma separated GPU ids; one worker process each")
    p.add_argument("-d", "--device", default="cuda")
    p.add_argument("-b", "--batch-size", type=int, default=4, help="tiles per forward")
    p.add_argument("--reader-threads", type=int, default=8, help="tile decode threads per worker")
    p.add_argument("--prefetch", type=int, default=4, help="batches read ahead of the GPU")
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--tf32", default="keep", choices=["keep", "off", "on"])
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--allow-arm-mismatch", action="store_true")
    p.add_argument("--log-every", type=int, default=25, help="log every N batches")
    # internal (worker processes)
    p.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--tiles-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--geo-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--shard-out", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    # ---------------- worker process: run the assigned shard and hand it back through a .npz
    if args.worker_index is not None:
        with open(args.geo_file) as f:
            geo = json.load(f)
        tiles = np.load(args.tiles_file)[args.worker_index:: args.worker_count]
        det, meta = run_shard(args, geo, tiles)
        np.savez(args.shard_out, det=det, meta=np.array(json.dumps(meta)))
        return 0

    # ---------------- parent: geometry, tissue mask, tile list
    if not os.path.isfile(args.wsi):
        sys.exit(f"not a file: {args.wsi}")
    if args.filter_size > args.patch_size or (args.patch_size - args.filter_size) % 2:
        sys.exit("--filter-size must be <= --patch-size and (patch - filter) must be even")
    if args.step_size != args.filter_size:
        print(f"[warn] --step-size {args.step_size} != --filter-size {args.filter_size}: the ownership "
              f"windows {'overlap (duplicates)' if args.step_size < args.filter_size else 'leave gaps'}",
              flush=True)

    slide = Slide(args.wsi, args.source_mpp)
    print(f"[slide] {slide.name}  {slide.width}x{slide.height}  levels={len(slide.level_dims)}  "
          f"mpp={slide.mpp if slide.mpp is None else round(slide.mpp, 4)} ({slide.mpp_source})", flush=True)

    scale = resolve_scale(slide, args.target_mpp, args.mpp_tolerance, args.patch_size)
    eff_ds = scale["eff_ds"]
    geo = {"patch": args.patch_size, "filter": args.filter_size, "step": args.step_size,
           "margin": (args.patch_size - args.filter_size) // 2, "eff_ds": eff_ds,
           "read_ratio": scale["read_ratio"], "level": scale["level"],
           "w_model": int(math.floor(slide.width / eff_ds)), "h_model": int(math.floor(slide.height / eff_ds))}
    print(f"[mpp] target={args.target_mpp} level={scale['level']} level_mpp={scale['level_mpp']:.4f} "
          f"effective_mpp={scale['effective_mpp']:.4f} read_ratio={scale['read_ratio']:.4f} "
          f"(level-0 px per model px = {eff_ds:.4f})", flush=True)
    print(f"[grid] model canvas {geo['w_model']}x{geo['h_model']}  patch={geo['patch']} "
          f"step={geo['step']} filter={geo['filter']} margin={geo['margin']}", flush=True)

    mask, mask_ds = tissue_mask(slide, args.seg_downsample, args.sat_min)
    tiles = enumerate_tiles(mask, mask_ds, eff_ds, args.step_size, geo["w_model"], geo["h_model"])
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
        print(f"[tiles] capped to --max-tiles {args.max_tiles}", flush=True)
    total_cells = math.ceil(geo["w_model"] / args.step_size) * math.ceil(geo["h_model"] / args.step_size)
    print(f"[tiles] {len(tiles):,} tissue tiles of {total_cells:,} grid cells", flush=True)
    if not len(tiles):
        sys.exit("[tiles] nothing to do - the tissue mask is empty")

    os.makedirs(args.output_dir, exist_ok=True)
    if "_" in args.tag:
        # Not a functional constraint - every consumer on the platform strips the whole suffix it was
        # given rather than splitting on '_' - but a single-underscore tag keeps the name splittable.
        print(f"[warn] --tag {args.tag!r} contains '_': the output name then has more than one underscore, "
              f"so the slide name can no longer be recovered by splitting on the last one", flush=True)
    # <slide file name incl. extension>_<tag>, cf. the platform's <slide>.i2syntax_LSPDETR_512.zst
    name = args.output_name or f"{slide.name}_{args.tag}"
    gpus = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]

    # ---------------- run: one worker subprocess per GPU, each on its own shard of the tile list
    work_dir = os.path.join(args.output_dir, f".{name}.shards")
    os.makedirs(work_dir, exist_ok=True)
    tiles_file, geo_file = os.path.join(work_dir, "tiles.npy"), os.path.join(work_dir, "geo.json")
    np.save(tiles_file, tiles)
    with open(geo_file, "w") as f:
        json.dump(geo, f)

    passthrough = ["--wsi", args.wsi, "--config", args.config, "--resume", args.resume,
                   "--weights", args.weights, "--output-dir", args.output_dir,
                   "--patch-size", str(args.patch_size), "--step-size", str(args.step_size),
                   "--filter-size", str(args.filter_size), "--conf-threshold", str(args.conf_threshold),
                   "--include-non-tumor", str(args.include_non_tumor), "--batch-size", str(args.batch_size),
                   "--reader-threads", str(args.reader_threads), "--prefetch", str(args.prefetch),
                   "--amp", args.amp, "--tf32", args.tf32, "--log-every", str(args.log_every),
                   "--device", args.device, "--target-mpp", str(args.target_mpp),
                   "--zst-convention", args.zst_convention]
    for flag, val in (("--source-mpp", args.source_mpp), ("--nms-iou-threshold", args.nms_iou_threshold),
                      ("--nms-score-threshold", args.nms_score_threshold),
                      ("--num-top-queries", args.num_top_queries), ("--use-nms", args.use_nms),
                      ("--class-agnostic-nms", args.class_agnostic_nms)):
        if val is not None:
            passthrough += [flag, str(val)]
    if args.update:
        passthrough += ["--update", *args.update]
    if args.no_warmup:
        passthrough += ["--no-warmup"]
    if args.allow_arm_mismatch:
        passthrough += ["--allow-arm-mismatch"]

    print(f"[run] {len(tiles):,} tiles over {len(gpus)} GPU(s): {', '.join(gpus)}  "
          f"batch={args.batch_size}", flush=True)
    procs, shard_files = [], []
    for i, gpu in enumerate(gpus):
        shard = os.path.join(work_dir, f"shard{i}.npz")
        shard_files.append(shard)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # torch >= 2.9 aborts on this box's malformed value
        cmd = [sys.executable, "-u", os.path.abspath(__file__), *passthrough,
               "--worker-index", str(i), "--worker-count", str(len(gpus)),
               "--tiles-file", tiles_file, "--geo-file", geo_file, "--shard-out", shard]
        procs.append(subprocess.Popen(cmd, env=env))
    rc = 0
    for i, p in enumerate(procs):
        if p.wait() != 0:
            print(f"[error] worker {i} (GPU {gpus[i]}) exited with {p.returncode}", flush=True)
            rc = 1
    if rc:
        for p in procs:
            if p.poll() is None:
                p.kill()
        return rc

    # ---------------- merge + write
    parts, model_meta = [], None
    for f in shard_files:
        with np.load(f, allow_pickle=False) as z:
            parts.append(z["det"])
            model_meta = model_meta or json.loads(str(z["meta"]))
    det = np.concatenate(parts) if parts else np.empty(0, dtype=DET_DTYPE)
    order = np.lexsort((det["x1"], det["y1"]))     # stable, shard-independent ordering
    det = det[order]

    # the parent never builds a model, so num_classes comes back from the shard's model meta
    num_classes = int((model_meta or {}).get("num_classes", 2))
    conv_name, table = resolve_convention(
        args.zst_convention, num_classes,
        [c.strip() for c in args.class_names.split(",")] if args.class_names else None)
    class_names = [t[3] for t in table]
    print(f"[classes] num_classes={num_classes} convention={conv_name}  " +
          "  ".join(f"{i}:{t[3]}=cat{t[0]}/nt{int(t[2])}" for i, t in enumerate(table)), flush=True)

    counts = {class_names[c]: int((det["label"] == c).sum()) for c in range(num_classes)}
    print(f"[detect] {len(det):,} detections  " + "  ".join(f"{k}={v:,}" for k, v in counts.items()), flush=True)

    base = os.path.join(args.output_dir, name)
    write_zst(det, slide, base + ".zst", table)
    if args.save_json:
        write_json(det, slide, base + ".json", class_names)
    meta = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "wsi": slide.path, "slide": slide.name,
            "width": int(slide.width), "height": int(slide.height), "mpp": slide.mpp,
            "mpp_source": slide.mpp_source, "bounds_offset": [slide.off_x, slide.off_y],
            "scale": scale, "geometry": geo, "tiles": int(len(tiles)), "grid_cells": int(total_cells),
            "conf_threshold": args.conf_threshold, "include_non_tumor": bool(args.include_non_tumor),
            "gpus": gpus, "batch_size": args.batch_size, "amp": args.amp, "tf32": args.tf32,
            "num_detections": int(len(det)), "class_totals": counts, "model": model_meta,
            "elapsed_sec": round(time.time() - t_start, 1), "cli": " ".join(sys.argv),
            "zst": {"format": "zstd(mapbox_vector_tile)", "layer": "default",
                    "extent": int(math.ceil(slide.height)),
                    "convention": conv_name,
                    "classes": [{"label": i, "name": t[3], "categoryId": t[0], "termId": t[1], "nt": bool(t[2])}
                                for i, t in enumerate(table)],
                    "geometry_note": "stored_y = height - centre_y (decodes back to image-space y)"},
            "output_name": name}
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {base}.meta.json", flush=True)

    for f in shard_files + [tiles_file, geo_file]:
        os.remove(f)
    os.rmdir(work_dir)
    print(f"[done] total {time.time() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
