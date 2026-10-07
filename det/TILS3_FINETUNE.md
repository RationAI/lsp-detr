# LSP-DETR-det: 2-class → 3-class fine-tune on TILs-images

**Result: macro-F1 0.474 → 0.795, lymphocyte F1 0.000 → 0.792, class-agnostic detection F1 unchanged at 0.90.** See §6; §7 covers whole-slide inference and the 3-class `.zst` encoding.

`ckpt/checkpoint0022.pth` (LSP-DETR-det trained on `combined_all_v1_bundle`, 0 = Non-tumor / 1 = Tumor)
extended to **0 = non_tumor / 1 = tumor / 2 = Lymphocyte** on the two hand-labelled patches in
`/workspace/TILs-images`.

Everything below is additive: no Dome file and no existing LSP-arm file changed behaviour except
`det/train.py` (one new optional CLI flag) and `det/lsp_det/lsp_criterion.py` (an import fallback).

```
det/
├── scripts/build_tils3_dataset.py     TILs-images labels -> det/data/tils3/ (train/val COCO + val tiles + grid whitelist)
├── scripts/expand_class_head.py       checkpoint 2-class -> 3-class (grows the shared decoder.class_head)
├── scripts/dist_train_tils3.sh        launcher
├── scripts/eval_tils3.py              per-class COCO AP + centroid-matched P/R/F1 + confusion matrix
├── lsp_det/grid_subset_wrapper.py     GridSubsetCropWrapper (multi-crop restricted to a whitelist of centre cells)
├── configs/dataset/tils3_detection.yml
├── configs/LSP-T-tils3.yml
└── data/tils3/                        the built bundle (train_coco.json, val_coco.json, val_tiles/, allowed_grids.json,
                                       build_report.json)
```

## 1. Data

Source: `/workspace/TILs-images/{DATA_KU_BC-1_43188_110656,SSMH_BRS_HE_006_69436_36043}_ki67.json`
plus the matching 4096×4096 PNGs in `/workspace/TILs-images/images`.

| source category | count (KU / SSMH) | treatment |
|---|---|---|
| 0 `non_tumor` | 7,740 / 30,554 | kept as class 0 |
| 1 `tumor` | 7,058 / 5,493 | kept as class 1 |
| 2 `Lymphocyte` | 7,237 / 25,659 | kept as class 2 |
| 3 `Exclusions` | 192 / 45 | **dropped** |
| 4 `Others_unmatched` | 635 / 3,632 | **dropped** (per instruction; mostly unverified cellpose-added cells) |

Dropped annotations become background supervision — that is the deliberate choice, not an oversight.

**Scale.** These patches are 0.5 µm/px and the median GT box is 12–14 px, matching the detector's own
`wh_prior_px [14, 14]` (the `combined_all_v1_bundle` per-axis bbox median). No resampling is applied
or needed; the labels themselves were produced by running a Dome Ki-67 detector on these patches at
native scale with the same window 1536 / step 672 geometry.

### Train / val split

The recipe crops 1536×1536 windows whose **centre 672×672** carries the GT, so a 4096 image is a 6×6
grid of non-overlapping centre cells; cell `(ix, iy)` owns `[ix·672, ix·672+672)` and is cropped as
`[ix·672−432, ix·672+1104)`. The split is done on that grid:

* **val** = cells `(1,1) (4,1) (1,4) (4,4)` of each image → 8 fixed 1536² tiles written to
  `data/tils3/val_tiles/`, GT restricted to the centre-672 window (the same convention as the
  `combined_all_v1_bundle` val split, so the val transform's `Resize [1536,1536]` is a no-op).
* **train** = the 16 cells that are neither a val cell nor a 4-neighbour of one, per image.
  `GridSubsetCropWrapper` feeds those linear grid indices to `RandomCropWithGridNoDummy`.

Two properties are asserted by the builder:

* **no GT leakage** — train and val GT come from disjoint centre cells;
* **< 50 % pixel overlap** — the closest allowed train crop is a *diagonal* neighbour of a val tile,
  i.e. `(1536−672)² / 1536² = 31.6 %` of its area (edge neighbours, which would be 56 %, are excluded).

Resulting sizes: **34,716** usable train GT (non_tumor 15,357 / tumor 6,173 / lymphocyte 13,186) and
**9,295** val GT (4,308 / 1,149 / 3,838) over 8 tiles.

```bash
python det/scripts/build_tils3_dataset.py \
  --src /workspace/TILs-images/DATA_KU_BC-1_43188_110656_ki67.json \
        /workspace/TILs-images/SSMH_BRS_HE_006_69436_36043_ki67.json
```

## 2. Checkpoint: 2 → 3 classes

`decoder.class_head` is a single `Linear(384, num_classes)` shared by all six decoder layers (sigmoid
VFL, no explicit background row), so growing the label set is exactly "append a row". Rows 0/1 keep
their trained values; row 2 starts at the port's own fresh-head initialisation (zero weight, bias
`−log 99`, i.e. focal prior p = 0.01) so it cannot disturb the two trained rows at step 0. Both
`model` and `ema.module` are expanded and every other tensor is asserted to keep its shape.

```bash
python det/scripts/expand_class_head.py -i ckpt/checkpoint0022.pth -o ckpt/checkpoint0022_3class.pth \
    --num-classes 3 --names non_tumor tumor Lymphocyte
```

The result is loaded with `train.py --tuning` (which reads the **EMA** weights and runs *before* the
new EMA is built), not `--resume`: the optimizer state belongs to the old 2-class label set.

**Arm.** `checkpoint0022.pth` carries no `init_report.json`, and `center_mode` leaves no trace in the
tensors (identical shapes). Measured on the val tiles before any training: `strict-local`
AP 0.237 / AP50 0.409 vs `movable-reference` AP 0.091 / AP50 0.231 → the checkpoint is
**strict-local**, matching `IMPLEMENTATION_LOG.md`'s "P4 launch (approved: ARM=strict-local)".

## 3. Recipe

Identical to `configs/LSP-T-combined.yml` except:

| | LSP-T-combined | LSP-T-tils3 |
|---|---|---|
| num_classes | 2 | 3 |
| weights | `hf-5class/model.safetensors` via `pretrained` | `ckpt/checkpoint0022_3class.pth` via `--tuning` (`pretrained: null`) |
| base lr / backbone lr | 2.5e-4 / 1.25e-5 | **5e-5 / 2.5e-6** (÷5: the trunk is converged and the set is two images) |
| warmup | 0 | **100 steps** linear |
| schedule | MultiStepLR [24, 30] γ 0.8 | MultiStepLR [55, 70] γ 0.5, 80 epochs |
| crops | `MultiCropDatasetWrapper`, 16 crops, any grid cell | `GridSubsetCropWrapper`, 48 crops, whitelist of 16 cells (each cell 3×/epoch) |
| batch | 8 × H100 | 4 GPUs × 1 (4 also divides the 8 val tiles, so `DistributedSampler` never pads — the `coco_eval_aitod.merge()` area-subset bug cannot apply) |

Everything else is unchanged: FP32 (AMP off), TF32 = machine default, grad clip 0.1, EMA 0.9999 /
warmups 1000, VFL α 0.75 γ 2 ×3 + L1 ×3 + GIoU ×1 over final + 5 aux with union-matched boxes,
patch-embed + Swin stage 1 frozen, top-2000 + class-agnostic NMS 0.7.

```bash
bash det/scripts/dist_train_tils3.sh                                   # 4 GPU, 80 ep, seed 0
NUM_GPUS=2 GPU_IDS=0,1 END_EPOCHS=2 bash det/scripts/dist_train_tils3.sh   # smoke
RESUME=<run>/last.pth bash det/scripts/dist_train_tils3.sh                 # resume
# knobs: NUM_GPUS GPU_IDS END_EPOCHS SEED OUTPUT_DIR TUNING ARM TF32_POLICY EXTRA_UPDATES MASTER_PORT
```

Two environment landmines on this box, both handled inside the launcher:

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=128` is malformed (`=` instead
  of `:` in the second token) and **SIGABRTs torch at CUDA init**; the launcher unsets it.
* a machine-wide `TF32=1` is exported, so the launcher's knob is named **`TF32_POLICY`** — a plain
  `TF32` would be inherited as `"1"` and rejected by `train.py --tf32`.

## 4. Evaluation

`det/scripts/eval_tils3.py` runs the same model / postprocessor as `train.py --test-only` in a single
process and adds what `log.txt` cannot show: **per-category** COCO AP, and centroid-matched
precision / recall / F1 per class with a confusion matrix (a prediction matches the nearest unused GT
centre within `--match-radius`, default 12 px = the label file's own Ki-67 match threshold).

```bash
python det/scripts/eval_tils3.py -r <run>/best_stg1.pth --weights ema --device cuda:0 --out <dir>
```

## 5. Inference

`det/inference.py` / `det/run_inference.sh` are already `num_classes`-generic (they read it from the
resolved config and `CLASS_COLORS` has an entry for class 2), so the 3-class checkpoint works as is:

```bash
cd det && RESUME=<run>/best_stg1.pth EXTRA_ARGS="--class-names non_tumor,tumor,Lymphocyte" \
    ./run_inference.sh /path/to/img_or_dir
```

`det/wsi_infer.py` still writes the 2-class viewer convention (tumour `1` / non-tumour `5`); emitting
the platform's 3-class encoding (lymphocyte `3` / `66d54cee…`, as `combine_tnt_lymph.py` produces)
directly from a 3-class model is a follow-up, not part of this fine-tune.

## 6. Result

Run: `det/output/tils3/[tils3]LSP-T_strict-local_x4_det-ft_80ep/`, 4 GPUs, seed 0, ~130 s/epoch.

**The run reached epoch 63 of 80, not 80.** At 2026-08-25 17:17 the kernel OOM-killer SIGKILLed rank 3
(`exitcode -9`) and torchrun tore the other three ranks down. This was **host RAM exhaustion from
outside this container** — at the time `MemAvailable` was 13.7 GB of 923 GB while the total RSS of every
process visible in this container was ~4 GB, i.e. ~860 GB was held by workloads in another PID
namespace. All checkpoints were verified intact afterwards (432 tensors, 3-row class head, finite,
EMA `updates` == epochs × 24). The missing 16 epochs do not change the deliverable: the first LR drop
had already landed (effective epoch 59, backbone 2.5e-6 → 1.25e-6) and epochs 55-63 sat at AP
0.390-0.403 with train loss flat at 9.57-9.72, all below the epoch-27 best.

Selected checkpoint: **`best_stg1.pth` (epoch 27, EMA)** — selected by the pre-registered criterion
(val COCO AP). `last.pth` (epoch 63) is within noise of it on the matched metrics and slightly worse on
COCO AP; picking it after the fact would be selecting on the same 8-tile val set twice.

### Val (8 held-out tiles, 9,295 GT: 4,308 non_tumor / 1,149 tumor / 3,838 lymphocyte)

COCO, single process, maxDets 2000:

| | AP | AP50 | AR@2000 |
|---|---|---|---|
| **before** (2-class checkpoint, class head grown but untrained) | 0.2367 | 0.4090 | 0.3429 |
| — non_tumor / tumor / Lymphocyte | 0.2075 / 0.5027 / — | 0.4107 / 0.8164 / — | |
| **after** (epoch 27 EMA) | **0.4392** | **0.7061** | **0.5232** |
| — non_tumor / tumor / Lymphocyte | 0.3491 / 0.5575 / **0.4110** | 0.5872 / 0.8472 / **0.6839** | |

Centroid-matched (score ≥ 0.4, radius 12 px = the label file's own Ki-67 match threshold):

| | precision | recall | F1 |
|---|---|---|---|
| detection, class-agnostic — before | 0.9170 | 0.8837 | 0.9001 |
| detection, class-agnostic — **after** | 0.9361 | 0.8718 | **0.9028** |
| non_tumor — before → after | 0.4711 → **0.7914** | 0.8387 → 0.6948 | 0.6033 → **0.7399** |
| tumor — before → after | 0.7747 → **0.8251** | 0.8677 → 0.8869 | 0.8186 → **0.8549** |
| Lymphocyte — before → after | — → **0.8131** | — → **0.7710** | 0.0000 → **0.7915** |
| **macro-F1** | | | 0.4739 → **0.7954** |

Confusion (rows GT, cols pred, matched pairs, after):

|  | non_tumor | tumor | Lymphocyte |
|---|---|---|---|
| non_tumor | 2993 | 105 | 567 |
| tumor | 78 | 1019 | 18 |
| Lymphocyte | 355 | 9 | **2959** |

Three things this says:

* **Detection was not traded away for the new class** — class-agnostic F1 0.9001 → 0.9028. The gain is
  purely the tumour/non-tumour/lymphocyte separation, not more or fewer cells found.
* **The tumour axis survived** — tumor F1 0.8186 → 0.8549, and tumour ↔ lymphocyte confusion is
  negligible (18 and 9 cells). Before the fine-tune 3,322 of 3,838 matched lymphocytes were called
  non_tumor; now 2,959 are called lymphocyte.
* **The residual error is the non_tumor ↔ lymphocyte boundary** (567 + 355 cells), which is also where
  the GT itself is softest: `non_tumor` here is "Others relabelled by a Ki-67 detector", not a directly
  annotated class.

Operating point: the score-threshold sweep is flat — macro-F1 0.792-0.796 over thr 0.35-0.60, best
0.7962 at 0.55, detection F1 peaking at 0.40. **The default 0.4 needs no tuning.**

### Caveats

* Two images from two slides, with a spatial (not slide-level) hold-out. These numbers say the fine-tune
  works *on this tissue*; they are **not** comparable to the leave-one-WSI-out numbers in
  `TILs-LAB/exp0`/`exp1` and do not establish cross-slide generalisation.
* Every val tile shares a slide with training crops. The split guarantees no GT and < 31.6 % pixel
  overlap, but not slide independence.
* `Others_unmatched` (cat 4) was dropped, so ~3-6 % of real nuclei are supervised as background. The
  detection recall above is measured against a GT that also excludes them.

## 6b. Reproducing on another machine

* Code: this branch (`det`) is self-contained for training (vendored Dome runtime, `det/requirements.txt`).
* Inputs not in git: the labelled patches `TILs-images/{images,*_ki67.json}` (`build_tils3_dataset.py --src … --img-dir …`),
  the 2-class source checkpoint `ckpt/checkpoint0022.pth` (`/mnt/nas2/LSP-DETR/TNT/best_stg1.pth` is the same arm's
  deployed TNT weight) and the built bundle `det/data/tils3/` (regenerated by the script; git-excluded).
* Deployed 3-class checkpoint (this run's `best_stg1.pth`, ep 29 EMA, val macro-F1 0.795):
  `/mnt/nas2/LSP-DETR/TILS3/best_stg1.pth` (716 MB, md5 fcd455c3…). `scripts/run_tils3_wsi.sh` falls back to it
  when `det/output/tils3/…/best_stg1.pth` is absent; `RESUME=` overrides.
* `.i2syntax` slides need a sibling clone of `AIVIS-inc/TILs-Inference` (Philips pixel-engine wrapper) next to this
  repo, or `TILS_ROOT=/path/to/TILs-Inference`; every other format needs only OpenSlide.

## 7. Whole-slide inference

`det/scripts/run_tils3_wsi.sh` runs the fine-tuned model over a directory of slides and writes the
platform's **3-class** viewer container. Outputs land in `/workspace/lsp-detr/results/tils3-wsi/` as
`<slide file name incl. extension>_LSPDETR-TNTL.{zst,json,meta.json,log}`.

```bash
bash det/scripts/run_tils3_wsi.sh                      # every slide in $WSI_DIR
bash det/scripts/run_tils3_wsi.sh /path/to/slide.svs   # or named slides
# knobs: WSI_DIR RESUME CONFIG OUTPUT_DIR TAG GPUS BATCH_SIZE CONF MAX_TILES FORCE
python det/scripts/summarize_tils3_wsi.py /workspace/lsp-detr/results/tils3-wsi   # counts + re-verify
```

### The .zst encoding

`wsi_infer.py` used to hardcode the 2-class arm's ids. It now carries a table of viewer conventions
(`ZST_CONVENTIONS`) selected by `--zst-convention` (default `auto`, i.e. by the model's `num_classes`):

| convention | label 0 | label 1 | label 2 |
|---|---|---|---|
| `tnt` (2-class) | Non-tumor `5` / `67bc099c32a0394aeaccc959` / nt=true | Tumor `1` / `66d54cd789181badfeac2d69` / nt=false | — |
| `tnt-lymph` (3-class) | non_tumor `2` / `66d54d2a89181badfeac2d75` / nt=true | tumor `1` / `66d54cd789181badfeac2d69` / nt=false | Lymphocyte `3` / `66d54cee89181badfeac2d6d` / nt=true |

`tnt-lymph` was taken from `/workspace/SSMH_BRS_HE_106.i2syntax_KongNet.zst` by decoding it (855,425
features, layer `default`, extent 91156, properties `imageId` / `categoryId` / `termId` / `nt` /
`positivity_rank`) and matches what `det/combine_tnt_lymph.py` already writes. Note it **remaps**
non-tumour from the 2-class arm's `5` / `67bc099c…` to `2` / `66d54d2a…`, and that a lymphocyte keeps
`nt=true` so the viewer's T/NT mode still reads it as non-tumour.

A convention whose class count disagrees with the model is a hard error, never a silent mis-map, and
`verify_zst` now re-decodes the finished file and checks **both** the coordinates and the
categoryId/termId/nt per label (a mis-mapped label draws the right dots in the wrong colour, which no
coordinate check can catch). 8 unit tests cover this in `tests/test_wsi_zst_convention.py`.

### Philips `.i2syntax`

OpenSlide cannot open these; they go through the Philips PixelEngine wrapper vendored in
`TILs-Inference/data/philips_pixel_engine`. The native SDK **is** installed on this box
(`/usr/local/lib/libpixelengine.so.5.1.0`, headers in `/usr/local/include/PhilipsPixelEngine`), but the
Python bindings are a compiled extension carrying an **interpreter ABI tag**, and only
`cpython-311` / `cpython-312` builds were present while this work runs on Python 3.13 — which surfaces
only as `UserWarning: Pixel engine bindings not available: No module named
'philips_pixel_engine.pixel_engine_bindings'` followed by `TypeError: 'NoneType' object is not
callable`, i.e. it reads like "no SDK" when the SDK is fine. Build one for the interpreter you use:

```bash
pip install "pybind11>=2.6.0"
cp -r /workspace/TILs-Inference/data/philips_pixel_engine/{setup.py,pixel_engine_wrapper.cpp,bindings} /tmp/ppe && cd /tmp/ppe
python setup.py build_ext --inplace     # SDK headers/libs are already in /usr/local
cp pixel_engine_bindings.cpython-*-x86_64-linux-gnu.so /workspace/TILs-Inference/data/philips_pixel_engine/
```

**The two formats are not the same canvas.** `SSMH_BRS_HE_106.i2syntax` is 160000×91156 at 0.25 µm/px
while `SSMH_BRS_HE_106.svs` is 128543×88339 at 0.2626 — the `.svs` is a cropped/resampled conversion.
The platform's own file for this slide (`SSMH_BRS_HE_106.i2syntax_KongNet.zst`) has extent 91156, i.e.
**the `.i2syntax` output is the one in the viewer's frame**; the `.svs` outputs are kept for
comparison but should not be loaded against `.i2syntax`-framed annotations.

### Run of 2026-08-25 — `/mnt/14T-2/tksong/tils-e2e-wsi`

10 files (8 distinct slides; 106/108 in both formats), `best_stg1.pth` (epoch 27, EMA), 6 GPUs,
conf 0.5 (the platform WSI recipe; our val sweep is flat over 0.35-0.60).

| slide | W×H | mpp | tiles | detections | non_tumor | tumor | Lymphocyte | min |
|---|---|---|---|---|---|---|---|---|
| Hagen-13674-21.tif | 77824×104448 | 0.2500 | 2,533 | 2,629,042 | 31.9 % | 16.7 % | 51.4 % | 3.9 |
| Hagen-25-25.tif | 90112×120832 | 0.2500 | 1,111 | 1,045,121 | 39.6 % | 17.8 % | 42.5 % | 2.0 |
| KU-304.mrxs | 83968×161792 | 0.2428 | 2,358 | 729,044 | 40.5 % | 47.1 % | 12.4 % | 4.0 |
| KU-305.mrxs | 88064×151296 | 0.2428 | 1,822 | 464,061 | 36.7 % | 46.0 % | 17.3 % | 3.8 |
| **SSMH_BRS_HE_106.i2syntax** | 160000×91156 | 0.2500 | 3,893 | 1,034,413 | 58.8 % | 18.5 % | 22.7 % | 3.7 |
| SSMH_BRS_HE_106.svs | 128543×88339 | 0.2626 | 3,828 | 1,028,274 | 55.3 % | 19.4 % | 25.3 % | 15.6 |
| **SSMH_BRS_HE_108.i2syntax** | 112000×89851 | 0.2500 | 3,027 | 568,339 | 56.1 % | 36.8 % | 7.1 % | 2.6 |
| SSMH_BRS_HE_108.svs | 115815×88339 | 0.2626 | 2,630 | 529,174 | 54.5 % | 38.2 % | 7.3 % | 2.5 |
| TIGER-129S.tif | 67456×45440 | 0.4567 | 3,107 | 687,902 | 47.2 % | 33.4 % | 19.4 % | 2.2 |
| TIGER-130S.tif | 41664×40576 | 0.4567 | 1,240 | 270,699 | 71.8 % | 6.0 % | 22.3 % | 1.1 |
| **total** | | | 25,549 | **8,986,069** | 44.8 % | 24.8 % | 30.4 % | 41 |

Every `.zst` was re-decoded after writing: feature count, extent (= level-0 slide height) and
per-category counts match `meta.json` exactly, and every feature carries the convention's
categoryId/termId/nt. The `.svs` and `.i2syntax` runs of the same slide agree on the class mix to
within ~3 pp despite being different scans — a consistency check in its own right.

### Three independent WSI-scale checks

**1. Same slide, before vs after the fine-tune** (`TIGER-129S.tif`, identical geometry — 3,107 of 6,868
tissue tiles in both runs; the pre-fine-tune run is kept in `results/tils3-wsi/_before-finetune/`):

| | total | tumour | non-tumour pool | lymphocyte |
|---|---|---|---|---|
| `ckpt/checkpoint0022.pth` (2-class) | 774,729 | 259,174 (33.5 %) | 515,555 | — |
| `best_stg1.pth` (3-class) | 687,902 | 229,693 (33.4 %) | 324,717 + 133,492 = 458,209 | 133,492 |

Every class shrinks by the same ~11 %, i.e. the fine-tuned model is uniformly a little more
conservative at conf 0.5 (matching the val measurement: precision 0.917 → 0.936, recall 0.884 → 0.872).
The **tumour share is preserved to 0.1 pp** (33.5 % → 33.4 %) — the tumour axis did not drift at
whole-slide scale — and 29.1 % of the non-tumour pool became lymphocyte.

**2. Same slide, same frame, different model** — `SSMH_BRS_HE_106.i2syntax` against the platform's
KongNet file (`det/scripts/compare_zst.py`, both extent 91156, so coordinates *are* comparable):

```
ours    1,034,413 detections   tumor 191,615 / non_tumor 608,019 / Lymphocyte 234,779
KongNet   855,425 detections   tumor 146,446 / non_tumor 448,030 / Lymphocyte 260,949

nearest KongNet detection to each of ours:  p50 2.24 px, p75 5.00 px, p90 25.96 px
within 12 px: 80.6 %      median offset of matched pairs: dx +1.00, dy +1.00 px
```

A wrong extent, a double y flip or a bounds offset would put the median offset in the thousands of
pixels; +1 px on both axes says the frame is right — **validated against a file this pipeline did not
write.** On the 833,362 matched pairs the two architectures agree on **89.3 %** of the class calls:

| reference \ ours | tumor | non_tumor | Lymphocyte |
|---|---|---|---|
| tumor | **127,001** | 16,888 | 928 |
| non_tumor | 8,734 | **413,805** | 16,010 |
| Lymphocyte | 9,343 | 37,076 | **203,577** |

Per class that is 87.8 % / 94.4 % / 81.4 % agreement. We are slightly more conservative on lymphocyte
(234,779 vs 260,949) while finding 21 % more cells overall — consistent with the val numbers
(lymphocyte precision 0.813 > recall 0.771). KongNet is a *reference, not ground truth*, so this
bounds the disagreement between two independent models; it does not decide which is right.

**3. The 2-class path still works** — the pre-fine-tune run above went through the same refactored
`write_zst` and came out as `convention=tnt` with categoryId 5/1, verified on decode. The
generalisation did not break the existing arm.

### What these numbers are not

The per-slide lymphocyte fraction ranges from 7.1 % to 51.4 %. Some of that is real tissue variation
(KU-304/305 are tumour-rich at 46-47 % tumour; TIGER-130S is 72 % non-tumour), but the model was
fine-tuned on **two patches from two slides**, so per-slide calibration on unseen scanners/stains is
untested. Before using these counts as TIL scores, validate on annotated regions from the slides in
question.
