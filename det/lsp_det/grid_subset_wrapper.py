"""``GridSubsetCropWrapper``: multi-crop wrapper that draws only from a *whitelist* of centre cells.

Dome's ``MultiCropDatasetWrapper`` expands each image into N crops and lets
``RandomCropWithGrid`` pick any of the image's ``ceil`` grid cells.  For the two-patch TILs
fine-tune we hold a few cells out as fixed validation tiles, so the training crops must never be
taken from those cells (nor from their edge neighbours, which would overlap a val tile by more than
half its area).  This wrapper is the same contract as Dome's - it exposes ``dataset`` /
``crops_per_image`` so ``dist_utils.warp_loader`` still picks ``MultiCropDistributedSampler`` and its
``global_idx = original_idx * crops_per_image + crop_idx`` indexing - but it sets
``target['grid_index']`` itself from the per-image whitelist instead of letting the transform choose.

``allowed_grids`` maps ``file_name`` -> list of *linear* grid indices ``iy * num_tiles_x + ix``
(``det/scripts/build_tils3_dataset.py`` writes it).  ``strategy='cycle'`` (default) walks the
whitelist deterministically so one epoch of ``crops_per_image = k * len(whitelist)`` covers every
allowed cell exactly k times; ``'random'`` samples uniformly instead.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.core import register

__all__ = ["GridSubsetCropWrapper"]


@register()
class GridSubsetCropWrapper(Dataset):
    __inject__ = ["dataset"]
    __share__ = []

    def __init__(self, dataset: Dataset = None, crops_per_image: int = 16,
                 allowed_grids: Optional[str] = None, strategy: str = "cycle",
                 seed: Optional[int] = None, **kwargs) -> None:
        super().__init__()
        if dataset is None and "type" in kwargs:
            from src.core import create
            dataset = create("dataset", {"dataset": kwargs.copy()})
        if dataset is None:
            raise ValueError("GridSubsetCropWrapper needs a `dataset`")
        assert strategy in ("cycle", "random"), strategy

        self.dataset = dataset
        self.crops_per_image = int(crops_per_image)
        self.strategy = strategy
        self.seed = seed
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed) if seed is not None else 0)

        if allowed_grids is None:
            raise ValueError("GridSubsetCropWrapper needs `allowed_grids` (json path or dict)")
        table: Dict[str, List[int]]
        if isinstance(allowed_grids, str):
            with open(os.path.expanduser(allowed_grids)) as f:
                table = json.load(f)
        else:
            table = dict(allowed_grids)

        # resolve file_name -> whitelist for every dataset index, in dataset order
        self.grids: List[List[int]] = []
        for idx in range(len(self.dataset)):
            image_id = self.dataset.ids[idx]
            fn = self.dataset.coco.loadImgs(image_id)[0]["file_name"]
            key = fn if fn in table else os.path.basename(fn)
            if key not in table:
                raise KeyError(f"no allowed_grids entry for {fn!r} (have {sorted(table)[:5]}...)")
            cells = sorted(int(c) for c in table[key])
            if not cells:
                raise ValueError(f"empty allowed-grid whitelist for {fn!r}")
            self.grids.append(cells)

        print(f"[GridSubsetCropWrapper] {len(self.dataset)} images x {self.crops_per_image} crops "
              f"= {len(self)} crops/epoch; strategy={strategy}; allowed cells per image="
              f"{[len(g) for g in self.grids]}")
        for idx, cells in enumerate(self.grids):
            image_id = self.dataset.ids[idx]
            fn = self.dataset.coco.loadImgs(image_id)[0]["file_name"]
            reps = self.crops_per_image / len(cells)
            print(f"  - {fn}: {len(cells)} cells {cells} ({reps:.2f} crops/cell per epoch)")

    # Dome's DataLoader.set_epoch forwards to the dataset (DetDataset's contract); a plain
    # torch Dataset would raise, so mirror it and pass the epoch on to the inner CocoDetection.
    def set_epoch(self, epoch) -> None:
        self._epoch = epoch
        inner = getattr(self.dataset, "set_epoch", None)
        if callable(inner):
            inner(epoch)

    @property
    def epoch(self):
        return getattr(self, "_epoch", -1)

    def __len__(self) -> int:
        return len(self.dataset) * self.crops_per_image

    def __getitem__(self, idx: int):
        original_idx, crop_idx = divmod(int(idx), self.crops_per_image)
        cells = self.grids[original_idx]
        if self.strategy == "cycle":
            grid_index = cells[crop_idx % len(cells)]
        else:
            grid_index = cells[int(torch.randint(0, len(cells), (1,), generator=self._rng).item())]

        image, target = self.dataset.load_item(original_idx)
        target["grid_index"] = grid_index
        target["crop_idx"] = crop_idx
        target["original_idx"] = original_idx

        transforms = getattr(self.dataset, "_transforms", None)
        if transforms is not None:
            result = transforms(image, target, self.dataset)
            if isinstance(result, tuple) and len(result) == 3:
                image, target, _ = result
            elif isinstance(result, tuple) and len(result) == 2:
                image, target = result
            else:
                raise TypeError(f"Unexpected transform output: {type(result)}")
        if not isinstance(target, dict):
            raise TypeError(f"After transforms, target should be dict, got {type(target)}")
        return image, target
