import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset
from stardist import star_distances
from torch import Tensor

from lsp_detr.misc import masks2centroids


class SPDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Tensor]]]):
    def __init__(
        self,
        data: Dataset,
        transforms: TransformsSeqType | None = None,
        n_rays: int = 64,
        allow_overlaps: bool = True,
    ) -> None:
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self.n_rays = n_rays
        self.allow_overlaps = allow_overlaps
        self._to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        sample = self.data[idx]
        image = sample["image"]
        masks = sample["instances"]
        labels = (
            sample["categories"]
            if "categories" in sample
            else np.zeros(masks.shape[-1])
        )

        # Apply transforms
        transformed = self.transforms(image=image, mask=masks)
        image = transformed["image"]
        masks = transformed["mask"].transpose(2, 0, 1)

        # Drop empty masks after augmentations
        keep = masks.any(axis=(1, 2))
        masks = masks[keep]
        labels = labels[keep]

        lower_bound, upper_bound = star_distances(masks, self.n_rays)
        if not self.allow_overlaps:
            upper_bound = lower_bound
        radial_distances = np.stack((lower_bound, upper_bound), axis=0)

        image = self._to_tensor(image=image)["image"]
        masks = torch.from_numpy(masks)

        return image, {
            "masks": masks,
            "labels": torch.from_numpy(labels).long(),
            "radial_distances": torch.from_numpy(radial_distances),
            "centroids": masks2centroids(masks, normalize=True),
            "tissue": self.data.features["tissue"].names[sample["tissue"]],
        }
