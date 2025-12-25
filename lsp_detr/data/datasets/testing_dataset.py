from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset
from torch import Tensor


class TestingDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self, data: Dataset, transforms: TransformsSeqType | None = None
    ) -> None:
        super().__init__()
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self._to_tensor = ToTensorV2(transpose_mask=True)
        self.categories = (
            self.data.features["categories"].feature.names
            if "categories" in self.data.features
            else []
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.data[idx]
        image = sample["image"]
        masks = sample["instances"]
        labels = (
            sample["categories"]
            if "categories" in sample
            else np.zeros(masks.shape[-1])
        )

        transformed = self.transforms(image=image, mask=masks)
        image = transformed["image"]
        mask = transformed["mask"]

        transformed = self._to_tensor(image=image, mask=mask)

        return transformed["image"], {
            "masks": transformed["mask"],
            "labels": torch.from_numpy(labels).long(),
            "tissue": self.data.features["tissue"].names[sample["tissue"]],
        }
