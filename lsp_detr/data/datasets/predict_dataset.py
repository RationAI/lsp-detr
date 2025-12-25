from typing import Any

import albumentations as A
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset
from PIL import Image
from torch import Tensor


class PredictDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(self, data: Dataset, transforms: TransformsSeqType | None) -> None:
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self._to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.data[idx]
        image = sample["image"]
        original_image = Image.fromarray(image)

        image = self.transforms(image=image)["image"]
        image = self._to_tensor(image=image)["image"]
        return image, {
            "original_image": original_image,
            "filename": f"{idx}.png",
        }
