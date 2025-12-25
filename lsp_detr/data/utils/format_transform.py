from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def format_transform(batch: dict[str, Any]) -> dict[str, Any]:
    def instance_to_mask(instances: list[Image.Image]) -> NDArray[np.uint8]:
        if instances:
            return np.array(instances).transpose(1, 2, 0).astype(np.uint8)
        return np.empty((256, 256, 0), dtype=np.uint8)

    if "image" in batch:
        batch["image"] = [np.array(img, dtype=np.uint8) for img in batch["image"]]

    if "instances" in batch:
        batch["instances"] = [instance_to_mask(mask) for mask in batch["instances"]]

    if "categories" in batch:
        batch["categories"] = [
            np.array(cat, dtype=np.uint8) for cat in batch["categories"]
        ]
    return batch
