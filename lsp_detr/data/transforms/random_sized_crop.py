from typing import Any

import albumentations as A
from numpy.typing import NDArray


class RandomSizedCrop(A.RandomSizedCrop):
    def apply_to_mask(
        self, mask: NDArray, crop_coords: tuple[int, int, int, int], **params: Any
    ) -> NDArray:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, crop_coords, **params)
