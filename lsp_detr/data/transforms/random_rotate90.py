from typing import Any

import albumentations as A
from numpy.typing import NDArray


class RandomRotate90(A.RandomRotate90):
    def apply_to_mask(
        self, mask: NDArray[Any], *args: Any, **params: Any
    ) -> NDArray[Any]:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, *args, **params)
