from typing import Any

import albumentations as A
from numpy.typing import NDArray


class ElasticTransform(A.ElasticTransform):
    def apply_to_mask(
        self, mask: NDArray, map_x: NDArray, map_y: NDArray, **params: Any
    ) -> NDArray:
        if mask.size == 0:
            return mask
        return super().apply_to_mask(mask, map_x, map_y, **params)
