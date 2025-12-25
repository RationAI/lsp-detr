# Modified by Matěj Pekár from https://github.com/hustvl/LKCell/blob/main/cell_segmentation/datasets/pannuke.py

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import WeightedRandomSampler


class WeightedTissueSampler(WeightedRandomSampler):
    def __init__(
        self, tissues: NDArray[np.uint8], gamma: float = 0.85, replacement: bool = True
    ) -> None:
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"

        weights = self.get_sampling_weights_tissue(tissues, gamma)
        super().__init__(weights.tolist(), len(tissues), replacement)

    @staticmethod
    def get_sampling_weights_tissue(
        tissues: NDArray[np.uint8], gamma: float = 1
    ) -> NDArray[np.float64]:
        _, counts = np.unique(tissues, return_counts=True)
        n = len(tissues)
        weights = n / (gamma * counts + (1 - gamma) * n)
        return weights[tissues]
