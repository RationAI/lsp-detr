import torch
from torch import Tensor


def masks2centroids(masks: Tensor, normalize: bool = False) -> Tensor:
    """Convert binary masks to centroids.

    Args:
        masks: Binary instance masks of shape [N, H, W].
        normalize: If True, normalize centroids to [0, 1] range.

    Returns:
        Tensor: Centroids of shape [N, 2], where 2 is (x, y) coordinates.
    """
    centroids = [torch.nonzero(mask).float().mean(0) for mask in masks]

    if centroids:
        centroids = torch.stack(centroids)
        if normalize:
            centroids[:, 0] /= masks.shape[1]
            centroids[:, 1] /= masks.shape[2]
        return centroids.flip(-1)  # (y, x) -> (x, y)

    return torch.empty((0, 2), dtype=torch.float32, device=masks.device)
