import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.misc import linear_assignment


class AggregatedJaccardIndex(Metric):
    """Aggregated Jaccard Index (AJI) metric.

    As described in the paper "A Multi-Organ Nucleus Segmentation Challenge" by Kumar et al.

    This is an efficient implementation of the AJI metric that uses the linear sum assignment algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "aji_accumulated", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the AJI metric state.

        Args:
            preds ([N, H, W]): Binary tensor of predicted instance segmentation masks.
            target ([M, H, W]): Binary tensor of target instance segmentation masks.
        """
        if target.numel() == 0:
            return

        preds = preds.flatten(1).float()
        target = target.flatten(1).float()

        # Compute iou between all pairs
        intersection = target @ preds.T
        union = target.sum(dim=1, keepdim=True) + preds.sum(dim=1) - intersection
        iou = intersection / union  # [M, N]

        # Perform linear sum assignment
        row_ind, col_ind = linear_assignment(-iou)

        # Sum intersection and union over matched pairs
        intersection_sum = intersection[row_ind, col_ind].sum()
        union_sum = preds.sum() + target.sum() - intersection_sum

        self.aji_accumulated += intersection_sum / union_sum
        self.count += 1

    def compute(self) -> Tensor:
        return self.aji_accumulated / self.count
