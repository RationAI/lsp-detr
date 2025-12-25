import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.misc import linear_assignment


def binary_panoptic_quality(
    preds: Tensor,
    targets: Tensor,
    iou_threshold: float = 0.5,
    masked: bool = False,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    preds = preds.flatten(1).float()
    targets = targets.flatten(1).float()

    # Compute iou between all pairs
    intersection = targets @ preds.T
    if masked:
        masked_preds = preds.bool() & ~targets.bool().any(0)
        union = targets.sum(dim=1, keepdim=True) + masked_preds.sum(dim=1)
    else:
        union = targets.sum(dim=1, keepdim=True) + preds.sum(dim=1) - intersection
    iou = intersection / union  # [M, N]

    # Perform linear sum assignment
    row_ind, col_ind = linear_assignment(-iou)
    iou = iou[row_ind, col_ind]

    # Filter assignments with iou <= threshold
    iou = iou[iou > iou_threshold]
    tp = iou.numel()

    dq = tp / (0.5 * preds.size(0) + 0.5 * targets.size(0) + eps)
    sq = iou.sum() / (tp + eps)
    pq = dq * sq

    return dq, sq.item(), pq.item()


class BinaryPanopticQuality(Metric):
    def __init__(self, iou_threshold: float = 0.5, masked: bool = False) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.masked = masked
        self.add_state("bDQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bSQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bPQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the bPQ metric state.

        Args:
            preds ([N, H, W]): Binary tensor of predicted instance segmentation masks.
            targets ([M, H, W]): Binary tensor of targets instance segmentation masks.
        """
        if targets.numel() == 0:
            return

        dq, sq, pq = binary_panoptic_quality(
            preds, targets, self.iou_threshold, self.masked
        )

        self.bDQ += dq
        self.bSQ += sq
        self.bPQ += pq
        self.count += 1

    def compute(self) -> dict[str, Tensor]:
        if self.masked:
            return {
                "bMDQ": self.bDQ / self.count,
                "bMSQ": self.bSQ / self.count,
                "bMPQ": self.bPQ / self.count,
            }

        return {
            "bDQ": self.bDQ / self.count,
            "bSQ": self.bSQ / self.count,
            "bPQ": self.bPQ / self.count,
        }
