import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.misc import linear_assignment


class AveragePrecision2018DSB(Metric):
    """Average Precision metric used in the 2018 Data Science Bowl.

    As described in the 2018 Data Science Bowl. IT DIFFERS FROM THE STANDARD AP.
    """

    def __init__(self, iou_thresholds: Tensor | list) -> None:
        super().__init__()
        self.iou_thresholds = (
            iou_thresholds
            if isinstance(iou_thresholds, Tensor)
            else torch.tensor(iou_thresholds)
        )
        self.add_state(
            "ap_accumulated", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the AP metric state.

        Args:
            preds ([N, H, W]): Binary tensor of predicted instance segmentation masks
            target ([M, H, W]): Binary tensor of target instance segmentation masks
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

        # Filter assignments with IoU above the thresholds
        iou_matches = iou[row_ind, col_ind] > self.iou_thresholds[:, None].to(
            iou
        )  # [num_thresholds, num_matches]

        # Compute true positives for each threshold
        tp = iou_matches.sum(dim=1)  # [num_thresholds]

        # Compute APs for all thresholds
        total = preds.size(0) + target.size(0) - tp
        self.ap_accumulated += torch.mean(tp / total)
        self.count += 1

    def compute(self) -> Tensor:
        return self.ap_accumulated / self.count
