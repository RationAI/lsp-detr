import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.metrics.binary_panoptic_quality import binary_panoptic_quality


class PanopticQuality(Metric):
    def __init__(
        self, num_classes: int, iou_threshold: float = 0.5, masked: bool = False
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.masked = masked

        self.add_state("mDQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mSQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mPQ", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> None:
        if targets["masks"].numel() == 0:
            return

        dqs = []
        sqs = []
        pqs = []

        for c in range(self.num_classes):
            if torch.any(targets["labels"] == c):
                dq, sq, pq = binary_panoptic_quality(
                    preds["masks"][preds["labels"] == c],
                    targets["masks"][targets["labels"] == c],
                    self.iou_threshold,
                    self.masked,
                )
                dqs.append(dq)
                sqs.append(sq)
                pqs.append(pq)

        self.mDQ += torch.mean(torch.tensor(dqs))
        self.mSQ += torch.mean(torch.tensor(sqs))
        self.mPQ += torch.mean(torch.tensor(pqs))
        self.count += 1

    def compute(self) -> dict[str, Tensor]:
        if self.masked:
            return {
                "mMDQ": self.mDQ / self.count,
                "mMSQ": self.mSQ / self.count,
                "mMPQ": self.mPQ / self.count,
            }

        return {
            "mDQ": self.mDQ / self.count,
            "mSQ": self.mSQ / self.count,
            "mPQ": self.mPQ / self.count,
        }
