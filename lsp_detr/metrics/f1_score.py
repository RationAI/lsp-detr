import torch
from torch import Tensor
from torchmetrics import Metric

from lsp_detr.misc import linear_assignment, masks2centroids


class F1Score(Metric):
    def __init__(self, radius: int = 12) -> None:
        super().__init__()
        self.radius = radius

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update the F1 metric state.

        Args:
            preds ([N, H, W]): Binary tensor of predicted instance segmentation masks.
            targets ([M, H, W]): Binary tensor of targets instance segmentation masks.
        """
        pred_centroids = masks2centroids(preds)
        target_centroids = masks2centroids(targets)

        dists = torch.cdist(
            pred_centroids,
            target_centroids,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",  # for numerical stability
        )  # [N, M]
        row_ind, col_ind = linear_assignment(dists)
        valid = dists[row_ind, col_ind] <= self.radius

        self.tp += valid.sum()
        self.fp += preds.size(0) - valid.sum()
        self.fn += targets.size(0) - valid.sum()

    def compute(self) -> dict[str, Tensor]:
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {"Precision": precision, "Recall": recall, "F1": f1_score}
