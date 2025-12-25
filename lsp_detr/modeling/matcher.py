# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
# Modified by Matěj Pekár from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
"""Modules to compute the matching cost and solve the corresponding LSAP."""

from collections.abc import Iterable
from typing import TypedDict

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from lsp_detr.misc import linear_assignment


class Outputs(TypedDict):
    logits: Tensor
    radial_distances: Tensor
    points: Tensor


class Target(TypedDict):
    labels: Tensor
    masks: Tensor
    centroids: Tensor
    radial_distances: Tensor


def l1_log_space_cost(
    outputs: Tensor, tgt_min_bound: Tensor, tgt_max_bound: Tensor
) -> Tensor:
    """Compute the L1 log-space loss for the radial distances."""
    # L1 minimum bound
    loss_min = F.relu(tgt_min_bound.log() - outputs)

    # L1 maximum bound
    loss_max = F.relu(outputs - tgt_max_bound.log())

    loss = torch.max(loss_min, loss_max)
    return loss.mean(-1).nan_to_num(1.0)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_point: float = 1.0,
        cost_radial_distances: float = 1.0,
        cost_inner: float = 9999,
    ) -> None:
        super().__init__()
        assert (
            cost_class != 0
            or cost_point != 0
            or cost_radial_distances != 0
            or cost_inner != 0
        ), "all costs can't be 0"

        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cost_radial_distances = cost_radial_distances
        self.cost_inner = cost_inner

    def compute_cost_labels(
        self,
        out_prob: Tensor,
        tgt_labels: Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> Tensor:
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        return pos_cost_class[:, tgt_labels] - neg_cost_class[:, tgt_labels]

    def compute_cost_inner(self, out_point: Tensor, tgt_mask: Tensor) -> Tensor:
        """Compute if points lie inside any mask.

        Args:
            out_point: Points tensor of shape (N,2) scaled 0-1, 2 is (x,y)
            tgt_mask: Binary masks tensor of shape (M,H,W)

        Returns:
            cost_inner: Tensor of shape (N,M) with 0 if point inside mask, 1 if outside
        """
        if tgt_mask.size(0) == 0:
            return torch.empty((out_point.size(0), 0), device=out_point.device)

        out_point = out_point[None, None] * 2 - 1  # Scale to [-1, 1]
        sampled_mask = F.grid_sample(tgt_mask[None], out_point, align_corners=False)
        return rearrange(-sampled_mask, "1 M 1 N -> N M")

    def compute_cost_radial_distances(
        self,
        out_point: Tensor,
        out_radial_distances: Tensor,
        tgt_radial_distances: Tensor,
    ) -> Tensor:
        """Compute L1 loss between predicted and target radial distances.

        We only compute the cost for each predicted point, as inner loss penalizes mismatched points.

        Args:
            out_point (N, 2): (x,y) coordinates scaled to 0-1
            out_radial_distances (N, R): R radial distances for each point
            tgt_radial_distances (2, R, H, W): Map of target radial distances

        Returns:
            cost_radial_distances (N, 1): L1 loss between predicted and target radial distances
        """
        grid = 2 * out_point - 1
        sampled = F.grid_sample(
            tgt_radial_distances,
            repeat(grid, "N two -> 2 1 N two"),
            align_corners=False,
            padding_mode="zeros",  # "border" results in NaN with Infs
        )  # [2, R, 1, N]

        min_bound, max_bound = rearrange(sampled, "two R 1 N -> two N 1 R")

        return l1_log_space_cost(
            outputs=rearrange(out_radial_distances, "N R -> N 1 R"),
            tgt_min_bound=min_bound,
            tgt_max_bound=max_bound,
        )

    @torch.no_grad()
    def forward(
        self, outputs: Outputs, targets: Iterable[Target]
    ) -> list[tuple[Tensor, Tensor]]:
        """Performs memory-friendly matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        indices: list[tuple[Tensor, Tensor]] = []

        for out_prob, out_radial_distances, out_points, target in zip(
            outputs["logits"].sigmoid(),
            outputs["radial_distances"],
            outputs["points"],
            targets,
            strict=True,
        ):
            tgt_mask = target["masks"].to(out_prob)

            cost_class = self.compute_cost_labels(out_prob, target["labels"])
            cost_inner = self.compute_cost_inner(out_points, tgt_mask)
            cost_point = torch.cdist(
                out_points, target["centroids"].to(out_points), p=1
            )
            cost_radial_distances = self.compute_cost_radial_distances(
                out_points,
                out_radial_distances,
                target["radial_distances"].to(out_radial_distances),
            )

            cost_matrix = (
                self.cost_class * cost_class
                + self.cost_point * cost_point
                + self.cost_radial_distances * cost_radial_distances
                + self.cost_inner * cost_inner
            )
            indices.append(linear_assignment(cost_matrix))

        return indices
