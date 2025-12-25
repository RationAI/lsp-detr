import torch
from torch import Tensor
from torch_linear_assignment import assignment_to_indices, batch_linear_assignment


@torch.compiler.disable
def linear_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    if cost.device.type == "mps":
        # MPS does not support batch_linear_assignment, so we need to use a workaround
        # by moving the cost matrix to CPU and then back to MPS.
        row_ind, col_ind = assignment_to_indices(
            batch_linear_assignment(cost[None].cpu())
        )
        return row_ind[0].to(cost.device), col_ind[0].to(cost.device)

    row_ind, col_ind = assignment_to_indices(batch_linear_assignment(cost[None]))
    return row_ind[0], col_ind[0]
