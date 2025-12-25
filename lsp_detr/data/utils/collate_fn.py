from typing import Any

import torch
from torch import Tensor


def collate_fn(batch: list[tuple[Tensor, Any]]) -> tuple[Tensor, list[Any]]:
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]
