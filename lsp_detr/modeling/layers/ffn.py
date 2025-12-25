import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    """FeedForward module.

    Taken from https://github.com/meta-llama/llama-models/blob/main/models/llama4/ffn.py
    """

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        """Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
