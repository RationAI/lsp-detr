from torch import nn


class MLP(nn.Sequential):
    """Very simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        assert num_layers > 1

        layers = []
        h = [hidden_dim] * (num_layers - 1)
        for n, k in zip([input_dim, *h], h, strict=False):
            layers.append(nn.Linear(n, k))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        super().__init__(*layers)
