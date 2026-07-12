from typing import TypedDict

from transformers import PretrainedConfig


class STAConfig(TypedDict):
    kernel: int
    q_tile: int
    kv_tile: int


class LSPDetrConfig(PretrainedConfig):
    model_type = "lsp_detr"

    def __init__(
        self,
        use_timm_backbone: bool = False,
        use_pretrained_backbone: bool = True,
        backbone: str = "microsoft/swinv2-tiny-patch4-window16-256",
        backbone_out_features: tuple[str, ...] = (
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        ),
        dim: int = 384,
        num_heads: int = 12,
        num_classes: int = 1,
        query_block_size: float = 14.222222222222223,  # 256 / 18
        feature_levels: tuple[int, ...] = (2, 1, 0),  # , 2, 1, 0),
        num_radial_distances: int = 64,
        self_sta_config: STAConfig | None = None,
        cross_sta_config: tuple[STAConfig, ...] = (
            {"kernel": 5, "q_tile": 3, "kv_tile": 8},
            {"kernel": 5, "q_tile": 3, "kv_tile": 4},
            {"kernel": 5, "q_tile": 3, "kv_tile": 2},
        ),
        **kwargs,
    ) -> None:
        if self_sta_config is None:
            self_sta_config = {"kernel": 3, "q_tile": 3, "kv_tile": 3}

        self.use_timm_backbone = use_timm_backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone = backbone
        self.backbone_out_features = list(backbone_out_features)
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.query_block_size = query_block_size
        self.feature_levels = feature_levels
        self.num_radial_distances = num_radial_distances
        self.self_sta_config = self_sta_config
        self.cross_sta_config = cross_sta_config

        super().__init__(**kwargs)
