import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.backbone_utils import load_backbone

from lsp_detr.configuration import LSPDetrConfig, STAConfig
from lsp_detr.modeling.layers import MLP, CayleySTRING, FeedForward


flex_attention = torch.compile(flex_attention, dynamic=True)


def generate_sta_mask(
    q_canvas_w: int,
    kv_canvas_hw: tuple[int, int],
    kernel: int,
    q_tile: int,
    kv_tile: int,
) -> _mask_mod_signature:
    q_canvas_tile_w = q_canvas_w // q_tile
    kv_canvas_tile_h = kv_canvas_hw[0] // kv_tile
    kv_canvas_tile_w = kv_canvas_hw[1] // kv_tile

    def q_tile_rescale(x: Tensor):
        # Computes round(x * (kv_canvas_tile_w - 1) / (q_canvas_tile_w - 1))
        scale_numerator = kv_canvas_tile_w - 1
        scale_denominator = q_canvas_tile_w - 1
        return (x * scale_numerator + scale_denominator // 2) // scale_denominator

    def get_tile_xy(
        idx: Tensor, tile_size: int, canvas_tile_w: int
    ) -> tuple[Tensor, Tensor]:
        tile_id = idx // (tile_size * tile_size)
        tile_x = tile_id % canvas_tile_w
        tile_y = tile_id // canvas_tile_w
        return tile_x, tile_y

    def sta_mask_2d(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_x_tile, q_y_tile = get_tile_xy(q_idx, q_tile, q_canvas_tile_w)
        kv_x_tile, kv_y_tile = get_tile_xy(kv_idx, kv_tile, kv_canvas_tile_w)

        q_x_tile = q_tile_rescale(q_x_tile)
        q_y_tile = q_tile_rescale(q_y_tile)

        center_x = q_x_tile.clamp(kernel // 2, (kv_canvas_tile_w - 1) - kernel // 2)
        center_y = q_y_tile.clamp(kernel // 2, (kv_canvas_tile_h - 1) - kernel // 2)

        # Apply kernel mask in canvas coordinates (not tile coordinates)
        x_mask = torch.abs(center_x - kv_x_tile) <= kernel // 2
        y_mask = torch.abs(center_y - kv_y_tile) <= kernel // 2

        return x_mask & y_mask

    return sta_mask_2d


@lru_cache
def create_sta_block_mask(
    q_len: int,
    kv_len: int,
    q_width: int,
    kv_width: int,
    kernel: int,
    q_tile: int,
    kv_tile: int,
) -> BlockMask:
    return create_block_mask(
        generate_sta_mask(
            q_width, (kv_len // kv_width, kv_width), kernel, q_tile, kv_tile
        ),
        B=None,
        H=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        Q_LEN=q_len,
        KV_LEN=kv_len,
        _compile=True,
    )


@torch.autocast("cuda", enabled=False)
def relative_to_absolute_pos(pos: Tensor, step_x: float, step_y: float) -> Tensor:
    pos = pos.sigmoid()
    h, w = pos.shape[1:3]

    anchor_x = torch.arange(w, dtype=torch.float32, device=pos.device) * step_x
    anchor_y = torch.arange(h, dtype=torch.float32, device=pos.device) * step_y

    absolute_x = pos[..., 0] * step_x + anchor_x
    absolute_y = pos[..., 1] * step_y + anchor_y.unsqueeze(1)
    return torch.stack((absolute_x, absolute_y), dim=-1)


class STAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        src_dim: int,
        num_heads: int,
        kernel: int,
        q_tile: int,
        kv_tile: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.kernel = kernel
        self.q_tile = q_tile
        self.kv_tile = kv_tile

        self.pe = CayleySTRING(dim // num_heads)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(src_dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def maybe_pad(self, x: Tensor, tile: int) -> Tensor:
        h, w = x.shape[1:3]
        pad_right = (tile - w % tile) % tile
        pad_bottom = (tile - h % tile) % tile
        return F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))

    def tile(self, x: Tensor, height: int, tile: int) -> tuple[Tensor, int, int]:
        x = rearrange(x, "b head (h w) dim -> b h w (head dim)", h=height)
        x = self.maybe_pad(x, tile)
        h, w = x.shape[1:3]
        x = rearrange(
            x,
            "b (n_h ts_h) (n_w ts_w) (h d) -> b h (n_h n_w ts_h ts_w) d",
            ts_h=tile,
            ts_w=tile,
            h=self.num_heads,
        )
        return x, h, w

    def forward(
        self, tgt: Tensor, src: Tensor, q_coords: Tensor, k_coords: Tensor
    ) -> Tensor:
        h, w = tgt.shape[1:3]

        q = rearrange(
            self.q(tgt), "b h w (head d) -> b head (h w) d", head=self.num_heads
        )
        k, v = rearrange(
            self.kv(src),
            "b h w (two head d) -> two b head (h w) d",
            two=2,
            head=self.num_heads,
        )

        # RoPE
        q = self.pe(q, q_coords)
        k = self.pe(k, k_coords)

        # tile
        q, q_h, q_w = self.tile(q, h, self.q_tile)
        k, _, kv_w = self.tile(k, src.shape[1], self.kv_tile)
        v, _, _ = self.tile(v, src.shape[1], self.kv_tile)

        # flex attention
        block_mask = create_sta_block_mask(
            q_len=q.shape[2],
            kv_len=k.shape[2],
            q_width=q_w,
            kv_width=kv_w,
            kernel=self.kernel,
            q_tile=self.q_tile,
            kv_tile=self.kv_tile,
        )
        x = flex_attention(q, k, v, block_mask=block_mask)

        # un-tile
        x = rearrange(
            x,
            "b h (n_h n_w ts_h ts_w) d -> b (n_h ts_h) (n_w ts_w) (h d)",
            n_h=q_h // self.q_tile,
            n_w=q_w // self.q_tile,
            ts_h=self.q_tile,
            ts_w=self.q_tile,
        )

        # remove padding
        x = x[:, :h, :w, :].contiguous()

        return self.wo(x)


class Layer(nn.Module):
    def __init__(
        self,
        dim: int,
        src_dim: int,
        num_heads: int,
        self_sta_config: STAConfig,
        cross_sta_config: STAConfig,
    ) -> None:
        super().__init__()

        self.self_attention = STAttention(
            dim,
            dim,
            num_heads,
            kernel=self_sta_config["kernel"],
            q_tile=self_sta_config["q_tile"],
            kv_tile=self_sta_config["kv_tile"],
        )
        self.self_attention_norm = nn.LayerNorm(dim)

        self.cross_attention = STAttention(
            dim,
            src_dim,
            num_heads,
            kernel=cross_sta_config["kernel"],
            q_tile=cross_sta_config["q_tile"],
            kv_tile=cross_sta_config["kv_tile"],
        )
        self.cross_attention_norm = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor
    ) -> Tensor:
        x = self.self_attention(tgt, tgt, tgt_coords, tgt_coords)
        tgt = self.self_attention_norm(tgt + x)

        x = self.cross_attention(tgt, src, tgt_coords, src_coords)
        tgt = self.cross_attention_norm(tgt + x)

        return self.ffn_norm(tgt + self.ffn(tgt))


class LSPTransformer(nn.Module):
    def __init__(self, config: LSPDetrConfig, feature_channels: list[int]) -> None:
        super().__init__()

        self.query_block_size = config.query_block_size
        self.num_radial_distances = config.num_radial_distances
        self.feature_levels = config.feature_levels
        self.num_classes = config.num_classes + 1

        self.layers = nn.ModuleList()
        for level in config.feature_levels:
            layer = Layer(
                dim=config.dim,
                src_dim=feature_channels[level],
                num_heads=config.num_heads,
                self_sta_config=config.self_sta_config,
                cross_sta_config=config.cross_sta_config[level],
            )
            self.layers.append(layer)

        # output heads
        self.class_head = nn.Linear(config.dim, self.num_classes)
        self.point_head = nn.ModuleList(
            MLP(config.dim, config.dim, 2, 3) for _ in config.feature_levels
        )
        self.radial_distances_head = nn.ModuleList(
            MLP(config.dim, config.dim, config.num_radial_distances, 3)
            for _ in config.feature_levels
        )

        self.init_weights()

    def init_weights(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_head.bias, bias_value)

        # initialize regression layers
        for head in self.point_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

        for head in self.radial_distances_head:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)

    def forward(
        self,
        tgt: Tensor,
        ref_points: Tensor,
        features: list[Tensor],
        height: int,
        width: int,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        src = []
        src_coords = []
        for feature in features:
            b, _, h, w = feature.shape
            coords = torch.zeros(b, h, w, 2, dtype=torch.float32, device=feature.device)
            coords = relative_to_absolute_pos(
                coords, step_x=math.ceil(width / w), step_y=math.ceil(height / h)
            )
            # the outputs from SwinV2 are already normalized
            src.append(rearrange(feature, "b c h w -> b h w c"))
            src_coords.append(rearrange(coords, "b h w pos -> b (h w) pos"))

        radial_distances = torch.full(
            (*tgt.shape[:3], self.num_radial_distances),
            math.log(self.query_block_size / 2),
            dtype=torch.float32,
            device=tgt.device,
        )

        logits_list: list[Tensor] = []
        ref_points_list: list[Tensor] = []
        radial_distances_list: list[Tensor] = []

        # for look forward twice
        new_ref_points = ref_points.clone()
        new_radial_distances = radial_distances.clone()

        for i, layer in enumerate(self.layers):
            tgt = layer(
                tgt=tgt,
                src=src[self.feature_levels[i]],
                tgt_coords=relative_to_absolute_pos(
                    ref_points, self.query_block_size, self.query_block_size
                ).flatten(1, 2),
                src_coords=src_coords[self.feature_levels[i]],
            )

            # output heads
            delta_point = self.point_head[i](tgt)
            delta_distances = self.radial_distances_head[i](tgt)
            logits = self.class_head(tgt)

            ref_points_list.append(
                relative_to_absolute_pos(
                    new_ref_points + delta_point,
                    step_x=self.query_block_size / width,
                    step_y=self.query_block_size / height,
                ).flatten(1, 2)
            )
            logits_list.append(logits.flatten(1, 2))
            radial_distances_list.append(
                torch.flatten(new_radial_distances + delta_distances, 1, 2)
            )

            new_ref_points = ref_points + delta_point
            new_radial_distances = radial_distances + delta_distances
            ref_points = new_ref_points.detach()
            radial_distances = new_radial_distances.detach()

        return {
            "logits": logits_list[-1],
            "points": ref_points_list[-1],
            "radial_distances": radial_distances_list[-1],
            "absolute_points": relative_to_absolute_pos(
                ref_points, self.query_block_size, self.query_block_size
            ).flatten(1, 2),
            "embeddings": tgt.flatten(1, 2),
            "aux_outputs": [
                {
                    "logits": a,
                    "points": b,
                    "radial_distances": c,
                }
                for a, b, c in zip(
                    logits_list[:-1],
                    ref_points_list[:-1],
                    radial_distances_list[:-1],
                    strict=True,
                )
            ],
        }


class FeatureSampling(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, points: Tensor, feature: Tensor) -> Tensor:
        x = F.grid_sample(self.reduction(feature), points * 2 - 1, align_corners=False)
        return self.norm(rearrange(x, "b c h w -> b h w c"))


class LSPDetrModel(PreTrainedModel):
    config_class = LSPDetrConfig

    def __init__(self, config: LSPDetrConfig) -> None:
        super().__init__(config)
        self.query_block_size = config.query_block_size

        self.backbone = load_backbone(config)
        _, *feature_channels, neck = self.backbone.num_features

        self.feature_sampling = FeatureSampling(neck, config.dim)
        self.decode_head = LSPTransformer(config, feature_channels)

    def forward(self, pixel_values: Tensor) -> dict[str, Tensor]:
        b, _, h, w = pixel_values.shape

        *features, neck = self.backbone(pixel_values).feature_maps

        ref_points = torch.zeros(
            b,
            math.ceil(h / self.query_block_size),
            math.ceil(w / self.query_block_size),
            2,
            dtype=torch.float32,
            device=neck.device,
        )  # center positions
        tgt = self.feature_sampling(
            relative_to_absolute_pos(
                ref_points, self.query_block_size, self.query_block_size
            ),
            neck,
        )

        return self.decode_head(tgt, ref_points, features, h, w)
