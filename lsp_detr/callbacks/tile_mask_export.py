from collections.abc import Iterable
from pathlib import Path

from lightning import Callback, LightningModule, Trainer
from PIL import ImageDraw
from torch import Tensor
from torch.types import Number


class TileMaskExport(Callback):
    def __init__(
        self, output_dir: str | Path, color_map: dict[Number, list[int]]
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_map = color_map

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Iterable[dict[str, Tensor]],
        batch: tuple[Tensor, list],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for b, output in enumerate(outputs):
            tile = batch[1][b]["original_image"].copy()
            canvas = ImageDraw.Draw(tile)

            for polygon, label in zip(
                output["polygons"], output["labels"], strict=True
            ):
                canvas.polygon(
                    xy=polygon.flatten().tolist(),
                    outline=tuple(self.color_map[label.item()]),
                    width=1,
                )

            tile.save(self.output_dir / batch[1][b]["filename"])
