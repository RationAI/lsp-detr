import re
from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import Tensor
from torch.nn import ModuleDict
from torchmetrics import Metric, MetricCollection

from lsp_detr.configuration import LSPDetrConfig
from lsp_detr.image_processing import LSPDetrImageProcessor
from lsp_detr.metrics import (
    AggregatedJaccardIndex,
    AveragePrecision2018DSB,
    BinaryPanopticQuality,
    F1Score,
    PanopticQuality,
)
from lsp_detr.modeling import SetCriterion
from lsp_detr.modeling.lsp_detr import LSPDetrModel


class NestedMetricCollection(ModuleDict):
    def __init__(self, metric: Metric | MetricCollection) -> None:
        super().__init__()
        self.metric = metric

    def update(self, *args: Any, key: str, **kwargs: Any) -> None:  # type: ignore[override]
        if key not in self:
            self.add_module(key, deepcopy(self.metric))

        self[key].update(*args, **kwargs)

    def compute(self) -> dict[str, Any]:
        divided_metrics = {k: v.compute() for k, v in self.items() if k != "metric"}

        out = defaultdict(list)
        for key, metrics in divided_metrics.items():
            out["key"].append(key)
            for subkey, value in metrics.items():
                out[subkey].append(value.item())

        return out

    def reset(self) -> None:
        for metric in self.values():
            metric.reset()


class DETMetaArch(LSPDetrModel, LightningModule):
    def __init__(
        self,
        warmup_epochs: int,
        num_classes: int,
        criterion: SetCriterion,
        **config,
    ) -> None:
        super().__init__(LSPDetrConfig(**config, num_classes=num_classes))
        self.criterion = criterion
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs

        self.backbone.train()  # to enable the drop path
        self.processor = LSPDetrImageProcessor()

        self.val_metrics = NestedMetricCollection(
            MetricCollection({"bPQ": BinaryPanopticQuality()})
        )

        self.binary_test_metrics = NestedMetricCollection(
            MetricCollection(
                {
                    "AJI": AggregatedJaccardIndex(),
                    "AP_0,5": AveragePrecision2018DSB(iou_thresholds=[0.5]),
                    "AP_0,7": AveragePrecision2018DSB(iou_thresholds=[0.7]),
                    "AP_0,9": AveragePrecision2018DSB(iou_thresholds=[0.9]),
                    "AP_0,5:0,05:0,9": AveragePrecision2018DSB(
                        iou_thresholds=torch.arange(0.5, 1, 0.05)
                    ),
                    "F1": F1Score(radius=12),
                    "bPQ": BinaryPanopticQuality(),
                    "bMPQ": BinaryPanopticQuality(masked=True),
                },
                compute_groups=False,
            )
        )
        self.multiclass_test_metric = NestedMetricCollection(
            MetricCollection(
                {
                    "mPQ": PanopticQuality(num_classes=num_classes),
                    "mMPQ": PanopticQuality(num_classes=num_classes, masked=True),
                },
                compute_groups=False,
            )
        )

        self.tissue_binary_test_metrics = NestedMetricCollection(
            MetricCollection(
                {
                    "AJI": AggregatedJaccardIndex(),
                    "AP_0,5": AveragePrecision2018DSB(iou_thresholds=[0.5]),
                    "AP_0,7": AveragePrecision2018DSB(iou_thresholds=[0.7]),
                    "AP_0,9": AveragePrecision2018DSB(iou_thresholds=[0.9]),
                    "AP_0,5:0,05:0,9": AveragePrecision2018DSB(
                        iou_thresholds=torch.arange(0.5, 1, 0.05)
                    ),
                    "F1": F1Score(radius=12),
                    "bPQ": BinaryPanopticQuality(),
                    "bMPQ": BinaryPanopticQuality(masked=True),
                },
                compute_groups=False,
            )
        )
        self.tissue_multiclass_test_metric = NestedMetricCollection(
            MetricCollection(
                {
                    "mPQ": PanopticQuality(num_classes=num_classes),
                    "mMPQ": PanopticQuality(num_classes=num_classes, masked=True),
                },
                compute_groups=False,
            )
        )
        self.category_test_metric = NestedMetricCollection(
            MetricCollection(
                {
                    "F1": F1Score(radius=12),
                    "PQ": BinaryPanopticQuality(),
                    "MPQ": BinaryPanopticQuality(masked=True),
                },
                compute_groups=False,
            )
        )
        self._aux_pattern = re.compile(r"_\d+$")

    def training_step(self, batch: tuple[Tensor, Any]) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_step=True,
            prog_bar=True,
        )

        losses = {
            key: loss * self.criterion.weight_dict[name]
            for key, loss in losses.items()
            if (name := self._aux_pattern.sub("", key)) in self.criterion.weight_dict
        }

        return sum(losses.values())  # type: ignore[return-value]

    def validation_step(self, batch: tuple[Tensor, Any]) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        losses = self.criterion(outputs, targets)
        self.log_dict(
            {f"validation/{k}": v for k, v in losses.items()},
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        results = self.processor.post_process(outputs)
        results = self.processor.post_process_instance(
            results, *inputs.shape[2:], allow_overlap=False
        )

        for b, result in enumerate(results):
            self.val_metrics.update(
                result["masks"], targets[b]["masks"], key=targets[b]["tissue"]
            )

    def on_validation_epoch_end(self) -> None:
        self.log("validation/bPQ", mean(self.val_metrics.compute()["bPQ"]))
        self.val_metrics.reset()

    def test_step(self, batch: tuple[Tensor, list[dict[str, Any]]]) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        results = self.processor.post_process(outputs)
        results = self.processor.post_process_instance(
            results, *inputs.shape[2:], allow_overlap=False
        )
        for b, result in enumerate(results):
            self.binary_test_metrics.update(
                result["masks"], targets[b]["masks"], key=self.trainer.datamodule.name
            )
            self.multiclass_test_metric.update(
                {"masks": result["masks"], "labels": result["labels"]},
                targets[b],
                key=self.trainer.datamodule.name,
            )

            self.tissue_binary_test_metrics.update(
                result["masks"], targets[b]["masks"], key=targets[b]["tissue"]
            )
            self.tissue_multiclass_test_metric.update(
                {"masks": result["masks"], "labels": result["labels"]},
                targets[b],
                key=targets[b]["tissue"],
            )
            for c, category in enumerate(
                self.trainer.test_dataloaders.dataset.categories
            ):
                self.category_test_metric.update(
                    result["masks"][result["labels"] == c],
                    targets[b]["masks"][targets[b]["labels"] == c],
                    key=category,
                )

    def on_test_epoch_end(self) -> None:
        self.logger.log_table(
            self.binary_test_metrics.compute() | self.multiclass_test_metric.compute(),
            "test_metrics.json",
        )
        self.logger.log_table(
            self.tissue_binary_test_metrics.compute()
            | self.tissue_multiclass_test_metric.compute(),
            f"{self.trainer.datamodule.name}_tissue_test_metrics.json",
        )
        self.logger.log_table(
            self.category_test_metric.compute(),
            f"{self.trainer.datamodule.name}_category_test_metrics.json",
        )
        self.binary_test_metrics.reset()
        self.multiclass_test_metric.reset()
        self.tissue_binary_test_metrics.reset()
        self.tissue_multiclass_test_metric.reset()
        self.category_test_metric.reset()

    def predict_step(
        self, batch: tuple[Tensor, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        inputs = batch[0]
        outputs = self(inputs)
        return self.processor.post_process(outputs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4 / self.trainer.accumulate_grad_batches,
            weight_decay=1e-4,
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.trainer.max_epochs,
            lr_min=1e-6 / self.trainer.accumulate_grad_batches,
            warmup_lr_init=1e-7 / self.trainer.accumulate_grad_batches,
            warmup_t=self.warmup_epochs,
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(
        self, scheduler: CosineLRScheduler, metric: Any | None
    ) -> None:
        scheduler.step(epoch=self.current_epoch)
