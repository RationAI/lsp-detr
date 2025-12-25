from collections.abc import Iterable
from typing import Any

import numpy as np
from albumentations.core.composition import TransformsSeqType
from datasets import DatasetDict, concatenate_datasets, load_dataset
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from lsp_detr.data.datasets import PredictDataset, SPDataset, TestingDataset
from lsp_detr.data.samplers import WeightedClassAndTissueSampler
from lsp_detr.data.utils import collate_fn, format_transform


class PanNuke(LightningDataModule):
    name = "pannuke"

    def __init__(
        self,
        batch_size: int,
        num_radial_distances: int,
        train_fold: list[int] | int | None = None,
        val_fold: int | None = None,
        test_fold: int | None = None,
        predict_fold: int | None = None,
        num_workers: int = 0,
        train_transforms: TransformsSeqType | None = None,
        eval_transforms: TransformsSeqType | None = None,
        allow_overlaps: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_radial_distances = num_radial_distances
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.predict_fold = predict_fold
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.allow_overlaps = allow_overlaps

    def setup(self, stage: str) -> None:
        ds: DatasetDict = load_dataset("RationAI/PanNuke")
        ds.set_transform(
            format_transform,
            columns=["image", "instances", "categories"],
            output_all_columns=True,
        )

        match stage:
            case "fit":
                data = (
                    concatenate_datasets([ds[f"fold{f}"] for f in self.train_fold])
                    if isinstance(self.train_fold, Iterable)
                    else ds[f"fold{self.train_fold}"]
                )
                self.train_dataset = SPDataset(
                    data,
                    self.train_transforms,
                    self.num_radial_distances,
                    allow_overlaps=self.allow_overlaps,
                )
                self.val_dataset = SPDataset(
                    ds[f"fold{self.val_fold}"],
                    self.eval_transforms,
                    self.num_radial_distances,
                    allow_overlaps=self.allow_overlaps,
                )
            case "validate":
                self.val_dataset = SPDataset(
                    ds[f"fold{self.val_fold}"],
                    self.eval_transforms,
                    self.num_radial_distances,
                    allow_overlaps=self.allow_overlaps,
                )
            case "test":
                self.test_dataset = TestingDataset(
                    ds[f"fold{self.test_fold}"], self.eval_transforms
                )
            case "predict":
                self.predict_dataset = PredictDataset(
                    ds[f"fold{self.predict_fold}"], self.eval_transforms
                )

    def train_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            pin_memory=True,
            sampler=WeightedClassAndTissueSampler(
                tissues=np.array(self.train_dataset.data["tissue"]),
                classes=self.train_dataset.data["categories"],
                num_classes=len(
                    self.train_dataset.data.features["categories"].feature.names
                ),
                num_samples=len(self.train_dataset),
            ),
        )

    def val_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self) -> Iterable[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
