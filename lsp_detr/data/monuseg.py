from collections.abc import Iterable
from typing import Any

from albumentations.core.composition import TransformsSeqType
from datasets import DatasetDict, load_dataset
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from lsp_detr.data.datasets import PredictDataset, SPDataset, TestingDataset
from lsp_detr.data.utils import collate_fn, format_transform


class MoNuSeg(LightningDataModule):
    name = "monuseg"

    def __init__(
        self,
        batch_size: int,
        num_radial_distances: int,
        predict_split: str = "test",
        val_size: float = 0.2,
        num_workers: int = 0,
        train_transforms: TransformsSeqType | None = None,
        eval_transforms: TransformsSeqType | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_radial_distances = num_radial_distances
        self.predict_split = predict_split
        self.val_size = val_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

    def setup(self, stage: str) -> None:
        ds: DatasetDict = load_dataset("RationAI/MoNuSeg")
        ds.set_transform(
            format_transform, columns=["image", "instances"], output_all_columns=True
        )

        match stage:
            case "fit":
                data = ds["train"].train_test_split(
                    test_size=self.val_size, stratify_by_column="tissue"
                )
                self.train_dataset = SPDataset(
                    data["train"], self.train_transforms, self.num_radial_distances
                )
                self.val_dataset = SPDataset(
                    data["test"], self.eval_transforms, self.num_radial_distances
                )
            case "validate":
                data = ds["train"].train_test_split(
                    test_size=self.val_size, stratify_by_column="tissue"
                )
                self.val_dataset = SPDataset(
                    data["test"], self.eval_transforms, self.num_radial_distances
                )
            case "test":
                self.test_dataset = TestingDataset(ds["test"], self.eval_transforms)
            case "predict":
                self.predict_dataset = PredictDataset(
                    ds[self.predict_split], self.eval_transforms
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
