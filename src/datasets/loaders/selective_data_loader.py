from dataclasses import dataclass

import numpy as np
import zarr
from sklearn.model_selection import train_test_split

from ...io import LoggerFactory
from .zarr_loader import ZarrLoader


@dataclass
class DatasetSplit:
    labels: np.ndarray
    indices: np.ndarray
    loader: ZarrLoader
    author_ids: np.ndarray
    author_names: list[str]


class SelectiveDataLoader:

    def __init__(
        self,
        dataset_path: str,
        logger: LoggerFactory,
        seed: int | None = None,
    ) -> None:
        root = zarr.open(dataset_path, mode="r")

        self.images_non_test = root["non_test"]["images"]
        self.labels_non_test = root["non_test"]["labels"][:]

        self.images_test = root["test"]["images"]
        self.labels_test = root["test"]["labels"][:]

        self.metadata = dict(root.attrs)
        self.author_names = self.metadata["author_names"]

        self.seed = seed
        self.authors = np.unique(self.labels_non_test)

        self.logger = logger

    def _select_author_indices(
        self, authors: np.ndarray | list[int] | range, labels: np.ndarray
    ) -> np.ndarray:
        if isinstance(authors, range):
            authors = np.arange(authors.start, authors.stop)
        else:
            authors = np.asarray(authors)

        mask = np.isin(labels, authors)
        return np.where(mask)[0]

    def _train_validation_split(
        self,
        indices: np.ndarray,
        validation_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_subset = self.labels_non_test[indices]

        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_fraction,
            stratify=y_subset,
            random_state=self.seed,
        )
        return train_idx, val_idx

    def build(
        self,
        authors: np.ndarray | list[int] | range,
        validation_fraction: float = 0.25,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:

        indices = self._select_author_indices(authors, self.labels_non_test)
        test_indices = self._select_author_indices(authors, self.labels_test)

        train_idx, val_idx = self._train_validation_split(indices, validation_fraction)

        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_indices = np.sort(test_indices)

        selected_author_ids = np.unique(self.labels_non_test[indices])
        selected_author_names = [self.author_names[i] for i in selected_author_ids]

        train_loader = ZarrLoader(images=self.images_non_test, indices=train_idx)
        val_loader = ZarrLoader(images=self.images_non_test, indices=val_idx)
        test_loader = ZarrLoader(images=self.images_test, indices=test_indices)

        self._log_stats(indices, train_idx, val_idx, test_indices)

        train = DatasetSplit(
            labels=self.labels_non_test,
            indices=train_idx,
            loader=train_loader,
            author_ids=selected_author_ids,
            author_names=selected_author_names,
        )
        val = DatasetSplit(
            labels=self.labels_non_test,
            indices=val_idx,
            loader=val_loader,
            author_ids=selected_author_ids,
            author_names=selected_author_names,
        )

        test = DatasetSplit(
            labels=self.labels_test,
            indices=test_indices,
            loader=test_loader,
            author_ids=selected_author_ids,
            author_names=selected_author_names,
        )

        return train, val, test

    def _log_stats(
        self,
        indices: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_indices: np.ndarray,
    ) -> None:

        self.logger.info(f"Total patches selected: {len(indices)}")
        self.logger.info(
            f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_indices)}"
        )

        unique, counts = np.unique(self.labels_non_test[indices], return_counts=True)
        self.logger.info(
            f"Avg non_test patches per author: {np.mean(counts):.2f} | "
            f"Min: {np.min(counts)} | Max: {np.max(counts)}"
        )

        unique_test, counts_test = np.unique(
            self.labels_test[test_indices], return_counts=True
        )
        self.logger.info(
            f"Avg test patches per author: {np.mean(counts_test):.2f} | "
            f"Min: {np.min(counts_test)} | Max: {np.max(counts_test)}"
        )

        self.logger.info(f"Image shape: {self.images_non_test.shape[1:]}")
        self.logger.info(f"Zarr chunk shape: {self.images_non_test.chunks}")
