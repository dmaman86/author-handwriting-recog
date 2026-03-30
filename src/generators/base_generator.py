import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf

from ..datasets import DatasetSplit
from .augmentation import AugmentImage


class BaseGenerator(ABC):

    def __init__(
        self,
        split: DatasetSplit,
        batch_size: int = 32,
        seed: int | None = None,
        augment: AugmentImage | None = None,
        shuffle_fraction: float = 1.0,
    ) -> None:

        if len(split.indices) == 0:
            raise ValueError("Input data indices cannot be empty.")
        if np.max(split.indices) >= len(split.labels):
            raise ValueError("Indices out of range")

        self.split = split
        self.loader = split.loader
        self.labels = split.labels
        self.indices = split.indices

        self.batch_size = batch_size
        self.seed = seed
        self.augment = augment
        self.shuffle_fraction = shuffle_fraction

        self.rng = np.random.default_rng(seed)

        self.num_patches = len(self.indices)
        self.steps_per_epoch = math.ceil(self.num_patches / self.batch_size)

        subset_labels = self.labels[self.indices]
        self.authors = np.unique(subset_labels)

        self.subset_author_indices = {
            author: self.indices[subset_labels == author].astype(np.int32)
            for author in self.authors
        }

        self.indices_tensor = tf.convert_to_tensor(self.indices, dtype=tf.int32)
        self.labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.authors_tensor = tf.convert_to_tensor(self.authors, dtype=tf.int32)

        max_author = int(np.max(self.labels)) + 1
        rows = [
            self.subset_author_indices.get(a, np.array([], dtype=np.int32))
            for a in range(max_author)
        ]
        self._author_index_table = tf.ragged.constant(rows, dtype=tf.int32)

    def _get_author_indices(self, author: tf.Tensor) -> tf.Tensor:
        return self._author_index_table[author]

    def _base_dataset(self) -> tf.data.Dataset:

        ds = tf.data.Dataset.from_tensor_slices(self.indices_tensor)

        if self.shuffle_fraction > 0:
            shuffle_size = int(self.num_patches * self.shuffle_fraction)
            ds = ds.shuffle(shuffle_size, seed=self.seed, reshuffle_each_iteration=True)

        ds = ds.map(
            lambda idx: (
                idx,
                tf.gather(self.labels_tensor, idx),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        return ds

    @abstractmethod
    def build(self) -> tf.data.Dataset:
        pass

    def get_config(self) -> dict[str, Any]:
        return {
            "num_samples": len(self.indices),
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "num_authors": len(self.authors),
            "patch_shape": self.loader.patch_shape,
            "authors": self.authors.tolist(),
            "shuffle_fraction": self.shuffle_fraction,
            "seed": self.seed,
            "augment": self.augment is not None,
        }
