import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf

from .augmentation.augmentation_builder import AugmentationBuilder
from .augmentation.batch_augmenter import BatchAugmenter
from .data.split_data import SplitData
from .transforms.batch_resizer import BatchResizer


class BaseGenerator(ABC):

    def __init__(
        self,
        data: SplitData,
        batch_size: int = 32,
        target_size: tuple[int, int] = (60, 53),
        shuffle: bool = True,
        use_augmentation: bool = False,
        seed: int | None = None,
    ) -> None:

        self.X = data.X
        self.y = data.y
        self.author_index = data.author_index
        self.authors_ids = data.authors_ids
        self.valid_authors = data.valid_authors

        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.augmenter: BatchAugmenter | None = None
        if use_augmentation:
            pipeline = AugmentationBuilder.build_default(seed=seed)
            self.augmenter = BatchAugmenter(pipeline)

        self.resizer = BatchResizer(target_size=target_size)

        self.num_patches = data.num_patches
        self.steps_per_epoch = math.ceil(self.num_patches / self.batch_size)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            random.shuffle(self.authors_ids)

    def _normalize_batch(self, batch: np.ndarray) -> np.ndarray:
        batch = batch.astype(np.float32)
        if batch.max() > 1.0:
            batch = batch / 255.0
        if batch.ndim == 3:
            batch = batch[..., np.newaxis]
        return batch

    @abstractmethod
    def _generator(self):
        pass

    @abstractmethod
    def _output_signature(self) -> tuple:
        pass

    def to_dataset(self) -> tuple[tf.data.Dataset, int]:
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=self._output_signature(),
        )
        return dataset.prefetch(tf.data.AUTOTUNE), self.steps_per_epoch

    def get_config(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "target_size": self.target_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_authors": len(self.authors_ids),
            "num_valid_authors": len(self.valid_authors),
            "num_patches": self.num_patches,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def get_statistics(self) -> dict[str, Any]:
        patches_per_author = [e - s for s, e in self.author_index.values()]
        return {
            "total_authors": len(self.authors_ids),
            "valid_authors": len(self.valid_authors),
            "total_patches": self.num_patches,
            "avg_patches_per_author": np.mean(patches_per_author),
            "min_patches": np.min(patches_per_author),
            "max_patches": np.max(patches_per_author),
        }
