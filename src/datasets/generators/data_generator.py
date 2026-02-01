import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .augmentation.augmentation_builder import AugmentationBuilder
from .augmentation.batch_augmenter import BatchAugmenter
from .transforms.batch_resizer import BatchResizer


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        data: dict[int, list[np.ndarray]],
        batch_size: int = 32,
        target_size: tuple[int, int] = (60, 53),
        shuffle: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.data = data
        self.batch_size: int = batch_size
        self.target_size = target_size
        self.shuffle: bool = shuffle
        self.seed: int | None = seed

        if seed is not None:
            np.random.seed(seed)

        self.resizer = BatchResizer(target_size)

        self.samples: list[tuple[np.ndarray, int]] = []
        for author_id, patches in data.items():
            for p in patches:
                self.samples.append((p, author_id))

        self.indices = np.arange(len(self.samples))

        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X, y = [], []
        for i in batch_idx:
            patch, author_id = self.samples[i]

            patch = self._normalize_patch(patch)

            X.append(patch)
            y.append(author_id)

        X = np.stack(X, axis=0)
        y = np.array(y, dtype=np.int32)

        X = self.resizer.resize(X)

        return X, y

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch)

        if patch.ndim == 2:
            patch = patch[..., np.newaxis]  # -> (H, W, 1)

        elif patch.ndim == 3:
            if patch.shape[0] == 1:
                patch = np.transpose(patch, (1, 2, 0))  # (1, H, W) -> (H, W, 1)

            if patch.shape[-1] != 1:
                patch = patch[..., :1]

        else:
            raise ValueError(f"Invalid patch shape: {patch.shape}")

        return patch.astype(np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            random.shuffle(self.samples)

