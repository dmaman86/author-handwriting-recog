import math
import random
from typing import Any

import numpy as np
import tensorflow as tf

from .augmentation.augmentation_builder import AugmentationBuilder
from .augmentation.batch_augmenter import BatchAugmenter
from .transforms.batch_resizer import BatchResizer


class PairGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        data: dict[int, list[np.ndarray]],
        authors_ids: list[int],
        valid_authors: list[int],
        batch_size: int = 32,
        target_size: tuple[int, int] = (60, 53),
        positive_ratio: float = 0.5,
        shuffle: bool = True,
        use_augmentation: bool = True,
        seed: int | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.data = data
        self.authors_ids = authors_ids
        self.valid_authors = valid_authors
        self.batch_size = batch_size
        self.target_size = target_size
        self.positive_ratio = positive_ratio
        self.shuffle = shuffle
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.augmenter = None
        if use_augmentation:
            pipeline = AugmentationBuilder.build_default(seed)
            self.augmenter = BatchAugmenter(pipeline)

        self.resizer = BatchResizer(target_size)

        self.num_patches = sum(len(patches) for patches in self.data.values())
        self.steps_per_epoch = math.ceil(self.num_patches / self.batch_size)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        X1, X2, y = self._generate_pairs()

        if self.augmenter is not None:
            X1, X2 = self.augmenter.augment_pairs(X1, X2, y)

        X1 = self.resizer.resize(X1)
        X2 = self.resizer.resize(X2)

        return (X1, X2), y

    def _generate_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        batch_x1: list[np.ndarray] = []
        batch_x2: list[np.ndarray] = []
        batch_y: list[int] = []

        num_positive = int(self.batch_size * self.positive_ratio)
        num_negative = self.batch_size - num_positive

        for _ in range(num_positive):
            author_id = random.choice(self.valid_authors)
            patches = self.data[author_id]
            p1, p2 = random.sample(patches, 2)

            batch_x1.append(p1)
            batch_x2.append(p2)
            batch_y.append(1)

        for _ in range(num_negative):
            a1, a2 = random.sample(self.authors_ids, 2)
            p1 = random.choice(self.data[a1])
            p2 = random.choice(self.data[a2])

            batch_x1.append(p1)
            batch_x2.append(p2)
            batch_y.append(0)

        X1 = np.array(batch_x1, dtype=np.float32)
        X2 = np.array(batch_x2, dtype=np.float32)
        y = np.array(batch_y, dtype=np.float32)

        return X1, X2, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            random.shuffle(self.authors_ids)

    def get_config(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "target_size": self.target_size,
            "positive_ratio": self.positive_ratio,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_authors": len(self.authors_ids),
            "num_valid_authors": len(self.valid_authors),
            "num_patches": self.num_patches,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def get_statistics(self) -> dict[str, Any]:
        patches_per_author = [len(patches) for patches in self.data.values()]

        return {
            "total_authors": len(self.authors_ids),
            "valid_authors": len(self.valid_authors),
            "total_patches": self.num_patches,
            "avg_patches_per_author": np.mean(patches_per_author),
            "min_patches": np.min(patches_per_author),
            "max_patches": np.max(patches_per_author),
            "positive_pairs_per_batch": int(self.batch_size * self.positive_ratio),
            "negative_pairs_per_batch": self.batch_size
            - int(self.batch_size * self.positive_ratio),
        }


# class PairGenerator(tf.keras.utils.Sequence):
#     """
#     Optimized PairGenerator with batch-wise operations.
#
#     Performance improvements:
#     - Batch-wise augmentation (no loops)
#     - Batch-wise resizing
#     - Pre-normalized data
#     """
#
#     def __init__(
#         self,
#         data_partition: Union[
#             dict[int, list[np.ndarray]], tuple[np.ndarray, np.ndarray]
#         ],
#         batch_size: int = 32,
#         target_size: tuple[int, int] = (60, 53),
#         positive_ratio: float = 0.5,
#         shuffle: bool = True,
#         use_augmentation: bool = True,
#         seed: int | None = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.batch_size: int = batch_size
#         self.target_size: tuple[int, int] = target_size
#         self.positive_ratio: float = positive_ratio
#         self.shuffle: bool = shuffle
#         self.use_augmentation: bool = use_augmentation
#         self.seed: int | None = seed
#
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#             tf.random.set_seed(seed)
#
#         # Handle both input formats
#         if isinstance(data_partition, dict):
#             self._init_from_dict(data_partition)
#         elif isinstance(data_partition, tuple) and len(data_partition) == 2:
#             self._init_from_arrays(data_partition)
#         else:
#             raise TypeError(
#                 f"train_partition must be dict or tuple got {type(data_partition)}"
#             )
#
#         self.num_patches = sum(len(patches) for patches in self.data_partition.values())
#
#         self.steps_per_epoch = math.ceil(self.num_patches / self.batch_size)
#
#         self._normalize_patches()
#
#         self._augmentation = self._build_augmentation() if use_augmentation else None
#
#     def _init_from_dict(self, data_partition: dict[int, list[np.ndarray]]) -> None:
#         """Initialize from legacy dict format."""
#         self.data_partition: dict[int, list[np.ndarray]] = data_partition
#         self.author_ids: list[int] = list(data_partition.keys())
#
#         self.valid_authors: list[int] = [
#             author_id
#             for author_id, patches in data_partition.items()
#             if len(patches) >= 2
#         ]
#
#         if len(self.valid_authors) < 2:
#             raise ValueError(
#                 "PairGenerator requires at least two authors "
#                 "with at least two patches each."
#             )
#
#     def _init_from_arrays(self, data_partition: tuple[np.ndarray, np.ndarray]) -> None:
#         """Initialize from (X, y) format from SelectiveDataLoader."""
#         X, y = data_partition
#
#         self.data_partition: dict[int, list[np.ndarray]] = {}
#
#         for patch, author_id in zip(X, y):
#             author_id = int(author_id)
#
#             if author_id not in self.data_partition:
#                 self.data_partition[author_id] = []
#
#             self.data_partition[author_id].append(patch)
#
#         self.author_ids: list[int] = list(self.data_partition.keys())
#
#         self.valid_authors: list[int] = [
#             author_id
#             for author_id, patches in self.data_partition.items()
#             if len(patches) >= 2
#         ]
#
#         if len(self.valid_authors) < 2:
#             raise ValueError(
#                 "PairGenerator requires at least two authors "
#                 "with at least two patches each."
#             )
#
#     def _normalize_patches(self) -> None:
#         """
#         Normalize all patches to [0, 1] once.
#         Avoids repeated normalization in __getitem__.
#         """
#         for author_id in self.data_partition:
#             normalized_patches = []
#             for patch in self.data_partition[author_id]:
#                 patch = patch.astype(np.float32)
#
#                 # Normalize if needed
#                 if patch.max() > 1.0:
#                     patch = patch / 255.0
#
#                 # Ensure channel dimension
#                 if len(patch.shape) == 2:
#                     patch = np.expand_dims(patch, axis=-1)
#
#                 normalized_patches.append(patch)
#
#             self.data_partition[author_id] = normalized_patches
#
#     def _build_augmentation(self) -> tf.keras.Sequential:
#         """Build augmentation pipeline."""
#         return tf.keras.Sequential(
#             [
#                 layers.RandomRotation(
#                     0.028, fill_mode="nearest", seed=self.seed  # ±10 degrees
#                 ),
#                 layers.RandomTranslation(
#                     0.05, 0.05, fill_mode="nearest", seed=self.seed
#                 ),
#                 layers.RandomZoom(0.1, fill_mode="nearest", seed=self.seed),
#                 layers.RandomContrast(0.1, seed=self.seed),
#             ],
#             name="pair_augmentation",
#         )
#
#     def __len__(self) -> int:
#         """Number of batches per epoch."""
#         return self.steps_per_epoch
#
#     def __getitem__(
#         self, index: int
#     ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
#         """
#         Generate one batch of pairs.
#
#         Optimized version:
#         - Collects all patches first
#         - Applies batch-wise resize
#         - Applies batch-wise augmentation
#         """
#         batch_x1: list[np.ndarray] = []
#         batch_x2: list[np.ndarray] = []
#         batch_y: list[int] = []
#
#         num_positive = int(self.batch_size * self.positive_ratio)
#         num_negative = self.batch_size - num_positive
#
#         # ---- Positive pairs (same author) ----
#         for _ in range(num_positive):
#             author_id = random.choice(self.valid_authors)
#             patches = self.data_partition[author_id]
#             p1, p2 = random.sample(patches, 2)
#
#             batch_x1.append(p1)
#             batch_x2.append(p2)
#             batch_y.append(1)
#
#         # ---- Negative pairs (different authors) ----
#         for _ in range(num_negative):
#             a1, a2 = random.sample(self.author_ids, 2)
#             p1 = random.choice(self.data_partition[a1])
#             p2 = random.choice(self.data_partition[a2])
#
#             batch_x1.append(p1)
#             batch_x2.append(p2)
#             batch_y.append(0)
#
#         X1 = np.array(batch_x1, dtype=np.float32)
#         X2 = np.array(batch_x2, dtype=np.float32)
#         y = np.array(batch_y, dtype=np.float32)
#
#         if self.use_augmentation and self._augmentation is not None:
#             X1, X2 = self._apply_augmentation_batch(X1, X2, y)
#
#         X1 = self._resize_batch(X1)
#         X2 = self._resize_batch(X2)
#
#         return (X1, X2), y
#
#     def _resize_batch(self, batch: np.ndarray) -> np.ndarray:
#         """
#         Resize entire batch at once.
#
#         Args:
#             batch: (batch_size, H, W, C)
#
#         Returns:
#             Resized batch (batch_size, target_H, target_W, C)
#         """
#         return tf.image.resize(batch, self.target_size, method="bilinear").numpy()
#
#     def _apply_augmentation_batch(
#         self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Apply augmentation to entire batch.
#
#         Strategy:
#         - Positive pairs: Apply SAME transformation to both (preserve similarity)
#         - Negative pairs: Apply DIFFERENT transformations
#
#         Args:
#             X1: First images (batch_size, H, W, C)
#             X2: Second images (batch_size, H, W, C)
#             y: Labels (batch_size,) - 1=positive, 0=negative
#
#         Returns:
#             Augmented (X1, X2)
#         """
#         X1_aug = []
#         X2_aug = []
#
#         positive_mask = y == 1
#
#         if np.any(positive_mask):
#             X1_pos = X1[positive_mask]
#             X2_pos = X2[positive_mask]
#
#             X_pos_concat = np.concatenate([X1_pos, X2_pos], axis=0)
#             X_pos_aug = self._augmentation(X_pos_concat, training=True).numpy()
#
#             n_pos = len(X1_pos)
#             X1_pos_aug = X_pos_aug[:n_pos]
#             X2_pos_aug = X_pos_aug[n_pos:]
#
#             X1_aug.append(X1_pos_aug)
#             X2_aug.append(X2_pos_aug)
#
#         negative_mask = y == 0
#         if np.any(negative_mask):
#             X1_neg = X1[negative_mask]
#             X2_neg = X2[negative_mask]
#
#             X1_neg_aug = self._augmentation(X1_neg, training=True).numpy()
#             X2_neg_aug = self._augmentation(X2_neg, training=True).numpy()
#
#             X1_aug.append(X1_neg_aug)
#             X2_aug.append(X2_neg_aug)
#
#         X1_final = np.zeros_like(X1)
#         X2_final = np.zeros_like(X2)
#
#         if len(X1_aug) > 0:
#             pos_idx = 0
#             neg_idx = 0
#             for i, is_positive in enumerate(positive_mask):
#                 if is_positive:
#                     X1_final[i] = (
#                         X1_aug[0][pos_idx]
#                         if len(X1_aug) > 0 and pos_idx < len(X1_aug[0])
#                         else X1[i]
#                     )
#                     X2_final[i] = (
#                         X2_aug[0][pos_idx]
#                         if len(X2_aug) > 0 and pos_idx < len(X2_aug[0])
#                         else X2[i]
#                     )
#                     pos_idx += 1
#                 else:
#                     aug_idx = 1 if len(X1_aug) > 1 else 0
#                     X1_final[i] = (
#                         X1_aug[aug_idx][neg_idx]
#                         if aug_idx < len(X1_aug) and neg_idx < len(X1_aug[aug_idx])
#                         else X1[i]
#                     )
#                     X2_final[i] = (
#                         X2_aug[aug_idx][neg_idx]
#                         if aug_idx < len(X2_aug) and neg_idx < len(X2_aug[aug_idx])
#                         else X2[i]
#                     )
#                     neg_idx += 1
#
#         return X1_final, X2_final
#
#     def on_epoch_end(self) -> None:
#         """Shuffle authors if enabled."""
#         if self.shuffle:
#             random.shuffle(self.author_ids)
