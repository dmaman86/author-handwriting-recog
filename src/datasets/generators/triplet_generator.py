import random

import numpy as np
import tensorflow as tf

from .base_generator import BaseGenerator
from .data.split_data import SplitData


class TripletGenerator(BaseGenerator):

    def __init__(
        self,
        data: SplitData,
        batch_size: int = 32,
        target_size: tuple[int, int] = (60, 53),
        shuffle: bool = True,
        use_augmentation: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            data=data,
            batch_size=batch_size,
            target_size=target_size,
            shuffle=shuffle,
            use_augmentation=use_augmentation,
            seed=seed,
        )

    def _generate_triplets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_anchors: list[np.ndarray] = []
        batch_positives: list[np.ndarray] = []
        batch_negatives: list[np.ndarray] = []

        for _ in range(self.batch_size):
            author_id = random.choice(self.valid_authors)
            start, end = self.author_index[author_id]
            idx_a, idx_p = random.sample(range(start, end), 2)

            neg_author = random.choice(self.authors_ids)
            while neg_author == author_id:
                neg_author = random.choice(self.authors_ids)
            s, e = self.author_index[neg_author]
            idx_n = random.randint(s, e - 1)

            batch_anchors.append(self.X[idx_a])
            batch_positives.append(self.X[idx_p])
            batch_negatives.append(self.X[idx_n])

        return (
            np.stack(batch_anchors),
            np.stack(batch_positives),
            np.stack(batch_negatives),
        )

    def _generator(self):
        while True:
            for _ in range(self.steps_per_epoch):
                anchors, positives, negatives = self._generate_triplets()

                anchors = self._normalize_batch(anchors)
                positives = self._normalize_batch(positives)
                negatives = self._normalize_batch(negatives)

                if self.augmenter is not None:
                    anchors = self.augmenter.augment_batch(anchors)
                    positives = self.augmenter.augment_batch(positives)
                    negatives = self.augmenter.augment_batch(negatives)

                anchors = self.resizer.resize(anchors)
                positives = self.resizer.resize(positives)
                negatives = self.resizer.resize(negatives)

                yield (anchors, positives, negatives), np.zeros(
                    self.batch_size, dtype=np.float32
                )

            self.on_epoch_end()

    def _output_signature(self) -> tuple:
        h, w = self.target_size
        img_spec = tf.TensorSpec(shape=(None, h, w, 1), dtype=tf.float32)
        return (
            (img_spec, img_spec, img_spec),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
