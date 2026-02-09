import numpy as np
import tensorflow as tf

from .base_generator import BaseGenerator
from .data.split_data import SplitData


class DataGenerator(BaseGenerator):

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

        self.indices = np.arange(self.num_patches)

    def _generator(self):
        while True:
            for i in range(self.steps_per_epoch):
                batch_idx = self.indices[
                    i * self.batch_size : (i + 1) * self.batch_size
                ]
                X = self._normalize_batch(self.X[batch_idx])
                y = self.y[batch_idx]

                if self.augmenter is not None:
                    X = self.augmenter.augment_batch(X)

                X = self.resizer.resize(X)

                yield X, y

            self.on_epoch_end()
            if self.shuffle:
                np.random.shuffle(self.indices)

    def _output_signature(self) -> tuple:
        h, w = self.target_size
        return (
            tf.TensorSpec(shape=(None, h, w, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
