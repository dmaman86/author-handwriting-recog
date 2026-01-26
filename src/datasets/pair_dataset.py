import math

import numpy as np
import tensorflow as tf

from .samplers.pair_index_sampler import PairIndexSampler


class PairDataset(tf.keras.utils.Sequence):

    def __init__(
        self,
        X: np.ndarray,
        sampler: PairIndexSampler,
        batch_size: int = 32,
    ) -> None:
        self.X = X
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        pairs, labels = self.sampler.sample(self.batch_size)

        X1: np.ndarray = np.stack([self.X[i] for i, _ in pairs])
        X2: np.ndarray = np.stack([self.X[j] for _, j in pairs])
        y: np.ndarray = np.array(labels, dtype=np.float32)

        return (X1, X2), y
