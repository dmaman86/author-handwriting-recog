import numpy as np
import tensorflow as tf


class BatchResizer:

    def __init__(
        self,
        target_size: tuple[int, int],
        method: str = "bilinear",
    ) -> None:
        self.target_size = target_size
        self.method = method

    def resize(self, batch: np.ndarray) -> np.ndarray:
        """
        Resize entire batch at once.

        Args:
            batch: (batch_size, H, W, C)

        Returns:
            Resized batch (batch_size, target_H, target_W, C)
        """
        return tf.image.resize(batch, self.target_size, method=self.method).numpy()
