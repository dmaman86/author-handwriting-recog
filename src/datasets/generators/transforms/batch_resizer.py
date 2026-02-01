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

    def resize(self, batch: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """
        Resize entire batch at once.

        Args:
            batch: (batch_size, H, W, C) or list of arrays with different shapes

        Returns:
            Resized batch (batch_size, target_H, target_W, C)
        """
        if isinstance(batch, list):
            resized = []
            for img in batch:
                # Add batch dimension for tf.image.resize
                img_resized = tf.image.resize(
                    np.expand_dims(img, axis=0), self.target_size, method=self.method
                )
                resized.append(img_resized[0].numpy())
            return np.array(resized, dtype=np.float32)

        return tf.image.resize(batch, self.target_size, method=self.method).numpy()
