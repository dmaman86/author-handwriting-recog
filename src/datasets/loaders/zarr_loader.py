import numpy as np
import tensorflow as tf
import zarr


class ZarrLoader:

    def __init__(
        self,
        images: zarr.Array,
        indices: np.ndarray,
    ) -> None:
        if len(indices) == 0:
            raise ValueError("Input data indices cannot be empty.")
        if np.max(indices) >= images.shape[0]:
            raise ValueError("Indices out of range")

        self.images = images
        self.indices = indices
        self.patch_shape = images.shape[1:]

    def get_images(self, idxs: tf.Tensor) -> tf.Tensor:
        def load_batch(i):
            imgs = self.images[i]
            return imgs.astype(np.float32)

        images = tf.numpy_function(
            load_batch,
            [idxs],
            tf.float32,
        )
        images.set_shape((None, *self.patch_shape))
        images = tf.expand_dims(images, axis=-1)
        return images

    def info(self) -> dict:
        return {
            "num_indices": len(self.indices),
            "patch_shape": self.patch_shape,
        }
