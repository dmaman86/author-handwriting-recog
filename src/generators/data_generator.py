import tensorflow as tf

from .base_generator import BaseGenerator


class DataGenerator(BaseGenerator):

    def _load_images(
        self, idxs: tf.Tensor, labels: tf.Tensor
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        images = self.loader.get_images(idxs)
        return images, (idxs, labels)

    def build(self) -> tf.data.Dataset:

        ds = self._base_dataset()

        ds = ds.batch(self.batch_size)

        ds = ds.map(
            self._load_images,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.augment is not None:
            ds = ds.map(
                self.augment,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
