import tensorflow as tf

from .base_generator import BaseGenerator

PairIdxs = tuple[tf.Tensor, tf.Tensor]
PairLabels = tuple[tf.Tensor, tf.Tensor, tf.Tensor]  # label1, label2, target
PairImages = tuple[tf.Tensor, tf.Tensor]


class PairGenerator(BaseGenerator):

    def _random_choice(
        self,
        values: tf.Tensor,
        fallback: tf.Tensor,
    ) -> tf.Tensor:
        values = tf.reshape(values, [-1])  # ensure values is 1D
        num = tf.shape(values)[0]
        return tf.cond(
            num > 0,
            lambda: tf.gather(
                values,
                tf.random.uniform([], 0, num, dtype=tf.int32),
            ),
            lambda: fallback,
        )

    def _make_pair(
        self,
        idx: tf.Tensor,
        label: tf.Tensor,
    ) -> tuple[PairIdxs, PairLabels]:
        rand = tf.random.uniform(())

        def same_author():
            indices = self._get_author_indices(label)
            indices = tf.boolean_mask(indices, indices != idx)

            idx2 = self._random_choice(indices, idx)
            return (
                (idx, idx2),
                (label, label, tf.constant(1, dtype=tf.int32)),
            )

        def different_author():
            other_authors = tf.boolean_mask(
                self.authors_tensor,
                self.authors_tensor != label,
            )
            author2 = self._random_choice(other_authors, label)
            indices = self._get_author_indices(author2)
            idx2 = self._random_choice(indices, idx)

            return (
                (idx, idx2),
                (label, author2, tf.constant(0, dtype=tf.int32)),
            )

        return tf.cond(
            rand < 0.5,
            same_author,
            different_author,
        )

    def _load_pair_images(
        self,
        pair_idxs: PairIdxs,
        pair_labels: PairLabels,
    ) -> tuple[PairImages, tuple[PairIdxs, PairLabels]]:
        idx1, idx2 = pair_idxs
        label1, label2, target = pair_labels

        img1 = self.loader.get_images(idx1)
        img2 = self.loader.get_images(idx2)

        return (
            (img1, img2),
            (
                (idx1, idx2),
                (label1, label2, target),
            ),
        )

    def build(self) -> tf.data.Dataset:
        ds = self._base_dataset()

        ds = ds.map(
            self._make_pair,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds = ds.batch(self.batch_size)

        ds = ds.map(
            self._load_pair_images,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.augment is not None:
            ds = ds.map(
                self.augment,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
