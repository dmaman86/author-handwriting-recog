import tensorflow as tf

from .base_generator import BaseGenerator

TripletIdxs = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletLabels = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletImages = tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class TripletGenerator(BaseGenerator):

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

    def _make_triplet(
        self, idx: tf.Tensor, label: tf.Tensor
    ) -> tuple[TripletIdxs, TripletLabels]:
        pos_indices = self._get_author_indices(label)
        pos_indices = tf.boolean_mask(pos_indices, pos_indices != idx)

        pos_idx = self._random_choice(pos_indices, idx)

        other_authors = tf.boolean_mask(
            self.authors_tensor, self.authors_tensor != label
        )

        neg_author = self._random_choice(other_authors, label)

        neg_indices = self._get_author_indices(neg_author)

        neg_idx = self._random_choice(neg_indices, idx)

        return (
            (idx, pos_idx, neg_idx),
            (label, label, neg_author),
        )

    def _load_triplet_images(
        self,
        triplet_idxs: TripletIdxs,
        triplet_labels: TripletLabels,
    ) -> tuple[TripletImages, tuple[TripletIdxs, TripletLabels]]:
        anchor_idx, pos_idx, neg_idx = triplet_idxs
        label, pos_label, neg_author = triplet_labels

        anchor = self.loader.get_images(anchor_idx)
        positive = self.loader.get_images(pos_idx)
        negative = self.loader.get_images(neg_idx)

        return (
            (anchor, positive, negative),
            (
                (anchor_idx, pos_idx, neg_idx),
                (label, pos_label, neg_author),
            ),
        )

    def build(self) -> tf.data.Dataset:
        ds = self._base_dataset()

        ds = ds.map(
            self._make_triplet,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        ds = ds.batch(self.batch_size)

        ds = ds.map(
            self._load_triplet_images,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.augment is not None:
            ds = ds.map(
                self.augment,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
