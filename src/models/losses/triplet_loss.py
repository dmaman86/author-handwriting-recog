import tensorflow as tf


class TripletLoss:

    def __init__(
        self,
        margin: float = 0.2,
    ) -> None:
        self.margin = margin

    def call(
        self,
        emb_a: tf.Tensor,
        emb_p: tf.Tensor,
        emb_n: tf.Tensor,
    ) -> tf.Tensor:
        d_ap = 1.0 - tf.reduce_sum(emb_a * emb_p, axis=-1)
        d_an = 1.0 - tf.reduce_sum(emb_a * emb_n, axis=-1)
        return tf.reduce_mean(tf.maximum(d_ap - d_an + self.margin, 0.0))

    def __call__(
        self,
        emb_a: tf.Tensor,
        emb_p: tf.Tensor,
        emb_n: tf.Tensor,
    ) -> tf.Tensor:
        return self.call(emb_a, emb_p, emb_n)


"""
class TripletLoss(tf.keras.layers.Layer):
    def __init__(self, margin: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.cosine_distance = CosineDistanceLayer()

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        emb_a, emb_p, emb_n = inputs

        d_ap = self.cosine_distance((emb_a, emb_p))
        d_an = self.cosine_distance((emb_a, emb_n))

        loss = tf.maximum(d_ap - d_an + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def compute_triplet_loss(
        self, emb_a: tf.Tensor, emb_p: tf.Tensor, emb_n: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        emb_a = tf.math.l2_normalize(emb_a, axis=-1)
        emb_p = tf.math.l2_normalize(emb_p, axis=-1)
        emb_n = tf.math.l2_normalize(emb_n, axis=-1)

        loss = self.call((emb_a, emb_p, emb_n))
        return loss, emb_a, emb_p, emb_n

    def get_config(self):
        return {**super().get_config(), "margin": self.margin}
"""
