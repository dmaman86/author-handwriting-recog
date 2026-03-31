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
