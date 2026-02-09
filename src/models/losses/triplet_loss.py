from __future__ import annotations

from typing import Any

import tensorflow as tf


class TripletLoss(tf.keras.losses.Loss):
    """
    Triplet loss for metric learning.
    Enforces:
        d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(
        self,
        margin: float = 1.0,
        name: str = "triplet_loss",
    ) -> None:
        super().__init__(name=name)
        self.margin = margin

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Args:
            y_true: Dummy tensor (not used).
            embeddings: Tensor of shape (batch_size, 3, embedding_dim)
            [anchor, positive, negative]
        Returns:
            Scalar tensor representing the triplet loss
        """
        anchor = y_pred[:, 0, :]  # (batch_size, embedding_dim)
        positive = y_pred[:, 1, :]  # (batch_size, embedding_dim)
        negative = y_pred[:, 2, :]  # (batch_size, embedding_dim)

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"margin": self.margin})
        return config
