from typing import Any

import tensorflow as tf


class BinaryCrossEntropyDistance(tf.keras.losses.Loss):

    def __init__(
        self,
        scale: float = 2.0,
        name: str = "binary_cross_entropy_distance",
    ) -> None:
        """
        Args:
            scale: Controls how strongly distances are separated.
                   Higher values make the sigmoid sharper.
        """
        super().__init__(name=name)
        self.scale: float = scale
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(
        self,
        y_true: tf.Tensor,
        embeddings: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute binary cross-entropy loss from embedding distances.

        Args:
            y_true: Stacked tensor of shape (batch_size, 3) from generator:
                   [:, 0] = pair_labels (1.0 = same class, 0.0 = different class)
                   [:, 1] = img1_labels (author_id as float32, unused by this loss)
                   [:, 2] = img2_labels (author_id as float32, unused by this loss)
            embeddings: Tensor of shape (batch_size, 2, embedding_dim)

        Returns:
            Scalar loss value

        Note:
            The loss only uses pair_labels (column 0). Author ID labels (columns 1-2)
            are ignored - they're only used by evaluators for per-author metrics.
        """
        # Extract only pair_labels from column 0
        pair_labels = y_true[:, 0]

        emb1: tf.Tensor = embeddings[:, 0, :]
        emb2: tf.Tensor = embeddings[:, 1, :]

        distances: tf.Tensor = tf.norm(
            emb1 - emb2,
            axis=1,
        )

        # Convert distance to similarity logit
        logits: tf.Tensor = -self.scale * distances

        loss: tf.Tensor = self._bce(pair_labels, logits)
        return loss

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"scale": self.scale})
        return config
