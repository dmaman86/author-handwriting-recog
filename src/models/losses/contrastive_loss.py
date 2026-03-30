from typing import Any, Literal

import tensorflow as tf


class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Contrastive loss for Siamese networks.
    L = y * d^2 + (1-y) * max(margin - d, 0)^2
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: Literal["euclidean", "cosine"] = "euclidean",
        name: str = "contrastive_loss",
    ) -> None:
        super().__init__(name=name)
        self.margin = margin
        self.distance = distance

    def _distance(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        if self.distance == "euclidean":
            return tf.linalg.norm(a - b, axis=1)
        elif self.distance == "cosine":
            similarity = tf.reduce_sum(a * b, axis=1)
            return 1.0 - similarity
        else:
            raise ValueError(f"Unsupported distance type: {self.distance}")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes contrastive loss for pair embeddings.

        Args:
            y_true: Stacked tensor of shape (batch_size, 3) from generator:
                   [:, 0] = pair_labels (1.0 = same class, 0.0 = different class)
                   [:, 1] = img1_labels (author_id as float32, unused by this loss)
                   [:, 2] = img2_labels (author_id as float32, unused by this loss)
            y_pred: Embeddings stacked as (batch_size, 2, embedding_dim)

        Returns:
            Contrastive loss per sample
        """
        # Extract only pair_labels from column 0
        # y_true shape: (batch_size, 3)
        # pair_labels shape: (batch_size,)
        pair_labels = y_true[:, 0]

        emb1: tf.Tensor = y_pred[:, 0, :]  # (batch_size, embedding_dim)
        emb2: tf.Tensor = y_pred[:, 1, :]  # (batch_size, embedding_dim)

        d = self._distance(emb1, emb2)
        positive_loss: tf.Tensor = pair_labels * tf.square(d)
        negative_loss: tf.Tensor = (1 - pair_labels) * tf.square(
            tf.maximum(self.margin - d, 0)
        )
        return positive_loss + negative_loss

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "margin": self.margin,
                "distance": self.distance,
            }
        )
        return config
