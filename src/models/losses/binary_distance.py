from typing import Any
import tensorflow as tf

class BinaryCrossEntropyDistance(tf.keras.losses.Loss):

    def __init__(
        self,
        scale: float = 10.0,
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
        Args:
            y_true: Tensor of shape (batch_size,)
                    1.0 for positive pairs, 0.0 for negative pairs
            embeddings: Tensor of shape (batch_size, 2, embedding_dim)

        Returns:
            Scalar loss value
        """
        emb1: tf.Tensor = embeddings[:, 0, :]
        emb2: tf.Tensor = embeddings[:, 1, :]

        distances: tf.Tensor = tf.norm(
            emb1 - emb2,
            axis=1,
        )

        # Convert distance to similarity logit
        logits: tf.Tensor = -self.scale * distances

        loss: tf.Tensor = self._bce(y_true, logits)
        return loss

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = super().get_config()
        config.update({"scale": self.scale})
        return config