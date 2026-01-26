from typing import Any
import tensorflow as tf

class ContrastiveLoss(tf.keras.losses.Loss):

    def __init__(self, margin: float = 1.0, name: str = "contrastive_loss") -> None:
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Args:
            y_true: Tensor of shape (batch_size,)
                    1.0 for positive pairs, 0.0 for negative pairs
            embeddings: Tensor of shape (batch_size, 2, embedding_dim)

        Returns:
            Scalar tensor representing the contrastive loss
        """

        emb1: tf.Tensor = embeddings[:, 0, :] # (batch_size, embedding_dim)
        emb2: tf.Tensor = embeddings[:, 1, :] # (batch_size, embedding_dim)

        distances: tf.Tensor = tf.norm(emb1 - emb2, axis=1)
        positive_loss: tf.Tensor = y_true * tf.square(distances)
        negative_loss: tf.Tensor = (1 - y_true) * tf.square(
            tf.maximum(self.margin - distances, 0)
        )
        loss: tf.Tensor = tf.reduce_mean(positive_loss + negative_loss)
        return loss
    
    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"margin": self.margin})
        return config