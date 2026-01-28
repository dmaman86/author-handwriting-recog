from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models


class TransformerEncoder(layers.Layer):
    """
    Standard Transformer Encoder block adapted for vision tokens.

    Input shape:
        (batch_size, num_tokens, embed_dim)

    Output shape:
        (batch_size, num_tokens, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        self.ffn = models.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ],
            name="ffn",
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")

        self.dropout1 = layers.Dropout(dropout_rate, name="dropout_attn")
        self.dropout2 = layers.Dropout(dropout_rate, name="dropout_ffn")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch, tokens, embed_dim)
            training: Training mode flag

        Returns:
            Tensor of shape (batch, tokens, embed_dim)
        """
        # Self-attention
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.norm2(out1 + ffn_output)

