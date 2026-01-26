import tensorflow as tf
from tensorflow.keras import layers

from .base_backbone import BaseBackbone
from .cnn_backbone import CNNBackbone
from .transformer_encoder import TransformerEncoder


class CNNTransformerBackbone(BaseBackbone):
    def __init__(
        self, num_layers: int = 2, num_heads: int = 4, ff_dim: int = 512
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.cnn_backbone = CNNBackbone()

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        x = self.cnn_backbone.build(inputs)

        _, h, w, c = x.shape
        if not all([h, w, c]):
            raise ValueError(
                f"Expected static spatial dims, got shape: {x.shape}. "
                f"Use Input(shape=...) with concrete values."
            )

        x = layers.Reshape((h * w, c))(x)

        positions: tf.Tensor = tf.range(start=0, limit=h * w, delta=1)
        pos_embed: tf.Tensor = layers.Embedding(input_dim=h * w, output_dim=c)(
            positions
        )
        x = x + pos_embed

        for i in range(self.num_layers):
            x = TransformerEncoder(
                embed_dim=c,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                name=f"transformer_encoder_{i}",
            )(x)

        return x
