import math

import tensorflow as tf

from .backbones import BaseBackbone
from .layers import L2Normalization


class EmbeddingNetwork:

    def __init__(
        self,
        backbone: BaseBackbone,
        input_shape: tuple[int, int, int],
        num_authors: int,
    ):
        self.backbone = backbone
        self.input_shape = input_shape
        self.num_authors = num_authors

    def build_model(self) -> tf.keras.Model:
        embedding_dim, projection_dim = self._embedding_config()

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        features = self.backbone.build(inputs)

        if len(features.shape) == 4:
            x = tf.keras.layers.GlobalAveragePooling2D()(features)
        elif len(features.shape) == 3:
            x = tf.keras.layers.GlobalAveragePooling1D()(features)
        else:
            raise ValueError("Unsupported feature shape")

        x = tf.keras.layers.Dense(projection_dim)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(embedding_dim)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = L2Normalization()(x)

        return tf.keras.Model(inputs, outputs=x, name="embedding_model")

    def _embedding_config(self) -> tuple[int, int]:
        base = int(32 * math.log2(self.num_authors))

        embedding_dim = int(round(base / 32) * 32)
        embedding_dim = max(128, min(embedding_dim, 224))

        projection_dim = embedding_dim * 2
        return embedding_dim, projection_dim
