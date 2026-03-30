from abc import ABC, abstractmethod

import tensorflow as tf

# from .backbones import BaseBackbone


class BaseMetricModel(ABC):

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        name: str = "metric_model",
    ) -> None:
        self.input_shape = input_shape
        self.name = name

    @abstractmethod
    def build(self, embedding_model: tf.keras.Model) -> tf.keras.Model:
        pass


'''
class L2Normalization(tf.keras.layers.Layer):
    """
    Layer that applies L2 normalization along the
    embedding dimension.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self) -> dict:
        return super().get_config()



class BaseMetricModel(ABC):

    def __init__(
        self,
        backbone: BaseBackbone,
        input_shape: tuple[int, int, int],
        embedding_dim: int = 256,
        seed: int | None = None,
        name: str = "metric_model",
    ) -> None:
        self.backbone = backbone
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.name = name

        self.embedding_model: tf.keras.Model | None = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        pass

    def _build_embedding_model(self) -> tf.keras.Model:
        if self.embedding_model is not None:
            return self.embedding_model
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs

        features = self.backbone.build(x)
        embeddings = self._embedding_head(features)
        return tf.keras.Model(
            inputs=inputs,
            outputs=embeddings,
            name="embedding_model",
        )

    def _embedding_head(self, features: tf.Tensor) -> tf.Tensor:
        if len(features.shape) == 4:
            x = tf.keras.layers.GlobalAveragePooling2D()(features)
        elif len(features.shape) == 3:
            x = tf.keras.layers.GlobalAveragePooling1D()(features)
        else:
            raise ValueError(f"Unsupported feature tensor rank: {len(features.shape)}")

        x = tf.keras.layers.Dense(self.embedding_dim, name="embedding_projection")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = L2Normalization(name="l2_normalize")(x)

        return x
'''
