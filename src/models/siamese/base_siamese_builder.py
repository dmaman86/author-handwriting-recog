from abc import ABC, abstractmethod

import tensorflow as tf

from ..embedding_network import EmbeddingNetwork


class BaseSiameseBuilder(ABC):

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        embedding_network: EmbeddingNetwork,
        name: str = "metric_model",
    ) -> None:
        self.input_shape = input_shape
        self.embedding_network = embedding_network
        self.name = name

        self._embedding_model: tf.keras.Model | None = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        pass

    @property
    def embedding_model(self) -> tf.keras.Model:
        if self._embedding_model is None:
            raise ValueError("Call build() before accessing embedding_model")
        return self._embedding_model
