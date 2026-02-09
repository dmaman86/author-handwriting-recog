import tensorflow as tf

from ..losses.binary_distance import BinaryCrossEntropyDistance
from ..siamese_model import build_siamese_model
from .base_model_trainer import BaseModelTrainer


class PairModelTrainer(BaseModelTrainer):

    def _build_siamese_model(self) -> tf.keras.Model:
        return build_siamese_model(
            embedding_model=self.embedding_model,
            input_shape=self.input_shape,
        )

    def _get_default_loss(self) -> tf.keras.losses.Loss:
        return BinaryCrossEntropyDistance(scale=2.0)
