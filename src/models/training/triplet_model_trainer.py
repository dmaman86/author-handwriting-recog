import tensorflow as tf

from ..build_triplet_model import build_triplet_model
from ..losses.triplet_loss import TripletLoss
from .base_model_trainer import BaseModelTrainer


class TripletModelTrainer(BaseModelTrainer):

    def _build_siamese_model(self) -> tf.keras.Model:
        return build_triplet_model(
            embedding_model=self.embedding_model,
            input_shape=self.input_shape,
        )

    def _get_default_loss(self) -> tf.keras.losses.Loss:
        return TripletLoss(margin=0.5)
