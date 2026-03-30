from typing import Any

import tensorflow as tf

from ..generators import TripletGenerator
from .base_metric_trainer import BaseMetricTrainer

TripletImages = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletEmbeddings = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletBatch = tuple[TripletImages, Any]


class TripletTrainer(BaseMetricTrainer):

    def __init__(
        self,
        train_generator: TripletGenerator,
        val_generator: TripletGenerator,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            train_generator=train_generator,
            val_generator=val_generator,
            **kwargs
        )

    def _compute_loss(self, triplet: TripletImages, training: bool):
        anchor, positive, negative = triplet

        emb_a, emb_p, emb_n = self.siamese_network(
            [anchor, positive, negative],
            training=training,
        )

        emb_a = tf.math.l2_normalize(emb_a, axis=-1)
        emb_p = tf.math.l2_normalize(emb_p, axis=-1)
        emb_n = tf.math.l2_normalize(emb_n, axis=-1)

        loss = self.loss_fn(emb_a, emb_p, emb_n)

        return loss, emb_a, emb_p, emb_n

    def _update_metrics(
        self,
        loss: tf.Tensor,
        embeddings: TripletEmbeddings,
    ) -> dict:
        emb_a, emb_p, emb_n = embeddings

        self.loss_tracker.update_state(loss)

        ap_sim = tf.reduce_sum(emb_a * emb_p, axis=-1)
        an_sim = tf.reduce_sum(emb_a * emb_n, axis=-1)
        margin = ap_sim - an_sim

        values = {
            "triplet_acc": tf.cast(margin > 0.0, tf.float32),
            "ap_sim": ap_sim,
            "an_sim": an_sim,
            "margin": margin,
        }

        for metric in self.user_metrics:
            if metric.name in values:
                metric.update_state(values[metric.name])

        return {
            "loss": self.loss_tracker.result(),
            **{m.name: m.result() for m in self.user_metrics},
        }

    def train_step(self, data: TripletBatch) -> dict[str, float]:
        (anchor, positive, negative), _ = data

        with tf.GradientTape() as tape:
            loss, emb_a, emb_p, emb_n = self._compute_loss(
                (anchor, positive, negative),
                training=True,
            )

        gradients = tape.gradient(
            loss,
            self.siamese_network.trainable_variables,
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_variables)
        )

        return self._update_metrics(loss, (emb_a, emb_p, emb_n))

    def test_step(self, data: TripletBatch) -> dict[str, float]:
        (anchor, positive, negative), _ = data

        loss, emb_a, emb_p, emb_n = self._compute_loss(
            (anchor, positive, negative),
            training=False,
        )
        return self._update_metrics(loss, (emb_a, emb_p, emb_n))
