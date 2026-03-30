import tensorflow as tf

from ..generators import PairGenerator
from .base_metric_trainer import BaseMetricTrainer

PairIdxs = tuple[tf.Tensor, tf.Tensor]
PairLabels = tuple[tf.Tensor, tf.Tensor, tf.Tensor]  # label1, label2, target
PairImages = tuple[tf.Tensor, tf.Tensor]
PairBatch = tuple[PairImages, tuple[PairIdxs, PairLabels]]


class PairTrainer(BaseMetricTrainer):

    def __init__(
        self,
        train_generator: PairGenerator,
        val_generator: PairGenerator,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            train_generator=train_generator,
            val_generator=val_generator,
            **kwargs
        )

        # self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _prepare_labels(self, labels: tf.Tensor) -> tf.Tensor:
        labels = tf.cast(labels, tf.float32)
        labels = tf.expand_dims(labels, -1)

        # convert {0, 1} -> {-1, 1}
        labels = 2.0 * labels - 1.0
        return labels

    def _compute_loss(
        self,
        pair: PairImages,
        labels: tf.Tensor,
        training: bool,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img1, img2 = pair
        similarity = self.siamese_network(
            [img1, img2],
            training=training,
        )
        loss = self.loss_fn(labels, similarity)
        return loss, similarity

    def _update_metrics(
        self, loss: tf.Tensor, labels: tf.Tensor, similarity: tf.Tensor
    ) -> dict:
        self.loss_tracker.update_state(loss)

        similarity = tf.squeeze(similarity, axis=-1)

        # labels: [-1, 1] -> [0, 1]
        labels_binary = tf.cast(labels > 0.0, tf.float32)

        # convert similarity -> binary predictions
        preds = tf.cast(similarity > 0.0, tf.float32)

        # similarity -> probability
        probs = (similarity + 1.0) / 2.0

        for metric in self.user_metrics:
            if metric.name in ["auc", "pr_auc"]:
                metric.update_state(labels_binary, probs)
            else:
                metric.update_state(labels_binary, preds)

        return {
            "loss": self.loss_tracker.result(),
            **{m.name: m.result() for m in self.user_metrics},
        }

    def train_step(self, data: PairBatch) -> dict[str, tf.Tensor]:
        (img1, img2), (_, (_, _, target)) = data

        labels = self._update_labels(target)

        with tf.GradientTape() as tape:
            loss, similarity = self._compute_loss(
                (img1, img2),
                labels,
                training=True,
            )

        gradients = tape.gradient(
            loss,
            self.siamese_network.trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        return self._update_metrics(loss, labels, similarity)

    def test_step(self, data: PairBatch) -> dict[str, tf.Tensor]:
        (img1, img2), (_, (_, _, target)) = data

        labels = self._update_labels(target)

        loss, similarity = self._compute_loss(
            (img1, img2),
            labels,
            training=False,
        )

        return self._update_metrics(loss, labels, similarity)
