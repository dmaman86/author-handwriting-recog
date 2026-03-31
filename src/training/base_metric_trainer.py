import tensorflow as tf

from ..generators import BaseGenerator
from ..models import BaseSiameseBuilder


class BaseMetricTrainer(tf.keras.Model):

    def __init__(
        self,
        siamese_builder: BaseSiameseBuilder,
        train_generator: BaseGenerator,
        val_generator: BaseGenerator,
        name: str = "base_metric_trainer",
    ) -> None:
        super().__init__(name=name)

        self.siamese_builder = siamese_builder
        self.train_generator = train_generator
        self.val_generator = val_generator

        self.siamese_network: tf.keras.Model | None = None

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.user_metrics: list[tf.keras.metrics.Metric] = []

    def initialize_model(self) -> None:

        self.siamese_network = self.siamese_builder.build()

        super().build(input_shape=self.siamese_network.input_shape)

    @property
    def embedding_model(self) -> tf.keras.Model:
        if self.siamese_network is None:
            raise ValueError(
                "The model must be built before accessing the embedding model."
            )
        return self.siamese_builder.embedding_model

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss=None,
        metrics=None,
        **kwargs,
    ):
        if self.siamese_network is None:
            raise ValueError("The model must be built before compiling.")
        super().compile(optimizer=optimizer, **kwargs)
        self.loss_fn = loss
        if metrics is not None:
            self.user_metrics = metrics

    def call(self, inputs):
        if self.siamese_network is None:
            raise ValueError("The model must be built before calling.")
        return self.siamese_network(inputs)

    def train_model(
        self,
        epochs: int,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        if self.siamese_network is None:
            raise ValueError("The model must be built before training.")

        train_data = self.train_generator.build()
        val_data = self.val_generator.build()

        steps_per_epoch = self.train_generator.steps_per_epoch
        val_steps = self.val_generator.steps_per_epoch

        return self.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=verbose,
        )

    @property
    def metrics(self):
        return [self.loss_tracker] + self.user_metrics

    def reset_metrics(self):
        self.loss_tracker.reset_state()
        for metric in self.user_metrics:
            metric.reset_state()

    def evaluate_dict(self, generator: BaseGenerator):
        dataset = generator.build()
        results = self.evaluate(dataset, return_dict=True)
        return {k: float(v) for k, v in results.items()}
