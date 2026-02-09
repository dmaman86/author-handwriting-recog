from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from ...io.logging import LoggerFactory
from ..build_model import build_embedding_model

GeneratorType = tf.data.Dataset | tf.keras.utils.Sequence


class BaseModelTrainer(ABC):

    def __init__(
        self,
        backbone,
        input_shape: tuple[int, int, int] = (60, 53, 1),
        embedding_dim: int = 256,
        model_name: str = "siamese",
        log_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize trainer.
        Args:
            backbone: Instance of BaseBackbone (CNNBackbone or CNNTransformerBackbone)
            input_shape: Input image shape (H, W, C)
            embedding_dim: Embedding space dimension
            model_name: Model name (for logging and saving)
            log_dir: Directory for logs (None = console only)
        """
        self.backbone = backbone
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.log_dir = Path(log_dir) if log_dir else None

        self.logger = LoggerFactory.get_logger(
            name=f"{self.model_name}_trainer",
            log_dir=self.log_dir,
            file_prefix=self.model_name,
        )

        self.embedding_model: tf.keras.Model | None = None
        self.siamese_model: tf.keras.Model | None = None
        self.history: tf.keras.callbacks.History | None = None

        self.logger.info(f"Model trainer initialized: {self.model_name}")
        self.logger.info(f"Input shape: {self.input_shape}")
        self.logger.info(f"Embedding dim: {self.embedding_dim}")

    def build(self) -> None:
        self.logger.info("Building models...")

        self.embedding_model = build_embedding_model(
            backbone=self.backbone,
            input_shape=self.input_shape,
            embedding_dim=self.embedding_dim,
            name=f"{self.model_name}_embedding_model",
        )

        self.siamese_model = self._build_siamese_model()

        total_params = self.siamese_model.count_params()
        trainable_params = sum(
            [tf.size(w).numpy() for w in self.siamese_model.trainable_weights]
        )

        self.logger.info(f"Model built: {self.siamese_model.name}")
        self.logger.info(f"     - Total parameters: {total_params}")
        self.logger.info(f"     - Trainable parameters: {trainable_params}")

    @abstractmethod
    def _build_siamese_model(self) -> tf.keras.Model:
        pass

    @abstractmethod
    def _get_default_loss(self) -> tf.keras.losses.Loss:
        pass

    def compile(
        self,
        loss: tf.keras.losses.Loss | None = None,
        learning_rate: float = 0.0001,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
    ) -> None:
        if self.siamese_model is None:
            self.build()

        self.logger.info("Compiling model...")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn = loss if loss is not None else self._get_default_loss()

        self.siamese_model.compile(optimizer=optimizer, loss=loss_fn)

        self.logger.info("Model compiled")
        self.logger.info(f"     - Optimizer: {optimizer}")
        self.logger.info(f"     - Learning rate: {learning_rate}")
        self.logger.info(f"     - Loss: {loss_fn.__class__.__name__}")

        self._log_loss_params(loss_fn)

    def _log_loss_params(self, loss_fn: tf.keras.losses.Loss) -> None:
        if hasattr(loss_fn, "margin"):
            self.logger.info(f"     - Loss margin: {loss_fn.margin}")
        if hasattr(loss_fn, "scale"):
            self.logger.info(f"     - Loss scale: {loss_fn.scale}")

    def train(
        self,
        train_generator: GeneratorType,
        val_generator: GeneratorType,
        epochs: int = 50,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
        callbacks: list | None = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        Args:
            train_generator: Training data generator (PairGenerator)
            val_generator: Validation data generator (PairGenerator)
            epochs: Number of epochs
            callbacks: Custom callbacks (None = defaults)
            verbose: Verbosity level (0, 1, 2)
        Returns:
            Training history
        """
        if self.siamese_model is None or not hasattr(self.siamese_model, "optimizer"):
            raise RuntimeError("Model must be compiled before training")

        train_steps = steps_per_epoch or len(train_generator)
        val_steps = validation_steps or len(val_generator)

        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"  - Train batches: {train_steps}")
        self.logger.info(f"  - Val batches: {val_steps}")

        if callbacks is None:
            from .training_callbacks import get_default_callbacks

            callbacks = get_default_callbacks(
                model_name=self.model_name,
                checkpoint_dir=Path("checkpoints") / self.model_name,
            )

        start_time = datetime.now()

        self.history = self.siamese_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=verbose,
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        self.logger.info(f"Training completed in {elapsed:.2f} seconds")
        self._log_final_metrics()

        return self.history

    def save(self, filepath: Path | str | None = None) -> None:
        """
        Save complete model.
        Args:
            filepath: Path to save model (None = auto-generate)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"models/{self.model_name}_{timestamp}.keras")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.siamese_model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")

    def _log_final_metrics(self) -> None:
        """Log final training metrics."""
        if self.history is None:
            return

        history = self.history.history

        last_epoch = len(history["loss"])

        self.logger.info(f"Final metrics (epoch {last_epoch})")
        self.logger.info(f"     - Train Loss: {history['loss'][-1]:.4f}")
        self.logger.info(f"     - Val loss: {history['val_loss'][-1]:.4f}")

        if "auc" in history:
            self.logger.info(f"     - Train AUC: {history['auc'][-1]:.4f}")
            self.logger.info(f"     - Val AUC: {history['val_auc'][-1]:.4f}")

        best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
        self.logger.info(f"     - Best epoch: {best_epoch}")
        self.logger.info(f"     - Val Loss: {min(history['val_loss']):.4f}")
