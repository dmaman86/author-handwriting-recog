from datetime import datetime
from pathlib import Path
from typing import Literal

import tensorflow as tf

from ...io.logging import LoggerFactory
from ..build_model import build_embedding_model
from ..build_triplet_model import build_triplet_model
from ..losses.binary_distance import BinaryCrossEntropyDistance
from ..losses.triplet_loss import TripletLoss
from ..siamese_model import build_siamese_model

ModelType = Literal["pair", "triplet"]
GeneratorType = tf.data.Dataset | tf.keras.utils.Sequence


class ModelTrainer:
    """
    Unified trainer for Siamese models.
    Features:
    - Automatic logging (console + file)
    - Automatic checkpoint saving
    - Configurable callbacks
    - Custom metrics
    Example:
        >>> from src.models.backbones import CNNBackbone
        >>> from src.datasets import PairGenerator
        >>>
        >>> backbone = CNNBackbone()
        >>> trainer = ModelTrainer(
        ...     backbone=backbone,
        ...     model_name="siamese_cnn",
        ...     log_dir="logs"
        ... )
        >>> trainer.build()
        >>> trainer.compile(learning_rate=0.0001)
        >>> history = trainer.train(train_gen, val_gen, epochs=50)
    """

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

    def build(self, model_type: ModelType = "pair") -> None:
        """
        Build complete Siamese model.
        Returns:
            Compiled Siamese model
        """
        self.model_type = model_type
        self.logger.info("Building models...")

        self.embedding_model = build_embedding_model(
            backbone=self.backbone,
            input_shape=self.input_shape,
            embedding_dim=self.embedding_dim,
            name=f"{self.model_name}_embedding_model",
        )

        if model_type == "pair":
            self.siamese_model = build_siamese_model(
                embedding_model=self.embedding_model,
                input_shape=self.input_shape,
            )
        elif model_type == "triplet":
            self.siamese_model = build_triplet_model(
                embedding_model=self.embedding_model,
                input_shape=self.input_shape,
            )

        total_params = self.siamese_model.count_params()
        trainable_params = sum(
            [tf.size(w).numpy() for w in self.siamese_model.trainable_weights]
        )

        self.logger.info(f"Model built: {self.siamese_model.name}")
        self.logger.info(f"Total params: {total_params}")
        self.logger.info(f"Trainable params: {trainable_params}")

    def compile(
        self,
        loss: tf.keras.losses.Loss | None = None,
        learning_rate: float = 0.0001,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
    ) -> None:
        """
        Compile model with loss and optimizer.
        Args:
            loss: Loss function instance (BinaryCrossEntropyDistance or ContrastiveLoss).
                  If None, defaults to BinaryCrossEntropyDistance(scale=10.0)
            learning_rate: Learning rate for optimizer
            optimizer: Custom optimizer (None = Adam with learning_rate)
        Examples:
            >>> from src.models.losses import BinaryCrossEntropyDistance, ContrastiveLoss
            >>>
            >>> # Option 1: Use default
            >>> trainer.compile()
            >>>
            >>> # Option 2: Binary Cross Entropy Distance
            >>> trainer.compile(loss=BinaryCrossEntropyDistance(scale=15.0))
            >>>
            >>> # Option 3: Contrastive Loss
            >>> trainer.compile(loss=ContrastiveLoss(margin=2.0))
        """
        if self.siamese_model is None:
            self.build()

        self.logger.info("Compiling model...")

        # Handle optimizer
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Handle loss function
        if loss is None:
            if self.model_type == "triplet":
                loss_fn = TripletLoss(margin=0.5)
            else:
                loss_fn = BinaryCrossEntropyDistance(scale=2.0)
        else:
            loss_fn = loss

        self.siamese_model.compile(optimizer=optimizer, loss=loss_fn)

        self.logger.info("Model compiled")
        self.logger.info(f"  - Optimizer: {optimizer.__class__.__name__}")
        self.logger.info(f"  - Learning rate: {learning_rate}")
        self.logger.info(f"  - Loss: {loss_fn.__class__.__name__}")

        # Log loss-specific parameters
        if hasattr(loss_fn, "scale"):
            self.logger.info(f"  - Loss scale: {loss_fn.scale}")
        if hasattr(loss_fn, "margin"):
            self.logger.info(f"  - Loss margin: {loss_fn.margin}")

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
