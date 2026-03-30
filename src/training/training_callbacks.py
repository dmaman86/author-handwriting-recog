from pathlib import Path

import tensorflow as tf

"""
for Pair model
callbacks = get_default_callbacks(
    model_name="siamese_pair_cnn",
    monitor="val_auc",
    mode="max",
)
for Triplet model
callbacks = get_default_callbacks(
    model_name="siamese_triplet_cnn",
    monitor="val_loss",
    mode="min",
)
"""


def get_default_callbacks(
    model_name: str,
    checkpoint_dir: Path | str | None = None,
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    start_from_epoch: int = 10,
    reduce_lr_patience: int = 5,
    min_lr: float = 1e-7,
) -> list[tf.keras.callbacks.Callback]:
    """
    Create default training callbacks.

    Args:
        model_name: Model name for checkpoint directory
        checkpoint_dir: Directory for checkpoints
        monitor: Metric to monitor (e.g. val_loss, val_auc)
        mode: "min" or "max"
        patience: Early stopping patience
        reduce_lr_patience: ReduceLROnPlateau patience
        min_lr: Minimum learning rate
    """

    callbacks = []

    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints") / model_name

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best.weights.h5"

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            start_from_epoch=start_from_epoch,
            mode=mode,
            verbose=1,
        )
    )

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
            verbose=1,
        )
    )

    return callbacks


class LoggingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for logging training progress.
    """

    def __init__(self, logger, log_every_n_epochs: int = 5):
        """
        Initialize callback.
        Args:
            logger: Logger instance
            log_every_n_epochs: Log detailed metrics every N epochs
        """
        super().__init__()
        self.logger = logger
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of epoch."""
        logs = logs or {}

        if (epoch + 1) % self.log_every_n_epochs == 0:
            self.logger.info(f"Epoch {epoch + 1}:")
            for key, value in logs.items():
                self.logger.info(f"  - {key}: {value:.4f}")
