import tensorflow as tf
from tensorflow.keras import layers


class AugmentationBuilder:

    @staticmethod
    def build_default(seed: int | None = None) -> tf.keras.Sequential:
        """Builds a default augmentation pipeline.

        Args:
            seed (int | None): Random seed for reproducibility.

        Returns:
            tf.keras.Sequential: Augmentation pipeline.
        """
        augmentation_layers = [
            layers.RandomRotation(0.028, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=seed),
            layers.RandomZoom(0.1, fill_mode="nearest", seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ]
        return tf.keras.Sequential(augmentation_layers, name="pair_augmentation")

    @staticmethod
    def build_custom(
        rotation: float = 0.028,
        translation: float = 0.05,
        zoom: float = 0.1,
        contrast: float = 0.1,
        seed: int | None = None,
    ) -> tf.keras.Sequential:
        augmentation_layers = [
            layers.RandomRotation(rotation, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(
                translation, translation, fill_mode="nearest", seed=seed
            ),
            layers.RandomZoom(zoom, fill_mode="nearest", seed=seed),
            layers.RandomContrast(contrast, seed=seed),
        ]

        return tf.keras.Sequential(augmentation_layers, name="custom_augmentation")

    @staticmethod
    def build_minimal(seed: int | None = None) -> tf.keras.Sequential:
        """Builds a minimal augmentation pipeline.

        Args:
            seed (int | None): Random seed for reproducibility.

        Returns:
            tf.keras.Sequential: Augmentation pipeline.
        """
        augmentation_layers = [
            layers.RandomRotation(0.014, fill_mode="nearest", seed=seed),  # ±5 degrees
        ]
        return tf.keras.Sequential(augmentation_layers, name="minimal_augmentation")
