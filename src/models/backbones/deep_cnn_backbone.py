import tensorflow as tf
from tensorflow.keras import layers

from .base_backbone import BaseBackbone


class DeepCNNBackbone(BaseBackbone):
    """
    Deeper CNN backbone for handwriting patches.
    Outputs a 4D feature map (batch, H', W', C).
    """

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        """
        Build deeper CNN feature extractor.

        Args:
            inputs: Keras Input tensor (batch, H, W, C)

        Returns:
            Feature map tensor (batch, H', W', C)
        """

        x = layers.Conv2D(96, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(384, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(384, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        return x
