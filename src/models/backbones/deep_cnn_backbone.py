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

        # Block 1 - Conv(96, 5x5, stride=2)
        x = layers.Conv2D(96, kernel_size=5, strides=2, padding="same", use_bias=False)(
            inputs
        )
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        # Block 2 - Conv(128) x 2
        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

        # Block 3 - Conv(256) x 2
        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.MaxPooling2D(pool_size=2)(x)

        # Block 4 - Conv(512)
        x = layers.Conv2D(512, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x
