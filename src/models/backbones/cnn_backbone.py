import tensorflow as tf
from tensorflow.keras import layers
from .base_backbone import BaseBackbone

class CNNBackbone(BaseBackbone):
    """
    CNN backbone for handwriting patches.
    Outputs a 4D feature map (batch, H', W', C).
    """

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        """
        Build CNN feature extractor.

        Args:
            inputs: Keras Input tensor (batch, H, W, C)

        Returns:
            Feature map tensor (batch, H', W', C)
        """
        
        x = layers.Conv2D(32, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x