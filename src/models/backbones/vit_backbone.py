import keras_hub
import tensorflow as tf

from .base_backbone import BaseBackbone


class ViTBackbone(BaseBackbone):

    def __init__(
        self,
        freeze_base: bool = True,
    ):
        self.freeze_base = freeze_base

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        backbone = keras_hub.models.Backbone.from_preset(
            "vit_base_patch16_224_imagenet"
        )

        backbone.trainable = not self.freeze_base

        return backbone(inputs)
