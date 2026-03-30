import tensorflow as tf

from .base_backbone import BaseBackbone


class MobileNetV2Backbone(BaseBackbone):

    def __init__(self, freeze_base: bool = True, fine_tune_at: int | None = None):
        self.freeze_base = freeze_base
        self.fine_tune_at = fine_tune_at

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=inputs.shape[1:],
        )

        if self.freeze_base:
            base_model.trainable = False
        elif self.fine_tune_at is not None:
            for layer in base_model.layers[: self.fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[self.fine_tune_at :]:
                layer.trainable = True
        else:
            base_model.trainable = True

        return base_model(inputs)
