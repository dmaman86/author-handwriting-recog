import tensorflow as tf

from .base_backbone import BaseBackbone


class EfficientNetV2Backbone(BaseBackbone):

    def __init__(
        self,
        variant: str = "B0",  # B0, B1, B2, B3, S, M, L
        freeze_base: bool = True,
        fine_tune_at: int | None = None,
    ):
        self.variant = variant
        self.freeze_base = freeze_base
        self.fine_tune_at = fine_tune_at

    def _get_model(self, input_shape):
        variant_map = {
            "B0": tf.keras.applications.EfficientNetV2B0,
            "B1": tf.keras.applications.EfficientNetV2B1,
            "B2": tf.keras.applications.EfficientNetV2B2,
            "B3": tf.keras.applications.EfficientNetV2B3,
            "S": tf.keras.applications.EfficientNetV2S,
            "M": tf.keras.applications.EfficientNetV2M,
            "L": tf.keras.applications.EfficientNetV2L,
        }

        if self.variant not in variant_map:
            raise ValueError(f"Unknown EfficientNetV2 variant: {self.variant}")

        return variant_map[self.variant](
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        base_model = self._get_model(inputs.shape[1:])

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
