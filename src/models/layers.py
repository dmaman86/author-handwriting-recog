import tensorflow as tf


class EuclideanDistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, y = inputs

        return tf.sqrt(
            tf.reduce_sum(
                tf.square(x - y),
                axis=1,
                keepdims=True,
            )
        )


class CosineDistanceLayer(tf.keras.layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, y = inputs

        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)

        distance = 1 - tf.reduce_sum(
            x * y,
            axis=-1,
            keepdims=True,
        )
        return distance

    def get_config(self):
        return super().get_config()


class CosineSimilarityLayer(tf.keras.layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, y = inputs

        similarity = tf.reduce_sum(
            x * y,
            axis=1,
            keepdims=True,
        )
        return similarity

    def get_config(self):
        return super().get_config()


class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def call(
        self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return ap_distance, an_distance


class ScaledCosineLayer(tf.keras.layers.Layer):

    def __init__(self, initial_scale: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale

    def build(self, input_shape):
        # Learnable scalar parameter
        self.scale = self.add_weight(
            name="scale",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: cosine similarity values in [-1, 1]
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_scale": self.initial_scale,
            }
        )
        return config


class L2Normalization(tf.keras.layers.Layer):
    """
    Layer that applies L2 normalization along the
    embedding dimension.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self) -> dict:
        return super().get_config()
