import tensorflow as tf

from .base_metric_model import BaseMetricModel


class SiameseTripletModel(BaseMetricModel):

    def build(self, embedding_model: tf.keras.Model) -> tf.keras.Model:

        anchor = tf.keras.layers.Input(shape=self.input_shape, name="anchor")
        positive = tf.keras.layers.Input(shape=self.input_shape, name="positive")
        negative = tf.keras.layers.Input(shape=self.input_shape, name="negative")

        emb_a = embedding_model(anchor)
        emb_p = embedding_model(positive)
        emb_n = embedding_model(negative)

        return tf.keras.Model(
            inputs=[anchor, positive, negative],
            outputs=[emb_a, emb_p, emb_n],
            name=self.name,
        )
