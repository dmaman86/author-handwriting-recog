import tensorflow as tf

from .base_metric_model import BaseMetricModel
from .layers import (CosineSimilarityLayer, EuclideanDistanceLayer,
                     ScaledCosineLayer)


class SiamesePairModel(BaseMetricModel):

    def build(self, embedding_model: tf.keras.Model) -> tf.keras.Model:
        input1 = tf.keras.layers.Input(shape=self.input_shape, name="image1")
        input2 = tf.keras.layers.Input(shape=self.input_shape, name="image2")

        embedding1 = embedding_model(input1)
        embedding2 = embedding_model(input2)

        # cosine similarity in [-1, 1]
        similarity = CosineSimilarityLayer(name="cosine_similarity")(
            [embedding1, embedding2]
        )

        """
        scaled_similarity = ScaledCosineLayer(
            initial_scale=8.0, name="scaled_similarity"
        )(similarity)

        probability = tf.keras.layers.Activation(
            "sigmoid", name="similarity_to_probability"
        )(scaled_similarity)
        """

        return tf.keras.Model(
            inputs=[input1, input2],
            outputs=similarity,
            name=self.name,
        )
