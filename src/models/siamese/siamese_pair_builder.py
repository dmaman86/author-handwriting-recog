import tensorflow as tf

from ..layers import CosineSimilarityLayer
from .base_siamese_builder import BaseSiameseBuilder


class SiamesePairBuilder(BaseSiameseBuilder):

    def build(self) -> tf.keras.Model:
        self._embedding_model = self.embedding_network.build_model()
        input1 = tf.keras.layers.Input(shape=self.input_shape, name="image1")
        input2 = tf.keras.layers.Input(shape=self.input_shape, name="image2")

        embedding1 = self._embedding_model(input1)
        embedding2 = self._embedding_model(input2)

        # cosine similarity in [-1, 1]
        similarity = CosineSimilarityLayer(name="cosine_similarity")(
            [embedding1, embedding2]
        )

        model = tf.keras.Model(
            inputs=[input1, input2],
            outputs=similarity,
            name=self.name,
        )
        return model
