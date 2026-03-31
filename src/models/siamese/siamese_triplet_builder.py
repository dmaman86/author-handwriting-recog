import tensorflow as tf

from .base_siamese_builder import BaseSiameseBuilder


class SiameseTripletBuilder(BaseSiameseBuilder):

    def build(self) -> tf.keras.Model:
        self._embedding_model = self.embedding_network.build_model()
        anchor = tf.keras.layers.Input(shape=self.input_shape, name="anchor")
        positive = tf.keras.layers.Input(shape=self.input_shape, name="positive")
        negative = tf.keras.layers.Input(shape=self.input_shape, name="negative")

        emb_a = self._embedding_model(anchor)
        emb_p = self._embedding_model(positive)
        emb_n = self._embedding_model(negative)

        model = tf.keras.Model(
            inputs=[anchor, positive, negative],
            outputs=[emb_a, emb_p, emb_n],
            name=self.name,
        )

        return model
