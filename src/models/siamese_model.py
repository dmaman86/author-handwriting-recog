import tensorflow as tf
from tensorflow.keras import layers, models, ops


def build_siamese_model(
        embedding_model: tf.keras.Model,
        input_shape: tuple[int, int, int] = (60, 53, 1),
) -> tf.keras.Model:
    
    input1 = layers.Input(shape=input_shape, name="image1")
    input2 = layers.Input(shape=input_shape, name="image2")
        
    embedding1 = embedding_model(input1)  # (batch, embedding_dim)
    embedding2 = embedding_model(input2)
        
    embeddings = ops.stack([embedding1, embedding2], axis=1)  # (batch, 2, embedding_dim)
        
    return models.Model(
        inputs=[input1, input2],
        outputs=embeddings,
        name="siamese_model"
    )
    