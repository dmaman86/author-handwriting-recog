import tensorflow as tf
from tensorflow.keras import layers

def embedding_head(features: tf.Tensor, embedding_dim: int = 256) -> tf.Tensor:
    if len(features.shape) == 4:
        x: tf.Tensor = layers.GlobalAveragePooling2D()(features)
    elif len(features.shape) == 3:
        x = layers.GlobalAveragePooling1D()(features)
    else:
        raise ValueError(
            f"Unsupported feature tensor rank: {len(features.shape)}"
        )
    x = layers.Dense(embedding_dim, name="embedding_projection")(x)
    x = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name="l2_normalize"
    )(x)
    return x