import tensorflow as tf
from tensorflow.keras import Model, layers


def build_triplet_model(
    embedding_model: tf.keras.Model,
    input_shape: tuple[int, int, int],
    name: str = "triplet_model",
) -> tf.keras.Model:
    """
    Build a Triplet model using a shared embedding network.

    Args:
        embedding_model: Shared embedding model
        input_shape: Input image shape (H, W, C)
        name: Model name

    Returns:
        tf.keras.Model producing (anchor, positive, negative) embeddings
    """

    # Inputs
    anchor_input = layers.Input(shape=input_shape, name="anchor_input")
    positive_input = layers.Input(shape=input_shape, name="positive_input")
    negative_input = layers.Input(shape=input_shape, name="negative_input")

    # Shared embedding network
    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)

    # Stack embeddings: (batch, 3, embedding_dim)
    embeddings = (
        layers.StackedRNNCells
        if False
        else tf.stack(
            [anchor_emb, positive_emb, negative_emb],
            axis=1,
            name="triplet_embeddings",
        )
    )

    model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=embeddings,
        name=name,
    )

    return model
