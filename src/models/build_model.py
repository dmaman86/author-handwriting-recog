import tensorflow as tf
from tensorflow.keras import layers, models
from .backbones.base_backbone import BaseBackbone
from .heads.embedding_head import embedding_head

def build_embedding_model(
    backbone: BaseBackbone,
    input_shape: tuple[int, int, int] = (60, 53, 1),
    embedding_dim: int = 256,
    name: str = "embedding_model"
) -> tf.keras.Model:
    """
    Build full embedding model from backbone + embedding head.
    """
    inputs = layers.Input(shape=input_shape)
    features = backbone.build(inputs)
    embeddings = embedding_head(features, embedding_dim)

    return models.Model(
        inputs=inputs, 
        outputs=embeddings,
        name=name)