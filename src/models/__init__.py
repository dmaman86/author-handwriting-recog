from .backbones.cnn_backbone import CNNBackbone
from .backbones.cnn_transformer_backbone import CNNTransformerBackbone
from .build_model import build_embedding_model
from .evaluation import (
    ModelComparator,
    ModelEvaluator,
    plot_comparison_metrics,
    plot_embeddings_tsne,
    plot_training_history,
)
from .losses.binary_distance import BinaryCrossEntropyDistance
from .losses.contrastive_loss import ContrastiveLoss
from .siamese_model import build_siamese_model
from .training import ModelTrainer, get_default_callbacks

__all__ = [
    "CNNBackbone",
    "CNNTransformerBackbone",
    "build_embedding_model",
    "ContrastiveLoss",
    "BinaryCrossEntropyDistance",
    "build_siamese_model",
    "ModelTrainer",
    "get_default_callbacks",
    "ModelEvaluator",
    "ModelComparator",
    "plot_training_history",
    "plot_comparison_metrics",
    "plot_embeddings_tsne",
]

