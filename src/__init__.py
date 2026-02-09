"""
Author Handwriting Recognition - Main Package

This package provides a complete pipeline for processing handwriting images,
extracting features, and training recognition models.

Main Components:
    - I/O: File operations, image processing, serialization
    - Processors: High-level processing pipelines
    - Models: Neural network architectures (if applicable)

Quick Start:
    from src.io import FileSystem, ImageTransformer, ImageAnalyzer
    from src.processors import AuthorProcessor, LineSegmentProcessor

    # Create services
    fs = FileSystem()
    transformer = ImageTransformer()
    analyzer = ImageAnalyzer()
    line_processor = LineSegmentProcessor()

    # Process authors
    processor = AuthorProcessor(
        ...,
        transformer=transformer,
        analyzer=analyzer,
        line_processor=line_processor
    )
"""

from .datasets import (AuthorDatasetBuilder, AuthorPartitionState,
                       DataGenerator, DataInitializer, PairDataset,
                       PairGenerator, PairIndexSampler, PatchNormalizer,
                       SelectiveDataLoader, SubsetBuilder, TripletGenerator)
# Re-export main classes for convenience
from .io import (FileSystem, ImageAnalyzer, ImageTransformer, PatchExtractor,
                 load, save)
from .io.logging import LoggerFactory
from .models import (BinaryCrossEntropyDistance, CNNBackbone,
                     CNNTransformerBackbone, ContrastiveLoss, ModelComparator,
                     ModelEvaluator, ModelTrainer, build_embedding_model,
                     build_siamese_model, get_default_callbacks,
                     plot_comparison_metrics, plot_embeddings_tsne,
                     plot_training_history)
from .processors import AuthorDataLoader, AuthorProcessor, LineSegmentProcessor

__all__ = [
    # Logging
    "LoggerFactory",
    # I/O Services
    "FileSystem",
    "ImageTransformer",
    "ImageAnalyzer",
    "PatchExtractor",
    # Serialization
    "save",
    "load",
    # Processors
    "AuthorProcessor",
    "LineSegmentProcessor",
    "AuthorDataLoader",
    # Datasets
    "SelectiveDataLoader",
    "AuthorDatasetBuilder",
    "SubsetBuilder",
    "PairGenerator",
    "DataGenerator",
    "TripletGenerator",
    "PairIndexSampler",
    "PairDataset",
    "DataInitializer",
    "PatchNormalizer",
    "AuthorPartitionState",
    # Models
    "CNNBackbone",
    "CNNTransformerBackbone",
    "build_embedding_model",
    "build_siamese_model",
    "ContrastiveLoss",
    "BinaryCrossEntropyDistance",
    "ContrastiveLoss",
    "ModelTrainer",
    "ModelComparator",
    "ModelEvaluator",
    "plot_comparison_metrics",
    "plot_embeddings_tsne",
    "plot_training_history",
    "get_default_callbacks",
]

__version__ = "0.1.0"
