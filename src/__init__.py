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

from .datasets import (AuthorDatasetBuilder, DataGenerator, DataInitializer,
                       PairDataset, PairGenerator, PairIndexSampler,
                       PatchNormalizer, SelectiveDataLoader)
# Re-export main classes for convenience
from .io import (FileSystem, ImageAnalyzer, ImageTransformer, PatchExtractor,
                 load, save)
from .io.logging import LoggerFactory
from .processors import AuthorDataLoader, AuthorProcessor, LineSegmentProcessor

# Optional: Import models only if TensorFlow is available
try:
    from .models import (BinaryCrossEntropyDistance, CNNBackbone,
                         CNNTransformerBackbone, ContrastiveLoss,
                         build_embedding_model, build_siamese_model)

    _MODELS_AVAILABLE = True
except ImportError:
    # TensorFlow not installed - models won't be available
    _MODELS_AVAILABLE = False
    CNNBackbone = None
    CNNTransformerBackbone = None
    build_embedding_model = None
    build_siamese_model = None
    ContrastiveLoss = None
    BinaryCrossEntropyDistance = None

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
    "PairGenerator",
    "DataGenerator",
    "PairIndexSampler",
    "PairDataset",
    "DataInitializer",
    "PatchNormalizer",
    # Models
    "CNNBackbone",
    "CNNTransformerBackbone",
    "build_embedding_model",
    "build_siamese_model",
    "ContrastiveLoss",
    "BinaryCrossEntropyDistance",
]

__version__ = "0.1.0"
