"""
I/O Module - Unified interface for file operations, serialization, and image processing.

This module provides:
    - File system operations (FileSystem)
    - Image processing (ImageTransformer, ImageAnalyzer, PatchExtractor)
    - Serialization (save, load with auto-format detection)

Usage:
    from src.io import save, load, FileSystem, ImageTransformer
    
    # File operations
    FileSystem.ensure_directory("./output")
    
    # Serialization (auto-detects format from extension)
    save("./output", my_data, "data.json")
    data = load("./output/data.json")
    
    # Image processing
    img = ImageTransformer.preprocess_pipeline(raw_img, target_size=(128, 128))
"""

# File system operations
from .file_system import FileSystem

# Image processing
from .image_ops import ImageTransformer, ImageAnalyzer, PatchExtractor

# Serialization - Initialize singleton and export functions
from .serializers import (
    save,
    load,
    get_by_extension,
    get_for_file,
    list_extensions,
    BaseSerializer,
    SerializerFactory,
)

from .exceptions import (
    DeserializationError,
    SerializationError,
    UnsupportedFormatError,
)

from .logging import LoggerFactory

__all__ = [
    # File system
    "FileSystem",
    # Image processing
    "ImageTransformer",
    "ImageAnalyzer", 
    "PatchExtractor",
    # Serialization functions (most commonly used)
    "save",
    "load",
    "get_by_extension",
    "get_for_file",
    "list_extensions",
    # Serialization classes (for advanced usage)
    "BaseSerializer",
    "SerializerFactory",
    # Exceptions
    "DeserializationError",
    "SerializationError",
    "UnsupportedFormatError",
    # Logging
    "LoggerFactory",
]