"""
Serializer submodule with auto-registration pattern.

This module handles automatic serializer registration and provides
a factory pattern for saving/loading files based on their extension.

Adding a new serializer:
    1. Create a new class inheriting from BaseSerializer
    2. Implement extensions(), save(), and load() methods
    3. Import it in this __init__.py (auto-registration happens automatically)

Note: For end-users, import from parent module:
    from src.io import save, load  # Preferred
    from src.io.serializers import save, load  # Also works
"""

from .base_serializer import BaseSerializer
from .serializer_factory import SerializerFactory

# Import all serializers to trigger auto-registration via metaclass
from .image_serializer import ImageSerializer
from .json_serializer import JsonSerializer
from .keras_serializer import KerasSerializer
from .mat_serializer import MatSerializer
from .npy_serializer import NpySerializer
from .pickle_serializer import PickleSerializer
from .zarr_serializer import ZarrSerializer

# Re-export factory methods as module-level convenience functions
save = SerializerFactory.save
load = SerializerFactory.load
get_by_extension = SerializerFactory.get_by_extension
get_for_file = SerializerFactory.get_for_file
list_extensions = SerializerFactory.list_extensions

__all__ = [
    # Core classes
    "BaseSerializer",
    "SerializerFactory",
    # Convenience functions
    "save",
    "load",
    "get_by_extension",
    "get_for_file",
    "list_extensions",
    # Concrete serializers (for type hints and advanced usage)
    "ImageSerializer",
    "JsonSerializer",
    "KerasSerializer",
    "MatSerializer",
    "NpySerializer",
    "PickleSerializer",
    "ZarrSerializer",
]