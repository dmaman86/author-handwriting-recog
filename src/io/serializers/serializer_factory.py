from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from .base_serializer import BaseSerializer
    from ..file_system import FileSystem

from ..exceptions import UnsupportedFormatError


class SerializerFactory:
    """
    Factory for managing serializers using a Registry Pattern
    with Dependency Injection.
    Serializers auto-register themselves via
    BaseSerializer.__init_subclass__,
    eliminating the need for manual registration and
    hardcoded dependencies.

    Benefits:
    - Open/Closed Principle: Add new serializers without modifying this class
    - Lazy Loading: Serializer instances created only when needed
    - Dependency Injection: FileSystem injected into all serializers
    - Testability: Easy to inject custom serializers for testing
    - Configurable: Support serializer-specific configuration
    """

    # Class-level registry: {extension: SerializerClass}
    _registry: dict[str, Type["BaseSerializer"]] = {}

    # Instance cache: {extension: serializer_instance}
    _instances: dict[str, "BaseSerializer"] = {}

    # Injected FileSystem dependency
    _file_system: "FileSystem | None" = None

    # Serializer configuration: {serializer_name: config_dict}
    _serializer_config: dict[str, dict[str, Any]] = {}

    @classmethod
    def initialize(
        cls,
        file_system: "FileSystem",
        serializer_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the factory with a FileSystem instance.

        This must be called before using save/load functions.

        Args:
            file_system: FileSystem instance to inject into serializers
            serializer_config: Optional configuration for specific serializers
                              Example: {
                                  'SomeSerializer': {
                                      'option1': 'value1',
                                      'option2': 'value2'
                                  }
                              }
        """
        cls._file_system = file_system
        cls._serializer_config = serializer_config or {}
        # Clear cache to force re-creation with new file_system
        cls._instances.clear()

    @classmethod
    def register(cls, serializer_class: Type["BaseSerializer"]) -> None:
        """
        Register a serializer class for its supported extensions.

        This method is called automatically by BaseSerializer.__init_subclass__
        when a new serializer class is defined.

        Args:
            serializer_class: The serializer class to register (not an instance)
        """
        # Store the class by its name
        # We'll instantiate it later when we have FileSystem
        cls._registry[serializer_class.__name__] = serializer_class

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure factory is initialized with FileSystem."""
        if cls._file_system is None:
            # Auto-initialize with default FileSystem for convenience
            from ..file_system import FileSystem

            cls._file_system = FileSystem()

    @classmethod
    def get_by_extension(cls, extension: str) -> "BaseSerializer":
        """
        Get a serializer instance for the given extension.

        Uses lazy instantiation: creates the serializer only on first use.

        Args:
            extension: File extension (with or without leading dot)
        Returns:
            Serializer instance for the extension
        Raises:
            ValueError: If no serializer is registered for the extension
        """
        cls._ensure_initialized()

        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"

        # Check if we already have an instance
        if ext in cls._instances:
            return cls._instances[ext]

        # Find serializer class that supports this extension
        assert cls._file_system is not None, "Factory not initialized"

        for serializer_class in cls._registry.values():
            # Get configuration for this serializer if available
            config = cls._serializer_config.get(serializer_class.__name__, {})

            # Create instance with configuration
            temp = serializer_class(cls._file_system, **config)

            if ext in [e.lower() for e in temp.extensions()]:
                cls._instances[ext] = temp
                return temp

        raise UnsupportedFormatError(
            f"No serializer found for extension: {extension}. "
            f"Available: {cls.list_extensions()}"
        )

    @classmethod
    def get_for_file(cls, file_path: str | Path) -> "BaseSerializer":
        """
        Get a serializer for the given file path based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Serializer instance for the file's extension
        """
        ext = Path(file_path).suffix.lower()
        return cls.get_by_extension(ext)

    @classmethod
    def save(cls, directory: str | Path, data: Any, filename: str) -> None:
        """
        Save data to a file using the appropriate serializer.

        Args:
            directory: Target directory
            data: Data to save
            filename: Filename with extension
        """
        ext = Path(filename).suffix.lower()
        serializer = cls.get_by_extension(ext)
        serializer.save(directory, data, filename)

    @classmethod
    def load(cls, file_path: str | Path) -> Any:
        """
        Load data from a file using the appropriate serializer.

        Args:
            file_path: Path to the file to load

        Returns:
            Loaded data
        """
        serializer = cls.get_for_file(file_path)
        return serializer.load(file_path)

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the instance cache. Useful for testing.
        """
        cls._instances.clear()

    @classmethod
    def list_extensions(cls) -> list[str]:
        """
        List all registered file extensions.

        Returns:
            Sorted list of supported extensions
        """
        cls._ensure_initialized()
        assert cls._file_system is not None

        extensions = []
        for serializer_class in cls._registry.values():
            # Get config for this serializer
            config = cls._serializer_config.get(serializer_class.__name__, {})
            temp = serializer_class(cls._file_system, **config)
            extensions.extend(temp.extensions())

        return sorted(set(ext.lower() for ext in extensions))
