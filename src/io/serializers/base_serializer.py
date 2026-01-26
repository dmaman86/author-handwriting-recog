from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..file_system import FileSystem


class BaseSerializer(ABC):
    """
    Base class for all serializers (JSON, Pickle, NPY,
    MAT, Images, Keras models, etc.)

    Responsibilities:
    - Define which extensions the serializer supports.
    - Provide a unified interface for saving and loading data.
    - Auto-register with SerializerFactory via __init_subclass__
    - Use injected FileSystem for file operations
    """

    def __init__(self, file_system: 'FileSystem'):
        """
        Initialize serializer with file system dependency.
        
        Args:
            file_system: FileSystem instance for file operations
        """
        self.fs = file_system

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register each concrete serializer subclass
        with the SerializerFactory when the class is defined.
        """
        super().__init_subclass__(**kwargs)
        
        # Avoid registering abstract intermediate classes
        if not getattr(cls, '__abstractmethods__', None):
            # Lazy import to avoid circular dependency
            from .serializer_factory import SerializerFactory
            SerializerFactory.register(cls)

    @abstractmethod
    def extensions(self) -> list[str]:
        """
        Return list of file extensions this serializer handles.
        
        Returns:
            List of extensions (e.g., ['.json', '.txt'])
        """
        pass

    @abstractmethod
    def save(self, directory: str | Path, data, filename: str):
        """
        Save data to the given directory with the provided filename.

        Args:
            directory: Target directory where the file will be stored.
            data: The object to be saved.
            filename: The name of the file, including extension
            (e.g. 'config.json').
        """
        pass

    @abstractmethod
    def load(self, file_path: str | Path):
        """
        Load and return data from the given file path.

        Args:
            file_path: Full path to the file.

        Returns:
            Loaded object, depending on the serializer.
        """
        pass

    def normalize_path(self, path: str | Path) -> Path:
        """
        Helper method: convert any path input to a Path object.
        """
        return self.fs.to_path(path)

