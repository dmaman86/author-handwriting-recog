import pickle
from pathlib import Path

from .base_serializer import BaseSerializer
from ..exceptions import SerializationError, DeserializationError

class PickleSerializer(BaseSerializer):
    """Serializer for Python pickle files."""

    def extensions(self) -> list[str]:
        return [".pkl", ".pickle"]

    def save(self, directory: str | Path, data, filename: str) -> None:
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename
        try:
            with file_path.open("wb") as f:
                pickle.dump(data, f)
        except (pickle.PicklingError, TypeError) as e:
            raise SerializationError(f"Failed to pickle data: {e}") from e
        except OSError as e:
            raise SerializationError(f"Failed to write pickle to {file_path}: {e}") from e
        
    def load(self, file_path: str | Path):
        p = self.fs.ensure_file_exists(file_path)  # Propaga FileNotFoundError
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            raise DeserializationError(f"Failed to unpickle {p}: {e}") from e
        except OSError as e:
            raise DeserializationError(f"Failed to read pickle from {p}: {e}") from e

