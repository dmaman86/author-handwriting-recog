import json
from pathlib import Path

from .base_serializer import BaseSerializer
from ..exceptions import SerializationError, DeserializationError


class JsonSerializer(BaseSerializer):
    """Serializer for JSON files."""

    def extensions(self) -> list[str]:
        return [".json"]

    def save(self, directory: str | Path, data: dict, filename: str) -> None:
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename

        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize data to JSON: {e}")
        except OSError as e:
            raise SerializationError(f"Failed to write JSON to {file_path}: {e}")

    def load(self, file_path: str | Path) -> dict:
        p = self.fs.ensure_file_exists(file_path)

        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON in {p}: {e}") from e
        except OSError as e:
            raise DeserializationError(f"Failed to read JSON from {p}: {e}") from e

