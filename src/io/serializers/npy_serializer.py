from pathlib import Path

import numpy as np

from .base_serializer import BaseSerializer


class NpySerializer(BaseSerializer):
    """Serializer for NumPy .npy files."""

    def extensions(self) -> list[str]:
        return [".npy"]

    def save(self, directory: str | Path, data: np.ndarray, filename: str) -> None:
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename

        np.save(file_path, data)

    def load(self, file_path: str | Path) -> np.ndarray:
        p = self.fs.ensure_file_exists(file_path)

        return np.load(p)

