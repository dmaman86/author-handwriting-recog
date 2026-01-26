from pathlib import Path

from scipy.io import loadmat

from .base_serializer import BaseSerializer


class MatSerializer(BaseSerializer):
    """Serializer for MATLAB .mat files (read-only)."""

    def extensions(self) -> list[str]:
        return [".mat"]

    def save(self, directory: str | Path, data: dict, filename: str) -> None:
        raise NotImplementedError("Saving .mat files is not supported.")

    def load(self, file_path: str | Path) -> dict:
        p = self.fs.ensure_file_exists(file_path)

        return loadmat(p)

