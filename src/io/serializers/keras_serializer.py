from pathlib import Path

from keras.models import load_model

from .base_serializer import BaseSerializer


class KerasSerializer(BaseSerializer):
    """Serializer for Keras/TensorFlow model files."""

    def extensions(self) -> list[str]:
        return [".keras"]

    def save(self, directory: str | Path, model, filename: str) -> None:
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename

        model.save(file_path)

    def load(self, file_path: str | Path):
        p = self.fs.ensure_file_exists(file_path)

        return load_model(p, compile=False)

