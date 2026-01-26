from pathlib import Path

import cv2
import numpy as np

from .base_serializer import BaseSerializer
from ..exceptions import SerializationError, DeserializationError


class ImageSerializer(BaseSerializer):
    """
    Serializer for image files using OpenCV.
    
    Supports common image formats: PNG, JPG, JPEG, BMP, TIFF.
    Automatically handles RGB to BGR conversion for OpenCV compatibility.
    """

    def extensions(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    def save(self, directory: str | Path, data: np.ndarray, filename: str) -> None:
        """
        Save image data to file.
        
        Args:
            directory: Directory path where to save the image
            data: Image data as numpy array
            filename: Name of the file to save
            
        Raises:
            SerializationError: If image save fails
        """
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename

        # Convert RGB to BGR for OpenCV
        if isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        try:
            success = cv2.imwrite(str(file_path), data)
            if not success:
                raise SerializationError(f"Failed to save image to {file_path}")
        except Exception as e:
            raise SerializationError(f"Failed to save image to {file_path}: {e}")

    def load(self, file_path: str | Path) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image data as numpy array (BGR format)
            
        Raises:
            FileNotFoundError: If file does not exist
            DeserializationError: If image load fails
        """
        p = self.fs.ensure_file_exists(file_path)

        try:
            image = cv2.imread(str(p))
            if image is None:
                raise DeserializationError(f"Failed to load image from {p}")
            return image
        except Exception as e:
            raise DeserializationError(f"Error loading image from {p}: {e}")