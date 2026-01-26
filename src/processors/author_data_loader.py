from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..io import load, ImageTransformer, DeserializationError

@dataclass(frozen=True)
class AuthorData:
    author_id: str
    image: np.ndarray  # preprocessed (binarized & inverted)
    lines: list[tuple[int, int]]
    test_area_top: int
    test_area_bottom: int
    scale_factor: float

    @property
    def num_lines(self) -> int:
        return len(self.lines)
    

class AuthorDataLoader:
    def __init__(self, images_dir: str | Path, mat_dir: str | Path, transformer: ImageTransformer) -> None:
        self.images_dir = Path(images_dir)
        self.mat_dir = Path(mat_dir)
        self.transformer = transformer

    def load_author_data(self, author_id: str) -> AuthorData:
        image = self._load_author_image(author_id)
        lines, top, bottom, scale = self._load_author_metadata(author_id)

        return AuthorData(
            author_id=author_id,
            image=image,
            lines=lines,
            test_area_top=top,
            test_area_bottom=bottom,
            scale_factor=scale,
        )

    def _load_author_image(self, author_id: str) -> np.ndarray:
        image_path = self.images_dir / f"{author_id}.jpg"
        try:
            raw_image = load(image_path)
        except FileNotFoundError:
            raise ValueError(f"Image file for author '{author_id}' not found at {image_path}")
        except DeserializationError as e:
            raise ValueError(f"Failed to load image for author '{author_id}': {e}")
        
        return self.transformer.binarize_and_invert(raw_image.copy())
    
    def _load_author_metadata(self, author_id: str) -> tuple[list[tuple[int, int]], int, int, float]:
        mat_path = self.mat_dir / f"{author_id}.mat"
        
        try:
            mat_info = load(mat_path)
        except (FileNotFoundError, DeserializationError) as e:
            raise ValueError(
                f"Missing or corrupted MAT file for author: {author_id}"
            ) from e
        
        # Parse MAT data
        try:
            peaks: np.ndarray = mat_info['peaks_indices'].flatten()
            scale: float = float(mat_info['SCALE_FACTOR'].flatten()[0])
            top: int = int(mat_info['top_test_area'].flatten()[0])
            bottom: int = int(mat_info['bottom_test_area'].flatten()[0])
            
            # Scale peaks and create line segments
            peaks_scaled = [int(p * scale) for p in peaks]
            lines = [
                (peaks_scaled[i], peaks_scaled[i + 1])
                for i in range(len(peaks_scaled) - 1)
            ]
            return lines, top, bottom, scale
            
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid MAT file format for author {author_id}: {e}"
            ) from e
        
    def exists(self, author_id: str) -> bool:
        image_path = self.images_dir / f"{author_id}.jpg"
        mat_path = self.mat_dir / f"{author_id}.mat"
        return image_path.exists() and mat_path.exists()