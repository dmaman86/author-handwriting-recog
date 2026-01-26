import random
import numpy as np

from ..io import (
    ImageAnalyzer,
    PatchExtractor
)
from .author_data_loader import AuthorDataLoader, AuthorData


class LineSegmentProcessor:
    """
    Handles processing of text line segments.
    
    Provides utilities for normalizing line heights and extracting
    segments from images with proper padding/cropping.
    """
    
    def normalize_height(
        self,
        start_y: int,
        end_y: int,
        target_height: int = 180
    ) -> tuple[int, int]:
        """
        Normalize line segment height to a target height.
        
        If the line is shorter than target, adds padding (centered).
        If the line is taller than target, crops from center.
        
        Args:
            start_y: Starting Y coordinate of the line
            end_y: Ending Y coordinate of the line
            target_height: Desired height in pixels
            
        Returns:
            Tuple of (normalized_start_y, normalized_end_y)
        """
        current_height = end_y - start_y

        if current_height == target_height:
            return start_y, end_y
        
        center = (start_y + end_y) // 2
        half = target_height // 2
        return center - half, center + half
    
    def extract_segment(
        self,
        image: np.ndarray,
        start_y: int,
        end_y: int,
        target_height: int = 180
    ) -> np.ndarray:
        """
        Extract a line segment from an image.
        
        Args:
            image: Source image
            start_y: Starting Y coordinate
            end_y: Ending Y coordinate
            normalize: Whether to normalize height
            target_height: Target height if normalizing
            
        Returns:
            Extracted segment (H, W) or (H, W, C)
        """
        h_img = image.shape[0]
        start_y, end_y = self.normalize_height(start_y, end_y, target_height)

        start_y = max(0, start_y)
        end_y = min(h_img, end_y)

        segment = image[start_y:end_y, :]
        return self._pad_or_crop(segment, target_height)
        
    def _pad_or_crop(self, segment: np.ndarray, target_height: int) -> np.ndarray:
        h, w = segment.shape[:2]

        if h == target_height:
            return segment
        
        if h > target_height:
            center = h // 2
            half = target_height // 2
            return segment[center - half:center + half, :]
        
        pad = target_height - h
        pad_top = pad // 2
        pad_bottom = pad - pad_top

        return np.pad(segment, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

class AuthorProcessor:
    """
    Processes author handwriting images to extract patches for training.
    
    This class orchestrates image loading, preprocessing, line segmentation,
    and patch extraction using injected dependencies.
    """

    def __init__(
        self,
        label_id: int,
        author: str,
        data_loader: AuthorDataLoader,
        analyzer: ImageAnalyzer,
        line_processor: LineSegmentProcessor,
        patch_size: tuple[int, int] = (180, 160),
        stride_x: int = 30,
        threshold: float = 0.95,
        val_size: float = 0.15,
        seed: int = 42
    ) -> None:
        """
        Initialize AuthorProcessor with dependencies and configuration.
        
        Args:
            label_id: Numeric label for this author
            author: Author identifier/name
            data_loader: AuthorDataLoader instance (handles all file I/O)
            analyzer: ImageAnalyzer instance for quality checks
            line_processor: LineSegmentProcessor for line normalization
            patch_size: (height, width) of extracted patches
            stride_x: Horizontal stride for sliding window
            threshold: Threshold for filtering empty patches
            val_size: Validation set size ratio
            seed: Random seed for reproducibility
        """
        # Configuration
        self.label_id = label_id
        self.author = author
        self.patch_size = patch_size
        self.stride_x = stride_x
        self.threshold = threshold
        self.val_size = val_size
        self.seed = seed
        
        # Injected dependencies (stateless services)
        self.data_loader = data_loader
        self.analyzer = analyzer
        self.line_processor = line_processor
        
        # Initialize patch extractor with dependency injection
        self.patch_extractor = PatchExtractor(
            image_analyzer=self.analyzer,
            patch_width=self.patch_size[1],
            stride_x=self.stride_x,
            filter_empty=True,
            empty_threshold=self.threshold
        )

        self.data: AuthorData = self.data_loader.load_author_data(author)
    
    def _extract_from_line(self, line_image: np.ndarray) -> list[np.ndarray]:
        patches = self.patch_extractor.extract_horizontal(line_image)
        return patches
    
    def _process_test_area(self) -> list[np.ndarray]:
        """
        Process test area and return patches directly (no file saving).
        
        Returns:
            List of patch arrays
        """
        # Extract raw segment
        raw_segment = self.data.image[
            self.data.test_area_top:self.data.test_area_bottom, :
        ]
        
        # Normalize height to match patch_size[0]
        segment = self.line_processor.extract_segment(
            self.data.image,
            self.data.test_area_top,
            self.data.test_area_bottom,
            target_height=self.patch_size[0]
        )
        
        patches = self._extract_from_line(segment)
        return patches
    
    def _process_lines(self) -> dict[int, list[np.ndarray]]:
        """
        Process all text lines and return patches directly (no file saving).
        
        Returns:
            Mapping of line_idx -> list of patch arrays
        """
        mapping: dict[int, list[np.ndarray]] = {}
        line_idx = 0

        for (start_y, end_y) in self.data.lines:
            segment = self.line_processor.extract_segment(
                self.data.image,
                start_y,
                end_y,
                target_height=self.patch_size[0]
            )
            
            patches = self._extract_from_line(segment)
            if len(patches) > 0:
                mapping[line_idx] = patches
                line_idx += 1

        return mapping
    
    def _build_partition(self) -> dict[str, list[np.ndarray]]:
        """
        Build partition using LINE-level split.
        Test line is fixed by protocol.
        Train/validation are split only by number of lines, not patches.
        """
        test_patches = self._process_test_area()
        line_map = self._process_lines()

        partition = { 'train': [], 'validation': [], 'test': test_patches }
        items = list(line_map.items())
        num_lines = len(items)

        if num_lines == 0:
            return partition
        if num_lines == 1:
            partition['train'].extend(items[0][1])
            return partition
        
        if num_lines == 2:
            partition['train'].extend(items[0][1])
            partition['validation'].extend(items[1][1])
            return partition
        
        rng = random.Random(self.seed)
        rng.shuffle(items)

        val_lines_count = max(1, int(round(num_lines * self.val_size)))

        val_items = items[:val_lines_count]
        train_items = items[val_lines_count:]

        for _, patches in val_items:
            partition['validation'].extend(patches)

        for _, patches in train_items:
            partition['train'].extend(patches)

        return partition
    
    def generate(self) -> tuple[dict[str, list[np.ndarray]], list[int]]:
        """
        Generate patches and labels for this author.
        
        OPTIMIZED: Returns patches directly in memory instead of saving to disk.
        This is much faster and simpler - just save the whole partition dict.
        
        Returns:
            tuple:
                - partition: dict with 'train', 'validation', 'test' keys, 
                  each containing list of patch arrays
                - labels: list of label_id for each patch (same length as total patches)
        """
        partition = self._build_partition()
        
        # Create labels list (one label per patch)
        total_patches = (
            len(partition['train']) + 
            len(partition['validation']) + 
            len(partition['test'])
        )
        labels = [self.label_id] * total_patches
        
        return partition, labels
