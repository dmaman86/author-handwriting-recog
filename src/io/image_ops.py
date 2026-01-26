import cv2
import numpy as np
import tensorflow as tf
from typing import Callable, List, Tuple, Generator


class ImageTransformer:    
    def to_gray(self, img: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale.
        
        Args:
            img: Input image (H, W, 3) or already grayscale (H, W)
            
        Returns:
            Grayscale image (H, W)
        """
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    
    def binarize_and_invert(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's binarization and invert colors.
        
        Useful for handwriting where we want white text on black background.
        
        Args:
            img: Input image (grayscale or RGB)
            
        Returns:
            Inverted binary image (H, W)
        """
        if len(img.shape) == 3:
            img = self.to_gray(img)
        _, binary_img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        inverted_img = cv2.bitwise_not(binary_img)
        return inverted_img
    
    def resize(
        self,
        img: np.ndarray, 
        target_size: Tuple[int, int],
        method: str = 'bilinear'
    ) -> np.ndarray:

        return tf.image.resize(img, target_size, method).numpy()
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            img: Input image with values in [0, 255]
            
        Returns:
            Normalized image with float32 values in [0, 1]
        """
        return img.astype(np.float32) / 255.0
    
    def add_channel_dimension(self, img: np.ndarray) -> np.ndarray:
        """
        Add channel dimension to grayscale image.
        
        Converts (H, W) to (H, W, 1) for model compatibility.
        
        Args:
            img: Input image
            
        Returns:
            Image with channel dimension if needed
        """
        if img.ndim == 2:
            return img[..., np.newaxis]
        return img


class ImageAnalyzer:
    """
    Provides image analysis and quality assessment operations.
    """

    def is_mostly_black(
        self,
        patch: np.ndarray, 
        threshold: float = 0.95,
        black_value: int = 10
    ) -> bool:
        """
        Check if a patch is mostly black (empty/background).
        
        Useful for filtering out non-informative patches during extraction.
        
        Args:
            patch: Image patch to analyze
            threshold: Minimum ratio of black pixels (0.0 to 1.0)
            black_value: Pixel value threshold to consider as black
            
        Returns:
            True if the patch is mostly black, False otherwise
        """
        total_pixels = patch.size
        black_pixels = np.sum(patch < black_value)
        return (black_pixels / total_pixels) >= threshold
    
    def calculate_ink_density(self, patch: np.ndarray) -> float:
        """
        Calculate the density of ink/writing in a patch.
        
        Args:
            patch: Image patch (assumes white ink on black background)
            
        Returns:
            Ratio of non-black pixels (0.0 to 1.0)
        """
        total_pixels = patch.size
        non_black_pixels = np.sum(patch >= 10)
        return non_black_pixels / total_pixels
    
    def has_sufficient_content(
        self,
        patch: np.ndarray,
        min_density: float = 0.05
    ) -> bool:
        """
        Check if patch has sufficient writing content.
        
        Args:
            patch: Image patch to check
            min_density: Minimum ink density required
            
        Returns:
            True if patch has enough content
        """
        density = self.calculate_ink_density(patch)
        return density >= min_density


class PatchExtractor:
    """
    Extracts sliding window patches from images for segmentation tasks.
    """
    
    def __init__(
        self,
        image_analyzer: ImageAnalyzer,
        patch_width: int = 160,
        stride_x: int = 30,
        filter_empty: bool = True,
        empty_threshold: float = 0.95,
    ):
        """
        Initialize patch extractor with configuration.
        
        Args:
            patch_width: Width of each extracted patch
            stride_x: Horizontal step size between patches
            filter_empty: Whether to filter out mostly-black patches
            empty_threshold: Threshold for considering a patch as empty
        """
        self.patch_width = patch_width
        self.stride_x = stride_x
        self.filter_empty = filter_empty
        self.empty_threshold = empty_threshold
        self.image_analyzer = image_analyzer
    
    def _should_keep_patch(self, patch: np.ndarray) -> bool:
        """
        Determine if a patch should be kept based on filter criteria.
        
        Args:
            patch: Image patch to evaluate
            
        Returns:
            True if patch should be kept, False if it should be filtered out
        """
        if not self.filter_empty:
            return True
        
        return self.image_analyzer.has_sufficient_content(patch, 1 - self.empty_threshold)
    
    def _extract_patches_generator(
        self, 
        segment: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Core extraction logic that yields patches with their positions.
        
        This is the single source of truth for patch extraction logic.
        
        Args:
            segment: Line segment (H, W) or (H, W, C)
            
        Yields:
            Tuples of (patch, x_start, x_end) for valid patches
        """
        H, W = segment.shape[:2]
        
        # Only extract patches where we have enough width
        # This ensures all patches have exactly patch_width width
        if W < self.patch_width:
            # Segment is too narrow, cannot extract any patches
            return
        
        for x in range(0, W - self.patch_width + 1, self.stride_x):
            patch = segment[:, x:x + self.patch_width]
            
            # Sanity check: ensure patch has correct width
            if patch.shape[1] != self.patch_width:
                continue
            
            if self._should_keep_patch(patch):
                yield (patch, x, x + self.patch_width)
    
    def extract_horizontal(self, segment: np.ndarray) -> List[np.ndarray]:
        """
        Extract horizontal sliding patches across a text line segment.
        
        Patch height equals the segment height. Patches slide horizontally
        with the configured stride.
        
        Args:
            segment: Line segment (H, W) or (H, W, C)
            
        Returns:
            List of extracted patches that pass the filter criteria
        """
        return [patch for patch, _, _ in self._extract_patches_generator(segment)]
    
    def extract_with_positions(
        self, 
        segment: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Extract patches along with their positions in the original segment.
        
        Args:
            segment: Line segment (H, W) or (H, W, C)
            
        Returns:
            List of tuples (patch, x_start, x_end)
        """
        return list(self._extract_patches_generator(segment))