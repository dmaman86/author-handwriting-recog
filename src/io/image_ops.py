import cv2
import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view


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

        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted_img = cv2.bitwise_not(binary_img)
        return inverted_img

    def resize(
        self,
        img: np.ndarray,
        target_size: tuple[int, int],
        method: str = "bilinear",
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
        black_value: int = 10,
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
        min_density: float = 0.05,
    ) -> bool:
        density = self.calculate_ink_density(patch)
        return density >= min_density


class PatchExtractor:
    """
    Extracts sliding window patches from images for segmentation tasks.
    """

    def __init__(
        self,
        image_analyzer: ImageAnalyzer,
        transformer: ImageTransformer,
        output_patch_dim: tuple[int, int] = (180, 180),  # (H, W)
        stride_dim: tuple[int, int] = (30, 30),  # (stride_y, stride_x)
        empty_threshold: float = 0.90,
    ) -> None:
        self.output_patch_height, self.output_patch_width = output_patch_dim
        self.stride_y, self.stride_x = stride_dim

        self.empty_threshold = empty_threshold
        self.min_density = 1 - empty_threshold

        self.image_analyzer = image_analyzer
        self.transformer = transformer

    def _get_patches(
        self,
        segment: np.ndarray,
    ) -> list[np.ndarray]:
        H, _ = segment.shape[:2]
        window_shape = (H, self.output_patch_width)
        windows = sliding_window_view(segment, window_shape)

        strided_windows = windows[:: self.stride_y, :: self.stride_x]

        patches = strided_windows.reshape(-1, H, self.output_patch_width)

        return list(patches)

    def _format_patch(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape[:2]
        if (h, w) == (
            self.output_patch_height,
            self.output_patch_width,
        ):
            return patch

        # tf.image.resize_with_pad expects 3D input,
        # so we add a channel dimension if needed
        patch_3d = patch[..., np.newaxis]

        formatted_patch = tf.image.resize_with_pad(
            patch_3d,
            self.output_patch_height,
            self.output_patch_width,
        ).numpy()

        # convert back to 2D if the original patch was 2D
        return np.squeeze(formatted_patch, axis=-1)

    def get_filtered_patches(
        self,
        segment: np.ndarray,
    ) -> list[np.ndarray]:
        patches = self._get_patches(segment)

        filtered_patches = []

        for patch in patches:
            if self.image_analyzer.has_sufficient_content(
                patch,
                min_density=self.min_density,
            ):
                formatted_patch = self._format_patch(patch)
                filtered_patches.append(formatted_patch)

        return filtered_patches
