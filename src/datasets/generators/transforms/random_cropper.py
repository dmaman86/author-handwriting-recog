import random

import numpy as np


class RandomCropper:

    def __init__(
        self, scale: tuple[float, float] = (0.8, 1.0), seed: int | None = None
    ) -> None:
        self.scale = scale
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def crop(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            h, w = image.shape
            top, left, crop_h, crop_w = self._calculate_crop_params(h, w)
            return image[top : top + crop_h, left : left + crop_w]
        elif len(image.shape) == 3:
            h, w, _ = image.shape
            top, left, crop_h, crop_w = self._calculate_crop_params(h, w)
            return image[top : top + crop_h, left : left + crop_w, :]
        else:
            return image

    def _calculate_crop_params(self, h: int, w: int) -> tuple[int, int, int, int]:
        target_area = h * w * random.uniform(*self.scale)

        aspect_ratio = random.uniform(0.9, 1.1)

        crop_h = int(np.sqrt(target_area * aspect_ratio))
        crop_w = int(np.sqrt(target_area / aspect_ratio))

        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0

        return top, left, crop_h, crop_w
