from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...io import ImageTransformer, PatchExtractor
from ..loaders import AuthorInfo


@dataclass(frozen=True)
class PatchDataset:
    images: np.ndarray  # shape: (N, H, W)
    labels: np.ndarray  # shape: (N,)


@dataclass(frozen=True)
class AuthorDataset:
    test: PatchDataset
    non_test: PatchDataset


class AuthorDatasetBuilder:

    def __init__(
        self,
        images_dir: str | Path,
        transformer: ImageTransformer,
        patch_extractor: PatchExtractor,
        load_fn: Callable[[Path], dict],
    ) -> None:
        self.images_dir = images_dir
        self.transformer = transformer
        self.patch_extractor = patch_extractor
        self.load_fn = load_fn

    def _load_author_image(self, author_id: str) -> np.ndarray:
        image_path = self.images_dir / f"{author_id}.jpg"
        raw_image = self.load_fn(image_path)

        return self.transformer.binarize_and_invert(raw_image.copy())

    def _build_test_data(self, image: np.ndarray, info: AuthorInfo) -> list[np.ndarray]:

        test_top = info.test_area_top
        test_bottom = info.test_area_bottom
        segment = image[test_top:test_bottom, :]
        return self.patch_extractor.get_filtered_patches(segment)

    def _clean_image(self, image: np.ndarray, info: AuthorInfo) -> np.ndarray:
        img = image.copy()

        test_top = info.test_area_top
        test_bottom = info.test_area_bottom

        img[test_top:test_bottom, :] = 0
        return img

    def _build_non_test_data(
        self, image: np.ndarray, info: AuthorInfo
    ) -> list[np.ndarray]:

        img = self._clean_image(image, info)
        patches: list[np.ndarray] = []
        half = info.line_height // 2

        for start, end in info.lines:
            center = (start + end) // 2
            offset = int(info.line_height * 0.15)
            center += offset
            new_start = center - half
            new_end = center + half
            line = img[new_start:new_end, :]
            patches.extend(self.patch_extractor.get_filtered_patches(line))
        return patches

    def _to_dataset(self, patches: list[np.ndarray], label_id: int) -> PatchDataset:

        images = np.stack(patches, axis=0).astype(np.uint8)
        labels = np.full((images.shape[0],), label_id, dtype=np.int32)

        return PatchDataset(images=images, labels=labels)

    def build(self, info: AuthorInfo, label_id: int) -> AuthorDataset:
        image = self._load_author_image(info.author_id)

        test_patches = self._build_test_data(image, info)
        non_test_patches = self._build_non_test_data(image, info)

        return AuthorDataset(
            test=self._to_dataset(test_patches, label_id),
            non_test=self._to_dataset(non_test_patches, label_id),
        )
