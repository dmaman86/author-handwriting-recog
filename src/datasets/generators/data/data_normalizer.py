import numpy as np


class PatchNormalizer:

    @staticmethod
    def normalize_partition(
        data: dict[int, list[np.ndarray]],
    ) -> dict[int, list[np.ndarray]]:
        normalized_data = {}

        for author_id, patches in data.items():
            normalized_data[author_id] = PatchNormalizer.normalize_batch(patches)

        return normalized_data

    @staticmethod
    def normalize_batch(patches: list[np.ndarray]) -> list[np.ndarray]:
        normalize = []

        for patch in patches:
            normalize.append(PatchNormalizer.normalize_single(patch))

        return normalize

    @staticmethod
    def normalize_single(patch: np.ndarray) -> np.ndarray:
        patch = patch.astype(np.float32)

        if patch.max() > 1.0:
            patch = patch / 255.0

        if len(patch.shape) == 2:
            patch = np.expand_dims(patch, axis=-1)

        return patch
