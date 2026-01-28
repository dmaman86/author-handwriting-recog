import numpy as np


class PatchNormalizer:

    def normalize_partition(
        self,
        data: dict[int, list[np.ndarray]],
    ) -> dict[int, list[np.ndarray]]:
        normalized_data = {}

        for author_id, patches in data.items():
            normalized_data[author_id] = self.normalize_batch(patches)

        return normalized_data

    def normalize_batch(self, patches: list[np.ndarray]) -> list[np.ndarray]:
        normalize = []

        for patch in patches:
            normalize.append(self.normalize_single(patch))

        return normalize

    def normalize_single(self, patch: np.ndarray) -> np.ndarray:
        patch = patch.astype(np.float32)

        if patch.max() > 1.0:
            patch = patch / 255.0

        if len(patch.shape) == 2:
            patch = np.expand_dims(patch, axis=-1)

        return patch
