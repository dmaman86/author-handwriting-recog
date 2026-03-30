import numpy as np


class EmbeddingGeometry:

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(norms, a_min=1e-8, a_max=None)

    @staticmethod
    def intra_inter_distance(
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        sample_size: int = 10000,
    ) -> tuple[float, float]:

        pairs = rng.integers(0, len(X), size=(sample_size, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

        i, j = pairs[:, 0], pairs[:, 1]

        dots = np.sum(X[i] * X[j], axis=1)
        dist = 1 - dots

        same = y[i] == y[j]

        intra = dist[same]
        inter = dist[~same]

        intra_mean = np.mean(intra) if intra.size > 0 else np.nan
        inter_mean = np.mean(inter) if inter.size > 0 else np.nan

        return intra_mean, inter_mean
