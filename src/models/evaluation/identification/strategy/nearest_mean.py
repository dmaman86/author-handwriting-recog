import numpy as np

from .base import IdentificationStrategy


class NearestMeanStrategy(IdentificationStrategy):

    def __init__(self, rejection_threshold: float | None = None) -> None:
        self.rejection_threshold = rejection_threshold

    def identify(
        self, probe_embeddings: np.ndarray, gallery: dict[int, np.ndarray]
    ) -> dict[str, any]:
        gallery_ids = np.array(list(gallery.keys()))
        gallery_matrix = np.stack(list(gallery.values()))

        distances = np.linalg.norm(
            probe_embeddings[:, np.newaxis, :] - gallery_matrix[np.newaxis, :, :],
            axis=2,
        )

        mean_distances = distances.mean(axis=0)
        best_idx = int(np.argmin(mean_distances))
        min_distance = float(mean_distances[best_idx])
        scores = dict(zip(gallery_ids.tolist(), mean_distances.tolist()))

        if (
            self.rejection_threshold is not None
            and min_distance > self.rejection_threshold
        ):
            return {
                "pred_author_id": None,
                "scores": scores,
                "min_distance": min_distance,
                "confidence": 0.0,
            }

        max_distance = mean_distances.max()
        if max_distance > 0:
            confidence = 1.0 - (min_distance / max_distance)
        else:
            confidence = 1.0

        return {
            "pred_author_id": int(gallery_ids[best_idx]),
            "scores": scores,
            "min_distance": min_distance,
            "confidence": float(confidence),
        }
