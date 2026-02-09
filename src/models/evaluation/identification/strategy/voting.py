import numpy as np

from .base import IdentificationStrategy


class VotingStrategy(IdentificationStrategy):

    def __init__(self, distance_threshold: float) -> None:
        self.distance_threshold = distance_threshold

    def identify(
        self, probe_embeddings: np.ndarray, gallery: dict[int, np.ndarray]
    ) -> dict[str, any]:
        votes: dict[int, int] = {}

        for author_id, ref_embeddings in gallery.items():
            distances = np.linalg.norm(
                probe_embeddings[:, None, :] - ref_embeddings[None, :, :], axis=2
            )
            votes[author_id] = int((distances < self.distance_threshold).sum())

        best_author = max(votes, key=votes.get)
        max_votes = votes[best_author]

        if max_votes == 0:
            return {"pred_author_id": None, "scores": votes, "confidence": 0.0}

        total_possible_votes = len(probe_embeddings) * len(gallery)
        confidence = (
            max_votes / total_possible_votes if total_possible_votes > 0 else 0.0
        )

        return {
            "pred_author_id": int(best_author),
            "scores": votes,
            "confidence": float(confidence),
        }
