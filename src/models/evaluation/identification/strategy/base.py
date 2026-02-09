from abc import ABC, abstractmethod

import numpy as np


class IdentificationStrategy(ABC):

    @abstractmethod
    def identify(
        self,
        probe_embeddings: np.ndarray,
        gallery: dict[int, np.ndarray],
    ) -> dict[str, any]:
        """
        Return:
        {
            "pred_author_id": int | None,
            "scores": dict[int, float],
        }
        """
        pass
