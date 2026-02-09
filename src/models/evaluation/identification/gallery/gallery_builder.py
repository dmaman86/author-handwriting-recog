from abc import ABC, abstractmethod

import numpy as np


class GalleryBuilder(ABC):
    @abstractmethod
    def build(
        self,
        embeddings: np.ndarray,
        author_ids: np.ndarray,
        gallery_ratio: float,
        min_probe: int,
        seed: int,
    ) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
        """
        Build gallery and probe sets.
        Returns:
            gallery: Dict mapping author_id to their gallery embedding
            probe_embeddings: Embeddings of probe samples
            probe_author_ids: Author IDs corresponding to probe embeddings
        """
        pass
