import numpy as np

from .gallery_builder import GalleryBuilder


class MeanGalleryBuilder(GalleryBuilder):

    def build(
        self,
        embeddings: np.ndarray,
        author_ids: np.ndarray,
        gallery_ratio: float,
        min_probe: int,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        gallery = {}
        probe_idx = []

        for author_id in np.unique(author_ids):
            idx = np.where(author_ids == author_id)[0]
            if len(idx) < min_probe + 1:
                continue

            rng.shuffle(idx)
            n_gallery = max(1, int(len(idx) * gallery_ratio))
            n_gallery = min(n_gallery, len(idx) - min_probe)

            gallery_idx = idx[:n_gallery]
            probe_idx.extend(idx[n_gallery:])

            gallery[author_id] = embeddings[gallery_idx].mean(axis=0)

        return gallery, embeddings[probe_idx], author_ids[probe_idx]
