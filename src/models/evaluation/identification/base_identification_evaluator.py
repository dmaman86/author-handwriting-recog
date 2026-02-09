from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from ....io.logging import LoggerFactory
from .gallery.gallery_builder import GalleryBuilder
from .strategy.base import IdentificationStrategy


class BaseIdentificationEvaluator(ABC):
    """
    Closed-set writer identification evaluator based on embeddings.
    """

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        strategy: IdentificationStrategy,
        gallery_builder: GalleryBuilder,
        gallery_ratio: float = 0.5,
        min_probe: int = 2,
        seed: int = 42,
        log_dir: str | None = None,
        model_name: str = "embedding_model",
    ) -> None:
        self.embedding_model = embedding_model
        self.strategy = strategy
        self.gallery_builder = gallery_builder
        self.gallery_ratio = gallery_ratio
        self.min_probe = min_probe
        self.seed = seed

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_identification",
            log_dir=log_dir,
            file_prefix=f"{model_name}_identification",
        )

    def evaluate(self, dataset: tf.data.Dataset, steps: int) -> dict:
        if steps <= 0:
            raise ValueError(f"Steps must be > 0, got {steps}")

        self.logger.info("Starting identification evaluation")
        self.logger.info(f"  Strategy: {self.strategy.__class__.__name__}")
        self.logger.info(f"  Gallery ratio: {self.gallery_ratio}")
        self.logger.info(f"  Min probe samples: {self.min_probe}")

        embeddings, author_ids = self._extract_embeddings(dataset, steps)

        gallery, probe_embeddings, probe_author_ids = self.gallery_builder.build(
            embeddings=embeddings,
            author_ids=author_ids,
            gallery_ratio=self.gallery_ratio,
            seed=self.seed,
            min_probe=self.min_probe,
        )

        return self._evaluate_identification(
            gallery, probe_embeddings, probe_author_ids
        )

    @abstractmethod
    def _evaluate_identification(
        self,
        gallery: dict[int, np.ndarray],
        probe_embeddings: np.ndarray,
        probe_author_ids: np.ndarray,
    ) -> dict:
        pass

    @abstractmethod
    def _get_mode_name(self) -> str:
        pass

    def _extract_embeddings(
        self, dataset: tf.data.Dataset, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info(f"Extracting embeddings from {steps} batches...")

        all_embeddings, all_author_ids = [], []

        for X, y in dataset.take(steps):
            emb = self.embedding_model(X, training=False)
            all_embeddings.append(emb.numpy())

            if isinstance(y, np.ndarray):
                all_author_ids.append(y)
            else:
                all_author_ids.append(y.numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        author_ids = np.concatenate(all_author_ids, axis=0)

        self.logger.info(
            f"Extracted {len(embeddings)} embeddings "
            f"from {len(np.unique(author_ids))} unique authors"
        )

        return embeddings, author_ids
