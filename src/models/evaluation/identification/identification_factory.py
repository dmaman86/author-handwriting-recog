from typing import Literal

import tensorflow as tf

from .base_identification_evaluator import BaseIdentificationEvaluator
from .gallery.gallery_builder import GalleryBuilder
from .seen_identification_evaluator import SeenIdentificationEvaluator
from .strategy.base import IdentificationStrategy
from .unseen_identification_evaluator import UnseenIdentificationEvaluator


class IdentificationFactory:

    @staticmethod
    def create(
        embedding_model: tf.keras.Model,
        strategy: IdentificationStrategy,
        gallery_builder: GalleryBuilder,
        mode: Literal["open-set", "closed-set"] = "closed-set",
        gallery_ratio: float = 0.5,
        min_probe: int = 2,
        seed: int = 42,
        log_dir: str | None = None,
        model_name: str = "embedding_model",
    ) -> BaseIdentificationEvaluator:
        if mode == "closed-set":
            return SeenIdentificationEvaluator(
                embedding_model=embedding_model,
                strategy=strategy,
                gallery_builder=gallery_builder,
                gallery_ratio=gallery_ratio,
                min_probe=min_probe,
                seed=seed,
                log_dir=log_dir,
                model_name=model_name,
            )
        elif mode == "open-set":
            return UnseenIdentificationEvaluator(
                embedding_model=embedding_model,
                strategy=strategy,
                gallery_builder=gallery_builder,
                gallery_ratio=gallery_ratio,
                min_probe=min_probe,
                seed=seed,
                log_dir=log_dir,
                model_name=model_name,
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'open-set' or 'closed-set'."
            )
