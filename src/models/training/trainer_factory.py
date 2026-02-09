from pathlib import Path
from typing import Literal

from .base_model_trainer import BaseModelTrainer
from .pair_model_trainer import PairModelTrainer
from .triplet_model_trainer import TripletModelTrainer


class ModelTrainerFactory:

    @staticmethod
    def create(
        backbone,
        input_shape: tuple[int, int, int] = (60, 53, 1),
        embedding_dim: int = 256,
        model_name: str = "model",
        model_type: Literal["pair", "triplet"] = "pair",
        log_dir: Path | str | None = None,
    ) -> BaseModelTrainer:
        if model_type == "pair":
            return PairModelTrainer(
                backbone=backbone,
                input_shape=input_shape,
                embedding_dim=embedding_dim,
                model_name=model_name,
                log_dir=log_dir,
            )
        elif model_type == "triplet":
            return TripletModelTrainer(
                backbone=backbone,
                input_shape=input_shape,
                embedding_dim=embedding_dim,
                model_name=model_name,
                log_dir=log_dir,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
