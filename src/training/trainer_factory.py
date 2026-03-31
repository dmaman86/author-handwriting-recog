from dataclasses import dataclass
from typing import Literal

from ..generators import BaseGenerator, PairGenerator, TripletGenerator
from ..models import BaseSiameseBuilder
from .base_metric_trainer import BaseMetricTrainer
from .pair_trainer import PairTrainer
from .triplet_trainer import TripletTrainer


@dataclass
class TrainerSpec:
    trainer_class: type[BaseMetricTrainer]
    generator_type: type[BaseGenerator]

    def validate_generator(self, name: str, generator: BaseGenerator) -> None:
        """
        Ensure the generator matches the expected type for this trainer.
        """
        if not isinstance(generator, self.generator_type):
            raise TypeError(
                f"{name} must be {self.generator_type.__name__}, "
                f"got {type(generator).__name__}"
            )


class ModelTrainerFactory:

    REGISTERED_TRAINERS: dict[str, TrainerSpec] = {
        "pair": TrainerSpec(PairTrainer, PairGenerator),
        "triplet": TrainerSpec(TripletTrainer, TripletGenerator),
    }

    @classmethod
    def create(
        cls,
        siamese_builder: BaseSiameseBuilder,
        train_generator: BaseGenerator,
        val_generator: BaseGenerator,
        model_name: str = "metric_model",
        model_type: Literal["pair", "triplet"] = "pair",
    ) -> BaseMetricTrainer:

        spec = cls.REGISTERED_TRAINERS[model_type]

        if spec is None:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Available: {list(cls.REGISTERED_TRAINERS.keys())}"
            )

        spec.validate_generator("train_generator", train_generator)
        spec.validate_generator("val_generator", val_generator)

        return spec.trainer_class(
            siamese_builder=siamese_builder,
            train_generator=train_generator,
            val_generator=val_generator,
            name=model_name,
        )
