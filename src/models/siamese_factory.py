from typing import Literal

from .base_metric_model import BaseMetricModel
from .siamese_pair_model import SiamesePairModel
from .siamese_triplet_model import SiameseTripletModel


class SiameseFactory:

    REGISTRY: dict[str, type[BaseMetricModel]] = {
        "pair": SiamesePairModel,
        "triplet": SiameseTripletModel,
    }

    @classmethod
    def create(
        cls,
        model_type: Literal["pair", "triplet"],
        input_shape: tuple[int, int, int],
    ) -> BaseMetricModel:
        if model_type not in cls.REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = cls.REGISTRY[model_type]

        return model_class(
            input_shape=input_shape,
            name=f"siamese_{model_type}_model",
        )
