from typing import Literal

from ..embedding_network import EmbeddingNetwork
from .base_siamese_builder import BaseSiameseBuilder
from .siamese_pair_builder import SiamesePairBuilder
from .siamese_triplet_builder import SiameseTripletBuilder


class SiameseFactory:

    REGISTRY: dict[str, type[BaseSiameseBuilder]] = {
        "pair": SiamesePairBuilder,
        "triplet": SiameseTripletBuilder,
    }

    @classmethod
    def create(
        cls,
        model_type: Literal["pair", "triplet"],
        input_shape: tuple[int, int, int],
        embedding_network: EmbeddingNetwork,
    ) -> BaseSiameseBuilder:
        if model_type not in cls.REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_builder = cls.REGISTRY[model_type]

        return model_builder(
            input_shape=input_shape,
            embedding_network=embedding_network,
            name=f"siamese_{model_type}_model",
        )
