from pathlib import Path
from typing import Literal

import tensorflow as tf

from .base_verification_evaluator import BaseVerificationEvaluator
from .pair_verification_evaluator import PairVerificationEvaluator
from .thresholding.base import ThresholdFinder
from .triplet_verification_evaluator import TripletVerificationEvaluator


class VerificationEvaluatorFactory:

    @staticmethod
    def create(
        model: tf.keras.Model,
        model_name: str,
        model_type: Literal["pair", "triplet"] = "pair",
        threshold_finder: ThresholdFinder | None = None,
        log_dir: Path | str | None = None,
    ) -> BaseVerificationEvaluator:
        if model_type == "pair":
            return PairVerificationEvaluator(
                model=model,
                model_name=model_name,
                threshold_finder=threshold_finder,
                log_dir=log_dir,
            )
        elif model_type == "triplet":
            return TripletVerificationEvaluator(
                model=model,
                model_name=model_name,
                threshold_finder=threshold_finder,
                log_dir=log_dir,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
