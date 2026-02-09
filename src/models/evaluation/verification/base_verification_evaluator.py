from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from ....io.logging import LoggerFactory
from .thresholding import (ThresholdFinder, ThresholdFinderFactory,
                           ThresholdStrategy)

GeneratorType = tf.data.Dataset | tf.keras.utils.Sequence


class BaseVerificationEvaluator(ABC):

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str,
        threshold_finder: ThresholdFinder | None = None,
        log_dir: Path | str | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.threshold_finder = threshold_finder

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_evaluator",
            log_dir=log_dir,
            file_prefix=f"{model_name}_eval",
        )

    def evaluate(
        self,
        generator: GeneratorType,
        steps: int | None = None,
        distance_threshold: float | None = None,
        threshold_strategy: ThresholdStrategy = "youden",
    ) -> dict:
        distances, y_true = self._compute_distances(generator, steps)

        if distance_threshold is None:
            finder = (
                self.threshold_finder
                if self.threshold_finder
                else ThresholdFinderFactory.create(threshold_strategy)
            )
            distance_threshold = finder.find(y_true, distances)

        results = self._compute_metrics(y_true, distances, distance_threshold)
        self._log_results(results)
        return results

    def _compute_distances(self, generator: GeneratorType, steps: int | None):
        distances, y_true = [], []

        dataset = self._get_dataset(generator, steps)
        distances, y_true = self._evaluate_dataset(dataset)

        return np.concatenate(distances), np.concatenate(y_true)

    def _get_dataset(self, generator: GeneratorType, steps: int | None):
        if isinstance(generator, tf.data.Dataset):
            if steps is None:
                card = tf.data.experimental.cardinality(generator).numpy()
                if card > 0:
                    steps = card
                else:
                    raise ValueError("Steps must be provided for infinite datasets")
            return generator.take(steps)
        return (generator[i] for i in range(len(generator)))

    @abstractmethod
    def _evaluate_dataset(self, dataset) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pass

    def _compute_metrics(self, y_true, distances, threshold):
        y_pred = (distances < threshold).astype(int)
        scores = -distances

        return {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[0],
            "recall": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[1],
            "f1": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[2],
            "auc": roc_auc_score(y_true, scores),
            "distances": distances,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def _log_results(self, results: dict[str, any]) -> None:
        self.logger.info("Verification evaluation completed")
        self.logger.info(f"  Threshold: {results['threshold']:.4f}")
        self.logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        self.logger.info(f"  AUC:       {results['auc']:.4f}")
        self.logger.info(f"  Precision: {results['precision']:.4f}")
        self.logger.info(f"  Recall:    {results['recall']:.4f}")
        self.logger.info(f"  F1-score:  {results['f1']:.4f}")
