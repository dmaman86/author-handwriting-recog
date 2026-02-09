"""
Comparator for multiple Siamese models.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ...io.logging import LoggerFactory
from .model_evaluator import ModelEvaluator, ThresholdStrategy


class ModelComparator:
    """
    Compare multiple Siamese models on the same test set.
    Uses ModelEvaluator internally for consistent evaluation logic.
    """

    def __init__(
        self,
        evaluators_dict: dict[str, ModelEvaluator],
        log_dir: Path | str | None = None,
    ):
        """
        Initialize comparator.

        Args:
            evaluators_dict: {'model_name': ModelEvaluator_instance, ...}
            log_dir: Directory for logs
        """
        self.evaluators = evaluators_dict
        self.results: dict[str, dict[str, Any]] = {}

        self.logger = LoggerFactory.get_logger(
            name="model_comparator", log_dir=log_dir, file_prefix="comparison"
        )

        self.logger.info(f"Comparator initialized with {len(evaluators_dict)} models")

    def evaluate_all(
        self,
        test_generator: tf.keras.utils.Sequence,
        distance_threshold: float | None = None,
        threshold_strategy: ThresholdStrategy = "youden",
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate all models on the same generator using pair-level evaluation.

        Args:
            test_generator: Test data generator (PairGenerator)
            distance_threshold: Threshold for binary classification.
                If None, each model finds its own optimal threshold
                using ``threshold_strategy``.
            threshold_strategy: Strategy for automatic threshold search.
                See ``ModelEvaluator.evaluate`` for details.

        Returns:
            Nested dictionary: {model_name: {metric: value, ...}, ...}
        """

        # Evaluate each model using its evaluator
        for name, evaluator in self.evaluators.items():
            self.logger.info(f"\nEvaluating: {name}")
            results = evaluator.evaluate(
                test_generator=test_generator,
                distance_threshold=distance_threshold,
                threshold_strategy=threshold_strategy,
            )
            self.results[name] = results

        self._print_summary()
        return self.results

    def evaluate_all_by_voting(
        self,
        test_data: dict[int, list[np.ndarray]],
        distance_threshold: float = 1.0,
        min_votes: int = 3,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate all models using voting at author level.

        Args:
            test_data: Dictionary {author_id: [sample1, sample2, ...]}
            distance_threshold: Threshold for binary classification
            min_votes: Minimum number of votes required for decision

        Returns:
            Nested dictionary: {model_name: {metric: value, ...}, ...}
        """

        # Evaluate each model using its evaluator
        for name, evaluator in self.evaluators.items():
            self.logger.info(f"\nEvaluating: {name}")
            results = evaluator.evaluate_by_voting(
                test_data=test_data,
                distance_threshold=distance_threshold,
                min_votes=min_votes,
            )
            self.results[name] = results

        self._print_summary()
        return self.results

    def get_best_model(self, metric: str = "f1") -> tuple[str, dict[str, float]]:
        """
        Get best model by metric.

        Args:
            metric: Metric to compare ('f1', 'accuracy', 'auc', etc.)

        Returns:
            (model_name, metrics_dict)
        """
        if not self.results:
            raise ValueError("Run evaluate_all() first")

        best = max(self.results.items(), key=lambda x: x[1][metric])
        self.logger.info(f"Best model by {metric}: {best[0]} = {best[1][metric]:.4f}")
        return best

    def _print_summary(self):
        """Print final summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("=" * 60)

        for name, metrics in self.results.items():
            self.logger.info(f"\n{name}")
            self.logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
            if "auc" in metrics:
                self.logger.info(f"   AUC:       {metrics['auc']:.4f}")
            self.logger.info(f"   Precision: {metrics['precision']:.4f}")
            self.logger.info(f"   Recall:    {metrics['recall']:.4f}")
            self.logger.info(f"   F1-Score:  {metrics['f1']:.4f}")

        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]["f1"])
        self.logger.info(f"\nBest model (by F1-Score): {best_model[0]}")
        self.logger.info("=" * 60)
