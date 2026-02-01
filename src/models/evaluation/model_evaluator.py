"""
Evaluator for individual and ensemble Siamese models.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from ...io.logging import LoggerFactory


class ModelEvaluator:
    """
    Evaluate individual Siamese model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str,
        embedding_model: tf.keras.Model | None = None,
        log_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            model: Trained Siamese model
            model_name: Model name for logging
            log_dir: Directory for logs
        """
        self.model = model
        self.embedding_model = embedding_model
        self.model_name = model_name

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_evaluator",
            log_dir=log_dir,
            file_prefix=f"{model_name}_eval",
        )

        self.results: dict[str, Any] = {}

    def evaluate(
        self,
        test_generator: tf.keras.utils.Sequence,
        distance_threshold: float = 1.0,
    ) -> dict[str, float]:
        """
        Evaluate model on test generator.

        Args:
            test_generator: Test data generator (PairGenerator)
            distance_threshold: Threshold for binary classification

        Returns:
            Dictionary with metrics
        """
        self.logger.info(f"Evaluating {self.model_name}...")

        # Extract data from generator
        y_true_all = []
        embeddings_all = []

        for i in range(len(test_generator)):
            (img1, img2), labels = test_generator[i]
            embeddings = self.model.predict([img1, img2], verbose=0)
            embeddings_all.append(embeddings)
            y_true_all.append(labels)

        y_true = np.concatenate(y_true_all, axis=0)
        embeddings = np.concatenate(embeddings_all, axis=0)

        # Calculate distances
        emb1 = embeddings[:, 0, :]
        emb2 = embeddings[:, 1, :]
        distances = np.linalg.norm(emb1 - emb2, axis=1)

        # Binary predictions
        y_pred = (distances < distance_threshold).astype(int)

        # Probabilities (convert distance to similarity)
        y_pred_proba = 1 / (1 + distances)

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0.0
            self.logger.warning("Could not calculate AUC")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # Store results
        self.results["pair_patches"] = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "distances": distances,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"  - Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  - AUC:       {auc:.4f}")
        self.logger.info(f"  - Precision: {precision:.4f}")
        self.logger.info(f"  - Recall:    {recall:.4f}")
        self.logger.info(f"  - F1-Score:  {f1:.4f}")

        return self.results["pair_patches"]

    def evaluate_by_voting(
        self,
        test_data: dict[int, list[np.ndarray]],
        distance_threshold: float = 1.0,
        min_votes: int = 3,
    ) -> dict[str, float]:
        """
        Evaluate model using majority voting at author level.

        For each author, compare all pairs of samples and use majority voting
        to determine if samples belong to the same author.

        Args:
            test_data: Dictionary {author_id: [sample1, sample2, ...]}
            distance_threshold: Threshold for binary classification
            min_votes: Minimum number of votes required for decision

        Returns:
            Dictionary with metrics

        Example:
            Given author A with 5 samples:
            - Compare all pairs: (s1,s2), (s1,s3), ..., (s4,s5) = 10 pairs
            - Each pair votes: same author (1) or different (0)
            - If majority votes "same author" → predict author A
        """
        self.logger.info(f"Evaluating {self.model_name} by voting...")
        self.logger.info(f"  - Distance threshold: {distance_threshold}")
        self.logger.info(f"  - Minimum votes: {min_votes}")

        y_true_authors = []
        y_pred_authors = []
        voting_details = []

        if self.embedding_model is None:
            raise ValueError("Embedding model is required for voting evaluation")

        # Evaluate each author
        for author_id, samples in test_data.items():
            if len(samples) < 2:
                self.logger.warning(
                    f"Author {author_id} has only {len(samples)} samples, skipping"
                )
                continue

            num_pairs = len(samples) * (len(samples) - 1) // 2
            if num_pairs < min_votes:
                self.logger.warning(
                    f"Author {author_id} has only {num_pairs} pairs (< {min_votes}), skipping"
                )
                continue

            samples_array = np.array(samples)  # (num_samples, H, W, C)
            embeddings = self.embedding_model.predict(samples_array, verbose=0)

            votes_same_author = 0
            votes_different_author = 0

            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    distance = np.linalg.norm(embeddings[i] - embeddings[j])
                    if distance < distance_threshold:
                        votes_same_author += 1
                    else:
                        votes_different_author += 1

            total_votes = votes_same_author + votes_different_author
            prediction = 1 if votes_same_author > votes_different_author else 0

            y_true_authors.append(1)  # Ground truth: all samples are from same author
            y_pred_authors.append(prediction)

            voting_details.append(
                {
                    "author_id": author_id,
                    "num_samples": len(samples),
                    "num_pairs": num_pairs,
                    "votes_same": votes_same_author,
                    "votes_different": votes_different_author,
                    "prediction": prediction,
                    "confidence": votes_same_author / total_votes,
                }
            )

        y_true_authors = np.array(y_true_authors)
        y_pred_authors = np.array(y_pred_authors)

        accuracy = accuracy_score(y_true_authors, y_pred_authors)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_authors, y_pred_authors, average="binary", zero_division=0
        )

        self.results["voting"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_true": y_true_authors,
            "y_pred": y_pred_authors,
            "voting_details": voting_details,
            "num_authors_evaluated": len(y_true_authors),
        }

        self.logger.info(f"Voting evaluation completed:")
        self.logger.info(f"  - Number of authors evaluated: {len(y_true_authors)}")
        self.logger.info(f"  - Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  - Precision: {precision:.4f}")
        self.logger.info(f"  - Recall:    {recall:.4f}")
        self.logger.info(f"  - F1-Score:  {f1:.4f}")

        correct_predictions = sum(1 for d in voting_details if d["prediction"] == 1)
        avg_confidence = np.mean([d["confidence"] for d in voting_details])
        self.logger.info(
            f"  - Correct predictions: {correct_predictions}/{len(voting_details)}"
        )
        self.logger.info(f"  - Average confidence: {avg_confidence:.4f}")

        return self.results["voting"]
