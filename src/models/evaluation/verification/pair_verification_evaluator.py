import numpy as np

from .base_verification_evaluator import BaseVerificationEvaluator


class PairVerificationEvaluator(BaseVerificationEvaluator):

    def _evaluate_dataset(self, dataset) -> tuple[list[np.ndarray], list[np.ndarray]]:
        distances_all, y_true_all = [], []

        for (img1, img2), labels in dataset:
            embeddings = self.model.predict([img1, img2], verbose=0)
            emb1 = embeddings[:, 0, :]
            emb2 = embeddings[:, 1, :]
            distances = np.linalg.norm(emb1 - emb2, axis=1)

            distances_all.append(distances)
            y_true_all.append(
                labels if isinstance(labels, np.ndarray) else labels.numpy()
            )

        return distances_all, y_true_all
