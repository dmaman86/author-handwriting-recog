import numpy as np

from .base_verification_evaluator import BaseVerificationEvaluator


class TripletVerificationEvaluator(BaseVerificationEvaluator):

    def _evaluate_dataset(self, dataset) -> tuple[list[np.ndarray], list[np.ndarray]]:
        distances_all, y_true_all = [], []

        for (anchor, positive, negative), _ in dataset:
            embeddings = self.model.predict([anchor, positive, negative], verbose=0)
            emb_anchor = embeddings[:, 0, :]
            emb_positive = embeddings[:, 1, :]
            emb_negative = embeddings[:, 2, :]

            pos_distances = np.sqrt(
                np.sum(np.square(emb_anchor - emb_positive), axis=1)
            )
            distances_all.append(pos_distances)
            y_true_all.append(np.ones(len(pos_distances)))

            neg_distances = np.sqrt(
                np.sum(np.square(emb_anchor - emb_negative), axis=1)
            )
            distances_all.append(neg_distances)
            y_true_all.append(np.zeros(len(neg_distances)))

        return distances_all, y_true_all
