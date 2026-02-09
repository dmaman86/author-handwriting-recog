import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .base import ThresholdFinder


class F1ThresholdFinder(ThresholdFinder):
    def __init__(self, n_candidates: int = 200) -> None:
        self.n_candidates = n_candidates

    def find(self, y_true: np.ndarray, distances: np.ndarray) -> float:
        candidates = np.linspace(distances.min(), distances.max(), self.n_candidates)
        best_t, best_f1 = candidates[0], -1

        for t in candidates:
            y_pred = (distances < t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t

        return best_t
