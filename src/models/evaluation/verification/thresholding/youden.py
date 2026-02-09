import numpy as np
from sklearn.metrics import roc_curve

from .base import ThresholdFinder


class YoudenThresholdFinder(ThresholdFinder):

    def find(self, y_true: np.ndarray, distances: np.ndarray) -> float:
        scores = -distances
        fpr, tpr, thr = roc_curve(y_true, scores)
        return -thr[np.argmax(tpr - fpr)]
