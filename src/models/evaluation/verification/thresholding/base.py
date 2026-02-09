from abc import ABC, abstractmethod

import numpy as np


class ThresholdFinder(ABC):
    @abstractmethod
    def find(self, y_true: np.ndarray, distances: np.ndarray) -> float:
        pass
