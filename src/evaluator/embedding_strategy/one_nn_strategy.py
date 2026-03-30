import numpy as np
import pandas as pd

from ..evaluator import EmbeddingContext
from .base_strategy import BaseStrategy


class OneNNStrategy(BaseStrategy):

    name = "1nn"

    def predict(self, context: EmbeddingContext) -> pd.DataFrame:

        X = context.X
        y = context.y

        sims = X @ X.T
        np.fill_diagonal(sims, -np.inf)

        best_idx = np.argmax(sims, axis=1)
        preds = y[best_idx]
        scores = sims[np.arange(len(X)), best_idx]

        return self._build_result(context, preds, scores)
