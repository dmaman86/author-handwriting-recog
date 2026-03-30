import numpy as np
import pandas as pd

from ..evaluator import EmbeddingContext
from .base_strategy import BaseStrategy


class TopKStrategy(BaseStrategy):

    name = "topk"

    def __init__(self, k: int = 5):
        self.k = k

    def predict(self, context: EmbeddingContext) -> pd.DataFrame:
        X = context.X
        y = context.y

        sims = X @ X.T
        np.fill_diagonal(sims, -np.inf)

        topk_idx = np.argsort(-sims, axis=1)[:, : self.k]

        preds = []
        scores = []

        for i in range(len(X)):
            authors = y[topk_idx[i]]
            sim_vals = sims[i, topk_idx[i]]

            values, counts = np.unique(authors, return_counts=True)
            pred = int(values[np.argmax(counts)])

            mask = authors == pred
            score = float(sim_vals[mask].mean())

            preds.append(pred)
            scores.append(score)

        return self._build_result(context, np.array(preds), np.array(scores))
