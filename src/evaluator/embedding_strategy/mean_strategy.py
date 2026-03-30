import numpy as np
import pandas as pd

from ..evaluator import EmbeddingContext
from .base_strategy import BaseStrategy


class MeanStrategy(BaseStrategy):
    name = "mean"

    def predict(self, context: EmbeddingContext) -> pd.DataFrame:

        X = context.X
        y = context.y

        authors = np.unique(y)
        author_to_idx = {a: np.where(y == a)[0] for a in authors}

        preds = []
        scores = []

        for i in range(len(X)):
            query = X[i]
            true_author = y[i]

            best_score = -np.inf
            best_author = None

            for author, idxs in author_to_idx.items():

                if author == true_author:
                    idxs = idxs[idxs != i]
                    if len(idxs) == 0:
                        continue

                refs = X[idxs]
                score = (refs @ query).mean()

                if score > best_score:
                    best_score = score
                    best_author = author

            preds.append(int(best_author))
            scores.append(float(best_score))

        return self._build_result(context, np.array(preds), np.array(scores))
