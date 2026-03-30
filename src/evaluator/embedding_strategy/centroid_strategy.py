import numpy as np
import pandas as pd

from ..evaluator import EmbeddingContext
from .base_strategy import BaseStrategy


class CentroidStrategy(BaseStrategy):
    name = "centroid"

    def predict(self, context: EmbeddingContext) -> pd.DataFrame:
        X = context.X
        y = context.y

        authors = np.unique(y)
        author_to_idx = {a: np.where(y == a)[0] for a in authors}
        preds, scores = [], []

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
                centroid = refs.mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-8

                score = float(query @ centroid)

                if score > best_score:
                    best_score = score
                    best_author = int(author)

            preds.append(int(best_author))
            scores.append(best_score)

        return self._build_result(
            context,
            np.array(preds),
            np.array(scores),
        )
