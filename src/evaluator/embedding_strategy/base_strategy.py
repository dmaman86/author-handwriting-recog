from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..evaluator import EmbeddingContext


class BaseStrategy(ABC):

    name: str

    @abstractmethod
    def predict(self, context: EmbeddingContext) -> pd.DataFrame:
        pass

    def _build_result(
        self,
        context: EmbeddingContext,
        preds: np.ndarray,
        scores: np.ndarray,
    ) -> pd.DataFrame:

        df = context.df.copy()
        df["author_pred"] = preds
        df["score"] = scores

        return df
