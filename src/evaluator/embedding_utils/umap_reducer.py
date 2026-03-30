import numpy as np
import pandas as pd
import umap


class UMAPReducer:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
        X = np.stack(df["embedding"].values)

        reducer = umap.UMAP(random_state=self.random_state)
        X_2d = reducer.fit_transform(X)

        df_umap = df.copy()
        df_umap["x"] = X_2d[:, 0]
        df_umap["y"] = X_2d[:, 1]

        return df_umap, reducer
