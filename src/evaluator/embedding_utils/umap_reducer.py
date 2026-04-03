import numpy as np
import pandas as pd
import umap


class UMAPReducer:

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
        X = np.stack(df["embedding"].values)

        reducer = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            metric="cosine",
        )
        emb = reducer.fit_transform(X)

        df_umap = df.copy()

        for i in range(self.n_components):
            df_umap[f"umap_{i}"] = emb[:, i]

        df_umap["x"] = emb[:, 0]
        df_umap["y"] = emb[:, 1]

        return df_umap, reducer
