import gc

import numpy as np

from .author_partition_state import AuthorPartitionState


class DataInitializer:

    def __init__(
        self,
        data_partition: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._build_data(data_partition)
        self._build_authord_ids()

    def _build_data(self, data_partition: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = data_partition

        self.data: dict[int, list[np.ndarray]] = {}
        for patch, author_id in zip(X, y):
            aid = int(author_id)
            self.data.setdefault(aid, []).append(patch)

        del X, y
        gc.collect()

    def _build_authord_ids(self) -> None:
        self.authors_ids = list(self.data.keys())
        self.valid_authors = [
            author_id for author_id, patches in self.data.items() if len(patches) >= 2
        ]
        self._validate()

    def _validate(self) -> None:
        if len(self.valid_authors) < 2:
            raise ValueError(
                "Data requires at least two authors " "with at least two patches each."
            )

    def to_state(self) -> AuthorPartitionState:
        return AuthorPartitionState(
            data=self.data,
            authors_ids=self.authors_ids,
            valid_authors=self.valid_authors,
        )
