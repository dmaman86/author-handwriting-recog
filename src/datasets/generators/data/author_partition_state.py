from dataclasses import dataclass
import numpy as np

@dataclass
class AuthorPartitionState:
    data: dict[int, list[np.ndarray]]
    authors_ids: list[int]
    valid_authors: list[int]

    def flatten_data(self, only_valid_authors: bool = True) -> tuple[np.ndarray, np.ndarray]:
        X: list[np.ndarray] = []
        y: list[int] = []

        authors = self.valid_authors if only_valid_authors else self.authors_ids

        for author_id in authors:
            for patch in self.data[author_id]:
                X.append(patch)
                y.append(author_id)

        return np.stack(X, axis=0), np.array(y, dtype=np.int32)