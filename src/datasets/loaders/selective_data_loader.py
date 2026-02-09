import gc
from collections.abc import Callable

import numpy as np
from tqdm import tqdm


class SelectiveDataLoader:
    def __init__(self, dataset_path: str, load_fn: Callable[[str], dict]) -> None:
        self.dataset_path = dataset_path
        self.load_fn = load_fn
        self._init_values()

    def _init_values(self) -> None:
        dataset = self.load_fn(self.dataset_path)

        self.num_authors = dataset["num_authors"]
        self.author_names = dataset["author_names"]

        # FIX: Normalizar author_indices keys a int
        # Zarr serializa dict keys como strings
        author_indices = dataset["author_indices"]
        self.author_indices = {}
        for split in ["train", "validation", "test"]:
            if split in author_indices:
                self.author_indices[split] = {
                    int(k): v for k, v in author_indices[split].items()
                }
            else:
                self.author_indices[split] = {}

        self.patch_shape = dataset["patch_shape"]
        self.metadata = dataset["metadata"]

        del dataset
        gc.collect()

    def build_author_maps(
        self, authors_id: list[int]
    ) -> tuple[dict[int, int], dict[int, int], list[str]]:
        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(authors_id)
        }
        local_to_global = {
            local_id: global_id for global_id, local_id in global_to_local.items()
        }
        author_names = [
            self.author_names[local_to_global[i]] for i in range(len(authors_id))
        ]
        return global_to_local, local_to_global, author_names

    def load_split_by_authors(
        self,
        split: str,
        authors_id: list[int],
        global_to_local: dict[int, int],
        show_progress_bar: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        dataset = self.load_fn(self.dataset_path)
        partition = dataset["partition"]
        author_indices = dataset["author_indices"]

        if split in author_indices:
            split_indices = {int(k): v for k, v in author_indices[split].items()}
        else:
            split_indices = {}

        del dataset, author_indices
        gc.collect()

        X: list[np.ndarray] = []
        y: list[int] = []

        iterator = (
            tqdm(authors_id, desc=f"Loading {split}")
            if show_progress_bar
            else authors_id
        )

        for global_id in iterator:
            if global_id not in split_indices:
                continue
            start, end = split_indices[global_id]
            patches = partition[split][start:end]

            local_id = global_to_local[global_id]
            X.extend(patches)
            y.extend([local_id] * len(patches))

        del partition, split_indices
        gc.collect()

        return np.array(X), np.array(y)

    def load_dataset_by_authors(
        self, authors_id: list[int] | range, show_progress_bar: bool = False
    ) -> dict:
        if isinstance(authors_id, range):
            authors_id = list(authors_id)

        global_to_local, local_to_global, author_names = self.build_author_maps(
            authors_id
        )

        result = {}
        for split in ["train", "validation", "test"]:
            result[split] = self.load_split_by_authors(
                split=split,
                authors_id=authors_id,
                global_to_local=global_to_local,
                show_progress_bar=show_progress_bar,
            )

        result["num_authors"] = len(authors_id)
        result["author_names"] = author_names
        result["patch_shape"] = self.patch_shape
        result["metadata"] = self.metadata
        result["global_to_local"] = global_to_local
        result["local_to_global"] = local_to_global

        return result

    def get_author_info(self, author_id: int) -> dict:
        self._validate_author_id(author_id)

        info = {
            "author_id": author_id,
            "author_name": self.author_names[author_id],
            "num_patches": {},
        }

        for split in ("train", "validation", "test"):
            if author_id in self.author_indices[split]:
                start, end = self.author_indices[split][author_id]
                info["num_patches"][split] = end - start
            else:
                info["num_patches"][split] = 0

        info["num_patches"]["total"] = sum(info["num_patches"].values())
        return info

    def get_author_name(self, label_id: int) -> str:
        self._validate_author_id(label_id)
        return self.author_names[label_id]

    def get_author_id(self, author_name: str) -> int:
        try:
            return self.author_names.index(author_name)
        except ValueError:
            raise ValueError(
                f"Author name '{author_name}' not found in the dataset."
                f" Available authors: {len(self.author_names)}"
            )

    def list_authors(self) -> list[tuple[int, str]]:
        return [(i, name) for i, name in enumerate(self.author_names)]

    def _validate_author_id(self, author_id: int) -> None:
        if author_id < 0 or author_id >= self.num_authors:
            raise IndexError(
                f"Author ID {author_id} out of range " f"[0, {self.num_authors - 1}]"
            )

