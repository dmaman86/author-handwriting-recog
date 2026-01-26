import numpy as np
import gc
from typing import Union
from collections.abc import Callable
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
        for split in ['train', 'validation', 'test']:
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

    def load_dataset_by_authors(
            self, 
            authors_id: list[int] | range,
            show_progress_bar: bool = False) -> dict:
        if isinstance(authors_id, range):
            authors_id = list(authors_id)

        # Load dataset
        dataset = self.load_fn(self.dataset_path)
        partition = dataset['partition']
        author_indices = dataset['author_indices']
        
        # FIX: Normalizar author_indices keys a int
        # Zarr puede serializar keys como strings
        author_indices_normalized = {}
        for split in ['train', 'validation', 'test']:
            if split in author_indices:
                author_indices_normalized[split] = {
                    int(k): v for k, v in author_indices[split].items()
                }
            else:
                author_indices_normalized[split] = {}
        
        author_indices = author_indices_normalized

        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(authors_id)
        }

        local_to_global = {
            local_id: global_id for global_id, local_id in global_to_local.items()
        }

        total_steps = len(authors_id) * 3
        pbar = (
            tqdm(total=total_steps, desc="Building subset dataset")
            if show_progress_bar
            else None
        )

        def build_split(split: str) -> tuple[np.ndarray, np.ndarray]:
            X: list[np.ndarray] = []
            y: list[int] = []

            for global_id in authors_id:
                if global_id not in author_indices[split]:
                    if pbar:
                        pbar.update(1)
                    continue
                    
                start, end = author_indices[split][global_id]
                patches = partition[split][start:end]

                local_id = global_to_local[global_id]
                X.extend(patches)
                y.extend([local_id] * len(patches))

                if pbar:
                    pbar.update(1)

            return np.array(X), np.array(y)
        
        X_train, y_train = build_split('train')
        X_validation, y_validation = build_split('validation')
        X_test, y_test = build_split('test')

        if pbar:
            pbar.close()

        del dataset, partition, author_indices
        gc.collect()

        return {
            'train': (X_train, y_train),
            'validation': (X_validation, y_validation),
            'test': (X_test, y_test),
            'num_authors': len(authors_id),
            'author_names': [self.author_names[local_to_global[i]] for i in range(len(authors_id))],
            'patch_shape': self.patch_shape,
            'metadata': self.metadata,
            'global_to_local': global_to_local,
            'local_to_global': local_to_global,
        }
    
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
                f"Author ID {author_id} out of range "
                f"[0, {self.num_authors - 1}]"
            )