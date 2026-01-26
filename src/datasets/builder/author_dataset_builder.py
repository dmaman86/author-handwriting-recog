import numpy as np
from typing import Callable, Dict, List, Tuple, Any

class AuthorDatasetBuilder:
    """
    Incrementally builds a dataset by processing authors one by one.

    Responsibilities:
    - Accept per-author processing calls
    - Aggregate partitions and labels
    - Track author index ranges per split
    - Assemble the final dataset structure
    """

    def __init__(
        self,
        authors_names: List[str],
        author_processor_factory: Callable[[int, str], Any],
        metadata: Dict[str, Any],
    ):
        self.authors_names = authors_names
        self.author_processor_factory = author_processor_factory
        self.metadata = metadata

        self.partition: Dict[str, List[np.ndarray]] = {
            "train": [],
            "validation": [],
            "test": [],
        }

        self.labels: List[int] = []

        self.author_indices: Dict[str, Dict[int, Tuple[int, int]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }

        self.current_pos: Dict[str, int] = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_author(self, label_id: int, author: str) -> None:
        """
        Process a single author and merge its data into the dataset.
        """
        processor = self.author_processor_factory(label_id, author)
        partition, labels = processor.generate()
        self._merge_author_data(label_id, partition, labels)

    def build_dataset(self, date_time: str) -> Dict[str, Any]:
        """
        Assemble and return the final dataset.
        """
        sample_patch = (
            self.partition["train"][0]
            if self.partition["train"]
            else None
        )

        dataset_metadata = dict(self.metadata)
        dataset_metadata["generated_at"] = date_time

        return {
            "partition": self.partition,
            "labels": self.labels,
            "author_indices": self.author_indices,
            "num_authors": len(self.authors_names),
            "author_names": self.authors_names,
            "patch_shape": (
                sample_patch.shape if sample_patch is not None else None
            ),
            "metadata": dataset_metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_author_data(
        self,
        label_id: int,
        partition: Dict[str, List],
        labels: List[int],
    ) -> None:
        """
        Merge one author's data into the global dataset.
        """
        for split in ("train", "validation", "test"):
            patches = partition.get(split, [])
            if not patches:
                continue

            start = self.current_pos[split]
            end = start + len(patches)

            self.author_indices[split][label_id] = (start, end)
            self.current_pos[split] = end

            self.partition[split].extend(patches)

        self.labels.extend(labels)