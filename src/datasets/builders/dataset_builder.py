import gc
import os
import re
from pathlib import Path
from typing import Callable

from ...io import ImageAnalyzer, ImageTransformer, PatchExtractor
from ..loaders import AuthorMetadataLoader
from .author_dataset_builder import AuthorDatasetBuilder


class DatasetBuilder:

    def __init__(
        self,
        authors: list[str],
        images_dir: str | Path,
        mat_dir: str | Path,
        patch_size: tuple[int, int],
        strides: tuple[int, int],
        threshold: float,
        analyzer: ImageAnalyzer,
        transformer: ImageTransformer,
        save_fn: Callable[[str, dict], None],
        load_fn: Callable[[Path], dict],
        directory: str,
        filename: str,
    ) -> None:
        self.authors = self._get_author_names(authors)

        self.patch_extractor = PatchExtractor(
            image_analyzer=analyzer,
            transformer=transformer,
            output_patch_dim=patch_size,
            stride_dim=strides,
            empty_threshold=threshold,
        )
        self.save_fn = save_fn
        self.author_metadata_loader = AuthorMetadataLoader(
            parent_path=mat_dir,
            load_fn=load_fn,
        )
        self.author_dataset_builder = AuthorDatasetBuilder(
            image_dir=images_dir,
            transformer=transformer,
            patch_extractor=self.patch_extractor,
            load_fn=load_fn,
        )
        self.directory = directory
        self.filename = filename

    def build_dataset(self, author: str, label_id: int) -> None:
        author_info = self.author_metadata_loader.build_author_data(
            author_id=author,
        )
        dataset = self.author_dataset_builder.build(
            info=author_info,
            label_id=label_id,
        )

        self.save_fn(
            directory=self.directory,
            data=dataset,
            filename=self.filename,
        )
        del author_info, dataset
        gc.collect()

    def _get_author_names(self, authors: list[str]) -> list[str]:
        # get only the file name without extension
        authors: list[str] = [
            os.path.splitext(os.path.basename(author))[0] for author in authors
        ]

        return sorted(authors, key=self._natural_key)

    def _natural_key(self, name: str) -> list:
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", name)
        ]
