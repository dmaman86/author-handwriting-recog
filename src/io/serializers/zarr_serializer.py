from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import zarr

from ..exceptions import DeserializationError, SerializationError
from .base_serializer import BaseSerializer

try:
    # Zarr 2.x
    DirectoryStore = zarr.DirectoryStore
except AttributeError:
    # Zarr 3.x
    try:
        from zarr.storage import DirectoryStore
    except ImportError:
        DirectoryStore = None


class ZarrSerializer(BaseSerializer):

    def __init__(
        self,
        file_system,
        compression_level: int = 5,
        chunk_size: int = 64,
    ) -> None:
        super().__init__(file_system)

        self.chunk_size = chunk_size

        self.compressor = numcodecs.Blosc(
            cname="zstd",
            clevel=compression_level,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )

    def extensions(self) -> list[str]:
        return [".zarr"]

    def save(self, directory: str | Path, data: Any, filename: str) -> None:
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename

        if DirectoryStore is None:
            raise SerializationError("Zarr DirectoryStore not available.")

        store = DirectoryStore(str(file_path))

        if self.fs.directory_exists(file_path):
            root = zarr.open_group(store=store, mode="a")
        else:
            root = zarr.group(store=store, overwrite=True)

        has_patches = all(key in data for key in ["test", "non_test"])
        has_metadata = "metadata" in data

        if not has_patches and not has_metadata:
            raise SerializationError("Nothing to save.")

        if has_patches:
            self._handle_append(root, data)
        if has_metadata:
            self._handle_metadata(root, data["metadata"])

    def _handle_append(
        self,
        root: zarr.Group,
        data: dict[str, Any],
    ) -> None:
        for split in ("test", "non_test"):
            split_data = data.get(split)
            images = split_data.get("images")
            labels = split_data.get("labels")
            self._append_split(root, split, images, labels)

    def _append_split(
        self,
        root: zarr.Group,
        split: str,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        group = root.require_group(split)

        if "images" not in group:
            group.create_dataset(
                "images",
                data=images,
                maxshape=(None, *images.shape[1:]),
                chunks=(self.chunk_size, *images.shape[1:]),
                compressor=self.compressor,
                dtype=images.dtype,
            )
        else:
            z_images = group["images"]

            old_size = z_images.shape[0]
            new_size = old_size + images.shape[0]

            new_shape = (new_size, *z_images.shape[1:])
            z_images.resize(new_shape)

            z_images[old_size:new_size] = images

        if "labels" not in group:
            group.create_dataset(
                "labels",
                data=labels,
                maxshape=(None,),
                chunks=(self.chunk_size,),
                compressor=self.compressor,
                dtype=labels.dtype,
            )
        else:
            z_labels = group["labels"]

            old_size = z_labels.shape[0]
            new_size = old_size + labels.shape[0]

            new_shape = (new_size,)
            z_labels.resize(new_shape)

            z_labels[old_size:new_size] = labels

    def _handle_metadata(self, root: zarr.Group, metadata: dict):
        root.attrs.update(metadata)

    def load(self, file_path: str | Path) -> dict[str, Any]:
        dataset_path = self.normalize_path(file_path)

        if not self.fs.directory_exists(dataset_path):
            raise DeserializationError("Dataset does not exist.")

        if DirectoryStore is None:
            raise DeserializationError("Zarr DirectoryStore not available.")

        store = DirectoryStore(str(dataset_path))
        root = zarr.open_group(store=store, mode="r")
        dataset = {}

        for split in ("test", "non_test"):
            if split in root:
                images = root[split]["images"]
                labels = root[split]["labels"]
                dataset[split] = (images, labels)

        dataset["metadata"] = dict(root.attrs)
        return dataset


class ZarrArrayWrapper:

    def __init__(self, zarr_array: zarr.Array) -> None:
        self.zarr_array = zarr_array

    def __len__(self):
        return self.zarr_array.shape[0]

    def __getitem__(self, key):
        return self.zarr_array[key]

    def __iter__(self):
        for i in range(len(self)):
            yield self.zarr_array[i]

    @property
    def shape(self):
        return self.zarr_array.shape

    @property
    def dtype(self):
        return self.zarr_array.dtype
