import shutil
from pathlib import Path


class FileSystem:
    """
    File system operations service.
    Provides a clean interface for common file and directory operations
    with proper error handling and path validation.

    This is a stateless service - all operations are pure functions
    that don't maintain internal state.
    """

    def to_path(self, path: str | Path) -> Path:
        """
        Convert a string path to a Path object if necessary.

        Args:
            path: String or Path object representing a file system path

        Returns:
            Path object
        """
        return Path(path) if isinstance(path, str) else path

    def ensure_file_exists(self, path: str | Path) -> Path:
        """
        Verify that a file exists and return its resolved path.

        Args:
            path: Path to the file to verify

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If the file does not exist
        """
        p = self.to_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return p.resolve()

    def ensure_directory(self, path: str | Path, remove_existing: bool = False) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        Args:
            path: Path to the directory
            remove_existing: If True, remove the directory if it already exists
        Returns:
            Resolved Path object to the directory
        """
        p = self.to_path(path)
        if p.exists() and remove_existing:
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    def list_files(self, directory: str | Path, extension: str) -> list[str]:
        """
        List all filenames (without extension) with the given extension.
        Args:
            directory: Directory to search in
            extension: File extension to filter by (with or without leading dot)
        Returns:
            List of file stems (filenames without extension)
        """
        dir_path = self.to_path(directory)
        if not dir_path.exists():
            return []

        ext = extension.lstrip(".")
        files = dir_path.glob(f"*.{ext}")

        return [f.stem for f in files if f.is_file()]

    def file_exists(self, path: str | Path) -> bool:
        """
        Check if a file exists.
        Args:
            path: Path to check
        Returns:
            True if the file exists, False otherwise
        """
        return self.to_path(path).is_file()

    def directory_exists(self, path: str | Path) -> bool:
        """
        Check if a directory exists.
        Args:
            path: Path to check
        Returns:
            True if the directory exists, False otherwise
        """
        return self.to_path(path).is_dir()

    def remove_file(self, path: str | Path) -> None:
        """
        Remove a file if it exists.
        Args:
            path: Path to the file to remove
        """
        p = self.to_path(path)
        if p.is_file():
            p.unlink()

    def remove_directory(self, path: str | Path) -> None:
        """
        Remove a directory and all its contents.
        Args:
            path: Path to the directory to remove
        """
        p = self.to_path(path)
        if p.is_dir():
            shutil.rmtree(p)
