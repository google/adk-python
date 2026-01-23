"""
Base protocol for file sources.

FileSource is an abstract base class that defines the interface for
loading files from various sources (local filesystem, cloud storage, etc.).
"""

from abc import ABC
from abc import abstractmethod
from typing import Iterator

from adk_rlm.files.base import FileMetadata
from adk_rlm.files.base import LoadedFile


class FileSource(ABC):
  """
  Protocol for file sources.

  Implementations should handle loading files from a specific source type
  (e.g., local filesystem, SharePoint, Google Drive, S3).
  """

  @property
  @abstractmethod
  def source_type(self) -> str:
    """Return source type identifier (e.g., 'local', 'sharepoint')."""
    ...

  @abstractmethod
  def resolve(self, path: str) -> list[str]:
    """
    Resolve a path pattern to concrete file paths.

    Supports glob patterns for sources that allow it.

    Args:
        path: File path or pattern (e.g., "*.pdf", "folder/**/*.docx")

    Returns:
        List of resolved file paths/URIs
    """
    ...

  @abstractmethod
  def load(self, path: str) -> LoadedFile:
    """
    Load a single file from the source.

    Args:
        path: Resolved file path (from resolve())

    Returns:
        LoadedFile with content and metadata
    """
    ...

  def get_metadata(self, path: str) -> FileMetadata:
    """
    Get metadata for a file without loading full content.

    Override this in subclasses for more efficient metadata-only access
    (e.g., HEAD requests for HTTP, stat() for local files).

    Default implementation loads the full file, which is inefficient.

    Args:
        path: File path to get metadata for

    Returns:
        FileMetadata for the file
    """
    return self.load(path).metadata

  def load_many(self, paths: list[str]) -> Iterator[LoadedFile]:
    """
    Load multiple files.

    Override for parallel loading in subclasses.

    Args:
        paths: List of file paths to load

    Yields:
        LoadedFile for each path
    """
    for path in paths:
      yield self.load(path)

  def exists(self, path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        path: File path to check

    Returns:
        True if file exists, False otherwise
    """
    try:
      resolved = self.resolve(path)
      return len(resolved) > 0
    except Exception:
      return False
