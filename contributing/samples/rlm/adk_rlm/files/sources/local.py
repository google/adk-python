"""
Local filesystem file source implementation.

Provides file loading from the local filesystem with glob pattern support.
"""

from datetime import datetime
from glob import glob
import mimetypes
import os
from pathlib import Path

from adk_rlm.files.base import FileMetadata
from adk_rlm.files.base import LoadedFile
from adk_rlm.files.sources.base import FileSource


class LocalFileSource(FileSource):
  """
  Load files from the local filesystem.

  Supports glob patterns for resolving multiple files.

  Example:
      ```python
      source = LocalFileSource(base_path="/path/to/docs")

      # Single file
      file = source.load("report.pdf")

      # Glob pattern
      paths = source.resolve("**/*.md")
      for path in paths:
          file = source.load(path)
      ```
  """

  def __init__(self, base_path: str | Path | None = None):
    """
    Initialize LocalFileSource.

    Args:
        base_path: Base directory for relative paths.
                   Defaults to current working directory.
    """
    if base_path is None:
      self.base_path = Path.cwd()
    else:
      self.base_path = Path(base_path).resolve()

  @property
  def source_type(self) -> str:
    """Return 'local' as the source type."""
    return "local"

  def resolve(self, path: str) -> list[str]:
    """
    Resolve path, supporting glob patterns.

    Args:
        path: File path or glob pattern (e.g., "*.pdf", "**/*.md")

    Returns:
        List of absolute file paths matching the pattern
    """
    # Handle absolute paths
    if os.path.isabs(path):
      full_path = Path(path)
    else:
      full_path = self.base_path / path

    # Check for glob patterns
    if any(c in str(full_path) for c in ["*", "?", "["]):
      matches = glob(str(full_path), recursive=True)
      # Filter to only files (not directories)
      return sorted([str(m) for m in matches if os.path.isfile(m)])

    # Single file path
    if full_path.exists() and full_path.is_file():
      return [str(full_path)]

    return []

  def get_metadata(self, path: str) -> FileMetadata:
    """
    Get metadata via stat() without reading file content.

    This is more efficient than load() for metadata-only access.

    Args:
        path: File path to get metadata for

    Returns:
        FileMetadata for the file
    """
    file_path = Path(path)

    if not file_path.is_absolute():
      file_path = self.base_path / file_path

    if not file_path.exists():
      raise FileNotFoundError(f"File not found: {path}")

    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return FileMetadata(
        name=file_path.name,
        path=str(file_path),
        source_type=self.source_type,
        size_bytes=stat.st_size,
        mime_type=mime_type,
        last_modified=datetime.fromtimestamp(stat.st_mtime),
        extra={
            "mode": stat.st_mode,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        },
    )

  def load(self, path: str) -> LoadedFile:
    """
    Load file from filesystem.

    Args:
        path: File path (absolute or relative to base_path)

    Returns:
        LoadedFile with content and metadata
    """
    file_path = Path(path)

    if not file_path.is_absolute():
      file_path = self.base_path / file_path

    if not file_path.exists():
      raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
      raise ValueError(f"Path is not a file: {path}")

    stat = file_path.stat()
    content = file_path.read_bytes()

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))

    return LoadedFile(
        metadata=FileMetadata(
            name=file_path.name,
            path=str(file_path),
            source_type=self.source_type,
            size_bytes=stat.st_size,
            mime_type=mime_type,
            last_modified=datetime.fromtimestamp(stat.st_mtime),
            extra={},
        ),
        content=content,
    )

  def exists(self, path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        path: File path to check

    Returns:
        True if file exists, False otherwise
    """
    file_path = Path(path)

    if not file_path.is_absolute():
      file_path = self.base_path / file_path

    return file_path.exists() and file_path.is_file()
