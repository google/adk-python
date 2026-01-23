"""
Base types for the file handling module.

This module defines the core data structures used throughout the file
handling system: FileMetadata, LoadedFile, and ParsedContent.
"""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FileMetadata:
  """Metadata about a loaded file."""

  name: str
  path: str  # Original path/URI
  source_type: str  # "local", "sharepoint", "gdrive", etc.
  size_bytes: int
  mime_type: str | None = None
  last_modified: datetime | None = None
  extra: dict[str, Any] = field(default_factory=dict)

  @property
  def size_kb(self) -> float:
    """File size in KB."""
    return self.size_bytes / 1024

  @property
  def size_mb(self) -> float:
    """File size in MB."""
    return self.size_bytes / (1024 * 1024)

  @property
  def extension(self) -> str:
    """File extension (lowercase, with leading dot)."""
    if "." in self.name:
      return "." + self.name.rsplit(".", 1)[-1].lower()
    return ""

  def to_dict(self) -> dict[str, Any]:
    """Convert to dictionary for serialization."""
    return {
        "name": self.name,
        "path": self.path,
        "source_type": self.source_type,
        "size_bytes": self.size_bytes,
        "mime_type": self.mime_type,
        "last_modified": (
            self.last_modified.isoformat() if self.last_modified else None
        ),
        "extra": self.extra,
    }


@dataclass
class LoadedFile:
  """A file loaded from a source with raw content."""

  metadata: FileMetadata
  content: bytes

  def as_text(self, encoding: str = "utf-8") -> str:
    """Decode content as text."""
    return self.content.decode(encoding)


@dataclass
class ParsedContent:
  """Parsed content from a file."""

  text: str  # Extracted text content
  metadata: dict[str, Any] = field(
      default_factory=dict
  )  # Parser-specific metadata
  chunks: list[str] | None = None  # Optional: pre-chunked content (e.g., pages)
  tables: list[dict[str, Any]] | None = None  # Optional: extracted tables
  images: list[bytes] | None = None  # Optional: extracted images

  @property
  def has_tables(self) -> bool:
    """Check if tables were extracted."""
    return self.tables is not None and len(self.tables) > 0

  @property
  def has_chunks(self) -> bool:
    """Check if content was pre-chunked."""
    return self.chunks is not None and len(self.chunks) > 0

  @property
  def chunk_count(self) -> int:
    """Number of chunks (0 if not chunked)."""
    return len(self.chunks) if self.chunks else 0

  @property
  def table_count(self) -> int:
    """Number of tables extracted (0 if none)."""
    return len(self.tables) if self.tables else 0
