"""
Lazy file loading with progressive disclosure.

This module provides LazyFile and LazyFileCollection classes that support
on-demand loading of file content at three levels:

- Level 0 (free): name, path, extension - from initial listing
- Level 1 (cheap): size, modified_date, mime_type - metadata request
- Level 2 (expensive): content, tables, chunks - full download + parse
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import fnmatch
from typing import Any
from typing import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from adk_rlm.files.base import FileMetadata
  from adk_rlm.files.base import LoadedFile
  from adk_rlm.files.base import ParsedContent
  from adk_rlm.files.parsers.base import FileParser
  from adk_rlm.files.sources.base import FileSource


@dataclass
class LazyFile:
  """
  A lazy file reference that loads content on first access.

  Progressive disclosure file access:
  - Level 0 (free): name, path, extension - from initial listing
  - Level 1 (cheap): size, modified_date, mime_type - metadata request
  - Level 2 (expensive): content, tables, chunks - full download + parse

  Metadata (name, path) is available immediately.
  Content and parsed data load on-demand.

  Example:
      ```python
      # Level 0 - instant, no I/O
      print(file.name)        # "report.pdf"
      print(file.extension)   # ".pdf"

      # Level 1 - stat/HEAD request
      print(file.size_kb)     # 1024.5
      print(file.modified_date)

      # Level 2 - full download + parse
      print(file.content[:100])  # First 100 chars
      print(file.tables)         # Extracted tables
      ```
  """

  path: str
  source: "FileSource"
  parser: "FileParser | None" = None

  # Cached data (loaded on demand)
  _metadata: "FileMetadata | None" = field(default=None, repr=False)
  _loaded: "LoadedFile | None" = field(default=None, repr=False)
  _parsed: "ParsedContent | None" = field(default=None, repr=False)

  # =========================================================================
  # Level 0: Always available (from path) - No I/O required
  # =========================================================================

  @property
  def name(self) -> str:
    """Filename - available without loading."""
    return self.path.split("/")[-1].split("\\")[-1]

  @property
  def extension(self) -> str:
    """File extension (lowercase, with leading dot) - available without loading."""
    if "." in self.name:
      return "." + self.name.rsplit(".", 1)[-1].lower()
    return ""

  @property
  def is_loaded(self) -> bool:
    """Check if content has been loaded."""
    return self._loaded is not None

  @property
  def is_parsed(self) -> bool:
    """Check if content has been parsed."""
    return self._parsed is not None

  @property
  def level(self) -> int:
    """
    Current loading level.

    - 0: path only (no I/O)
    - 1: metadata loaded
    - 2: content parsed
    """
    if self._parsed is not None:
      return 2
    if self._metadata is not None:
      return 1
    return 0

  # =========================================================================
  # Level 1: Metadata (lazy, cached) - Cheap I/O (stat/HEAD)
  # =========================================================================

  def _ensure_metadata(self) -> None:
    """Load just metadata (not full content)."""
    if self._metadata is None:
      # If we already have the loaded file, use its metadata
      if self._loaded is not None:
        self._metadata = self._loaded.metadata
      else:
        # Sources can implement get_metadata() for efficiency
        self._metadata = self.source.get_metadata(self.path)

  @property
  def metadata(self) -> "FileMetadata":
    """Full file metadata - Level 1 (triggers metadata load)."""
    self._ensure_metadata()
    assert self._metadata is not None
    return self._metadata

  @property
  def size(self) -> int:
    """File size in bytes - Level 1."""
    return self.metadata.size_bytes

  @property
  def size_kb(self) -> float:
    """File size in KB - Level 1."""
    return self.size / 1024

  @property
  def size_mb(self) -> float:
    """File size in MB - Level 1."""
    return self.size / (1024 * 1024)

  @property
  def modified_date(self) -> datetime | None:
    """Last modified date - Level 1."""
    return self.metadata.last_modified

  @property
  def mime_type(self) -> str | None:
    """MIME type - Level 1."""
    return self.metadata.mime_type

  # =========================================================================
  # Level 2: Full content (lazy, cached) - Expensive I/O (full download)
  # =========================================================================

  def _ensure_loaded(self) -> None:
    """Load file if not already loaded."""
    if self._loaded is None:
      self._loaded = self.source.load(self.path)
      # Also set metadata from loaded file
      self._metadata = self._loaded.metadata

  def _ensure_parsed(self) -> None:
    """Parse file if not already parsed."""
    self._ensure_loaded()
    if self._parsed is None:
      if self.parser is None:
        raise ValueError(f"No parser configured for {self.name}")
      assert self._loaded is not None
      self._parsed = self.parser.parse(self._loaded)

  @property
  def raw_content(self) -> bytes:
    """Raw file bytes - Level 2 (triggers download)."""
    self._ensure_loaded()
    assert self._loaded is not None
    return self._loaded.content

  @property
  def content(self) -> str:
    """Parsed text content - Level 2 (triggers download + parse)."""
    self._ensure_parsed()
    assert self._parsed is not None
    return self._parsed.text

  @property
  def tables(self) -> list[dict[str, Any]] | None:
    """Extracted tables - Level 2 (triggers download + parse)."""
    self._ensure_parsed()
    assert self._parsed is not None
    return self._parsed.tables

  @property
  def chunks(self) -> list[str] | None:
    """Pre-chunked content (e.g., pages) - Level 2 (triggers download + parse)."""
    self._ensure_parsed()
    assert self._parsed is not None
    return self._parsed.chunks

  @property
  def parsed_metadata(self) -> dict[str, Any]:
    """Parser-specific metadata - Level 2 (triggers download + parse)."""
    self._ensure_parsed()
    assert self._parsed is not None
    return self._parsed.metadata

  # =========================================================================
  # Utility methods
  # =========================================================================

  def read(self, encoding: str = "utf-8") -> str:
    """
    Read raw content as text - Level 2 (triggers download only, no parse).

    This is useful when you want raw text without parsing overhead.

    Args:
        encoding: Text encoding to use

    Returns:
        File content as string
    """
    self._ensure_loaded()
    assert self._loaded is not None
    return self._loaded.content.decode(encoding)

  def preload_metadata(self) -> "LazyFile":
    """
    Eagerly load metadata (Level 1).

    Useful for batch metadata loading.

    Returns:
        self (for chaining)
    """
    self._ensure_metadata()
    return self

  def preload(self) -> "LazyFile":
    """
    Eagerly load and parse content (Level 2).

    Useful when you know you'll need the content.

    Returns:
        self (for chaining)
    """
    self._ensure_parsed()
    return self

  def __str__(self) -> str:
    level_names = {0: "path", 1: "metadata", 2: "content"}
    return (
        f"<LazyFile '{self.name}' level={self.level}"
        f" ({level_names[self.level]})>"
    )

  def __repr__(self) -> str:
    return self.__str__()


@dataclass
class LazyFileCollection:
  """
  A collection of lazy files with helpful access patterns.

  Provides filtering and batch operations without loading files.

  Example:
      ```python
      files = LazyFileCollection([...])

      # No loading required
      print(files.names)                    # All filenames
      pdfs = files.by_extension(".pdf")     # Filter by extension

      # Selective loading
      for pdf in pdfs[:3]:                  # Only load first 3
          print(pdf.content)
      ```
  """

  files: list[LazyFile] = field(default_factory=list)

  def __len__(self) -> int:
    return len(self.files)

  def __iter__(self) -> Iterator[LazyFile]:
    return iter(self.files)

  def __getitem__(self, idx: int | slice) -> LazyFile | list[LazyFile]:
    result = self.files[idx]
    if isinstance(idx, slice):
      return result
    return result

  def __bool__(self) -> bool:
    return len(self.files) > 0

  # =========================================================================
  # Level 0 operations (no I/O)
  # =========================================================================

  @property
  def names(self) -> list[str]:
    """List all filenames (no loading required)."""
    return [f.name for f in self.files]

  @property
  def paths(self) -> list[str]:
    """List all file paths (no loading required)."""
    return [f.path for f in self.files]

  @property
  def extensions(self) -> set[str]:
    """Set of all file extensions (no loading required)."""
    return {f.extension for f in self.files}

  def by_extension(self, ext: str) -> "LazyFileCollection":
    """
    Filter files by extension (no loading required).

    Args:
        ext: Extension to filter by (with or without leading dot)

    Returns:
        New LazyFileCollection with matching files
    """
    if not ext.startswith("."):
      ext = "." + ext
    ext = ext.lower()
    return LazyFileCollection([f for f in self.files if f.extension == ext])

  def by_name(self, pattern: str) -> "LazyFileCollection":
    """
    Filter files by name pattern (no loading required).

    Uses fnmatch for glob-style matching.

    Args:
        pattern: Glob pattern (e.g., "report*.pdf", "*2024*")

    Returns:
        New LazyFileCollection with matching files
    """
    return LazyFileCollection(
        [f for f in self.files if fnmatch.fnmatch(f.name, pattern)]
    )

  def search(self, keyword: str) -> "LazyFileCollection":
    """
    Search for files with keyword in name (case-insensitive).

    Args:
        keyword: Keyword to search for in filename

    Returns:
        New LazyFileCollection with matching files
    """
    keyword_lower = keyword.lower()
    return LazyFileCollection(
        [f for f in self.files if keyword_lower in f.name.lower()]
    )

  # =========================================================================
  # Status tracking
  # =========================================================================

  @property
  def loaded_count(self) -> int:
    """Count of files that have been loaded (Level 2)."""
    return sum(1 for f in self.files if f.is_loaded)

  @property
  def parsed_count(self) -> int:
    """Count of files that have been parsed (Level 2)."""
    return sum(1 for f in self.files if f.is_parsed)

  @property
  def metadata_count(self) -> int:
    """Count of files with metadata loaded (Level 1+)."""
    return sum(1 for f in self.files if f.level >= 1)

  # =========================================================================
  # Batch loading operations
  # =========================================================================

  def load_all_metadata(self) -> "LazyFileCollection":
    """
    Eagerly load metadata for all files (Level 1).

    Returns:
        self (for chaining)
    """
    for f in self.files:
      f.preload_metadata()
    return self

  def load_all(self) -> "LazyFileCollection":
    """
    Eagerly load and parse all files (Level 2).

    Warning: This loads all files into memory. Use with caution.

    Returns:
        self (for chaining)
    """
    for f in self.files:
      f.preload()
    return self

  def get_all_content(self) -> list[str]:
    """
    Get parsed content from all files.

    Triggers loading for any unloaded files.

    Returns:
        List of text content from all files
    """
    return [f.content for f in self.files]

  # =========================================================================
  # Statistics (may require Level 1)
  # =========================================================================

  @property
  def total_size(self) -> int:
    """Total size in bytes of all files (triggers metadata load)."""
    return sum(f.size for f in self.files)

  @property
  def total_size_mb(self) -> float:
    """Total size in MB of all files (triggers metadata load)."""
    return self.total_size / (1024 * 1024)

  def summary(self) -> str:
    """
    Get a summary of the collection.

    Returns summary without triggering any loading.
    """
    ext_counts: dict[str, int] = {}
    for f in self.files:
      ext = f.extension or "(no ext)"
      ext_counts[ext] = ext_counts.get(ext, 0) + 1

    lines = [f"LazyFileCollection with {len(self.files)} files:"]
    for ext, count in sorted(ext_counts.items()):
      lines.append(f"  {ext}: {count}")
    lines.append(f"  Loaded: {self.loaded_count}/{len(self.files)}")
    return "\n".join(lines)

  def __str__(self) -> str:
    return (
        f"<LazyFileCollection [{len(self.files)} files, {self.loaded_count}"
        " loaded]>"
    )

  def __repr__(self) -> str:
    return self.__str__()
