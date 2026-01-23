"""
FileLoader orchestrator for ADK-RLM.

Coordinates file loading from various sources and parsing into
content usable by the RLM system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adk_rlm.files.base import LoadedFile
from adk_rlm.files.base import ParsedContent
from adk_rlm.files.lazy import LazyFile
from adk_rlm.files.lazy import LazyFileCollection
from adk_rlm.files.parsers.base import FileParser
from adk_rlm.files.parsers.pdf import PDFParser
from adk_rlm.files.parsers.text import TextParser
from adk_rlm.files.sources.base import FileSource
from adk_rlm.files.sources.local import LocalFileSource


@dataclass
class FileSpec:
  """
  Specification for a file to load.

  Allows explicit control over source selection.

  Example:
      ```python
      # Auto-detect source
      spec = FileSpec(path="report.pdf")

      # Explicit source
      spec = FileSpec(path="doc.pdf", source=my_source)
      ```
  """

  path: str
  source: FileSource | None = None  # None = auto-detect


class FileLoader:
  """
  Orchestrates file loading and parsing.

  Handles:
  - Auto-detecting file sources from paths/URIs
  - Resolving glob patterns
  - Loading files from various sources
  - Parsing files into text content
  - Creating lazy file collections for efficient access

  Example:
      ```python
      loader = FileLoader()

      # Eager loading - parse immediately
      contents = loader.load_files(["report.pdf", "data.csv"])

      # Lazy loading - parse on demand
      files = loader.create_lazy_files(["report.pdf", "*.md"])
      for f in files:
          print(f.name)        # No I/O
          print(f.content)     # Triggers load + parse
      ```
  """

  def __init__(
      self,
      sources: dict[str, FileSource] | None = None,
      parsers: list[FileParser] | None = None,
      base_path: str | Path | None = None,
  ):
    """
    Initialize FileLoader.

    Args:
        sources: Dictionary of named file sources.
                 Default includes "local" source.
        parsers: List of file parsers.
                 Default includes TextParser and PDFParser.
        base_path: Base path for local file source.
    """
    # Default sources
    self.sources: dict[str, FileSource] = sources or {
        "local": LocalFileSource(base_path),
    }

    # Default parsers (order matters - first matching parser wins)
    self.parsers: list[FileParser] = parsers or [
        TextParser(),
        PDFParser(),
    ]

  def register_source(self, name: str, source: FileSource) -> None:
    """
    Register a file source.

    Args:
        name: Name to register source under
        source: FileSource implementation
    """
    self.sources[name] = source

  def register_parser(self, parser: FileParser) -> None:
    """
    Register a file parser.

    New parsers are added to the end of the list.

    Args:
        parser: FileParser implementation
    """
    self.parsers.append(parser)

  def _detect_source(self, path: str) -> FileSource:
    """
    Auto-detect the appropriate source for a path.

    Args:
        path: File path or URI

    Returns:
        Appropriate FileSource for the path
    """
    if path.startswith("sharepoint://"):
      source = self.sources.get("sharepoint")
      if source is None:
        raise ValueError(
            "SharePoint source not configured. Register a SharePointSource with"
            " loader.register_source('sharepoint', source)"
        )
      return source

    elif path.startswith("gdrive://"):
      source = self.sources.get("gdrive")
      if source is None:
        raise ValueError(
            "Google Drive source not configured. Register a GoogleDriveSource"
            " with loader.register_source('gdrive', source)"
        )
      return source

    elif path.startswith("s3://"):
      source = self.sources.get("s3")
      if source is None:
        raise ValueError(
            "S3 source not configured. "
            "Register an S3Source with loader.register_source('s3', source)"
        )
      return source

    elif path.startswith("gs://"):
      source = self.sources.get("gcs")
      if source is None:
        raise ValueError(
            "GCS source not configured. Register a GCSFileSource with"
            " loader.register_source('gcs', source)"
        )
      return source

    elif path.startswith(("http://", "https://")):
      source = self.sources.get("http")
      if source is None:
        raise ValueError(
            "HTTP source not configured. "
            "Register an HTTPSource with loader.register_source('http', source)"
        )
      return source

    else:
      # Default to local filesystem
      return self.sources["local"]

  def _find_parser(self, file: LoadedFile) -> FileParser:
    """
    Find appropriate parser for a file.

    Args:
        file: LoadedFile to find parser for

    Returns:
        FileParser that can handle the file

    Raises:
        ValueError: If no parser found
    """
    for parser in self.parsers:
      if parser.can_parse(file):
        return parser
    raise ValueError(f"No parser found for file: {file.metadata.name}")

  def _find_parser_by_path(self, path: str) -> FileParser | None:
    """
    Find appropriate parser based on file path/extension.

    Args:
        path: File path

    Returns:
        FileParser if found, None otherwise
    """
    # Get extension from path
    name = path.split("/")[-1].split("\\")[-1]
    if "." not in name:
      return None

    ext = "." + name.rsplit(".", 1)[-1].lower()

    for parser in self.parsers:
      if ext in parser.supported_extensions:
        return parser
    return None

  def load_files(
      self,
      files: list[str | FileSpec],
  ) -> list[ParsedContent]:
    """
    Load and parse multiple files (eager loading).

    Args:
        files: List of file paths, URIs, or FileSpecs

    Returns:
        List of ParsedContent objects
    """
    results: list[ParsedContent] = []

    for file_ref in files:
      # Normalize to FileSpec
      if isinstance(file_ref, str):
        file_ref = FileSpec(path=file_ref)

      # Detect source
      source = file_ref.source or self._detect_source(file_ref.path)

      # Resolve patterns (e.g., globs)
      resolved_paths = source.resolve(file_ref.path)

      # Load and parse each file
      for path in resolved_paths:
        loaded = source.load(path)
        parser = self._find_parser(loaded)
        parsed = parser.parse(loaded)
        results.append(parsed)

    return results

  def create_lazy_files(
      self,
      files: list[str | FileSpec],
  ) -> LazyFileCollection:
    """
    Create lazy file references (deferred loading).

    Files are not loaded until their content is accessed.

    Args:
        files: List of file paths, URIs, or FileSpecs

    Returns:
        LazyFileCollection with lazy file references
    """
    lazy_files: list[LazyFile] = []

    for file_ref in files:
      # Normalize to FileSpec
      if isinstance(file_ref, str):
        file_ref = FileSpec(path=file_ref)

      # Detect source
      source = file_ref.source or self._detect_source(file_ref.path)

      # Resolve patterns
      resolved_paths = source.resolve(file_ref.path)

      # Create lazy file for each resolved path
      for path in resolved_paths:
        parser = self._find_parser_by_path(path)
        lazy_file = LazyFile(
            path=path,
            source=source,
            parser=parser,
        )
        lazy_files.append(lazy_file)

    return LazyFileCollection(lazy_files)

  def load_single(self, path: str) -> ParsedContent:
    """
    Load and parse a single file.

    Args:
        path: File path or URI

    Returns:
        ParsedContent for the file
    """
    results = self.load_files([path])
    if not results:
      raise FileNotFoundError(f"File not found: {path}")
    return results[0]

  def create_lazy_file(self, path: str) -> LazyFile:
    """
    Create a single lazy file reference.

    Args:
        path: File path or URI

    Returns:
        LazyFile reference
    """
    collection = self.create_lazy_files([path])
    if not collection:
      raise FileNotFoundError(f"File not found: {path}")
    return collection[0]

  def build_context(
      self,
      files: list[str | FileSpec],
      lazy: bool = True,
  ) -> dict[str, Any]:
    """
    Build a context dictionary for RLM consumption.

    Args:
        files: List of file paths, URIs, or FileSpecs
        lazy: If True, use lazy loading. If False, load immediately.

    Returns:
        Context dictionary with files and metadata
    """
    if lazy:
      file_collection = self.create_lazy_files(files)
      return {
          "files": file_collection,
          "file_count": len(file_collection),
          "file_names": file_collection.names,
      }
    else:
      parsed_files = self.load_files(files)
      if len(parsed_files) == 1:
        return {
            "content": parsed_files[0].text,
            "metadata": parsed_files[0].metadata,
            "tables": parsed_files[0].tables,
        }
      else:
        return {
            "files": [
                {
                    "content": pf.text,
                    "metadata": pf.metadata,
                    "tables": pf.tables,
                }
                for pf in parsed_files
            ],
            "file_count": len(parsed_files),
        }
