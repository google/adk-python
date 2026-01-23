"""
Text file parser implementation.

Handles plain text files including .txt, .md, .json, .yaml, .csv, etc.
"""

import csv
from io import StringIO
import json
from pathlib import Path
from typing import Any

from adk_rlm.files.base import LoadedFile
from adk_rlm.files.base import ParsedContent
from adk_rlm.files.parsers.base import FileParser


class TextParser(FileParser):
  """
  Parse plain text files.

  Supports various text-based formats including:
  - Plain text (.txt)
  - Markdown (.md, .markdown)
  - JSON (.json)
  - YAML (.yaml, .yml)
  - CSV/TSV (.csv, .tsv)
  - Code files (.py, .js, .ts, etc.)
  - Log files (.log)
  - XML/HTML (.xml, .html)
  """

  # Text file extensions
  TEXT_EXTENSIONS = [
      ".txt",
      ".md",
      ".markdown",
      ".json",
      ".yaml",
      ".yml",
      ".csv",
      ".tsv",
      ".log",
      ".xml",
      ".html",
      ".htm",
      ".rst",
      ".py",
      ".js",
      ".ts",
      ".jsx",
      ".tsx",
      ".java",
      ".c",
      ".cpp",
      ".h",
      ".hpp",
      ".go",
      ".rs",
      ".rb",
      ".php",
      ".sh",
      ".bash",
      ".zsh",
      ".sql",
      ".r",
      ".scala",
      ".kt",
      ".swift",
      ".css",
      ".scss",
      ".less",
      ".toml",
      ".ini",
      ".cfg",
      ".conf",
      ".properties",
      ".env",
  ]

  # Text MIME types
  TEXT_MIME_TYPES = [
      "text/plain",
      "text/markdown",
      "text/x-markdown",
      "application/json",
      "text/yaml",
      "application/x-yaml",
      "text/csv",
      "text/tab-separated-values",
      "text/html",
      "application/xml",
      "text/xml",
      "text/x-python",
      "application/javascript",
      "text/javascript",
  ]

  @property
  def supported_extensions(self) -> list[str]:
    """Return list of supported file extensions."""
    return self.TEXT_EXTENSIONS

  @property
  def supported_mime_types(self) -> list[str]:
    """Return list of supported MIME types."""
    return self.TEXT_MIME_TYPES

  def parse(self, file: LoadedFile) -> ParsedContent:
    """
    Parse text file.

    Provides special handling for structured formats like JSON and CSV.

    Args:
        file: LoadedFile with raw content

    Returns:
        ParsedContent with text and optional structured data
    """
    try:
      text = file.as_text()
    except UnicodeDecodeError:
      # Try common encodings
      for encoding in ["utf-8", "latin-1", "cp1252", "ascii"]:
        try:
          text = file.content.decode(encoding)
          break
        except UnicodeDecodeError:
          continue
      else:
        # Last resort: decode with replacement
        text = file.content.decode("utf-8", errors="replace")

    ext = Path(file.metadata.name).suffix.lower()
    metadata: dict[str, Any] = {"format": ext, "encoding": "utf-8"}
    tables: list[dict[str, Any]] | None = None

    # Special handling for structured formats
    if ext == ".json":
      text, metadata = self._handle_json(text, metadata)
    elif ext in [".csv", ".tsv"]:
      text, metadata, tables = self._handle_csv(text, ext, metadata)
    elif ext in [".yaml", ".yml"]:
      metadata = self._handle_yaml(text, metadata)

    return ParsedContent(
        text=text,
        metadata=metadata,
        chunks=None,
        tables=tables,
        images=None,
    )

  def _handle_json(
      self, text: str, metadata: dict[str, Any]
  ) -> tuple[str, dict[str, Any]]:
    """Handle JSON files - pretty print for readability."""
    try:
      data = json.loads(text)
      metadata["json_type"] = type(data).__name__
      if isinstance(data, list):
        metadata["item_count"] = len(data)
      elif isinstance(data, dict):
        metadata["keys"] = list(data.keys())[:20]  # First 20 keys
      # Pretty print for better readability
      text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except json.JSONDecodeError as e:
      metadata["parse_error"] = str(e)
    return text, metadata

  def _handle_csv(
      self, text: str, ext: str, metadata: dict[str, Any]
  ) -> tuple[str, dict[str, Any], list[dict[str, Any]] | None]:
    """Handle CSV/TSV files - extract tables."""
    delimiter = "\t" if ext == ".tsv" else ","
    tables: list[dict[str, Any]] = []

    try:
      reader = csv.DictReader(StringIO(text), delimiter=delimiter)
      for row in reader:
        tables.append(dict(row))

      if tables:
        metadata["row_count"] = len(tables)
        metadata["columns"] = list(tables[0].keys())
    except Exception as e:
      metadata["parse_error"] = str(e)
      tables = []

    return text, metadata, tables if tables else None

  def _handle_yaml(self, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Handle YAML files - add metadata about structure."""
    try:
      import yaml

      data = yaml.safe_load(text)
      metadata["yaml_type"] = type(data).__name__
      if isinstance(data, dict):
        metadata["keys"] = list(data.keys())[:20]
    except ImportError:
      metadata["yaml_parse"] = "yaml library not available"
    except Exception as e:
      metadata["parse_error"] = str(e)
    return metadata
