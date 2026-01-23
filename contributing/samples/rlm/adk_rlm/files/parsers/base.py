"""
Base protocol for file parsers.

FileParser is an abstract base class that defines the interface for
parsing files of various formats into text and structured content.
"""

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from adk_rlm.files.base import LoadedFile
from adk_rlm.files.base import ParsedContent


class FileParser(ABC):
  """
  Protocol for file format parsers.

  Implementations should handle parsing specific file formats
  (e.g., text, PDF, Office documents) into text content.
  """

  @property
  @abstractmethod
  def supported_extensions(self) -> list[str]:
    """
    Return list of supported file extensions.

    Extensions should include the leading dot and be lowercase.
    Example: [".txt", ".md", ".json"]
    """
    ...

  @property
  @abstractmethod
  def supported_mime_types(self) -> list[str]:
    """
    Return list of supported MIME types.

    Example: ["text/plain", "text/markdown", "application/json"]
    """
    ...

  @abstractmethod
  def parse(self, file: LoadedFile) -> ParsedContent:
    """
    Parse file content into text and structured data.

    Args:
        file: LoadedFile with raw content

    Returns:
        ParsedContent with extracted text and metadata
    """
    ...

  def can_parse(self, file: LoadedFile) -> bool:
    """
    Check if this parser can handle the file.

    Uses file extension and MIME type to determine compatibility.

    Args:
        file: LoadedFile to check

    Returns:
        True if this parser can handle the file
    """
    ext = Path(file.metadata.name).suffix.lower()
    mime = file.metadata.mime_type

    return ext in self.supported_extensions or (
        mime is not None and mime in self.supported_mime_types
    )
