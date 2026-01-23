"""
File handling module for ADK-RLM.

This module provides functionality for loading and parsing files from
various sources (local filesystem, cloud storage) and formats (text, PDF).

Features:
- Progressive disclosure via lazy loading (Level 0/1/2)
- Pluggable file sources (local, SharePoint, GDrive, S3, HTTP)
- Pluggable file parsers (text, PDF, Office documents)
- Glob pattern support for batch file operations

Example:
    ```python
    from adk_rlm.files import FileLoader, LocalFileSource

    # Basic usage with local files
    loader = FileLoader()
    files = loader.create_lazy_files(["./docs/**/*.pdf"])

    # Level 0 - no I/O
    for f in files:
        print(f.name, f.extension)

    # Level 1 - metadata only
    large_files = [f for f in files if f.size_mb > 10]

    # Level 2 - full content
    for f in large_files:
        print(f.content[:1000])
    ```
"""

from adk_rlm.files.base import FileMetadata
from adk_rlm.files.base import LoadedFile
from adk_rlm.files.base import ParsedContent
from adk_rlm.files.lazy import LazyFile
from adk_rlm.files.lazy import LazyFileCollection
from adk_rlm.files.loader import FileLoader
from adk_rlm.files.loader import FileSpec
from adk_rlm.files.parsers import FileParser
from adk_rlm.files.parsers import PDFParser
from adk_rlm.files.parsers import TextParser
from adk_rlm.files.sources import FileSource
from adk_rlm.files.sources import LocalFileSource

__all__ = [
    # Base types
    "FileMetadata",
    "LoadedFile",
    "ParsedContent",
    # Lazy loading
    "LazyFile",
    "LazyFileCollection",
    # Loader
    "FileLoader",
    "FileSpec",
    # Sources
    "FileSource",
    "LocalFileSource",
    # Parsers
    "FileParser",
    "TextParser",
    "PDFParser",
]
