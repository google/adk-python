"""
File source implementations for ADK-RLM.

This module provides various file source implementations for loading files
from different locations (local filesystem, cloud storage, etc.).
"""

from adk_rlm.files.sources.base import FileSource
from adk_rlm.files.sources.local import LocalFileSource

# Optional GCS support (requires google-cloud-storage)
try:
  from adk_rlm.files.sources.gcs import GCSFileSource
  from adk_rlm.files.sources.gcs import RetryConfig
except ImportError:
  GCSFileSource = None  # type: ignore
  RetryConfig = None  # type: ignore

__all__ = [
    "FileSource",
    "LocalFileSource",
    "GCSFileSource",
    "RetryConfig",
]
