"""
ADK-RLM: Recursive Language Models implemented with Google ADK.

This package provides an implementation of Recursive Language Models (RLM)
using Google's Agent Development Kit (ADK) framework.

Features:
- RLM (Recursive Language Model) pattern with Gemini
- ADK-native agent with streaming events
- File handling with lazy loading and progressive disclosure
- Support for local files, PDFs, and text formats
"""

# Import stdlib logging before any adk_rlm imports to avoid shadowing
# by the local adk_rlm.logging module
import logging as _logging

# Configure library logging on import
# Logs warnings and above to stderr by default
_logger = _logging.getLogger(__name__)
if not _logger.handlers:
  _handler = _logging.StreamHandler()
  _handler.setFormatter(
      _logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  )
  _logger.addHandler(_handler)
  _logger.setLevel(_logging.WARNING)


def configure_logging(
    level: int = _logging.WARNING,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
  """
  Configure logging for the adk_rlm package.

  This reconfigures the library's logging level and format.
  By default, the library logs WARNING and above to stderr.

  Args:
      level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
      format: Log message format string.

  Example:
      import logging
      import adk_rlm

      # Enable debug logging for adk_rlm
      adk_rlm.configure_logging(level=logging.DEBUG)

      # Or configure manually:
      logging.getLogger("adk_rlm").setLevel(logging.DEBUG)
  """
  logger = _logging.getLogger(__name__)
  logger.setLevel(level)

  # Update existing handler's level and format
  for handler in logger.handlers:
    handler.setLevel(level)
    handler.setFormatter(_logging.Formatter(format))


from adk_rlm.agents.rlm_agent import RLMAgent
from adk_rlm.code_executor import RLMCodeExecutor
from adk_rlm.events import RLMEventData
from adk_rlm.events import RLMEventType
from adk_rlm.files import FileLoader
from adk_rlm.files import FileMetadata
from adk_rlm.files import FileParser
from adk_rlm.files import FileSource
from adk_rlm.files import LazyFile
from adk_rlm.files import LazyFileCollection
from adk_rlm.files import LoadedFile
from adk_rlm.files import LocalFileSource
from adk_rlm.files import ParsedContent
from adk_rlm.files import PDFParser
from adk_rlm.files import TextParser
from adk_rlm.main import completion
from adk_rlm.main import RLM
from adk_rlm.types import CodeBlock
from adk_rlm.types import ModelUsageSummary
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMChatCompletion
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
from adk_rlm.types import UsageSummary

__all__ = [
    # Main class and convenience function
    "RLM",
    "completion",
    # Logging configuration
    "configure_logging",
    # ADK components
    "RLMAgent",
    "RLMCodeExecutor",
    # Event types
    "RLMEventType",
    "RLMEventData",
    # RLM types
    "CodeBlock",
    "ModelUsageSummary",
    "REPLResult",
    "RLMChatCompletion",
    "RLMIteration",
    "RLMMetadata",
    "UsageSummary",
    # File handling
    "FileLoader",
    "FileMetadata",
    "FileParser",
    "FileSource",
    "LazyFile",
    "LazyFileCollection",
    "LoadedFile",
    "LocalFileSource",
    "ParsedContent",
    "PDFParser",
    "TextParser",
]

__version__ = "0.1.0"
