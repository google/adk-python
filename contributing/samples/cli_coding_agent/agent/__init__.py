"""Coding agent package for file system operations with ADK."""

import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(
    level=logging.INFO, log_file='coding_agent.log', quiet_adk=True
):
  """Configure logging for the coding agent package.

  Args:
      level: Logging level (default: INFO)
      log_file: Path to log file (default: 'coding_agent.log')
      quiet_adk: If True, set ADK loggers to WARNING level (default: True)
  """
  # Create formatters
  detailed_formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
  )

  # Console handler (simple format)
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(level)
  console_handler.setFormatter(detailed_formatter)

  # File handler (detailed format)
  log_path = Path(log_file)
  file_handler = logging.FileHandler(log_path)
  file_handler.setLevel(logging.DEBUG)  # Log everything to file
  file_handler.setFormatter(detailed_formatter)

  # Configure root logger
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.DEBUG)
  root_logger.addHandler(console_handler)
  root_logger.addHandler(file_handler)

  # Quiet down noisy libraries
  if quiet_adk:
    logging.getLogger('google_adk').setLevel(logging.WARNING)
    logging.getLogger('google.genai').setLevel(logging.WARNING)
    logging.getLogger('google_genai.types').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)

  return logging.getLogger(__name__)


# Initialize logging
logger = setup_logging()
logger.info('Coding agent package initialized')

# Import agent after logging is configured
from . import agent

__all__ = ['agent', 'logger', 'setup_logging']
