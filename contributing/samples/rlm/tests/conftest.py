"""
Pytest configuration and fixtures for ADK-RLM tests.
"""

import os
from pathlib import Path

import pytest

# Skip E2E tests unless explicitly enabled
E2E_ENABLED = os.getenv("RLM_E2E_TESTS", "false").lower() == "true"


def pytest_configure(config):
  """Configure custom pytest markers."""
  config.addinivalue_line(
      "markers", "e2e: mark test as end-to-end (requires real LLM)"
  )


def pytest_collection_modifyitems(config, items):
  """Skip E2E tests unless explicitly enabled."""
  if not E2E_ENABLED:
    skip_e2e = pytest.mark.skip(
        reason="E2E tests disabled (set RLM_E2E_TESTS=true)"
    )
    for item in items:
      if "e2e" in item.keywords:
        item.add_marker(skip_e2e)


@pytest.fixture
def mock_llm_query():
  """Create a mock llm_query function for testing."""

  def _mock_llm_query(
      prompt: str,
      context=None,
      model: str | None = None,
      recursive: bool = True,
  ) -> str:
    return f"Mock response for: {prompt[:50]}..."

  return _mock_llm_query


@pytest.fixture
def mock_llm_query_batched(mock_llm_query):
  """Create a mock llm_query_batched function for testing."""

  def _mock_batched(
      prompts: list[str],
      contexts=None,
      model: str | None = None,
      recursive: bool = False,
      max_concurrent: int = 3,
  ) -> list[str]:
    return [mock_llm_query(p, model=model) for p in prompts]

  return _mock_batched


@pytest.fixture
def temp_log_dir(tmp_path):
  """Create a temporary directory for log files."""
  log_dir = tmp_path / "logs"
  log_dir.mkdir()
  return str(log_dir)


@pytest.fixture
def sample_context():
  """Return a sample context string for testing."""
  return (
      "This is a sample context with some text. It contains important"
      " information about testing. The magic number is 42."
  )


@pytest.fixture
def sample_context_dict():
  """Return a sample context dict for testing."""
  return {
      "title": "Test Document",
      "content": "This is the main content of the document.",
      "metadata": {"author": "Test Author", "date": "2024-01-01"},
  }


@pytest.fixture
def sample_context_list():
  """Return a sample context list for testing."""
  return [
      "First chunk of text with some information.",
      "Second chunk of text with more details.",
      "Third chunk of text with conclusions.",
  ]


@pytest.fixture
def fixtures_dir():
  """Return the path to the test fixtures directory."""
  return Path(__file__).parent / "fixtures"
