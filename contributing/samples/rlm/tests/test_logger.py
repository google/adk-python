"""
Tests for JSONL logging.
"""

import json
from pathlib import Path

from adk_rlm.logging.rlm_logger import RLMLogger
from adk_rlm.types import CodeBlock
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
import pytest


class TestRLMLogger:
  """Tests for RLMLogger."""

  def test_creates_log_file(self, temp_log_dir):
    """Logger creates log file."""
    logger = RLMLogger(temp_log_dir)

    assert Path(logger.log_file_path).parent.exists()

  def test_log_metadata(self, temp_log_dir):
    """Log metadata as first entry."""
    logger = RLMLogger(temp_log_dir)
    metadata = RLMMetadata(
        root_model="gemini-pro",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={"model_name": "gemini-pro"},
        environment_type="local",
        environment_kwargs={},
    )
    logger.log_metadata(metadata)

    # Read log file
    with open(logger.log_file_path) as f:
      lines = f.readlines()

    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"] == "metadata"
    assert entry["root_model"] == "gemini-pro"

  def test_log_iteration(self, temp_log_dir):
    """Log iteration."""
    logger = RLMLogger(temp_log_dir)
    iteration = RLMIteration(
        prompt="test prompt",
        response="test response",
        code_blocks=[],
        final_answer=None,
        iteration_time=1.0,
    )
    logger.log(iteration)

    with open(logger.log_file_path) as f:
      lines = f.readlines()

    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"] == "iteration"
    assert entry["iteration"] == 1
    assert entry["response"] == "test response"

  def test_iteration_count(self, temp_log_dir):
    """Iteration counter increments."""
    logger = RLMLogger(temp_log_dir)

    for i in range(3):
      iteration = RLMIteration(
          prompt="", response=f"response {i}", code_blocks=[]
      )
      logger.log(iteration)

    assert logger.iteration_count == 3

  def test_log_with_code_blocks(self, temp_log_dir):
    """Log iteration with code blocks."""
    logger = RLMLogger(temp_log_dir)

    result = REPLResult(
        stdout="42", stderr="", locals={"x": 42}, execution_time=0.1
    )
    code_block = CodeBlock(code="print(42)", result=result)
    iteration = RLMIteration(
        prompt="test",
        response="Let me calculate",
        code_blocks=[code_block],
        iteration_time=0.5,
    )
    logger.log(iteration)

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    assert len(entry["code_blocks"]) == 1
    assert entry["code_blocks"][0]["code"] == "print(42)"
    assert entry["code_blocks"][0]["result"]["stdout"] == "42"

  def test_metadata_logged_once(self, temp_log_dir):
    """Metadata only logged once."""
    logger = RLMLogger(temp_log_dir)
    metadata = RLMMetadata(
        root_model="gemini-pro",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
    )

    logger.log_metadata(metadata)
    logger.log_metadata(metadata)  # Second call should be ignored

    with open(logger.log_file_path) as f:
      lines = f.readlines()

    metadata_entries = [l for l in lines if '"type": "metadata"' in l]
    assert len(metadata_entries) == 1

  def test_get_log_path(self, temp_log_dir):
    """Get log path."""
    logger = RLMLogger(temp_log_dir)
    path = logger.get_log_path()

    assert path == logger.log_file_path
    assert temp_log_dir in path
