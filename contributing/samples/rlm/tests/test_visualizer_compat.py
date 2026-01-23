"""
Visualizer compatibility tests.

These tests verify that ADK-RLM JSONL output is compatible with the
original RLM visualizer by validating the log schema.
"""

import json
from pathlib import Path

from adk_rlm.logging.rlm_logger import RLMLogger
from adk_rlm.types import CodeBlock
from adk_rlm.types import ModelUsageSummary
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMChatCompletion
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
from adk_rlm.types import UsageSummary
import pytest


class TestVisualizerSchemaCompatibility:
  """Tests that verify JSONL output matches visualizer expectations."""

  def test_metadata_schema(self, temp_log_dir):
    """Verify metadata entry matches expected schema."""
    logger = RLMLogger(temp_log_dir)
    metadata = RLMMetadata(
        root_model="gemini-3-pro-preview",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={"model_name": "gemini-3-pro-preview"},
        environment_type="local",
        environment_kwargs={},
        other_backends=["gemini-3-flash-preview"],
    )
    logger.log_metadata(metadata)

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    # Required fields for visualizer
    assert entry["type"] == "metadata"
    assert "timestamp" in entry
    assert "root_model" in entry
    assert "max_depth" in entry
    assert "max_iterations" in entry
    assert "backend" in entry
    assert "backend_kwargs" in entry
    assert "environment_type" in entry
    assert "environment_kwargs" in entry
    assert "other_backends" in entry

  def test_iteration_schema(self, temp_log_dir):
    """Verify iteration entry matches expected schema."""
    logger = RLMLogger(temp_log_dir)

    # Create an iteration with code blocks and sub-calls
    sub_call = RLMChatCompletion(
        root_model="gemini-3-flash-preview",
        prompt="What is 2+2?",
        response="4",
        usage_summary=UsageSummary(
            model_usage_summaries={
                "gemini-3-flash-preview": ModelUsageSummary(
                    total_calls=1, total_input_tokens=10, total_output_tokens=5
                )
            }
        ),
        execution_time=0.5,
    )

    result = REPLResult(
        stdout="Output here",
        stderr="",
        locals={"x": 42, "y": "test"},
        execution_time=0.1,
        rlm_calls=[sub_call],
    )

    code_block = CodeBlock(code="x = 42\nprint(x)", result=result)

    iteration = RLMIteration(
        prompt=[{"role": "user", "content": "test"}],
        response="Let me calculate...\n```repl\nx = 42\nprint(x)\n```",
        code_blocks=[code_block],
        final_answer=None,
        iteration_time=1.5,
    )
    logger.log(iteration)

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    # Required fields for visualizer iteration
    assert entry["type"] == "iteration"
    assert "iteration" in entry
    assert "timestamp" in entry
    assert "prompt" in entry
    assert "response" in entry
    assert "code_blocks" in entry
    assert "final_answer" in entry
    assert "iteration_time" in entry

    # Check code block structure
    assert len(entry["code_blocks"]) == 1
    cb = entry["code_blocks"][0]
    assert "code" in cb
    assert "result" in cb

    # Check result structure
    result_entry = cb["result"]
    assert "stdout" in result_entry
    assert "stderr" in result_entry
    assert "locals" in result_entry
    assert "execution_time" in result_entry
    assert "rlm_calls" in result_entry

    # Check rlm_calls (sub-calls) structure
    assert len(result_entry["rlm_calls"]) == 1
    call = result_entry["rlm_calls"][0]
    assert "root_model" in call
    assert "prompt" in call
    assert "response" in call
    assert "usage_summary" in call
    assert "execution_time" in call

  def test_full_log_file_structure(self, temp_log_dir):
    """Test complete log file with metadata and multiple iterations."""
    logger = RLMLogger(temp_log_dir)

    # Log metadata
    metadata = RLMMetadata(
        root_model="gemini-3-pro-preview",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={"model_name": "gemini-3-pro-preview"},
        environment_type="local",
        environment_kwargs={},
    )
    logger.log_metadata(metadata)

    # Log multiple iterations
    for i in range(3):
      result = REPLResult(
          stdout=f"Output {i}",
          stderr="",
          locals={},
          execution_time=0.1,
      )
      iteration = RLMIteration(
          prompt=f"Iteration {i}",
          response=f"Response {i}",
          code_blocks=[CodeBlock(code=f"print({i})", result=result)],
          final_answer="Final" if i == 2 else None,
          iteration_time=0.5,
      )
      logger.log(iteration)

    # Verify structure
    with open(logger.log_file_path) as f:
      lines = f.readlines()

    assert len(lines) == 4  # 1 metadata + 3 iterations

    entries = [json.loads(line) for line in lines]
    assert entries[0]["type"] == "metadata"
    for i, entry in enumerate(entries[1:], 1):
      assert entry["type"] == "iteration"
      assert entry["iteration"] == i

  def test_field_naming_consistency(self):
    """Verify we use the same field names as original RLM."""
    # REPLResult must use 'rlm_calls' not 'llm_calls'
    result = REPLResult(
        stdout="",
        stderr="",
        locals={},
        execution_time=0.1,
        rlm_calls=[],
    )
    d = result.to_dict()
    assert "rlm_calls" in d
    assert "llm_calls" not in d

    # RLMIteration must have specific fields
    iteration = RLMIteration(
        prompt="test",
        response="response",
        code_blocks=[],
        final_answer=None,
        iteration_time=1.0,
    )
    d = iteration.to_dict()
    assert "prompt" in d
    assert "response" in d
    assert "code_blocks" in d
    assert "final_answer" in d
    assert "iteration_time" in d

    # RLMMetadata must have specific fields
    metadata = RLMMetadata(
        root_model="model",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={},
        environment_type="local",
        environment_kwargs={},
    )
    d = metadata.to_dict()
    assert "root_model" in d
    assert "max_depth" in d
    assert "max_iterations" in d
    assert "backend" in d
    assert "backend_kwargs" in d
    assert "environment_type" in d
    assert "environment_kwargs" in d

  def test_json_serializable(self, temp_log_dir):
    """All log entries must be valid JSON."""
    logger = RLMLogger(temp_log_dir)

    # Create complex structures
    metadata = RLMMetadata(
        root_model="gemini-pro",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={"nested": {"key": "value"}},
        environment_type="local",
        environment_kwargs={"list": [1, 2, 3]},
    )
    logger.log_metadata(metadata)

    # Log with complex locals
    result = REPLResult(
        stdout="test",
        stderr="",
        locals={
            "list_var": [1, 2, 3],
            "dict_var": {"a": 1, "b": {"c": 2}},
            "tuple_var": (1, 2),
        },
        execution_time=0.1,
    )
    iteration = RLMIteration(
        prompt="test",
        response="response",
        code_blocks=[CodeBlock(code="pass", result=result)],
    )
    logger.log(iteration)

    # Verify all entries are valid JSON
    with open(logger.log_file_path) as f:
      for line in f:
        entry = json.loads(line)
        # Re-serialize to ensure no issues
        json.dumps(entry)


class TestVisualizerFieldTypes:
  """Tests for correct field types in log entries."""

  def test_timestamp_is_iso_format(self, temp_log_dir):
    """Timestamps should be ISO format strings."""
    from datetime import datetime

    logger = RLMLogger(temp_log_dir)
    logger.log_metadata(
        RLMMetadata(
            root_model="model",
            max_depth=1,
            max_iterations=30,
            backend="gemini",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
        )
    )

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    # Should be parseable as ISO timestamp
    timestamp = entry["timestamp"]
    datetime.fromisoformat(timestamp)

  def test_iteration_number_is_integer(self, temp_log_dir):
    """Iteration number should be an integer."""
    logger = RLMLogger(temp_log_dir)
    logger.log(
        RLMIteration(
            prompt="test",
            response="response",
            code_blocks=[],
        )
    )

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    assert isinstance(entry["iteration"], int)
    assert entry["iteration"] == 1

  def test_execution_times_are_floats(self, temp_log_dir):
    """Execution times should be floats."""
    logger = RLMLogger(temp_log_dir)

    result = REPLResult(
        stdout="",
        stderr="",
        locals={},
        execution_time=0.123456,
    )
    iteration = RLMIteration(
        prompt="test",
        response="response",
        code_blocks=[CodeBlock(code="pass", result=result)],
        iteration_time=1.5,
    )
    logger.log(iteration)

    with open(logger.log_file_path) as f:
      entry = json.loads(f.readline())

    assert isinstance(entry["iteration_time"], float)
    assert isinstance(
        entry["code_blocks"][0]["result"]["execution_time"], float
    )
