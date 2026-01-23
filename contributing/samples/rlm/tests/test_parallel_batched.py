"""
Tests for parallel batched queries (llm_query_batched with recursive=True).

This module tests the parallel execution of recursive RLM child agents,
including batch metadata propagation and iteration linking.
"""

import time
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from adk_rlm.code_executor import RLMCodeExecutor
from adk_rlm.events import RLMEventData
from adk_rlm.events import RLMEventType
import pytest


class TestParallelRecursiveBatchedExecution:
  """Tests for parallel execution of recursive batched queries."""

  def test_parallel_recursive_method_exists(self):
    """Verify _run_parallel_recursive method exists."""
    executor = RLMCodeExecutor()
    assert hasattr(executor, "_run_parallel_recursive")

  def test_run_recursive_rlm_accepts_batch_params(self):
    """Verify _run_recursive_rlm accepts batch metadata parameters."""
    import inspect

    executor = RLMCodeExecutor()
    sig = inspect.signature(executor._run_recursive_rlm)
    params = list(sig.parameters.keys())

    assert "parallel_batch_id" in params
    assert "batch_index" in params
    assert "batch_size" in params


class TestBatchMetadataEventData:
  """Tests for batch metadata in RLMEventData."""

  def test_event_data_has_batch_fields(self):
    """Verify RLMEventData has parallel batch fields."""
    event_data = RLMEventData(
        event_type=RLMEventType.ITERATION_START,
        parallel_batch_id="test-batch-123",
        batch_index=0,
        batch_size=3,
    )

    assert event_data.parallel_batch_id == "test-batch-123"
    assert event_data.batch_index == 0
    assert event_data.batch_size == 3

  def test_event_data_to_dict_includes_batch_fields(self):
    """Verify to_dict includes batch fields when set."""
    event_data = RLMEventData(
        event_type=RLMEventType.ITERATION_START,
        iteration=1,
        parallel_batch_id="batch-abc",
        batch_index=2,
        batch_size=5,
    )

    data = event_data.to_dict()
    assert data["parallel_batch_id"] == "batch-abc"
    assert data["batch_index"] == 2
    assert data["batch_size"] == 5

  def test_event_data_to_dict_excludes_none_batch_fields(self):
    """Verify to_dict excludes batch fields when None."""
    event_data = RLMEventData(
        event_type=RLMEventType.ITERATION_START,
        iteration=1,
    )

    data = event_data.to_dict()
    assert "parallel_batch_id" not in data
    assert "batch_index" not in data
    assert "batch_size" not in data


class TestLoggerBatchTracking:
  """Tests for batch tracking in RLMLogger."""

  def test_logger_accepts_batch_params(self):
    """Verify logger.log accepts batch metadata parameters."""
    import inspect

    from adk_rlm.logging.rlm_logger import RLMLogger

    sig = inspect.signature(RLMLogger.log)
    params = list(sig.parameters.keys())

    assert "parent_iteration" in params
    assert "parent_block_index" in params
    assert "parallel_batch_id" in params
    assert "batch_index" in params
    assert "batch_size" in params

  def test_logger_writes_batch_metadata(self, temp_log_dir):
    """Verify logger writes batch metadata to log file."""
    import json

    from adk_rlm.logging.rlm_logger import RLMLogger
    from adk_rlm.types import RLMIteration

    logger = RLMLogger(log_dir=temp_log_dir)
    iteration = RLMIteration(
        prompt=[{"role": "user", "content": "test"}],
        response="test response",
        code_blocks=[],
    )

    logger.log(
        iteration,
        depth=1,
        agent_name="rlm_agent_depth_1_0",
        parent_agent="rlm_agent",
        parent_iteration=2,
        parent_block_index=0,
        parallel_batch_id="batch-123",
        batch_index=1,
        batch_size=3,
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["parent_iteration"] == 2
    assert entry["parent_block_index"] == 0
    assert entry["parallel_batch_id"] == "batch-123"
    assert entry["batch_index"] == 1
    assert entry["batch_size"] == 3


class TestParallelExecutionBehavior:
  """Tests for actual parallel execution behavior."""

  def test_parallel_batched_preserves_order(self):
    """Verify results are returned in original prompt order."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    # Mock _run_recursive_rlm to return predictable results with delays
    call_order = []
    original_run = executor._run_recursive_rlm

    def mock_run(prompt, model, context_obj=None, **kwargs):
      call_order.append(prompt)
      # Simulate varying execution times
      if "Q1" in prompt:
        time.sleep(0.05)
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3"]
    results = executor._run_parallel_recursive(prompts, None, "test-model")

    # Results should be in original order
    assert results[0] == "Result for Q1"
    assert results[1] == "Result for Q2"
    assert results[2] == "Result for Q3"

  def test_parallel_batched_generates_batch_id(self):
    """Verify parallel execution generates a batch ID."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    batch_ids_seen = []

    def mock_run(
        prompt, model, context_obj=None, parallel_batch_id=None, **kwargs
    ):
      batch_ids_seen.append(parallel_batch_id)
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3"]
    executor._run_parallel_recursive(prompts, None, "test-model")

    # All calls should have the same batch ID
    assert len(batch_ids_seen) == 3
    assert batch_ids_seen[0] is not None
    assert batch_ids_seen[0] == batch_ids_seen[1] == batch_ids_seen[2]
    # Batch ID should be a valid UUID format
    import uuid

    uuid.UUID(batch_ids_seen[0])  # Will raise if invalid

  def test_parallel_batched_passes_batch_index(self):
    """Verify parallel execution passes correct batch indices."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    indices_seen = {}

    def mock_run(prompt, model, context_obj=None, batch_index=None, **kwargs):
      indices_seen[prompt] = batch_index
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3"]
    executor._run_parallel_recursive(prompts, None, "test-model")

    assert indices_seen["Q1"] == 0
    assert indices_seen["Q2"] == 1
    assert indices_seen["Q3"] == 2

  def test_parallel_batched_passes_batch_size(self):
    """Verify parallel execution passes correct batch size."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    sizes_seen = []

    def mock_run(prompt, model, context_obj=None, batch_size=None, **kwargs):
      sizes_seen.append(batch_size)
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3", "Q4"]
    executor._run_parallel_recursive(prompts, None, "test-model")

    assert all(s == 4 for s in sizes_seen)

  def test_parallel_batched_passes_contexts(self):
    """Verify contexts are passed correctly to each child."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    contexts_seen = {}

    def mock_run(prompt, model, context_obj=None, **kwargs):
      contexts_seen[prompt] = context_obj
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3"]
    contexts = ["Context A", "Context B", "Context C"]
    executor._run_parallel_recursive(prompts, contexts, "test-model")

    assert contexts_seen["Q1"] == "Context A"
    assert contexts_seen["Q2"] == "Context B"
    assert contexts_seen["Q3"] == "Context C"

  def test_parallel_batched_handles_exceptions(self):
    """Verify parallel execution handles individual failures gracefully."""
    executor = RLMCodeExecutor(
        current_depth=0,
        max_depth=1,
    )

    def mock_run(prompt, model, context_obj=None, **kwargs):
      if "Q2" in prompt:
        raise ValueError("Simulated failure")
      return f"Result for {prompt}"

    executor._run_recursive_rlm = mock_run

    prompts = ["Q1", "Q2", "Q3"]
    results = executor._run_parallel_recursive(prompts, None, "test-model")

    assert results[0] == "Result for Q1"
    assert "Error" in results[1]
    assert "Simulated failure" in results[1]
    assert results[2] == "Result for Q3"


class TestIterationLinking:
  """Tests for linking child agents to spawning iteration."""

  def test_ancestry_includes_iteration_and_block(self):
    """Verify ancestry entry includes iteration and block_index."""
    executor = RLMCodeExecutor(
        parent_agent="rlm_agent",
    )

    executor.set_iteration_context(iteration=3, block_index=1)
    entry = executor._get_current_ancestry_entry()

    assert entry["agent"] == "rlm_agent"
    assert entry["iteration"] == 3
    assert entry["block_index"] == 1

  def test_child_ancestry_chain_preserved(self):
    """Verify child agents receive full ancestry chain."""
    parent_ancestry = [
        {"agent": "rlm_agent", "depth": 0, "iteration": 1, "block_index": 0}
    ]

    executor = RLMCodeExecutor(
        parent_agent="rlm_agent_depth_1_0",
        ancestry=parent_ancestry,
        current_depth=1,
    )

    executor.set_iteration_context(iteration=2, block_index=0)
    entry = executor._get_current_ancestry_entry()

    # Current entry should reflect current agent's context
    assert entry["agent"] == "rlm_agent_depth_1_0"
    assert entry["iteration"] == 2
    assert entry["block_index"] == 0

    # Full ancestry should include parent + current
    full_ancestry = executor._ancestry + [entry]
    assert len(full_ancestry) == 2
    assert full_ancestry[0]["agent"] == "rlm_agent"
    assert full_ancestry[1]["agent"] == "rlm_agent_depth_1_0"
