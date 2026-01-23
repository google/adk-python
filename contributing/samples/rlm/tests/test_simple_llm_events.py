"""
Tests for non-recursive (simple) LLM call events and logging.

These tests verify that when llm_query() or llm_query_batched() is called with
recursive=False, proper events are emitted and calls are logged.
"""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

from adk_rlm.code_executor import RLMCodeExecutor
from adk_rlm.events import RLMEventType
from adk_rlm.logging.rlm_logger import RLMLogger
import pytest


class TestLoggerSimpleLLMCall:
  """Tests for RLMLogger.log_simple_llm_call method."""

  def test_log_simple_llm_call_success(self, temp_log_dir):
    """Log a successful simple LLM call."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="What is 2+2?",
        response="The answer is 4.",
        model="gemini-3-flash-preview",
        execution_time_ms=150.5,
        depth=0,
        agent_name="rlm_agent",
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["prompt"] == "What is 2+2?"
    assert entry["response"] == "The answer is 4."
    assert entry["model"] == "gemini-3-flash-preview"
    assert entry["execution_time_ms"] == 150.5
    assert entry["depth"] == 0
    assert entry["agent_name"] == "rlm_agent"
    assert entry["recursive"] is False
    assert entry["success"] is True
    assert "error" not in entry

  def test_log_simple_llm_call_failure(self, temp_log_dir):
    """Log a failed simple LLM call."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="What is 2+2?",
        response="Error: LLM query failed - Connection timeout",
        model="gemini-3-flash-preview",
        execution_time_ms=5000.0,
        error="Connection timeout",
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["success"] is False
    assert entry["error"] == "Connection timeout"
    assert "Error:" in entry["response"]

  def test_log_simple_llm_call_with_batch_metadata(self, temp_log_dir):
    """Log a simple LLM call with batch metadata."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="Query 1",
        response="Response 1",
        model="gemini-3-flash-preview",
        execution_time_ms=100.0,
        batch_index=0,
        batch_size=3,
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["batch_index"] == 0
    assert entry["batch_size"] == 3

  def test_log_simple_llm_call_with_parent_context(self, temp_log_dir):
    """Log a simple LLM call with parent iteration context."""
    logger = RLMLogger(temp_log_dir)
    logger.log_simple_llm_call(
        prompt="Sub query",
        response="Sub response",
        model="gemini-3-flash-preview",
        execution_time_ms=100.0,
        parent_iteration=2,
        parent_block_index=1,
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["parent_iteration"] == 2
    assert entry["parent_block_index"] == 1

  def test_log_simple_llm_call_truncates_long_prompts(self, temp_log_dir):
    """Long prompts are truncated in summary but preserved in full."""
    logger = RLMLogger(temp_log_dir)
    long_prompt = "x" * 1000
    logger.log_simple_llm_call(
        prompt=long_prompt,
        response="Short response",
        model="test-model",
        execution_time_ms=100.0,
    )

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert len(entry["prompt"]) == 500
    assert len(entry["prompt_full"]) == 1000


class TestCodeExecutorEmitSubLLMEvent:
  """Tests for RLMCodeExecutor._emit_sub_llm_event method."""

  def test_emit_sub_llm_start_event(self):
    """Emit SUB_LLM_START event."""
    executor = RLMCodeExecutor(
        sub_model="gemini-3-flash-preview",
        current_depth=0,
        max_depth=5,
        parent_agent="rlm_agent",
    )
    executor._current_iteration = 2
    executor._current_block_index = 1

    executor._emit_sub_llm_event(
        RLMEventType.SUB_LLM_START,
        model="gemini-3-flash-preview",
        prompt="Test prompt",
    )

    # Check event was queued
    assert not executor._event_queue.empty()
    event = executor._event_queue.get()

    metadata = event.custom_metadata
    assert metadata["event_type"] == RLMEventType.SUB_LLM_START.value
    assert metadata["model"] == "gemini-3-flash-preview"
    assert metadata["prompt_preview"] == "Test prompt"
    assert metadata["iteration"] == 2
    assert metadata["block_index"] == 1
    assert metadata["agent_name"] == "rlm_agent"
    assert metadata["agent_depth"] == 0
    assert metadata["metadata"]["recursive"] is False

  def test_emit_sub_llm_end_event_success(self):
    """Emit SUB_LLM_END event on success."""
    executor = RLMCodeExecutor(sub_model="test-model")

    executor._emit_sub_llm_event(
        RLMEventType.SUB_LLM_END,
        model="test-model",
        response="The answer is 42.",
        execution_time_ms=150.0,
    )

    event = executor._event_queue.get()
    metadata = event.custom_metadata

    assert metadata["event_type"] == RLMEventType.SUB_LLM_END.value
    assert metadata["response_preview"] == "The answer is 42."
    assert metadata["response_full"] == "The answer is 42."
    assert metadata["execution_time_ms"] == 150.0
    assert metadata.get("error") is None

  def test_emit_sub_llm_end_event_failure(self):
    """Emit SUB_LLM_END event on failure."""
    executor = RLMCodeExecutor(sub_model="test-model")

    executor._emit_sub_llm_event(
        RLMEventType.SUB_LLM_END,
        model="test-model",
        error="API rate limit exceeded",
        execution_time_ms=100.0,
    )

    event = executor._event_queue.get()
    metadata = event.custom_metadata

    assert metadata["event_type"] == RLMEventType.SUB_LLM_END.value
    assert metadata["error"] == "API rate limit exceeded"
    assert metadata.get("response_preview") is None

  def test_emit_sub_llm_event_with_batch_metadata(self):
    """Emit SUB_LLM event with batch metadata."""
    executor = RLMCodeExecutor(sub_model="test-model")

    executor._emit_sub_llm_event(
        RLMEventType.SUB_LLM_START,
        model="test-model",
        prompt="Batch query",
        batch_index=2,
        batch_size=5,
    )

    event = executor._event_queue.get()
    metadata = event.custom_metadata

    assert metadata["batch_index"] == 2
    assert metadata["batch_size"] == 5


class TestCodeExecutorSimpleLLMCall:
  """Tests for RLMCodeExecutor._simple_llm_call method."""

  def test_simple_llm_call_emits_events(self):
    """Simple LLM call emits START and END events."""
    executor = RLMCodeExecutor(sub_model="test-model")

    # Mock the genai client (fresh client is created inside _simple_llm_call)
    mock_response = MagicMock()
    mock_response.text = "Mocked response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("adk_rlm.code_executor.genai.Client", return_value=mock_client):
      result = executor._simple_llm_call("Test prompt", "test-model")

    assert result == "Mocked response"

    # Should have 2 events: START and END
    events = []
    while not executor._event_queue.empty():
      events.append(executor._event_queue.get())

    assert len(events) == 2

    start_event = events[0]
    end_event = events[1]

    assert (
        start_event.custom_metadata["event_type"]
        == RLMEventType.SUB_LLM_START.value
    )
    assert (
        end_event.custom_metadata["event_type"]
        == RLMEventType.SUB_LLM_END.value
    )
    assert end_event.custom_metadata["response_full"] == "Mocked response"
    assert end_event.custom_metadata.get("error") is None

  def test_simple_llm_call_emits_events_on_error(self):
    """Simple LLM call emits events even when it fails."""
    executor = RLMCodeExecutor(sub_model="test-model")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")

    with patch("adk_rlm.code_executor.genai.Client", return_value=mock_client):
      result = executor._simple_llm_call("Test prompt", "test-model")

    assert "Error: LLM query failed" in result
    assert "API error" in result

    events = []
    while not executor._event_queue.empty():
      events.append(executor._event_queue.get())

    assert len(events) == 2

    end_event = events[1]
    assert (
        end_event.custom_metadata["event_type"]
        == RLMEventType.SUB_LLM_END.value
    )
    assert end_event.custom_metadata["error"] == "API error"

  def test_simple_llm_call_logs_to_jsonl(self, temp_log_dir):
    """Simple LLM call logs to JSONL logger."""
    logger = RLMLogger(temp_log_dir)
    executor = RLMCodeExecutor(
        sub_model="test-model",
        logger=logger,
        parent_agent="rlm_agent",
    )
    executor._current_iteration = 1
    executor._current_block_index = 0

    mock_response = MagicMock()
    mock_response.text = "Logged response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("adk_rlm.code_executor.genai.Client", return_value=mock_client):
      executor._simple_llm_call("Logged prompt", "test-model")

    with open(logger.get_log_path()) as f:
      entry = json.loads(f.readline())

    assert entry["type"] == "simple_llm_call"
    assert entry["prompt_full"] == "Logged prompt"
    assert entry["response_full"] == "Logged response"
    assert entry["agent_name"] == "rlm_agent"
    assert entry["parent_iteration"] == 1
    assert entry["success"] is True


class TestCodeExecutorBatchedNonRecursive:
  """Tests for llm_query_batched with recursive=False."""

  def test_batched_non_recursive_emits_events(self):
    """Batched non-recursive calls emit events for each query."""
    executor = RLMCodeExecutor(sub_model="test-model")

    mock_response = MagicMock()
    mock_response.text = "Batch response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20

    # Mock the fresh client created inside run_all()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content.return_value = mock_response

    llm_query_batched = executor._create_llm_query_batched_fn()

    with patch("adk_rlm.code_executor.genai.Client", return_value=mock_client):
      results = llm_query_batched(
          ["Query 1", "Query 2", "Query 3"],
          recursive=False,
      )

    assert len(results) == 3

    # Collect all events
    events = []
    while not executor._event_queue.empty():
      events.append(executor._event_queue.get())

    # Should have 2 events per query (START + END) = 6 events
    assert len(events) == 6

    start_events = [
        e
        for e in events
        if e.custom_metadata["event_type"] == RLMEventType.SUB_LLM_START.value
    ]
    end_events = [
        e
        for e in events
        if e.custom_metadata["event_type"] == RLMEventType.SUB_LLM_END.value
    ]

    assert len(start_events) == 3
    assert len(end_events) == 3

    # Check batch metadata
    for event in events:
      assert event.custom_metadata["batch_size"] == 3
      assert event.custom_metadata["batch_index"] in [0, 1, 2]

  def test_batched_non_recursive_logs_all_calls(self, temp_log_dir):
    """Batched non-recursive calls log all queries."""
    logger = RLMLogger(temp_log_dir)
    executor = RLMCodeExecutor(
        sub_model="test-model",
        logger=logger,
    )

    mock_response = MagicMock()
    mock_response.text = "Batch response"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20

    # Mock the fresh client created inside run_all()
    mock_client = MagicMock()
    mock_client.aio.models.generate_content.return_value = mock_response

    llm_query_batched = executor._create_llm_query_batched_fn()

    with patch("adk_rlm.code_executor.genai.Client", return_value=mock_client):
      llm_query_batched(["Q1", "Q2"], recursive=False)

    with open(logger.get_log_path()) as f:
      entries = [json.loads(line) for line in f]

    assert len(entries) == 2
    assert all(e["type"] == "simple_llm_call" for e in entries)
    assert all(e["batch_size"] == 2 for e in entries)

    batch_indices = {e["batch_index"] for e in entries}
    assert batch_indices == {0, 1}
