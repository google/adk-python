"""
Tests for data types and serialization.
"""

import json

from adk_rlm.types import CodeBlock
from adk_rlm.types import ModelUsageSummary
from adk_rlm.types import QueryMetadata
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMChatCompletion
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
from adk_rlm.types import UsageSummary
import pytest


class TestModelUsageSummary:
  """Tests for ModelUsageSummary."""

  def test_to_dict(self):
    """Serialize to dict."""
    usage = ModelUsageSummary(
        total_calls=5, total_input_tokens=1000, total_output_tokens=500
    )
    d = usage.to_dict()

    assert d["total_calls"] == 5
    assert d["total_input_tokens"] == 1000
    assert d["total_output_tokens"] == 500

  def test_from_dict(self):
    """Deserialize from dict."""
    d = {
        "total_calls": 5,
        "total_input_tokens": 1000,
        "total_output_tokens": 500,
    }
    usage = ModelUsageSummary.from_dict(d)

    assert usage.total_calls == 5
    assert usage.total_input_tokens == 1000
    assert usage.total_output_tokens == 500

  def test_round_trip(self):
    """Serialize then deserialize."""
    original = ModelUsageSummary(
        total_calls=10, total_input_tokens=2000, total_output_tokens=1000
    )
    restored = ModelUsageSummary.from_dict(original.to_dict())

    assert restored.total_calls == original.total_calls
    assert restored.total_input_tokens == original.total_input_tokens
    assert restored.total_output_tokens == original.total_output_tokens


class TestUsageSummary:
  """Tests for UsageSummary."""

  def test_to_dict(self):
    """Serialize to dict."""
    usage = UsageSummary(
        model_usage_summaries={
            "gemini-pro": ModelUsageSummary(
                total_calls=3, total_input_tokens=500, total_output_tokens=200
            ),
            "gemini-flash": ModelUsageSummary(
                total_calls=10, total_input_tokens=1000, total_output_tokens=500
            ),
        }
    )
    d = usage.to_dict()

    assert "gemini-pro" in d["model_usage_summaries"]
    assert "gemini-flash" in d["model_usage_summaries"]

  def test_total_properties(self):
    """Test total properties."""
    usage = UsageSummary(
        model_usage_summaries={
            "model1": ModelUsageSummary(
                total_calls=3, total_input_tokens=500, total_output_tokens=200
            ),
            "model2": ModelUsageSummary(
                total_calls=7, total_input_tokens=500, total_output_tokens=300
            ),
        }
    )

    assert usage.total_calls == 10
    assert usage.total_input_tokens == 1000
    assert usage.total_output_tokens == 500


class TestREPLResult:
  """Tests for REPLResult."""

  def test_to_dict(self):
    """Serialize to dict."""
    result = REPLResult(
        stdout="Hello",
        stderr="",
        locals={"x": 42, "y": "test"},
        execution_time=0.5,
        rlm_calls=[],
    )
    d = result.to_dict()

    assert d["stdout"] == "Hello"
    assert d["stderr"] == ""
    assert d["execution_time"] == 0.5
    assert "x" in d["locals"]

  def test_serialize_complex_locals(self):
    """Locals with complex types are serialized safely."""
    import re

    result = REPLResult(
        stdout="",
        stderr="",
        locals={
            "x": 42,
            "func": lambda: None,
            "module": re,
            "nested": {"a": [1, 2, 3]},
        },
        execution_time=0.1,
    )
    d = result.to_dict()

    # Should not raise, should convert to string representations
    json_str = json.dumps(d)
    assert json_str  # Valid JSON

  def test_str_representation(self):
    """String representation."""
    result = REPLResult(
        stdout="output",
        stderr="error",
        locals={"x": 1},
        execution_time=0.123,
        rlm_calls=[],
    )
    s = str(result)

    assert "REPLResult" in s
    assert "0.123" in s


class TestCodeBlock:
  """Tests for CodeBlock."""

  def test_to_dict(self):
    """Serialize to dict."""
    result = REPLResult(
        stdout="42", stderr="", locals={"x": 42}, execution_time=0.1
    )
    block = CodeBlock(code="x = 42\nprint(x)", result=result)
    d = block.to_dict()

    assert d["code"] == "x = 42\nprint(x)"
    assert "result" in d
    assert d["result"]["stdout"] == "42"


class TestRLMIteration:
  """Tests for RLMIteration."""

  def test_to_dict(self):
    """Serialize to dict."""
    result = REPLResult(stdout="", stderr="", locals={}, execution_time=0.1)
    iteration = RLMIteration(
        prompt="test prompt",
        response="test response",
        code_blocks=[CodeBlock(code="x = 1", result=result)],
        final_answer="final",
        iteration_time=1.5,
    )
    d = iteration.to_dict()

    assert d["prompt"] == "test prompt"
    assert d["response"] == "test response"
    assert len(d["code_blocks"]) == 1
    assert d["final_answer"] == "final"
    assert d["iteration_time"] == 1.5

  def test_from_dict(self):
    """Deserialize from dict."""
    d = {
        "prompt": "prompt",
        "response": "response",
        "code_blocks": [],
        "final_answer": None,
        "iteration_time": 2.0,
    }
    iteration = RLMIteration.from_dict(d)

    assert iteration.prompt == "prompt"
    assert iteration.response == "response"
    assert iteration.iteration_time == 2.0


class TestRLMMetadata:
  """Tests for RLMMetadata."""

  def test_to_dict(self):
    """Serialize to dict."""
    metadata = RLMMetadata(
        root_model="gemini-pro",
        max_depth=1,
        max_iterations=30,
        backend="gemini",
        backend_kwargs={"model_name": "gemini-pro"},
        environment_type="local",
        environment_kwargs={},
        other_backends=["gemini-flash"],
    )
    d = metadata.to_dict()

    assert d["root_model"] == "gemini-pro"
    assert d["max_iterations"] == 30
    assert d["backend"] == "gemini"
    assert d["other_backends"] == ["gemini-flash"]


class TestRLMChatCompletion:
  """Tests for RLMChatCompletion."""

  def test_to_dict(self):
    """Serialize to dict."""
    completion = RLMChatCompletion(
        root_model="gemini-pro",
        prompt="test prompt",
        response="test response",
        usage_summary=UsageSummary(),
        execution_time=5.0,
    )
    d = completion.to_dict()

    assert d["root_model"] == "gemini-pro"
    assert d["response"] == "test response"
    assert d["execution_time"] == 5.0


class TestQueryMetadata:
  """Tests for QueryMetadata."""

  def test_string_context(self):
    """Metadata for string context."""
    meta = QueryMetadata("Hello, World!")

    assert meta.context_type == "str"
    assert meta.context_total_length == 13
    assert meta.context_lengths == [13]

  def test_dict_context(self):
    """Metadata for dict context."""
    meta = QueryMetadata({"key1": "value1", "key2": "value2"})

    assert meta.context_type == "dict"
    assert len(meta.context_lengths) == 2

  def test_list_context(self):
    """Metadata for list context."""
    meta = QueryMetadata(["chunk1", "chunk2", "chunk3"])

    assert meta.context_type == "list"
    assert len(meta.context_lengths) == 3

  def test_empty_list(self):
    """Metadata for empty list."""
    meta = QueryMetadata([])

    assert meta.context_type == "list"
    assert meta.context_total_length == 0
