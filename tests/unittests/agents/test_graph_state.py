"""Tests for GraphState accessors and state_utils parsing functions."""

from __future__ import annotations

import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.adk.agents.graph.graph_state import GraphState
from google.adk.agents.graph.graph_state import PydanticJSONEncoder
from google.adk.agents.graph.state_utils import parse_state_value
from google.adk.agents.graph.state_utils import state_value_as_dict
from google.adk.agents.graph.state_utils import state_value_as_str
from pydantic import BaseModel
import pytest

# ── Test models ──────────────────────────────────────────────────────


class ReviewResult(BaseModel):
  decision: str
  reasoning: str


class NestedModel(BaseModel):
  name: str
  tags: List[str] = []
  metadata: Optional[Dict[str, Any]] = None


class StrictModel(BaseModel):
  value: int
  label: str


# ── parse_state_value ────────────────────────────────────────────────


class TestParseStateValue:

  def test_dict_to_model(self):
    raw = {"decision": "approve", "reasoning": "looks good"}
    result = parse_state_value(raw, ReviewResult)
    assert result is not None
    assert result.decision == "approve"
    assert result.reasoning == "looks good"

  def test_json_string_to_model(self):
    raw = json.dumps({"decision": "reject", "reasoning": "needs work"})
    result = parse_state_value(raw, ReviewResult)
    assert result is not None
    assert result.decision == "reject"

  def test_none_returns_default(self):
    default = ReviewResult(decision="skip", reasoning="default")
    result = parse_state_value(None, ReviewResult, default=default)
    assert result is default

  def test_none_returns_none_when_no_default(self):
    result = parse_state_value(None, ReviewResult)
    assert result is None

  def test_invalid_dict_returns_default(self):
    raw = {"wrong_key": 123}
    default = ReviewResult(decision="fallback", reasoning="bad data")
    result = parse_state_value(raw, ReviewResult, default=default)
    assert result is default

  def test_invalid_json_string_returns_default(self):
    raw = "not valid json at all"
    result = parse_state_value(raw, ReviewResult)
    assert result is None

  def test_unexpected_type_returns_default(self):
    result = parse_state_value(42, ReviewResult)
    assert result is None

  def test_unexpected_type_with_default(self):
    default = ReviewResult(decision="x", reasoning="y")
    result = parse_state_value(42, ReviewResult, default=default)
    assert result is default

  def test_nested_model(self):
    raw = {"name": "test", "tags": ["a", "b"], "metadata": {"key": "val"}}
    result = parse_state_value(raw, NestedModel)
    assert result is not None
    assert result.name == "test"
    assert result.tags == ["a", "b"]
    assert result.metadata == {"key": "val"}

  def test_json_string_nested_model(self):
    raw = json.dumps({"name": "test", "tags": ["x"]})
    result = parse_state_value(raw, NestedModel)
    assert result is not None
    assert result.name == "test"

  def test_empty_dict_with_required_fields(self):
    result = parse_state_value({}, StrictModel)
    assert result is None

  def test_list_type_returns_default(self):
    result = parse_state_value([1, 2, 3], ReviewResult)
    assert result is None

  def test_bool_type_returns_default(self):
    result = parse_state_value(True, ReviewResult)
    assert result is None


# ── state_value_as_str ───────────────────────────────────────────────


class TestStateValueAsStr:

  def test_string_value(self):
    assert state_value_as_str("hello") == "hello"

  def test_int_value(self):
    assert state_value_as_str(42) == "42"

  def test_float_value(self):
    assert state_value_as_str(3.14) == "3.14"

  def test_none_returns_default(self):
    assert state_value_as_str(None) == ""

  def test_none_custom_default(self):
    assert state_value_as_str(None, "N/A") == "N/A"

  def test_dict_value(self):
    result = state_value_as_str({"key": "val"})
    assert "key" in result

  def test_list_value(self):
    result = state_value_as_str([1, 2, 3])
    assert result == "[1, 2, 3]"

  def test_bool_value(self):
    assert state_value_as_str(True) == "True"
    assert state_value_as_str(False) == "False"

  def test_empty_string(self):
    assert state_value_as_str("") == ""


# ── state_value_as_dict ──────────────────────────────────────────────


class TestStateValueAsDict:

  def test_dict_value(self):
    raw = {"key": "val", "num": 1}
    assert state_value_as_dict(raw) == {"key": "val", "num": 1}

  def test_json_string(self):
    raw = json.dumps({"a": 1, "b": 2})
    assert state_value_as_dict(raw) == {"a": 1, "b": 2}

  def test_invalid_json_returns_default(self):
    assert state_value_as_dict("not json") == {}

  def test_invalid_json_custom_default(self):
    default = {"mode": "auto"}
    assert state_value_as_dict("not json", default=default) == default

  def test_none_returns_default(self):
    assert state_value_as_dict(None) == {}

  def test_none_custom_default(self):
    assert state_value_as_dict(None, default={"x": 1}) == {"x": 1}

  def test_non_dict_non_string_returns_default(self):
    assert state_value_as_dict(42) == {}

  def test_json_list_string_returns_default(self):
    """JSON list is valid JSON but not a dict."""
    assert state_value_as_dict("[1, 2, 3]") == {}

  def test_empty_dict(self):
    assert state_value_as_dict({}) == {}

  def test_empty_json_object(self):
    assert state_value_as_dict("{}") == {}

  def test_nested_dict(self):
    raw = {"outer": {"inner": "val"}}
    result = state_value_as_dict(raw)
    assert result["outer"]["inner"] == "val"


# ── GraphState.get_parsed ───────────────────────────────────────────


class TestGraphStateGetParsed:

  def test_dict_value(self):
    state = GraphState(
        data={"review": {"decision": "approve", "reasoning": "OK"}}
    )
    result = state.get_parsed("review", ReviewResult)
    assert result is not None
    assert result.decision == "approve"

  def test_json_string_value(self):
    state = GraphState(
        data={"review": json.dumps({"decision": "reject", "reasoning": "bad"})}
    )
    result = state.get_parsed("review", ReviewResult)
    assert result is not None
    assert result.decision == "reject"

  def test_missing_key(self):
    state = GraphState(data={})
    result = state.get_parsed("missing", ReviewResult)
    assert result is None

  def test_missing_key_with_default(self):
    default = ReviewResult(decision="default", reasoning="n/a")
    state = GraphState(data={})
    result = state.get_parsed("missing", ReviewResult, default=default)
    assert result is default

  def test_invalid_dict(self):
    state = GraphState(data={"bad": {"wrong": 123}})
    result = state.get_parsed("bad", ReviewResult)
    assert result is None

  def test_unexpected_type(self):
    state = GraphState(data={"val": 42})
    result = state.get_parsed("val", ReviewResult)
    assert result is None


# ── GraphState.get_str ──────────────────────────────────────────────


class TestGraphStateGetStr:

  def test_string_value(self):
    state = GraphState(data={"text": "hello"})
    assert state.get_str("text") == "hello"

  def test_non_string_value(self):
    state = GraphState(data={"num": 42})
    assert state.get_str("num") == "42"

  def test_missing_key(self):
    state = GraphState(data={})
    assert state.get_str("missing") == ""

  def test_missing_key_custom_default(self):
    state = GraphState(data={})
    assert state.get_str("missing", default="N/A") == "N/A"

  def test_none_value(self):
    state = GraphState(data={"val": None})
    assert state.get_str("val") == ""


# ── GraphState.get_dict ─────────────────────────────────────────────


class TestGraphStateGetDict:

  def test_dict_value(self):
    state = GraphState(data={"config": {"mode": "fast"}})
    assert state.get_dict("config") == {"mode": "fast"}

  def test_json_string_value(self):
    state = GraphState(data={"config": '{"mode": "fast"}'})
    assert state.get_dict("config") == {"mode": "fast"}

  def test_invalid_json_string(self):
    state = GraphState(data={"config": "not json"})
    assert state.get_dict("config") == {}

  def test_missing_key(self):
    state = GraphState(data={})
    assert state.get_dict("missing") == {}

  def test_missing_key_custom_default(self):
    state = GraphState(data={})
    assert state.get_dict("missing", default={"x": 1}) == {"x": 1}

  def test_non_dict_non_string(self):
    state = GraphState(data={"val": 42})
    assert state.get_dict("val") == {}


# ── GraphState.data_to_json ─────────────────────────────────────────


class ChildModel(BaseModel):
  score: float
  label: str


class TestDataToJson:

  def test_plain_dict(self):
    state = GraphState(data={"key": "val", "num": 1})
    result = state.data_to_json()
    parsed = json.loads(result)
    assert parsed == {"key": "val", "num": 1}

  def test_nested_pydantic(self):
    state = GraphState(data={"result": ChildModel(score=0.95, label="good")})
    result = state.data_to_json()
    parsed = json.loads(result)
    assert parsed["result"]["score"] == 0.95
    assert parsed["result"]["label"] == "good"

  def test_empty_data(self):
    state = GraphState()
    result = state.data_to_json()
    assert json.loads(result) == {}

  def test_indent(self):
    state = GraphState(data={"a": 1})
    compact = state.data_to_json(indent=0)
    assert "\n" in compact  # indent=0 still adds newlines
    no_indent = json.dumps({"a": 1})
    assert len(compact) >= len(no_indent)


# ── PydanticJSONEncoder ─────────────────────────────────────────────


class TestPydanticJSONEncoder:

  def test_encodes_model(self):
    model = ReviewResult(decision="yes", reasoning="ok")
    result = json.dumps(model, cls=PydanticJSONEncoder)
    parsed = json.loads(result)
    assert parsed["decision"] == "yes"

  def test_encodes_nested_model(self):
    data = {"child": ChildModel(score=0.5, label="mid")}
    result = json.dumps(data, cls=PydanticJSONEncoder)
    parsed = json.loads(result)
    assert parsed["child"]["score"] == 0.5

  def test_plain_types(self):
    data = {"str": "a", "int": 1, "float": 1.5, "bool": True, "null": None}
    result = json.dumps(data, cls=PydanticJSONEncoder)
    parsed = json.loads(result)
    assert parsed == data

  def test_non_serializable_raises(self):
    with pytest.raises(TypeError):
      json.dumps({"obj": object()}, cls=PydanticJSONEncoder)
