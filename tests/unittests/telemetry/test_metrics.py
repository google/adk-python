# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=protected-access

from unittest import mock

from google.adk.telemetry import _metrics
from google.adk.telemetry import _token_usage
from google.genai import types
from opentelemetry import metrics
import pytest


@pytest.fixture(name="mock_meter_setup")
def _mock_meter_setup(monkeypatch):
  """Sets up mock meter and histograms for testing."""
  mock_meter = mock.MagicMock()
  agent_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  workflow_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  tool_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  client_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  client_token_usage_hist = mock.MagicMock(spec=metrics.Histogram)
  input_tokens_hist = mock.MagicMock(spec=metrics.Histogram)
  output_tokens_hist = mock.MagicMock(spec=metrics.Histogram)
  total_tokens_hist = mock.MagicMock(spec=metrics.Histogram)
  cache_read_input_tokens_hist = mock.MagicMock(spec=metrics.Histogram)
  reasoning_output_tokens_hist = mock.MagicMock(spec=metrics.Histogram)
  tool_input_tokens_hist = mock.MagicMock(spec=metrics.Histogram)

  agent_duration_hist.name = "agent_invocation_duration"
  workflow_duration_hist.name = "workflow_invocation_duration"
  tool_duration_hist.name = "tool_execution_duration"
  client_duration_hist.name = "client_operation_duration"
  client_token_usage_hist.name = "client_token_usage"
  input_tokens_hist.name = "invoke_agent_input_tokens"
  output_tokens_hist.name = "invoke_agent_output_tokens"
  total_tokens_hist.name = "invoke_agent_total_tokens"
  cache_read_input_tokens_hist.name = "invoke_agent_cache_read_input_tokens"
  reasoning_output_tokens_hist.name = "invoke_agent_reasoning_output_tokens"
  tool_input_tokens_hist.name = "invoke_agent_tool_input_tokens"

  def create_histogram_side_effect(name, **_kwargs):
    if name == "gen_ai.invoke_agent.duration":
      return agent_duration_hist
    elif name == "gen_ai.invoke_workflow.duration":
      return workflow_duration_hist
    elif name == "gen_ai.execute_tool.duration":
      return tool_duration_hist
    elif name == "gen_ai.client.operation.duration":
      return client_duration_hist
    elif name == "gen_ai.client.token.usage":
      return client_token_usage_hist
    elif name == "adk.experimental.invoke_agent.input_tokens":
      return input_tokens_hist
    elif name == "adk.experimental.invoke_agent.output_tokens":
      return output_tokens_hist
    elif name == "adk.experimental.invoke_agent.total_tokens":
      return total_tokens_hist
    elif name == "adk.experimental.invoke_agent.cache_read.input_tokens":
      return cache_read_input_tokens_hist
    elif name == "adk.experimental.invoke_agent.reasoning.output_tokens":
      return reasoning_output_tokens_hist
    elif name == "adk.experimental.invoke_agent.tool.input_tokens":
      return tool_input_tokens_hist
    raise ValueError(f"Unknown metric name: {name}")

  mock_meter.create_histogram.side_effect = create_histogram_side_effect

  # Re-initialize the module-level variables in _metrics with mocked histograms
  monkeypatch.setattr(_metrics, "meter", mock_meter)
  monkeypatch.setattr(
      _metrics, "_agent_invocation_duration", agent_duration_hist
  )
  monkeypatch.setattr(
      _metrics, "_workflow_invocation_duration", workflow_duration_hist
  )
  monkeypatch.setattr(_metrics, "_tool_execution_duration", tool_duration_hist)
  monkeypatch.setattr(
      _metrics, "_client_operation_duration", client_duration_hist
  )
  monkeypatch.setattr(_metrics, "_client_token_usage", client_token_usage_hist)
  monkeypatch.setattr(_metrics, "_invoke_agent_input_tokens", input_tokens_hist)
  monkeypatch.setattr(
      _metrics, "_invoke_agent_output_tokens", output_tokens_hist
  )
  monkeypatch.setattr(_metrics, "_invoke_agent_total_tokens", total_tokens_hist)
  monkeypatch.setattr(
      _metrics,
      "_invoke_agent_cache_read_input_tokens",
      cache_read_input_tokens_hist,
  )
  monkeypatch.setattr(
      _metrics,
      "_invoke_agent_reasoning_output_tokens",
      reasoning_output_tokens_hist,
  )
  monkeypatch.setattr(
      _metrics, "_invoke_agent_tool_input_tokens", tool_input_tokens_hist
  )

  return {
      "meter": mock_meter,
      "agent_duration": agent_duration_hist,
      "workflow_duration": workflow_duration_hist,
      "tool_duration": tool_duration_hist,
      "client_duration": client_duration_hist,
      "client_token_usage": client_token_usage_hist,
      "input_tokens": input_tokens_hist,
      "output_tokens": output_tokens_hist,
      "total_tokens": total_tokens_hist,
      "cache_read_input_tokens": cache_read_input_tokens_hist,
      "reasoning_output_tokens": reasoning_output_tokens_hist,
      "tool_input_tokens": tool_input_tokens_hist,
  }


def test_record_agent_invocation_duration(mock_meter_setup):
  """Tests record_agent_invocation_duration records correctly."""
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1.0,
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  args, kwargs = agent_duration_hist.record.call_args
  assert args[0] == 1.0
  want_attributes = {"gen_ai.agent.name": "test_agent"}
  assert kwargs["attributes"] == want_attributes


def test_record_agent_invocation_duration_with_error(mock_meter_setup):
  """Tests record_agent_invocation_duration records error correctly."""
  test_error = ValueError("agent failed")
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1.0,
      error=test_error,
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  _, kwargs = agent_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


def test_record_workflow_invocation_duration_root(mock_meter_setup):
  """Tests record_workflow_invocation_duration omits nested for the root."""
  _metrics.record_workflow_invocation_duration(
      workflow_name="my_workflow",
      elapsed_s=1.0,
      nested=False,
  )
  hist = mock_meter_setup["workflow_duration"]
  hist.record.assert_called_once()
  args, kwargs = hist.record.call_args
  assert args[0] == 1.0
  assert kwargs["attributes"] == {
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.workflow.name": "my_workflow",
  }


def test_record_workflow_invocation_duration_nested_with_error(
    mock_meter_setup,
):
  """Tests record_workflow_invocation_duration records nested + error."""
  _metrics.record_workflow_invocation_duration(
      workflow_name="nested_workflow",
      elapsed_s=2.0,
      nested=True,
      error=ValueError("boom"),
  )
  hist = mock_meter_setup["workflow_duration"]
  hist.record.assert_called_once()
  _, kwargs = hist.record.call_args
  assert kwargs["attributes"]["gen_ai.workflow.nested"] is True
  assert kwargs["attributes"]["error.type"] == "ValueError"


def test_record_tool_execution_duration(mock_meter_setup):
  """Tests record_tool_execution_duration records correctly."""
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_tool_type",
      "test_agent",
      0.5,
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  args, kwargs = tool_duration_hist.record.call_args
  assert args[0] == 0.5
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.tool.name": "test_tool",
      "gen_ai.tool.type": "test_tool_type",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_tool_execution_duration_with_error(mock_meter_setup):
  """Tests record_tool_execution_duration records error correctly."""
  test_error = ValueError("tool failed")
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_tool_type",
      "test_agent",
      0.5,
      error=test_error,
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  _, kwargs = tool_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


def test_record_tool_execution_duration_with_detected_error_type(
    mock_meter_setup,
):
  """A failure reported in the tool response still labels the metric."""
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_tool_type",
      "test_agent",
      0.5,
      error_type="MCP_TOOL_ERROR",
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  _, kwargs = tool_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "MCP_TOOL_ERROR"


def test_record_tool_execution_duration_error_takes_precedence(
    mock_meter_setup,
):
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_tool_type",
      "test_agent",
      0.5,
      error=ValueError("tool failed"),
      error_type="MCP_TOOL_ERROR",
  )
  _, kwargs = mock_meter_setup["tool_duration"].record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("claude-sonnet-4-5", "anthropic"),
        ("anthropic/claude-sonnet-4-5", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("gemini-2.0-flash", "gemini"),
        ("test-model", "gemini"),
    ],
)
def test_record_client_operation_duration_provider_follows_model(
    mock_meter_setup, model, expected_provider
):
  """The provider name follows the served model, not just the deployment env."""
  llm_request = mock.MagicMock(
      contents=[types.Content(parts=[types.Part(text="hello")])],
      model=model,
  )
  _metrics.record_client_operation_duration(
      agent_name="test_agent",
      elapsed_s=0.1,
      llm_request=llm_request,
      responses=[],
  )
  _, kwargs = mock_meter_setup["client_duration"].record.call_args
  assert kwargs["attributes"]["gen_ai.provider.name"] == expected_provider


def test_record_client_operation_duration(mock_meter_setup):
  """Tests record_client_operation_duration records correctly."""
  llm_request = mock.MagicMock(
      contents=[types.Content(parts=[types.Part(text="hello")])],
      model="test-model",
  )
  response = mock.MagicMock(
      content=types.Content(parts=[types.Part(text="hello response")])
  )
  _metrics.record_client_operation_duration(
      agent_name="test_agent",
      elapsed_s=0.1,
      llm_request=llm_request,
      responses=[response],
  )
  client_duration_hist = mock_meter_setup["client_duration"]
  client_duration_hist.record.assert_called_once()
  args, kwargs = client_duration_hist.record.call_args
  assert args[0] == 0.1
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gemini",
      "gen_ai.request.model": llm_request.model,
      "gen_ai.response.model": response.model_version,
  }
  assert kwargs["attributes"] == want_attributes


def test_record_client_token_usage(mock_meter_setup):
  """Tests record_client_token_usage records correctly under different usage conditions."""
  llm_request = mock.MagicMock(
      contents=[types.Content(parts=[types.Part(text="hello")])],
      model="test-model",
  )
  response = mock.MagicMock(
      content=types.Content(parts=[types.Part(text="hello response")]),
      model_version="test-model-v1",
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=20,
          candidates_token_count=30,
          tool_use_prompt_token_count=5,
          thoughts_token_count=10,
      ),
  )
  _metrics.record_client_token_usage(
      agent_name="test_agent",
      llm_request=llm_request,
      responses=[response],
  )
  client_token_usage_hist = mock_meter_setup["client_token_usage"]
  assert client_token_usage_hist.record.call_count == 2

  base_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gemini",
      "gen_ai.request.model": "test-model",
      "gen_ai.response.model": "test-model-v1",
  }

  input_call = None
  output_call = None

  for args, kwargs in client_token_usage_hist.record.call_args_list:
    token_type = kwargs.get("attributes", {}).get("gen_ai.token.type")
    if token_type == "input":
      input_call = (args, kwargs)
    elif token_type == "output":
      output_call = (args, kwargs)

  assert input_call is not None, "Missing 'input' token usage record"
  assert output_call is not None, "Missing 'output' token usage record"

  # Verify input tokens (prompt_token_count + tool_use_prompt_token_count)
  assert input_call[0][0] == 25
  assert input_call[1]["attributes"] == base_attributes | {
      "gen_ai.token.type": "input"
  }

  # Verify output tokens (candidates_token_count + thoughts_token_count)
  assert output_call[0][0] == 40
  assert output_call[1]["attributes"] == base_attributes | {
      "gen_ai.token.type": "output"
  }


@pytest.fixture(name="call_count_histograms")
def _call_count_histograms(monkeypatch):
  """Redirects the two per-invocation call-count histograms."""
  inference_calls_hist = mock.MagicMock(spec=metrics.Histogram)
  tool_calls_hist = mock.MagicMock(spec=metrics.Histogram)
  inference_calls_hist.name = "invoke_agent_inference_calls"
  tool_calls_hist.name = "invoke_agent_tool_calls"

  monkeypatch.setattr(
      _metrics, "_invoke_agent_inference_calls", inference_calls_hist
  )
  monkeypatch.setattr(_metrics, "_invoke_agent_tool_calls", tool_calls_hist)

  return {
      "inference_calls": inference_calls_hist,
      "tool_calls": tool_calls_hist,
  }


def test_record_invoke_agent_inference_calls(call_count_histograms):
  """The count is recorded verbatim, dimensioned only by the agent."""
  _metrics.record_invoke_agent_inference_calls("test_agent", 3)

  inference_calls_hist = call_count_histograms["inference_calls"]
  inference_calls_hist.record.assert_called_once()
  args, kwargs = inference_calls_hist.record.call_args
  assert args[0] == 3
  assert kwargs["attributes"] == {"gen_ai.agent.name": "test_agent"}
  # The two counts are separate instruments and must not cross over.
  call_count_histograms["tool_calls"].record.assert_not_called()


def test_record_invoke_agent_tool_calls(call_count_histograms):
  """The count is recorded verbatim, dimensioned only by the agent."""
  _metrics.record_invoke_agent_tool_calls("test_agent", 7)

  tool_calls_hist = call_count_histograms["tool_calls"]
  tool_calls_hist.record.assert_called_once()
  args, kwargs = tool_calls_hist.record.call_args
  assert args[0] == 7
  assert kwargs["attributes"] == {"gen_ai.agent.name": "test_agent"}
  call_count_histograms["inference_calls"].record.assert_not_called()


def test_record_invoke_agent_call_counts_records_zero(call_count_histograms):
  """Zero is a real observation -- an invocation that called nothing.

  Skipping it would leave the zero bucket empty and bias the distribution
  upwards.
  """
  _metrics.record_invoke_agent_inference_calls("test_agent", 0)
  _metrics.record_invoke_agent_tool_calls("test_agent", 0)

  assert call_count_histograms["inference_calls"].record.call_args[0][0] == 0
  assert call_count_histograms["tool_calls"].record.call_args[0][0] == 0


def test_record_invoke_agent_token_usage(mock_meter_setup):
  """Each token bucket is recorded once, keyed by agent, zeros included."""
  # Recording genuine zeros is what keeps "what share of invocations read
  # nothing from cache" answerable. An invocation that called no model is kept
  # out by its caller, which never builds an `InvocationTokenTotals` at all.
  input_tokens = 1000
  output_tokens = 200
  cache_read_input_tokens = 750
  _metrics.record_invoke_agent_token_usage(
      "sub_agent",
      _token_usage.InvocationTokenTotals(
          input_tokens=input_tokens,
          output_tokens=output_tokens,
          cache_read_input_tokens=cache_read_input_tokens,
          reasoning_output_tokens=0,
          tool_input_tokens=0,
      ),
  )

  want = {
      "input_tokens": input_tokens,
      "output_tokens": output_tokens,
      "total_tokens": input_tokens + output_tokens,
      "cache_read_input_tokens": cache_read_input_tokens,
      "reasoning_output_tokens": 0,
      "tool_input_tokens": 0,
  }
  for bucket, want_value in want.items():
    hist = mock_meter_setup[bucket]
    hist.record.assert_called_once()
    args, kwargs = hist.record.call_args
    assert args[0] == want_value, f"wrong value for {bucket}"
    assert kwargs["attributes"] == {
        "gen_ai.agent.name": "sub_agent"
    }, f"wrong attributes for {bucket}"
