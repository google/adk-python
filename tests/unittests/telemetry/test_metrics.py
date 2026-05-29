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
from google.genai import types
from opentelemetry import metrics
import pytest


@pytest.fixture(name="mock_meter_setup")
def _mock_meter_setup(monkeypatch):
  """Sets up mock meter and histograms for testing."""
  mock_meter = mock.MagicMock()
  agent_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  tool_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  request_size_hist = mock.MagicMock(spec=metrics.Histogram)
  response_size_hist = mock.MagicMock(spec=metrics.Histogram)
  steps_hist = mock.MagicMock(spec=metrics.Histogram)
  client_duration_hist = mock.MagicMock(spec=metrics.Histogram)
  client_token_usage_hist = mock.MagicMock(spec=metrics.Histogram)

  agent_duration_hist.name = "agent_invocation_duration"
  tool_duration_hist.name = "tool_execution_duration"
  request_size_hist.name = "agent_request_size"
  response_size_hist.name = "agent_response_size"
  steps_hist.name = "agent_workflow_steps"
  client_duration_hist.name = "client_operation_duration"
  client_token_usage_hist.name = "client_token_usage"

  def create_histogram_side_effect(name, **_kwargs):
    if name == "gen_ai.agent.invocation.duration":
      return agent_duration_hist
    elif name == "gen_ai.tool.execution.duration":
      return tool_duration_hist
    elif name == "gen_ai.agent.request.size":
      return request_size_hist
    elif name == "gen_ai.agent.response.size":
      return response_size_hist
    elif name == "gen_ai.agent.workflow.steps":
      return steps_hist
    elif name == "gen_ai.client.operation.duration":
      return client_duration_hist
    elif name == "gen_ai.client.token.usage":
      return client_token_usage_hist
    raise ValueError(f"Unknown metric name: {name}")

  mock_meter.create_histogram.side_effect = create_histogram_side_effect

  # Re-initialize the module-level variables in _metrics with mocked histograms
  monkeypatch.setattr(_metrics, "meter", mock_meter)
  monkeypatch.setattr(
      _metrics, "_agent_invocation_duration", agent_duration_hist
  )
  monkeypatch.setattr(_metrics, "_tool_execution_duration", tool_duration_hist)
  monkeypatch.setattr(_metrics, "_agent_request_size", request_size_hist)
  monkeypatch.setattr(_metrics, "_agent_response_size", response_size_hist)
  monkeypatch.setattr(_metrics, "_agent_workflow_steps", steps_hist)
  monkeypatch.setattr(
      _metrics, "_client_operation_duration", client_duration_hist
  )
  monkeypatch.setattr(_metrics, "_client_token_usage", client_token_usage_hist)

  return {
      "meter": mock_meter,
      "agent_duration": agent_duration_hist,
      "tool_duration": tool_duration_hist,
      "request_size": request_size_hist,
      "response_size": response_size_hist,
      "steps": steps_hist,
      "client_duration": client_duration_hist,
      "client_token_usage": client_token_usage_hist,
  }


def test_record_agent_request_size(mock_meter_setup):
  """Tests record_agent_request_size records correctly."""
  user_content = "hello"
  _metrics.record_agent_request_size(
      "test_agent", types.Content(parts=[types.Part(text=user_content)])
  )
  request_size_hist = mock_meter_setup["request_size"]
  request_size_hist.record.assert_called_once()
  args, kwargs = request_size_hist.record.call_args
  assert args[0] == len(user_content)
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_agent_invocation_duration(mock_meter_setup):
  """Tests record_agent_invocation_duration records correctly."""
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1000.0,
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  args, kwargs = agent_duration_hist.record.call_args
  assert args[0] == 1000.0
  want_attributes = {"gen_ai.agent.name": "test_agent"}
  assert kwargs["attributes"] == want_attributes


def test_record_agent_invocation_duration_with_error(mock_meter_setup):
  """Tests record_agent_invocation_duration records error correctly."""
  test_error = ValueError("agent failed")
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1000.0,
      error=test_error,
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  _, kwargs = agent_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


def test_record_agent_response_size(mock_meter_setup):
  """Tests record_agent_response_size records correctly."""
  response_text = "response"
  event = mock.MagicMock(
      author="test_agent",
      content=types.Content(parts=[types.Part(text=response_text)]),
  )
  _metrics.record_agent_response_size("test_agent", [event])
  response_size_hist = mock_meter_setup["response_size"]
  response_size_hist.record.assert_called_once()
  args, kwargs = response_size_hist.record.call_args
  assert args[0] == len(response_text)
  want_attributes = {"gen_ai.agent.name": "test_agent"}
  assert kwargs["attributes"] == want_attributes


def test_record_agent_workflow_steps(mock_meter_setup):
  """Tests record_agent_workflow_steps records correctly."""
  _metrics.record_agent_workflow_steps(
      "test_agent",
      [
          mock.MagicMock(author="test_agent"),
          mock.MagicMock(author="test_agent"),
          mock.MagicMock(author="other_agent"),
      ],
  )
  steps_hist = mock_meter_setup["steps"]
  steps_hist.record.assert_called_once()
  args, kwargs = steps_hist.record.call_args
  assert args[0] == 2
  want_attributes = {"gen_ai.agent.name": "test_agent"}
  assert kwargs["attributes"] == want_attributes


def test_record_tool_execution_duration(mock_meter_setup):
  """Tests record_tool_execution_duration records correctly."""
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_agent",
      500.0,
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  args, kwargs = tool_duration_hist.record.call_args
  assert args[0] == 500.0
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.tool.name": "test_tool",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_tool_execution_duration_with_error(mock_meter_setup):
  """Tests record_tool_execution_duration records error correctly."""
  test_error = ValueError("tool failed")
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_agent",
      500.0,
      error=test_error,
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  _, kwargs = tool_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


@pytest.mark.parametrize(
    "content,expected_size",
    [
        (None, 0),
        (types.Content(parts=[types.Part(text="hello")]), 5),
        (
            types.Content(
                parts=[
                    types.Part(text="hello"),
                    types.Part(text=" world"),
                ]
            ),
            11,
        ),
        (
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png", data=b"12345"
                        )
                    )
                ]
            ),
            5,
        ),
        (
            types.Content(
                parts=[
                    types.Part(text="hello"),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png", data=b"12345"
                        )
                    ),
                ]
            ),
            10,
        ),
    ],
    ids=[
        "none_content",
        "simple_text",
        "multi_text",
        "inline_data",
        "mixed_content",
    ],
)
def test_get_content_size(content, expected_size):
  assert _metrics._get_content_size(content) == expected_size


def test_record_client_operation_duration(mock_meter_setup):
  """Tests record_client_operation_duration records correctly."""
  llm_request = mock.MagicMock(
      contents=[types.Content(parts=[types.Part(text="hello")])]
  )
  response = mock.MagicMock(
      content=types.Content(parts=[types.Part(text="hello response")])
  )
  _metrics.record_client_operation_duration(
      agent_name="test_agent",
      elapsed_ms=100.0,
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
