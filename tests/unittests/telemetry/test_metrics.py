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

  agent_duration_hist.name = "agent_invocation_duration"
  tool_duration_hist.name = "tool_execution_duration"
  request_size_hist.name = "agent_request_size"
  response_size_hist.name = "agent_response_size"
  steps_hist.name = "agent_workflow_steps"

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

  return {
      "meter": mock_meter,
      "agent_duration": agent_duration_hist,
      "tool_duration": tool_duration_hist,
      "request_size": request_size_hist,
      "response_size": response_size_hist,
      "steps": steps_hist,
  }


def test_record_agent_request_size(mock_meter_setup):
  """Tests record_agent_request_size records correctly."""
  _metrics.record_agent_request_size(
      "test_agent", types.Content(parts=[types.Part(text="hello")])
  )
  request_size_hist = mock_meter_setup["request_size"]
  request_size_hist.record.assert_called_once()
  args, kwargs = request_size_hist.record.call_args
  assert args[0] == 5  # len('hello')
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.input.type": "text",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_agent_invocation_duration(mock_meter_setup):
  """Tests record_agent_invocation_duration records correctly."""
  event = mock.MagicMock(
      author="test_agent",
      content=types.Content(parts=[types.Part(text="hello response")]),
  )
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1000.0,
      types.Content(parts=[types.Part(text="hello")]),
      [event],
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  args, kwargs = agent_duration_hist.record.call_args
  assert args[0] == 1000.0
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.input.type": "text",
      "gen_ai.output.type": "text",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_agent_invocation_duration_with_error(mock_meter_setup):
  """Tests record_agent_invocation_duration records error correctly."""
  test_error = ValueError("agent failed")
  event = mock.MagicMock(
      author="test_agent",
      content=types.Content(parts=[types.Part(text="hello response")]),
  )
  _metrics.record_agent_invocation_duration(
      "test_agent",
      1000.0,
      types.Content(parts=[types.Part(text="hello")]),
      [event],
      error=test_error,
  )
  agent_duration_hist = mock_meter_setup["agent_duration"]
  agent_duration_hist.record.assert_called_once()
  _, kwargs = agent_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"
  assert kwargs["attributes"]["gen_ai.output.type"] == "text"


def test_record_agent_response_size(mock_meter_setup):
  """Tests record_agent_response_size records correctly."""
  event = mock.MagicMock(
      author="test_agent",
      content=types.Content(parts=[types.Part(text="response")]),
  )
  _metrics.record_agent_response_size("test_agent", [event])
  response_size_hist = mock_meter_setup["response_size"]
  response_size_hist.record.assert_called_once()
  args, kwargs = response_size_hist.record.call_args
  assert args[0] == 8  # len('response')
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.output.type": "text",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_agent_workflow_steps(mock_meter_setup):
  """Tests record_agent_workflow_steps records correctly."""
  _metrics.record_agent_workflow_steps("test_agent", 5)
  steps_hist = mock_meter_setup["steps"]
  steps_hist.record.assert_called_once()
  args, kwargs = steps_hist.record.call_args
  assert args[0] == 5
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_tool_execution_duration(mock_meter_setup):
  """Tests record_tool_execution_duration records correctly."""
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_agent",
      500.0,
      types.Content(parts=[types.Part(text="input")]),
      types.Content(parts=[types.Part(text="result")]),
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  args, kwargs = tool_duration_hist.record.call_args
  assert args[0] == 500.0
  want_attributes = {
      "gen_ai.agent.name": "test_agent",
      "gen_ai.tool.name": "test_tool",
      "gen_ai.input.type": "text",
      "gen_ai.output.type": "text",
  }
  assert kwargs["attributes"] == want_attributes


def test_record_tool_execution_duration_with_error(mock_meter_setup):
  """Tests record_tool_execution_duration records error correctly."""
  test_error = ValueError("tool failed")
  _metrics.record_tool_execution_duration(
      "test_tool",
      "test_agent",
      500.0,
      types.Content(parts=[types.Part(text="input")]),
      None,
      error=test_error,
  )
  tool_duration_hist = mock_meter_setup["tool_duration"]
  tool_duration_hist.record.assert_called_once()
  _, kwargs = tool_duration_hist.record.call_args
  assert kwargs["attributes"]["error.type"] == "ValueError"


@pytest.mark.parametrize(
    "content,expected",
    [
        (None, "unknown"),
        (types.Content(parts=[types.Part(text="hello")]), "text"),
        (
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(mime_type="image/jpeg", data=b"")
                    )
                ]
            ),
            "image",
        ),
        (
            types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(
                            mime_type="video/mp4", file_uri=""
                        )
                    )
                ]
            ),
            "video",
        ),
        (
            types.Content(
                parts=[
                    types.Part(text="hello"),
                    types.Part(
                        inline_data=types.Blob(mime_type="image/png", data=b"")
                    ),
                ]
            ),
            "image,text",
        ),
        (
            types.Content(
                parts=[types.Part(text="hello"), types.Part(text="world")]
            ),
            "text",
        ),
        (
            types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(mime_type="invalid", data=b"")
                    )
                ]
            ),
            "text",
        ),
        (types.Content(parts=[]), "unknown"),
    ],
    ids=[
        "none_content",
        "simple_text",
        "inline_image",
        "file_video",
        "combo",
        "deduplication",
        "invalid_mime",
        "empty_parts",
    ],
)
def test_get_modality_from_content_parameterized(content, expected):
  """Tests _get_modality_from_content with various inputs."""
  assert _metrics._get_modality_from_content(content) == expected


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
