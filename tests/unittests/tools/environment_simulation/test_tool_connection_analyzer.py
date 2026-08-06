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

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.environment_simulation import tool_connection_analyzer
from google.adk.tools.environment_simulation.tool_connection_analyzer import ToolConnectionAnalyzer
from google.adk.tools.environment_simulation.tool_connection_map import ToolConnectionMap
from google.genai import types
import pytest


def _make_analyzer(response_text: str) -> ToolConnectionAnalyzer:
  """Builds a ToolConnectionAnalyzer whose LLM yields ``response_text``."""

  async def fake_generate_content_async(request):
    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=response_text)],
        )
    )

  mock_llm = MagicMock()
  mock_llm.generate_content_async = fake_generate_content_async

  with patch.object(
      tool_connection_analyzer, "LLMRegistry", autospec=True
  ) as mock_registry:
    mock_registry.return_value.resolve.return_value = MagicMock(
        return_value=mock_llm
    )
    return ToolConnectionAnalyzer(
        llm_name="fake-model",
        llm_config=types.GenerateContentConfig(),
    )


def _make_tool(name: str) -> MagicMock:
  """Builds a tool whose declaration produces a non-empty schema."""
  tool = MagicMock()
  tool._get_declaration.return_value = types.FunctionDeclaration(name=name)
  return tool


@pytest.mark.asyncio
class TestToolConnectionAnalyzerAnalyze:
  """Test cases for the analyze method of ToolConnectionAnalyzer."""

  async def test_malformed_json_returns_empty_map_without_crashing(
      self, caplog
  ):
    """Regression test: a non-JSON LLM response must not raise NameError.

    The JSONDecodeError handler logs the captured exception, so the ``except``
    clause must bind it (``as e``). Without the binding the handler itself
    raised ``NameError: name 'e' is not defined``, masking the real parse
    failure and crashing ``analyze()``.
    """
    analyzer = _make_analyzer("this is not json at all")
    tool = _make_tool("create_ticket")

    with caplog.at_level(logging.WARNING):
      result = await analyzer.analyze([tool])

    assert isinstance(result, ToolConnectionMap)
    assert result.stateful_parameters == []
    assert "Failed to parse tool connection analysis" in caplog.text

  async def test_valid_json_is_parsed_into_connection_map(self):
    """A well-formed JSON response is parsed into a ToolConnectionMap."""
    response_text = (
        '{"stateful_parameters": [{"parameter_name": "ticket_id",'
        ' "creating_tools": ["create_ticket"], "consuming_tools":'
        ' ["get_ticket"]}]}'
    )
    analyzer = _make_analyzer(response_text)
    tool = _make_tool("create_ticket")

    result = await analyzer.analyze([tool])

    assert isinstance(result, ToolConnectionMap)
    assert len(result.stateful_parameters) == 1
    parameter = result.stateful_parameters[0]
    assert parameter.parameter_name == "ticket_id"
    assert parameter.creating_tools == ["create_ticket"]
    assert parameter.consuming_tools == ["get_ticket"]

  async def test_fenced_json_is_stripped_before_parsing(self):
    """A response wrapped in a Markdown code fence is parsed correctly."""
    response_text = (
        "```json\n"
        '{"stateful_parameters": [{"parameter_name": "order_id",'
        ' "creating_tools": ["create_order"], "consuming_tools":'
        ' ["get_order"]}]}\n'
        "```"
    )
    analyzer = _make_analyzer(response_text)
    tool = _make_tool("create_order")

    result = await analyzer.analyze([tool])

    assert isinstance(result, ToolConnectionMap)
    assert len(result.stateful_parameters) == 1
    assert result.stateful_parameters[0].parameter_name == "order_id"
