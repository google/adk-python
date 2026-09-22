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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "typesafe_sdk", reason="Requires typesafe-sdk (google-adk[typesafe])"
)

from google.adk.agents.invocation_context import InvocationContext
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.integrations.typesafe import TypesafeClassifierTool
from google.adk.sessions.session import Session
from google.adk.tools.tool_context import ToolContext
from typesafe_sdk import Choice
from typesafe_sdk import ChoiceAnswer
from typesafe_sdk import SystemOneResponse
from typesafe_sdk import Usage


@pytest.fixture
def mock_tool_context() -> ToolContext:
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  mock_invocation_context.session.state = MagicMock()
  return ToolContext(invocation_context=mock_invocation_context)


def _make_tool(classify_mock: AsyncMock) -> TypesafeClassifierTool:
  classifier = TypesafeClassifier(client=AsyncMock())
  classifier.classify = classify_mock
  return TypesafeClassifierTool(
      name="classify_ticket_category",
      description="Classifies a support ticket into billing/technical/other.",
      classifier=classifier,
      question=Choice(
          instructions="What is this support ticket about?",
          criteria={"billing": None, "technical": None, "other": None},
      ),
  )


def test_tool_exposes_the_configured_name_and_description():
  tool = _make_tool(AsyncMock())

  assert tool.name == "classify_ticket_category"
  assert (
      tool.description
      == "Classifies a support ticket into billing/technical/other."
  )


def test_declaration_shown_to_the_model_uses_the_configured_name():
  """Regression test: the declared name must match tool.name.

  FunctionTool derives a declaration's name from the wrapped callable's
  __name__, which is always "_classify" for this tool. Without patching it
  onto the declaration, the model is shown a tool called "_classify" while
  ADK dispatches by `tool.name`, so every call to the advertised name fails.
  """
  tool = _make_tool(AsyncMock())

  declaration = tool._get_declaration()

  assert declaration.name == "classify_ticket_category"
  assert (
      declaration.description
      == "Classifies a support ticket into billing/technical/other."
  )
  # Whichever schema shape is active, the "state" parameter must be present.
  if declaration.parameters is not None:
    assert "state" in declaration.parameters.properties
  else:
    assert "state" in declaration.parameters_json_schema["properties"]


@pytest.mark.asyncio
async def test_run_async_passes_state_through_and_returns_the_answer(
    mock_tool_context,
):
  classify_mock = AsyncMock(
      return_value=SystemOneResponse(
          model="jev-latest",
          usage=Usage(),
          answers={
              "result": ChoiceAnswer(
                  choice="billing",
                  confidence=0.9,
                  probabilities={"billing": 0.9},
              )
          },
      )
  )
  tool = _make_tool(classify_mock)

  result = await tool.run_async(
      args={"state": "I was charged twice."}, tool_context=mock_tool_context
  )

  classify_mock.assert_awaited_once()
  assert classify_mock.await_args.kwargs["state"] == "I was charged twice."
  assert result["choice"] == "billing"
