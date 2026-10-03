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

from google.genai import types
import pytest

pytest.importorskip(
    "typesafe_sdk", reason="Requires typesafe-sdk (google-adk[typesafe])"
)

from google.adk.integrations.typesafe import create_model_router_callback
from google.adk.integrations.typesafe import create_tool_gate_callback
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.models.llm_request import LlmRequest
from typesafe_sdk import ChoiceAnswer
from typesafe_sdk import NoulAnswer
from typesafe_sdk import SystemOneResponse
from typesafe_sdk import Usage


def _classifier_returning(answer_id: str, answer) -> TypesafeClassifier:
  classifier = TypesafeClassifier(client=AsyncMock())
  classifier.classify = AsyncMock(
      return_value=SystemOneResponse(
          model="jev-latest", usage=Usage(), answers={answer_id: answer}
      )
  )
  return classifier


def _request_with_user_text(text: str) -> LlmRequest:
  return LlmRequest(
      model="gemini-2.5-flash",
      contents=[
          types.Content(role="user", parts=[types.Part.from_text(text=text)])
      ],
  )


@pytest.mark.asyncio
async def test_model_router_rewrites_the_model_on_a_matching_choice():
  classifier = _classifier_returning(
      "route",
      ChoiceAnswer(choice="complex", confidence=0.8, probabilities={}),
  )
  callback = create_model_router_callback(
      classifier=classifier,
      criteria={"simple": None, "complex": None},
      model_map={"complex": "gemini-2.5-pro", "simple": "gemini-2.5-flash"},
  )
  llm_request = _request_with_user_text("Plan my multi-city trip in detail.")

  result = await callback(MagicMock(), llm_request)

  assert result is None
  assert llm_request.model == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_model_router_leaves_the_model_untouched_without_a_map_entry():
  classifier = _classifier_returning(
      "route", ChoiceAnswer(choice="unmapped", confidence=0.5, probabilities={})
  )
  callback = create_model_router_callback(
      classifier=classifier,
      criteria={"simple": None},
      model_map={"simple": "gemini-2.5-flash"},
  )
  llm_request = _request_with_user_text("hello")

  await callback(MagicMock(), llm_request)

  assert llm_request.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_model_router_is_a_no_op_without_a_user_turn():
  classifier = TypesafeClassifier(client=AsyncMock())
  classifier.classify = AsyncMock()
  callback = create_model_router_callback(
      classifier=classifier, criteria={"a": None}, model_map={"a": "model-a"}
  )
  llm_request = LlmRequest(model="gemini-2.5-flash", contents=[])

  result = await callback(MagicMock(), llm_request)

  assert result is None
  classifier.classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_gate_blocks_a_tool_call_at_or_above_the_threshold():
  classifier = _classifier_returning("risk", NoulAnswer(noul=0.9))
  callback = create_tool_gate_callback(
      classifier=classifier, instructions="Is this risky?", threshold=0.5
  )
  tool = MagicMock()
  tool.name = "delete_account"

  result = await callback(tool, {"user_id": "123"}, MagicMock())

  assert result is not None
  assert "error" in result


@pytest.mark.asyncio
async def test_tool_gate_passes_a_tool_call_below_the_threshold():
  classifier = _classifier_returning("risk", NoulAnswer(noul=0.1))
  callback = create_tool_gate_callback(
      classifier=classifier, instructions="Is this risky?", threshold=0.5
  )
  tool = MagicMock()
  tool.name = "get_weather"

  result = await callback(tool, {"city": "NYC"}, MagicMock())

  assert result is None
