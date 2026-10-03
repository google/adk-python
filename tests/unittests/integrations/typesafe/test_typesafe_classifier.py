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

import pytest

pytest.importorskip(
    "typesafe_sdk", reason="Requires typesafe-sdk (google-adk[typesafe])"
)

from google.adk.integrations.typesafe import TypesafeClassifier
from typesafe_sdk import Choice
from typesafe_sdk import ChoiceAnswer
from typesafe_sdk import Noul
from typesafe_sdk import NoulAnswer
from typesafe_sdk import Score
from typesafe_sdk import ScoreAnswer
from typesafe_sdk import SystemOneResponse
from typesafe_sdk import Usage


def _fake_response() -> SystemOneResponse:
  return SystemOneResponse(
      model="jev-latest",
      usage=Usage(input_tokens=10, output_tokens=5),
      answers={
          "category": ChoiceAnswer(
              choice="billing", confidence=0.9, probabilities={"billing": 0.9}
          )
      },
  )


@pytest.mark.asyncio
async def test_classify_forwards_state_and_questions_to_the_client():
  mock_client = AsyncMock()
  mock_client.system_one.return_value = _fake_response()
  classifier = TypesafeClassifier(client=mock_client)
  question = Choice(
      instructions="What is this about?",
      criteria={"billing": None, "technical": None},
  )

  response = await classifier.classify(
      state="I was charged twice.", questions={"category": question}
  )

  mock_client.system_one.assert_awaited_once_with(
      state="I was charged twice.", questions={"category": question}
  )
  assert response.answers["category"].choice == "billing"


@pytest.mark.asyncio
async def test_classify_reuses_the_same_client_across_calls():
  mock_client = AsyncMock()
  mock_client.system_one.return_value = _fake_response()
  classifier = TypesafeClassifier(client=mock_client)
  question = Choice(instructions="?", criteria={"a": None})

  await classifier.classify(state="one", questions={"q": question})
  await classifier.classify(state="two", questions={"q": question})

  assert mock_client.system_one.await_count == 2


@pytest.mark.asyncio
async def test_classify_handles_all_three_question_primitives_in_one_call():
  """Noul, Choice, and Score can be asked together and round-trip correctly."""
  mock_client = AsyncMock()
  mock_client.system_one.return_value = SystemOneResponse(
      model="jev-latest",
      usage=Usage(input_tokens=10, output_tokens=5),
      answers={
          "is_urgent": NoulAnswer(noul=0.8),
          "category": ChoiceAnswer(
              choice="billing", confidence=0.9, probabilities={"billing": 0.9}
          ),
          "severity": ScoreAnswer(
              score=2,
              confidence=0.7,
              legend={0: "low", 1: "medium", 2: "high"},
              probabilities={0: 0.1, 1: 0.2, 2: 0.7},
          ),
      },
  )
  classifier = TypesafeClassifier(client=mock_client)
  questions = {
      "is_urgent": Noul(instructions="Is this urgent?"),
      "category": Choice(
          instructions="What is this about?",
          criteria={"billing": None, "technical": None},
      ),
      "severity": Score(
          instructions="How severe is this?",
          criteria=["low", "medium", "high"],
      ),
  }

  response = await classifier.classify(
      state="I was charged twice.", questions=questions
  )

  mock_client.system_one.assert_awaited_once_with(
      state="I was charged twice.", questions=questions
  )
  assert response.answers["is_urgent"].noul == 0.8
  assert response.answers["category"].choice == "billing"
  assert response.answers["severity"].score == 2


def test_classify_raises_a_descriptive_error_without_typesafe_sdk(monkeypatch):
  import sys

  monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
  for module_name in list(sys.modules):
    if module_name.startswith("google.adk.integrations.typesafe"):
      monkeypatch.delitem(sys.modules, module_name, raising=False)

  with pytest.raises(ImportError, match=r"google-adk\[typesafe\]"):
    import google.adk.integrations.typesafe  # noqa: F401
