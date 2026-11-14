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

"""Tests for SkillToolset's classification_config.

Verifies that a confident classifier loads the skill for a turn before the
model's first step, and that anything else leaves the choice to the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.labs.classifiers import BaseClassifier
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ChoiceAnswer
from google.adk.labs.classifiers import ClassifierInput
from google.adk.labs.classifiers import ClassifierResponse
from google.adk.labs.classifiers import Question
from google.adk.labs.classifiers import RefusalAnswer
from google.adk.models.llm_request import LlmRequest
from google.adk.skills import models
from google.adk.tools.skill_toolset import SkillClassificationConfig
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.utils.content_utils import SKIP_THOUGHT_SIGNATURE_VALIDATOR
from google.genai import types
from pydantic import Field
import pytest

from .. import testing_utils

_BILLING = models.Skill(
    frontmatter=models.Frontmatter(
        name="billing",
        description="Invoices, refunds, or charges.",
        metadata={"adk_additional_tools": ["lookup_invoice"]},
    ),
    instructions="Look up the invoice before answering.",
)
_WEATHER = models.Skill(
    frontmatter=models.Frontmatter(
        name="weather", description="Weather forecasts."
    ),
    instructions="Give the forecast in Celsius.",
)


def lookup_invoice(invoice_id: str) -> dict[str, str]:
  """Looks up an invoice."""
  return {"invoice_id": invoice_id, "status": "paid"}


class _FixedClassifier(BaseClassifier):
  """Answers every question with a fixed choice and records each call."""

  model: str = "fixed"
  choice: str = "_none"
  probability: float = 1.0
  refuse: bool = False
  error: Exception | None = None
  calls: list[dict[str, Any]] = Field(default_factory=list)

  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    self.calls.append({"input": input, "questions": dict(questions)})
    if self.error is not None:
      raise self.error
    answers = {}
    for name, question in questions.items():
      assert isinstance(question, Choice)
      if self.refuse:
        answers[name] = RefusalAnswer()
        continue
      labels = list(question.criteria)
      rest = (1.0 - self.probability) / (len(labels) - 1)
      answers[name] = ChoiceAnswer(
          choice=self.choice,
          probabilities={
              label: self.probability if label == self.choice else rest
              for label in labels
          },
      )
    return ClassifierResponse(answers=answers)


def _make_toolset(
    classifier: BaseClassifier | None, **kwargs: Any
) -> SkillToolset:
  return SkillToolset(
      skills=[_BILLING, _WEATHER],
      additional_tools=[lookup_invoice],
      classification_config=(
          SkillClassificationConfig(classifier=classifier)
          if classifier is not None
          else None
      ),
      **kwargs,
  )


def _make_runner(
    toolset: SkillToolset, responses: list[Any], **agent_kwargs: Any
) -> tuple[testing_utils.InMemoryRunner, testing_utils.MockModel]:
  model = testing_utils.MockModel.create(responses=responses)
  agent = LlmAgent(name="root", model=model, tools=[toolset], **agent_kwargs)
  return testing_utils.InMemoryRunner(root_agent=agent), model


def _declared_tool_names(request: LlmRequest) -> set[str]:
  return {
      declaration.name
      for tool in request.config.tools or []
      if isinstance(tool, types.Tool)
      for declaration in tool.function_declarations or []
  }


def _load_skill_responses(
    request: LlmRequest, name: str = "load_skill"
) -> list[dict[str, Any]]:
  return [
      part.function_response.response
      for content in request.contents
      for part in content.parts or []
      if part.function_response and part.function_response.name == name
  ]


async def test_confident_classification_saves_the_load_skill_step():
  """The model's first step already has the classified skill's instructions."""
  toolset = _make_toolset(_FixedClassifier(choice="billing"))
  runner, model = _make_runner(toolset, ["Your invoice is paid."])

  await runner.run_async("Was I charged twice?")

  [request] = model.requests
  [response] = _load_skill_responses(request)
  assert response["skill_name"] == "billing"
  assert response["instructions"] == "Look up the invoice before answering."


async def test_classified_skill_tools_are_declared_in_the_first_step():
  """The classified skill's additional tools are offered to the first step."""
  toolset = _make_toolset(_FixedClassifier(choice="billing"))
  runner, model = _make_runner(toolset, ["Your invoice is paid."])

  await runner.run_async("Was I charged twice?")

  assert "lookup_invoice" in _declared_tool_names(model.requests[0])


async def test_preload_is_recorded_as_a_load_skill_call():
  """The session shows a load_skill call marked as coming from classification."""
  toolset = _make_toolset(_FixedClassifier(choice="billing"))
  runner, _ = _make_runner(toolset, ["Your invoice is paid."])

  events = await runner.run_async("Was I charged twice?")

  call_event = next(e for e in events if e.get_function_calls())
  [part] = call_event.content.parts
  assert call_event.author == "root"
  assert part.function_call.name == "load_skill"
  assert part.function_call.args == {"skill_name": "billing"}
  assert part.thought_signature == SKIP_THOUGHT_SIGNATURE_VALIDATOR
  assert call_event.custom_metadata == {
      "adk_skill_classification": {"skill_name": "billing", "probability": 1.0}
  }


async def test_preload_runs_tool_callbacks():
  """Tool callbacks see the preload as they would a load the model made."""
  seen = []

  def before_tool(tool, args, tool_context):
    seen.append((tool.name, args))

  toolset = _make_toolset(_FixedClassifier(choice="weather"))
  runner, _ = _make_runner(
      toolset, ["Sunny."], before_tool_callback=before_tool
  )

  await runner.run_async("Will it rain?")

  assert seen == [("load_skill", {"skill_name": "weather"})]


async def test_classified_skill_stays_active_in_later_turns():
  """A preloaded skill is active, so its tools stay declared next turn."""
  classifier = _FixedClassifier(choice="billing")
  runner, model = _make_runner(_make_toolset(classifier), ["first", "second"])
  await runner.run_async("Was I charged twice?")
  classifier.choice = "_none"

  await runner.run_async("Thanks.")

  assert "lookup_invoice" in _declared_tool_names(model.requests[1])


async def test_classifier_input_is_the_text_message_with_every_skill():
  """The classifier gets the message text and one label per skill plus none."""
  classifier = _FixedClassifier()
  runner, _ = _make_runner(_make_toolset(classifier), ["ok"])

  await runner.run_async("Was I charged twice?")

  [call] = classifier.calls
  assert call["input"] == "Was I charged twice?"
  assert call["questions"]["skill"].criteria == {
      "billing": "Invoices, refunds, or charges.",
      "weather": "Weather forecasts.",
      "_none": "None of the other labels applies.",
  }


async def test_multimodal_message_is_classified_as_content():
  """A message with non-text parts reaches the classifier as Content."""
  classifier = _FixedClassifier()
  runner, _ = _make_runner(_make_toolset(classifier), ["ok"])
  message = types.Content(
      role="user",
      parts=[
          types.Part.from_text(text="What is this charge?"),
          types.Part.from_bytes(data=b"png", mime_type="image/png"),
      ],
  )

  await runner.run_async(message)

  [call] = classifier.calls
  assert isinstance(call["input"], types.Content)
  assert len(call["input"].parts) == 2


@pytest.mark.parametrize(
    "classifier_kwargs",
    [
        {"choice": "_none"},
        {"choice": "billing", "probability": 0.4},
        {"refuse": True},
        {"error": RuntimeError("classifier down")},
        {"choice": "not-a-skill"},
    ],
    ids=["no_skill", "below_threshold", "refusal", "error", "unknown_label"],
)
async def test_unconfident_classification_leaves_the_choice_to_the_model(
    classifier_kwargs,
):
  """Short of a confident skill choice, the turn runs as without a classifier."""
  toolset = _make_toolset(_FixedClassifier(**classifier_kwargs))
  runner, model = _make_runner(toolset, ["Hello."])

  events = await runner.run_async("Hi")

  assert not any(e.get_function_calls() for e in events)
  assert not _load_skill_responses(model.requests[0])


async def test_classifies_once_per_turn():
  """Later model steps of the same turn do not classify again."""
  classifier = _FixedClassifier(choice="billing")
  runner, model = _make_runner(
      _make_toolset(classifier),
      [
          types.Part.from_function_call(
              name="lookup_invoice", args={"invoice_id": "42"}
          ),
          "Invoice 42 is paid.",
      ],
  )

  await runner.run_async("Was invoice 42 charged twice?")

  assert len(model.requests) == 2
  assert len(classifier.calls) == 1


async def test_active_skill_is_not_loaded_again():
  """Classifying an already active skill emits no second load."""
  classifier = _FixedClassifier(choice="billing")
  runner, _ = _make_runner(_make_toolset(classifier), ["first", "second"])
  await runner.run_async("Was I charged twice?")

  events = await runner.run_async("And the month before?")

  assert len(classifier.calls) == 2
  assert not any(e.get_function_calls() for e in events)


async def test_without_classification_config_the_model_loads_skills():
  """No classification_config keeps model-driven loading."""
  runner, model = _make_runner(_make_toolset(None), ["Hello."])

  await runner.run_async("Was I charged twice?")

  assert not _load_skill_responses(model.requests[0])


async def test_filtered_out_load_skill_skips_classification():
  """With load_skill filtered out, the classifier is never asked."""
  classifier = _FixedClassifier(choice="billing")
  toolset = _make_toolset(classifier, tool_filter=["list_skills"])
  runner, _ = _make_runner(toolset, ["Hello."])

  await runner.run_async("Was I charged twice?")

  assert not classifier.calls


async def test_preload_uses_the_prefixed_tool_name():
  """The preload calls load_skill under the toolset's tool name prefix."""
  toolset = _make_toolset(
      _FixedClassifier(choice="weather"), tool_name_prefix="kb"
  )
  runner, model = _make_runner(toolset, ["Sunny."])

  await runner.run_async("Will it rain?")

  [response] = _load_skill_responses(model.requests[0], name="kb_load_skill")
  assert response["skill_name"] == "weather"


async def test_cloned_toolset_keeps_classifying():
  """clone_with_updated_skills carries the classification config over."""
  toolset = _make_toolset(_FixedClassifier(choice="weather"))
  clone = toolset.clone_with_updated_skills([_WEATHER])
  runner, model = _make_runner(clone, ["Sunny."])

  await runner.run_async("Will it rain?")

  [response] = _load_skill_responses(model.requests[0])
  assert response["skill_name"] == "weather"


def test_config_rejects_out_of_range_threshold():
  with pytest.raises(ValueError, match="threshold"):
    SkillClassificationConfig(classifier=_FixedClassifier(), threshold=1.5)


def test_config_rejects_a_non_classifier():
  with pytest.raises(TypeError, match="BaseClassifier"):
    SkillClassificationConfig(classifier="gemini")  # type: ignore[arg-type]


def test_classification_runs_before_contents_are_built():
  """The processor precedes contents, so the preload is in the same request."""
  names = [p.name for p in SingleFlow().request_processors]

  assert names.index("skill_classification") < names.index("contents")
