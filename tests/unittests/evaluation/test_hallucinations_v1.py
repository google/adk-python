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

import asyncio
import json

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import HallucinationsCriterion
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.hallucinations_v1 import _parse_sentences
from google.adk.evaluation.hallucinations_v1 import _parse_validation_results
from google.adk.evaluation.hallucinations_v1 import HallucinationsV1Evaluator
from google.adk.models.llm_response import LlmResponse
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import pytest


@pytest.fixture
def mock_llm_registry(mocker):
  """Mocks LLMRegistry to avoid actual model loading during tests."""
  MockLLMRegistry = mocker.patch(
      "google.adk.evaluation.hallucinations_v1.LLMRegistry"
  )
  mock_llm_cls = mocker.MagicMock()
  MockLLMRegistry.return_value.resolve.return_value = mock_llm_cls
  yield mock_llm_cls.return_value


@pytest.fixture
def hallucinations_metric(mock_llm_registry):
  """Provides a HallucinationsV1Evaluator instance for testing."""
  judge_model_options = JudgeModelOptions(
      judge_model="gemini-2.5-flash",
      judge_model_config=genai_types.GenerateContentConfig(temperature=0),
      num_samples=1,
  )
  criterion = HallucinationsCriterion(
      threshold=0.5,
      judge_model_options=judge_model_options,
      evaluate_intermediate_nl_responses=True,
  )
  eval_metric = EvalMetric(
      metric_name="hallucinations_v1", threshold=0.5, criterion=criterion
  )
  metric = HallucinationsV1Evaluator(eval_metric)
  return metric


class TestParseSentences:
  """Test cases for parsing sentences from segmenter response."""

  def test_parse_sentences_empty(self):
    """Tests _parse_sentences method with empty text."""
    text_empty = ""
    assert not _parse_sentences(text_empty)

  def test_parse_sentences_no_sentence(self):
    """Tests _parse_sentences method with no sentence."""
    text_no_sentence = "This is a sentence."
    assert not _parse_sentences(text_no_sentence)

  def test_parse_sentences_one_sentence(self):
    """Tests _parse_sentences method with one sentence."""
    text_one_sentence = "<sentence>This is a sentence.</sentence>"
    assert _parse_sentences(text_one_sentence) == ["This is a sentence."]

  def test_parse_sentences_multiple_sentences(self):
    """Tests _parse_sentences method with multiple sentences."""
    text_multiple_sentences = (
        "<sentence>Sentence 1.</sentence><sentence>Sentence 2.</sentence>"
    )
    assert _parse_sentences(text_multiple_sentences) == [
        "Sentence 1.",
        "Sentence 2.",
    ]

  def test_parse_sentences_with_bullets(self):
    """Tests _parse_sentences method with sentences containing bullets."""
    text_with_bullets = """<sentence>There are three kinds of fruits:</sentence>
<sentence>1. Apples are red.</sentence>
<sentence>2. Bananas are green.</sentence>
<sentence>3. Pears are purple.</sentence>"""
    assert _parse_sentences(text_with_bullets) == [
        "There are three kinds of fruits:",
        "1. Apples are red.",
        "2. Bananas are green.",
        "3. Pears are purple.",
    ]

  def test_parse_sentences_with_newlines(self):
    """Tests _parse_sentences method with sentences containing newlines."""
    text_with_newlines = """<sentence>This is a sentence with
\n\nnewlines.</sentence>
<sentence>This sentence won't be parsed because tag is misspelled</stenence>"""
    assert _parse_sentences(text_with_newlines) == [
        "This is a sentence with\n\n\nnewlines."
    ]


class TestParseValidationResults:
  """Test cases for parsing validation results from LLM response."""

  def test_parse_validation_results(self):
    """Tests _parse_validation_results method."""
    text = """sentence: Apples are red.
label: supported
rationale: The context explicitly states that apples are red.
supporting_excerpt: Apples are red fruits.
contradicting_excerpt: null

sentence: Bananas are green.
label: contradictory
rationale: The context states that bananas are yellow, not green.
supporting_excerpt: null
contradicting_excerpt: Bananas are yellow fruits.

sentence: Pears are purple.
label: disputed
rationale: The context states that pears are purple but it also states that pears are blue.
supporting_excerpt: Pears are purple fruits
contradicting_excerpt: Pears are blue fruits
"""
    expected = [
        {
            "sentence": "Apples are red.",
            "label": "supported",
            "rationale": "The context explicitly states that apples are red.",
            "supporting_excerpt": "Apples are red fruits.",
            "contradicting_excerpt": None,
        },
        {
            "sentence": "Bananas are green.",
            "label": "contradictory",
            "rationale": (
                "The context states that bananas are yellow, not green."
            ),
            "supporting_excerpt": None,
            "contradicting_excerpt": "Bananas are yellow fruits.",
        },
        {
            "sentence": "Pears are purple.",
            "label": "disputed",
            "rationale": (
                "The context states that pears are purple but it also states"
                " that pears are blue."
            ),
            "supporting_excerpt": "Pears are purple fruits",
            "contradicting_excerpt": "Pears are blue fruits",
        },
    ]
    assert _parse_validation_results(text) == expected

  def test_parse_validation_results_empty(self):
    """Tests _parse_validation_results with empty input."""
    text = ""
    assert not _parse_validation_results(text)


class TestEvaluateNlResponse:
  """Test cases for _evaluate_nl_response method."""

  def _create_genai_response(self, text, mocker):
    response_mock = mocker.MagicMock()
    response_mock.content = genai_types.Content(
        parts=[genai_types.Part(text=text)]
    )
    return response_mock

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_unexpected_labels(
      self, hallucinations_metric, mocker
  ):
    """Tests _evaluate_nl_response with unexpected labels."""
    metric = hallucinations_metric
    seg_response = self._create_genai_response(
        "<sentence>sentence 1</sentence><sentence>sentence 2</sentence>", mocker
    )
    val_response_text = """sentence: sentence 1
label:
rationale: r1
supporting_excerpt: null
contradicting_excerpt: null

sentence: sentence 2
label: unexpected
rationale: r2
supporting_excerpt: null
contradicting_excerpt: null
"""
    val_response = self._create_genai_response(val_response_text, mocker)

    async def seg_gen():
      yield seg_response

    async def val_gen():
      yield val_response

    metric._judge_model.generate_content_async = mocker.MagicMock(
        side_effect=[
            seg_gen(),
            val_gen(),
        ]
    )
    score, _ = await metric._evaluate_nl_response("nl", "ctx")
    assert score is None

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_missing_label(
      self, hallucinations_metric, mocker
  ):
    """Tests _evaluate_nl_response with missing labels in validation results."""
    metric = hallucinations_metric
    seg_response = self._create_genai_response(
        "<sentence>sentence 1</sentence>", mocker
    )
    val_response = self._create_genai_response("val_response", mocker)

    async def seg_gen():
      yield seg_response

    async def val_gen():
      yield val_response

    metric._judge_model.generate_content_async = mocker.MagicMock(
        side_effect=[
            seg_gen(),
            val_gen(),
        ]
    )
    score, _ = await metric._evaluate_nl_response("nl", "ctx")
    assert score is None

  async def _score_validator_reply(self, metric, mocker, validator_text):
    """Scores a two-sentence response against a scripted validator reply."""
    seg_response = self._create_genai_response(
        "<sentence>S1.</sentence><sentence>S2.</sentence>", mocker
    )
    val_response = self._create_genai_response(validator_text, mocker)

    async def seg_gen():
      yield seg_response

    async def val_gen():
      yield val_response

    metric._judge_model.generate_content_async = mocker.MagicMock(
        side_effect=[seg_gen(), val_gen()]
    )
    score, _ = await metric._evaluate_nl_response("nl", "ctx")
    return score

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_scores_every_labelled_sentence(
      self, hallucinations_metric, mocker
  ):
    """A complete reply is averaged over all sentences."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S2.
label: contradictory
rationale: r2
supporting_excerpt: null
contradicting_excerpt: e2
""",
    )
    assert score == 0.5

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_accepts_backticked_labels(
      self, hallucinations_metric, mocker
  ):
    """Labels echoed in backticks, as the prompt lists them, are counted."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: `supported`
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S2.
label: `contradictory`
rationale: r2
supporting_excerpt: null
contradicting_excerpt: e2
""",
    )
    assert score == 0.5

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_not_scored_when_sentence_omitted(
      self, hallucinations_metric, mocker
  ):
    """A sentence the validator leaves out does not drop out of the score."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null
""",
    )
    assert score is None

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_not_scored_when_a_label_is_unusable(
      self, hallucinations_metric, mocker
  ):
    """One unusable label makes the response unscored, not a 1.0."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S2.
label: false
rationale: r2
supporting_excerpt: null
contradicting_excerpt: e2
""",
    )
    assert score is None

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_not_scored_when_blocks_merge(
      self, hallucinations_metric, mocker
  ):
    """A block missing its last line must not swallow the next block."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1

sentence: S2.
label: contradictory
rationale: r2
supporting_excerpt: null
contradicting_excerpt: e2
""",
    )
    assert score is None

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_not_scored_with_extra_block(
      self, hallucinations_metric, mocker
  ):
    """An extra block with an unusable label cannot stand in for a sentence."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S1 again.
label: false
rationale: r2
supporting_excerpt: null
contradicting_excerpt: null

sentence: S2.
label: supported
rationale: r3
supporting_excerpt: e3
contradicting_excerpt: null
""",
    )
    assert score is None

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_not_scored_when_a_sentence_repeats(
      self, hallucinations_metric, mocker, caplog
  ):
    """A block repeating one sentence cannot stand in for an omitted one."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null
""",
    )
    assert score is None
    assert "block 1 does not echo sentence 1" in caplog.text

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_matches_sentences_with_tags_and_quotes(
      self, hallucinations_metric, mocker
  ):
    """An echo keeping the sentence tags or adding quotes still matches."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence: <sentence>S1.</sentence>
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: "S2."
label: supported
rationale: r2
supporting_excerpt: e2
contradicting_excerpt: null
""",
    )
    assert score == 1.0

  @pytest.mark.asyncio
  async def test_evaluate_nl_response_matches_sentences_ignoring_whitespace(
      self, hallucinations_metric, mocker
  ):
    """An echo differing only in case and spacing still matches."""
    score = await self._score_validator_reply(
        hallucinations_metric,
        mocker,
        """sentence:   s1.
label: supported
rationale: r1
supporting_excerpt: e1
contradicting_excerpt: null

sentence: S2.
label: supported
rationale: r2
supporting_excerpt: e2
contradicting_excerpt: null
""",
    )
    assert score == 1.0


@pytest.fixture
def create_context_data():
  """Provides data for TestCreateContext."""
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[
                  genai_types.Tool(
                      function_declarations=[
                          genai_types.FunctionDeclaration(name="tool1")
                      ]
                  )
              ],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query.")]
  )
  events = [
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          id="1", name="tool1", args={}
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          id="1",
                          name="tool1",
                          response={"result": "tool1 response"},
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(text="Intermediate NL response."),
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          id="2", name="tool1", args={}
                      )
                  ),
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          id="2",
                          name="tool1",
                          response={"result": "tool1 response 2"},
                      )
                  )
              ]
          ),
      ),
  ]
  invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(invocation_events=events),
  )
  return app_details, events, invocation


class TestCreateContext:
  """Test cases for creating the context in the validator prompt."""

  def test_create_context_for_intermediate_step(
      self, hallucinations_metric, create_context_data
  ):
    """Tests _create_context_for_step method."""
    app_details, events, invocation = create_context_data
    context = hallucinations_metric._create_context_for_step(
        app_details, invocation, events[:2]
    )
    expected_context = R"""Developer instructions:
root:
Root agent instructions.

User prompt:
User query.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool1"
          }
        ]
      }
    ]
  }
}

Grounding metadata:
No grounding metadata was provided.

tool_calls:
[
  {
    "id": "1",
    "args": {},
    "name": "tool1"
  }
]

tool_outputs:
[
  {
    "id": "1",
    "name": "tool1",
    "response": {
      "result": "tool1 response"
    }
  }
]
    """
    assert context.strip() == expected_context.strip()

  def test_create_context_for_final_step(
      self, hallucinations_metric, create_context_data
  ):
    """Tests _create_context_for_step method."""
    app_details, events, invocation = create_context_data
    context = hallucinations_metric._create_context_for_step(
        app_details, invocation, events
    )
    expected_context = R"""Developer instructions:
root:
Root agent instructions.

User prompt:
User query.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool1"
          }
        ]
      }
    ]
  }
}

Grounding metadata:
No grounding metadata was provided.

tool_calls:
[
  {
    "id": "1",
    "args": {},
    "name": "tool1"
  }
]

tool_outputs:
[
  {
    "id": "1",
    "name": "tool1",
    "response": {
      "result": "tool1 response"
    }
  }
]

Intermediate NL response.

tool_calls:
[
  {
    "id": "2",
    "args": {},
    "name": "tool1"
  }
]

tool_outputs:
[
  {
    "id": "2",
    "name": "tool1",
    "response": {
      "result": "tool1 response 2"
    }
  }
]
    """
    assert context.strip() == expected_context.strip()

  def test_create_context_for_step_includes_grounding_metadata(
      self,
      hallucinations_metric: HallucinationsV1Evaluator,
      create_context_data: tuple[AppDetails, list[InvocationEvent], Invocation],
  ) -> None:
    """Sentences grounded only via grounding_metadata must reach the judge."""
    app_details, events, invocation = create_context_data
    grounding_event = InvocationEvent(
        author="root",
        content=None,
        grounding_metadata=genai_types.GroundingMetadata(
            web_search_queries=["recent AI news"]
        ),
    )
    context = hallucinations_metric._create_context_for_step(
        app_details, invocation, events + [grounding_event]
    )
    grounding_section = context.split("Grounding metadata:\n")[1].split(
        "\n\ntool_calls:"
    )[0]
    parsed = json.loads(grounding_section)
    assert parsed["grounding_metadata"][0]["author"] == "root"
    assert parsed["grounding_metadata"][0]["grounding_metadata"][
        "web_search_queries"
    ] == ["recent AI news"]

  def test_get_steps_to_evaluate_includes_intermediate_grounding_metadata(
      self, hallucinations_metric: HallucinationsV1Evaluator
  ) -> None:
    """Intermediate NL steps include their own event's grounding_metadata."""
    invocation = Invocation(
        user_content=genai_types.Content(
            parts=[genai_types.Part(text="What's new in AI?")]
        ),
        intermediate_data=InvocationEvents(
            invocation_events=[
                InvocationEvent(
                    author="search_agent",
                    content=genai_types.Content(
                        parts=[
                            genai_types.Part(text="Grounded sub-agent step.")
                        ]
                    ),
                    grounding_metadata=genai_types.GroundingMetadata(
                        web_search_queries=["recent AI news"]
                    ),
                ),
            ]
        ),
        final_response=genai_types.Content(
            parts=[genai_types.Part(text="Final response.")]
        ),
    )
    steps = hallucinations_metric._get_steps_to_evaluate(invocation)
    assert len(steps) == 2
    assert steps[0].nl_response == "Grounded sub-agent step."
    assert "recent AI news" in steps[0].context
    assert "Grounded sub-agent step." not in steps[0].context
    assert steps[1].nl_response == "Final response."
    assert "recent AI news" in steps[1].context
    assert "Grounded sub-agent step." in steps[1].context

  def test_create_context_for_step_skips_thought_parts(
      self,
      hallucinations_metric: HallucinationsV1Evaluator,
      create_context_data: tuple[AppDetails, list[InvocationEvent], Invocation],
  ) -> None:
    """Thought parts of earlier events are not shown to the judge as context."""
    app_details, events, invocation = create_context_data
    thought_event = InvocationEvent(
        author="root",
        content=genai_types.Content(
            parts=[
                genai_types.Part(text="tool1 looks reliable.", thought=True),
                genai_types.Part(text="tool1 answered."),
            ]
        ),
    )

    context = hallucinations_metric._create_context_for_step(
        app_details, invocation, events + [thought_event]
    )

    assert "tool1 answered." in context
    assert "tool1 looks reliable." not in context

  def test_get_steps_to_evaluate_skips_thought_parts(
      self, hallucinations_metric: HallucinationsV1Evaluator
  ) -> None:
    """Thought parts are not evaluated as NL responses."""
    invocation = Invocation(
        user_content=genai_types.Content(
            parts=[genai_types.Part(text="What's new in AI?")]
        ),
        intermediate_data=InvocationEvents(
            invocation_events=[
                InvocationEvent(
                    author="root",
                    content=genai_types.Content(
                        parts=[
                            genai_types.Part(
                                text="I should search first.", thought=True
                            ),
                            genai_types.Part(text="Let me search."),
                        ]
                    ),
                ),
            ]
        ),
        final_response=genai_types.Content(
            parts=[
                genai_types.Part(text="Summarize the results.", thought=True),
                genai_types.Part(text="Final response."),
            ]
        ),
    )

    steps = hallucinations_metric._get_steps_to_evaluate(invocation)

    assert [step.nl_response for step in steps] == [
        "Let me search.",
        "Final response.",
    ]


@pytest.fixture
def agent_tree_data():
  """Provides data for TestEvaluateInvocationsAgentTree."""
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[
                  genai_types.Tool(
                      function_declarations=[
                          genai_types.FunctionDeclaration(name="tool_root")
                      ]
                  )
              ],
          ),
          "agent1": AgentDetails(
              name="agent1",
              instructions="Agent1 instructions.",
              tool_declarations=[
                  genai_types.Tool(
                      function_declarations=[
                          genai_types.FunctionDeclaration(name="tool_agent1")
                      ]
                  )
              ],
          ),
          "agent2": AgentDetails(
              name="agent2",
              instructions="Agent2 instructions.",
              tool_declarations=[],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query for agent tree.")]
  )
  events = [
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[genai_types.Part(text="Hi, I am root.")]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          name="tool_root", args={}
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          name="tool_root",
                          response={"result": "tool_root response"},
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="agent1",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          name="tool_agent1", args={"q": 1}
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="agent1",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          name="tool_agent1", response={"r": 2}
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="agent2",
          content=genai_types.Content(
              parts=[genai_types.Part(text="Agent2 response.")]
          ),
      ),
  ]
  invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(invocation_events=events),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent tree response.")]
      ),
  )
  expected_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent tree response.")]
      ),
  )
  return invocation, expected_invocation


class TestEvaluateInvocationsAgentTree:
  """Test cases for agent tree."""

  @pytest.mark.asyncio
  async def test_evaluate_invocations_multi_agents(
      self, hallucinations_metric, agent_tree_data, mocker
  ):
    """Tests evaluate_invocations with agent tree and checks contexts."""
    invocation, expected_invocation = agent_tree_data
    metric = hallucinations_metric
    expected_context0 = R"""Developer instructions:
root:
Root agent instructions.

agent1:
Agent1 instructions.

agent2:
Agent2 instructions.

User prompt:
User query for agent tree.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool_root"
          }
        ]
      }
    ],
    "agent1": [
      {
        "function_declarations": [
          {
            "name": "tool_agent1"
          }
        ]
      }
    ],
    "agent2": []
  }
}

Grounding metadata:
No grounding metadata was provided."""
    expected_context5 = R"""Developer instructions:
root:
Root agent instructions.

agent1:
Agent1 instructions.

agent2:
Agent2 instructions.

User prompt:
User query for agent tree.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool_root"
          }
        ]
      }
    ],
    "agent1": [
      {
        "function_declarations": [
          {
            "name": "tool_agent1"
          }
        ]
      }
    ],
    "agent2": []
  }
}

Grounding metadata:
No grounding metadata was provided.

Hi, I am root.

tool_calls:
[
  {
    "args": {},
    "name": "tool_root"
  }
]

tool_outputs:
[
  {
    "name": "tool_root",
    "response": {
      "result": "tool_root response"
    }
  }
]

tool_calls:
[
  {
    "args": {
      "q": 1
    },
    "name": "tool_agent1"
  }
]

tool_outputs:
[
  {
    "name": "tool_agent1",
    "response": {
      "r": 2
    }
  }
]"""
    expected_context6 = R"""Developer instructions:
root:
Root agent instructions.

agent1:
Agent1 instructions.

agent2:
Agent2 instructions.

User prompt:
User query for agent tree.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool_root"
          }
        ]
      }
    ],
    "agent1": [
      {
        "function_declarations": [
          {
            "name": "tool_agent1"
          }
        ]
      }
    ],
    "agent2": []
  }
}

Grounding metadata:
No grounding metadata was provided.

Hi, I am root.

tool_calls:
[
  {
    "args": {},
    "name": "tool_root"
  }
]

tool_outputs:
[
  {
    "name": "tool_root",
    "response": {
      "result": "tool_root response"
    }
  }
]

tool_calls:
[
  {
    "args": {
      "q": 1
    },
    "name": "tool_agent1"
  }
]

tool_outputs:
[
  {
    "name": "tool_agent1",
    "response": {
      "r": 2
    }
  }
]

Agent2 response.
"""

    async def mock_evaluate_nl_response(nl_response, context):
      if nl_response == "Hi, I am root.":
        assert context.strip() == expected_context0.strip()
        return 1.0, json.dumps(
            [{"sentence": "Hi, I am root.", "label": "supported"}]
        )
      elif nl_response == "Agent2 response.":
        assert context.strip() == expected_context5.strip()
        return 0.5, json.dumps(
            [{"sentence": "Agent2 response.", "label": "unsupported"}]
        )
      elif nl_response == "Final agent tree response.":
        assert context.strip() == expected_context6.strip()
        return 0.0, json.dumps([{
            "sentence": "Final agent tree response.",
            "label": "contradictory",
        }])
      return None, "error"

    mocker.patch(
        "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
        side_effect=mock_evaluate_nl_response,
    )
    result = await metric.evaluate_invocations(
        [invocation], [expected_invocation]
    )

    assert result.overall_score == pytest.approx(0.5)
    assert len(result.per_invocation_results) == 1
    per_invocation_result = result.per_invocation_results[0]
    assert per_invocation_result.score == pytest.approx(0.5)

  @pytest.mark.asyncio
  async def test_evaluate_invocations_agent_tree_skip_intermediate(
      self, mock_llm_registry, agent_tree_data, mocker
  ):
    """Tests evaluate_invocations with agent tree skipping intermediate steps."""
    invocation, expected_invocation = agent_tree_data
    judge_model_options = JudgeModelOptions(
        judge_model="gemini-2.5-flash",
        judge_model_config=genai_types.GenerateContentConfig(temperature=0),
        num_samples=1,
    )
    criterion = HallucinationsCriterion(
        threshold=0.5,
        judge_model_options=judge_model_options,
        evaluate_intermediate_nl_responses=False,
    )
    eval_metric = EvalMetric(
        metric_name="hallucinations_v1", threshold=0.5, criterion=criterion
    )
    metric = HallucinationsV1Evaluator(eval_metric)
    expected_context = R"""Developer instructions:
root:
Root agent instructions.

agent1:
Agent1 instructions.

agent2:
Agent2 instructions.

User prompt:
User query for agent tree.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "tool_root"
          }
        ]
      }
    ],
    "agent1": [
      {
        "function_declarations": [
          {
            "name": "tool_agent1"
          }
        ]
      }
    ],
    "agent2": []
  }
}

Grounding metadata:
No grounding metadata was provided.

Hi, I am root.

tool_calls:
[
  {
    "args": {},
    "name": "tool_root"
  }
]

tool_outputs:
[
  {
    "name": "tool_root",
    "response": {
      "result": "tool_root response"
    }
  }
]

tool_calls:
[
  {
    "args": {
      "q": 1
    },
    "name": "tool_agent1"
  }
]

tool_outputs:
[
  {
    "name": "tool_agent1",
    "response": {
      "r": 2
    }
  }
]

Agent2 response.
"""

    async def mock_evaluate_nl_response(nl_response, context):
      # Expect only the final response to be evaluated.
      assert nl_response == "Final agent tree response."
      assert context.strip() == expected_context.strip()
      return 0.0, json.dumps([{
          "sentence": "Final agent tree response.",
          "label": "contradictory",
      }])

    mocker.patch(
        "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
        side_effect=mock_evaluate_nl_response,
    )
    result = await metric.evaluate_invocations(
        [invocation], [expected_invocation]
    )

    assert result.overall_score == 0.0
    assert len(result.per_invocation_results) == 1
    per_invocation_result = result.per_invocation_results[0]
    assert per_invocation_result.score == 0.0


@pytest.fixture
def time_weather_data():
  """Provides data for TestEvaluateInvocationsTimeWeather."""
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions=(
                  "You are an agent that can get the current time and weather."
              ),
              tool_declarations=[
                  genai_types.Tool(
                      function_declarations=[
                          genai_types.FunctionDeclaration(
                              name="get_current_time",
                          ),
                          genai_types.FunctionDeclaration(name="get_weather"),
                      ]
                  )
              ],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[
          genai_types.Part(
              text="Get the current time and weather of San Francisco."
          )
      ]
  )
  response1 = (
      "The time in San Francisco is currently 10:30am PST. The date is"
      " September 21, 2025. I will now get the weather."
  )
  response2 = (
      "It is currently September 19, 2025, 10:30am PST in San Francisco. The"
      " weather is 65F with partly cloudy skies."
  )
  events = [
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          name="get_current_time",
                          args={"location": "San Francisco, CA"},
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          name="get_current_time",
                          response={"time": "10:30 AM PST Sep 19, 2025"},
                      )
                  )
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(text=response1),
                  genai_types.Part(
                      function_call=genai_types.FunctionCall(
                          name="get_weather",
                          args={
                              "location": "San Francisco, CA",
                              "time": "10:30 AM PST Sep 19, 2025",
                          },
                      )
                  ),
              ]
          ),
      ),
      InvocationEvent(
          author="root",
          content=genai_types.Content(
              parts=[
                  genai_types.Part(
                      function_response=genai_types.FunctionResponse(
                          name="get_weather",
                          response={"weather": "Partly cloudy, 65F"},
                      )
                  )
              ]
          ),
      ),
  ]
  invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(invocation_events=events),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=response2)]
      ),
  )
  return invocation, response1, response2


class TestEvaluateInvocationsTimeWeather:
  """Test cases for time/weather agent."""

  @pytest.mark.asyncio
  async def test_evaluate_invocations_time_weather(
      self, hallucinations_metric, time_weather_data, mocker
  ):
    """Tests evaluate_invocations with time/weather agent."""
    invocation, response1, response2 = time_weather_data
    metric = hallucinations_metric
    expected_context_1 = R"""Developer instructions:
root:
You are an agent that can get the current time and weather.

User prompt:
Get the current time and weather of San Francisco.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "get_current_time"
          },
          {
            "name": "get_weather"
          }
        ]
      }
    ]
  }
}

Grounding metadata:
No grounding metadata was provided.

tool_calls:
[
  {
    "args": {
      "location": "San Francisco, CA"
    },
    "name": "get_current_time"
  }
]

tool_outputs:
[
  {
    "name": "get_current_time",
    "response": {
      "time": "10:30 AM PST Sep 19, 2025"
    }
  }
]
"""
    expected_context_2 = R"""Developer instructions:
root:
You are an agent that can get the current time and weather.

User prompt:
Get the current time and weather of San Francisco.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "get_current_time"
          },
          {
            "name": "get_weather"
          }
        ]
      }
    ]
  }
}

Grounding metadata:
No grounding metadata was provided.

tool_calls:
[
  {
    "args": {
      "location": "San Francisco, CA"
    },
    "name": "get_current_time"
  }
]

tool_outputs:
[
  {
    "name": "get_current_time",
    "response": {
      "time": "10:30 AM PST Sep 19, 2025"
    }
  }
]

The time in San Francisco is currently 10:30am PST. The date is September 21, 2025. I will now get the weather.

tool_calls:
[
  {
    "args": {
      "location": "San Francisco, CA",
      "time": "10:30 AM PST Sep 19, 2025"
    },
    "name": "get_weather"
  }
]

tool_outputs:
[
  {
    "name": "get_weather",
    "response": {
      "weather": "Partly cloudy, 65F"
    }
  }
]
"""

    async def mock_evaluate_nl_response(nl_response, context):
      if nl_response == response1:
        assert context.strip() == expected_context_1.strip()
        sentence1, sentence2, sentence3, _ = response1.split(".")
        return 2.0 / 3.0, json.dumps([
            {"sentence": sentence1, "label": "supported"},
            {"sentence": sentence2, "label": "contradictory"},
            {"sentence": sentence3, "label": "supported"},
        ])
      elif nl_response == response2:
        assert context.strip() == expected_context_2.strip()
        sentence1, sentence2, _ = response2.split(".")
        return 1.0, json.dumps([
            {"sentence": sentence1, "label": "supported"},
            {"sentence": sentence2, "label": "supported"},
        ])
      return None, "error"

    mocker.patch(
        "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
        side_effect=mock_evaluate_nl_response,
    )
    result = await metric.evaluate_invocations([invocation], [invocation])

    assert result.overall_score == pytest.approx(5 / 6)
    assert len(result.per_invocation_results) == 1
    per_invocation_result = result.per_invocation_results[0]
    assert per_invocation_result.score == pytest.approx(5 / 6)

  @pytest.mark.asyncio
  async def test_evaluate_invocations_time_weather_skip_intermediate(
      self, mock_llm_registry, time_weather_data, mocker
  ):
    """Tests evaluate_invocations with time/weather agent."""
    invocation, _, response2 = time_weather_data
    judge_model_options = JudgeModelOptions(
        judge_model="gemini-2.5-flash",
        judge_model_config=genai_types.GenerateContentConfig(temperature=0),
        num_samples=1,
    )
    criterion = HallucinationsCriterion(
        threshold=0.5,
        judge_model_options=judge_model_options,
        evaluate_intermediate_nl_responses=False,
    )
    eval_metric = EvalMetric(
        metric_name="hallucinations_v1", threshold=0.5, criterion=criterion
    )
    metric = HallucinationsV1Evaluator(eval_metric)
    expected_context = R"""Developer instructions:
root:
You are an agent that can get the current time and weather.

User prompt:
Get the current time and weather of San Francisco.

Tool definitions:
{
  "tool_declarations": {
    "root": [
      {
        "function_declarations": [
          {
            "name": "get_current_time"
          },
          {
            "name": "get_weather"
          }
        ]
      }
    ]
  }
}

Grounding metadata:
No grounding metadata was provided.

tool_calls:
[
  {
    "args": {
      "location": "San Francisco, CA"
    },
    "name": "get_current_time"
  }
]

tool_outputs:
[
  {
    "name": "get_current_time",
    "response": {
      "time": "10:30 AM PST Sep 19, 2025"
    }
  }
]

The time in San Francisco is currently 10:30am PST. The date is September 21, 2025. I will now get the weather.

tool_calls:
[
  {
    "args": {
      "location": "San Francisco, CA",
      "time": "10:30 AM PST Sep 19, 2025"
    },
    "name": "get_weather"
  }
]

tool_outputs:
[
  {
    "name": "get_weather",
    "response": {
      "weather": "Partly cloudy, 65F"
    }
  }
]
"""

    async def mock_evaluate_nl_response(nl_response, context):
      # Expect only the final response to be evaluated.
      assert nl_response == response2
      assert context.strip() == expected_context.strip()
      sentence1, sentence2, _ = response2.split(".")
      return 1.0, json.dumps([
          {"sentence": sentence1, "label": "supported"},
          {"sentence": sentence2, "label": "supported"},
      ])

    mocker.patch(
        "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
        side_effect=mock_evaluate_nl_response,
    )
    result = await metric.evaluate_invocations([invocation], [invocation])

    assert result.overall_score == 1.0
    assert len(result.per_invocation_results) == 1
    per_invocation_result = result.per_invocation_results[0]
    assert per_invocation_result.score == 1.0


@pytest.mark.asyncio
async def test_evaluate_invocations_success_path(hallucinations_metric, mocker):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query.")]
  )
  actual_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(text="Intermediate NL response."),
                      ]
                  ),
              ),
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(
                              text="Another intermediate NL response."
                          ),
                      ]
                  ),
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )
  expected_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )

  async def mock_evaluate_nl_response(nl_response, context):
    if nl_response == "Intermediate NL response.":
      return 1.0, json.dumps(
          [{"sentence": "Intermediate NL response.", "label": "supported"}]
      )
    elif nl_response == "Another intermediate NL response.":
      return 0.5, json.dumps([{
          "sentence": "Another intermediate NL response.",
          "label": "unsupported",
      }])
    elif nl_response == "Final response.":
      return 0.0, json.dumps(
          [{"sentence": "Final response.", "label": "contradictory"}]
      )
    return None, "error"

  mocker.patch(
      "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
      side_effect=mock_evaluate_nl_response,
  )
  result = await metric.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )

  assert result.overall_score == pytest.approx(0.5)
  assert len(result.per_invocation_results) == 1
  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_evaluate_invocations_no_nl_response(hallucinations_metric):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query.")]
  )
  actual_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(
                              function_call=genai_types.FunctionCall(
                                  name="tool1", args={}
                              )
                          )
                      ]
                  ),
              ),
          ]
      ),
      final_response=None,
  )
  expected_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
  )

  result = await metric.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )
  assert result.overall_score is None
  assert len(result.per_invocation_results) == 1
  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.score is None
  assert per_invocation_result.eval_status == EvalStatus.NOT_EVALUATED


@pytest.mark.asyncio
async def test_evaluate_all_invocations_not_evaluated(
    hallucinations_metric, mocker
):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query.")]
  )
  actual_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(text="Intermediate NL response."),
                      ]
                  ),
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )
  expected_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )

  async def mock_evaluate_nl_response(nl_response, context):
    return None, "Judge model error."

  mocker.patch(
      "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
      side_effect=mock_evaluate_nl_response,
  )
  result = await metric.evaluate_invocations(
      [actual_invocation, actual_invocation],
      [expected_invocation, expected_invocation],
  )

  assert len(result.per_invocation_results) == 2
  assert result.per_invocation_results[0].score is None
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
  )
  assert result.per_invocation_results[1].score is None
  assert (
      result.per_invocation_results[1].eval_status == EvalStatus.NOT_EVALUATED
  )
  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED


@pytest.mark.asyncio
async def test_evaluate_invocations_partial_failure(
    hallucinations_metric, mocker
):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )
  user_content = genai_types.Content(
      parts=[genai_types.Part(text="User query.")]
  )
  actual_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(text="Intermediate NL response."),
                      ]
                  ),
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )
  expected_invocation = Invocation(
      app_details=app_details,
      user_content=user_content,
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )

  async def mock_evaluate_nl_response(nl_response, context):
    if nl_response == "Intermediate NL response.":
      return 0.8, json.dumps(
          [{"sentence": "Intermediate NL response.", "label": "supported"}]
      )
    elif nl_response == "Final response.":
      return None, "some error during evaluation"
    return None, "error"

  mocker.patch(
      "google.adk.evaluation.hallucinations_v1.HallucinationsV1Evaluator._evaluate_nl_response",
      side_effect=mock_evaluate_nl_response,
  )
  result = await metric.evaluate_invocations(
      [actual_invocation], [expected_invocation]
  )

  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  assert len(result.per_invocation_results) == 1
  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.score is None
  assert per_invocation_result.eval_status == EvalStatus.NOT_EVALUATED


@pytest.mark.asyncio
async def test_evaluate_invocations_unscored_step_still_fails_invocation(
    hallucinations_metric, mocker
):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )
  actual_invocation = Invocation(
      app_details=app_details,
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User query.")]
      ),
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[genai_types.Part(text=text)]
                  ),
              )
              for text in ("Unscored response.", "Hallucinated response.")
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Hallucinated final response.")]
      ),
  )

  async def mock_evaluate_nl_response(nl_response, context):
    if nl_response == "Unscored response.":
      return None, "some error during evaluation"
    return 0.0, "[]"

  mocker.patch.object(
      metric, "_evaluate_nl_response", side_effect=mock_evaluate_nl_response
  )
  result = await metric.evaluate_invocations([actual_invocation])

  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.score == pytest.approx(1 / 3)
  assert per_invocation_result.eval_status == EvalStatus.FAILED
  assert result.overall_score == pytest.approx(1 / 3)
  assert result.overall_eval_status == EvalStatus.FAILED


@pytest.mark.asyncio
async def test_evaluate_invocations_unscored_invocation_leaves_case_unscored(
    hallucinations_metric, mocker
):
  metric = hallucinations_metric
  app_details = AppDetails(
      agent_details={
          "root": AgentDetails(
              name="root",
              instructions="Root agent instructions.",
              tool_declarations=[],
          ),
      },
  )

  def make_invocation(final_text):
    return Invocation(
        app_details=app_details,
        user_content=genai_types.Content(
            parts=[genai_types.Part(text="User query.")]
        ),
        final_response=(
            genai_types.Content(parts=[genai_types.Part(text=final_text)])
            if final_text
            else None
        ),
    )

  async def mock_evaluate_nl_response(nl_response, context):
    if nl_response == "Good response.":
      return 1.0, "[]"
    if nl_response == "Hallucinated response.":
      return 0.0, "[]"
    return None, "some error during evaluation"

  mocker.patch.object(
      metric, "_evaluate_nl_response", side_effect=mock_evaluate_nl_response
  )

  result = await metric.evaluate_invocations([
      make_invocation("Good response."),
      make_invocation("Bad response."),
  ])
  assert result.per_invocation_results[0].score == 1.0
  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED

  # The case still fails when it would with the unscored response at 1.0.
  result = await metric.evaluate_invocations([
      make_invocation("Hallucinated response."),
      make_invocation("Hallucinated response."),
      make_invocation("Bad response."),
  ])
  assert result.per_invocation_results[2].score is None
  assert result.overall_score == pytest.approx(1 / 3)
  assert result.overall_eval_status == EvalStatus.FAILED

  # An invocation with no response to judge does not unscore the case.
  result = await metric.evaluate_invocations([
      make_invocation("Good response."),
      make_invocation(None),
  ])
  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_judge_model_config",
    [
        None,
        genai_types.GenerateContentConfig(temperature=0.3),
    ],
    ids=["default_config", "user_supplied_config"],
)
async def test_evaluate_nl_response_judge_request_disables_afc(
    mock_llm_registry, user_judge_model_config, mocker
):
  # Both the segmenter and validator judge requests must disable google-genai's
  # automatic function calling regardless of whether the caller supplied a
  # judge_model_config, since the judge never calls tools and leaving AFC on
  # only produces a spurious per-request warning on every eval run.
  del mock_llm_registry  # fixture only needed for its side effect.
  judge_model_options = JudgeModelOptions(
      judge_model="gemini-2.5-flash",
      judge_model_config=user_judge_model_config,
      num_samples=1,
  )
  criterion = HallucinationsCriterion(
      threshold=0.5,
      judge_model_options=judge_model_options,
      evaluate_intermediate_nl_responses=True,
  )
  metric = HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1", threshold=0.5, criterion=criterion
      )
  )

  captured_requests = []

  async def mock_generate(llm_request):
    captured_requests.append(llm_request)
    response = mocker.MagicMock()
    response.content = genai_types.Content(
        parts=[genai_types.Part(text="<sentence>s</sentence>")]
    )
    yield response

  metric._judge_model.generate_content_async = mock_generate
  await metric._evaluate_nl_response("nl", "ctx")

  # Both the segmenter and validator should have been called.
  assert len(captured_requests) == 2
  for request in captured_requests:
    assert request.config.automatic_function_calling is not None
    assert request.config.automatic_function_calling.disable is True


async def test_evaluate_invocations_runs_judge_calls_up_to_parallelism_limit(
    mock_llm_registry,
):
  """Judge calls run concurrently, never more than parallelism_limit at once.

  Setup: parallelism_limit=3 and four single-response invocations. Each judge
    call holds its slot until three calls are in flight.
  Act: Evaluate the invocations.
  Assert: The most judge calls in flight at once is three.
  """
  metric = HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1",
          threshold=0.5,
          criterion=HallucinationsCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash", parallelism_limit=3
              ),
          ),
      )
  )
  in_flight = 0
  max_in_flight = 0
  all_slots_filled = asyncio.Event()

  async def mock_generate(llm_request):
    del llm_request
    nonlocal in_flight, max_in_flight
    in_flight += 1
    max_in_flight = max(max_in_flight, in_flight)
    if in_flight == 3:
      all_slots_filled.set()
    try:
      await asyncio.wait_for(all_slots_filled.wait(), timeout=1.0)
      # Yield so a call beyond the limit, if unbounded, would start now.
      await asyncio.sleep(0.01)
      yield LlmResponse(
          content=genai_types.Content(
              parts=[genai_types.Part(text="<sentence>s</sentence>")]
          )
      )
    finally:
      in_flight -= 1

  mock_llm_registry.generate_content_async = mock_generate
  invocations = [
      Invocation(
          invocation_id=f"inv_{i}",
          user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text=f"answer {i}")]
          ),
      )
      for i in range(4)
  ]

  await metric.evaluate_invocations(invocations)

  assert max_in_flight == 3


async def test_evaluate_invocations_keeps_input_order_on_out_of_order_finish(
    mock_llm_registry,
):
  """Results keep input order when a later invocation finishes first.

  Setup: parallelism_limit=3 and three single-sentence invocations. The first
    invocation's validator call waits until the other two validators finish.
    The first answer is supported; the other two are contradictory.
  Act: Evaluate the invocations.
  Assert: The first invocation finished last, yet results are in input order,
    each with its own score.
  """
  metric = HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1",
          threshold=0.5,
          criterion=HallucinationsCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash", parallelism_limit=3
              ),
          ),
      )
  )
  labels = {
      "answer_alpha": "supported",
      "answer_bravo": "contradictory",
      "answer_charlie": "contradictory",
  }
  finished = []
  others_finished = asyncio.Event()

  async def mock_generate(llm_request):
    prompt = llm_request.contents[0].parts[0].text
    sentence = next(s for s in labels if s in prompt)
    if "**Your Sentence Segmentation Output:**" in prompt:
      text = f"<sentence>{sentence}</sentence>"
    else:
      if sentence == "answer_alpha":
        await asyncio.wait_for(others_finished.wait(), timeout=1.0)
      finished.append(sentence)
      if len(finished) == 2:
        others_finished.set()
      text = (
          f"sentence: {sentence}\n"
          f"label: {labels[sentence]}\n"
          "rationale: test rationale\n"
          "supporting_excerpt: excerpt\n"
          "contradicting_excerpt: null"
      )
    yield LlmResponse(
        content=genai_types.Content(parts=[genai_types.Part(text=text)])
    )

  mock_llm_registry.generate_content_async = mock_generate
  invocations = [
      Invocation(
          invocation_id=sentence,
          user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text=sentence)]
          ),
      )
      for sentence in labels
  ]

  result = await metric.evaluate_invocations(invocations)

  assert finished[-1] == "answer_alpha"
  assert [
      r.actual_invocation.invocation_id for r in result.per_invocation_results
  ] == ["answer_alpha", "answer_bravo", "answer_charlie"]
  assert [r.score for r in result.per_invocation_results] == [1.0, 0.0, 0.0]


async def test_evaluate_invocations_cancels_in_flight_judge_calls_on_error(
    mock_llm_registry, mocker
):
  """An invocation that raises cancels judge calls still in flight.

  Setup: parallelism_limit=2. The first invocation's judge call blocks, and
    collecting the steps of the second invocation raises.
  Act: Evaluate the invocations.
  Assert: The error propagates and the blocked judge call was cancelled.
  """
  metric = HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1",
          threshold=0.5,
          criterion=HallucinationsCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash", parallelism_limit=2
              ),
          ),
      )
  )
  cancelled = False

  async def mock_generate(llm_request):
    del llm_request
    nonlocal cancelled
    try:
      await asyncio.wait_for(asyncio.Event().wait(), timeout=2.0)
      yield LlmResponse()
    except asyncio.CancelledError:
      cancelled = True
      raise

  mock_llm_registry.generate_content_async = mock_generate
  get_steps_to_evaluate = metric._get_steps_to_evaluate

  def get_steps_or_raise(actual):
    if actual.invocation_id == "broken":
      raise RuntimeError("bad invocation")
    return get_steps_to_evaluate(actual)

  mocker.patch.object(
      metric, "_get_steps_to_evaluate", side_effect=get_steps_or_raise
  )
  invocations = [
      Invocation(
          invocation_id=invocation_id,
          user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="answer")]
          ),
      )
      for invocation_id in ["blocked", "broken"]
  ]

  with pytest.raises(RuntimeError, match="bad invocation"):
    await metric.evaluate_invocations(invocations)

  assert cancelled


def _make_metric_with_num_samples(
    num_samples: int,
) -> HallucinationsV1Evaluator:
  return HallucinationsV1Evaluator(
      EvalMetric(
          metric_name="hallucinations_v1",
          threshold=0.5,
          criterion=HallucinationsCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash", num_samples=num_samples
              ),
          ),
      )
  )


def _single_response_invocation() -> Invocation:
  return Invocation(
      user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final response.")]
      ),
  )


_SUPPORTED_VALIDATOR_REPLY = """sentence: Final response.
label: supported
rationale: r
supporting_excerpt: e
contradicting_excerpt: null
"""


async def test_evaluate_invocations_retries_transient_judge_error(
    mock_llm_registry, caplog
):
  """A single transient judge failure is retried instead of losing the case.

  Setup: num_samples=3. The first segmenter call raises; later calls succeed.
  Act: Evaluate one invocation.
  Assert: The invocation is scored, and the failed attempt is logged.
  """
  metric = _make_metric_with_num_samples(3)
  calls = 0

  async def mock_generate(llm_request):
    nonlocal calls
    calls += 1
    if calls == 1:
      raise RuntimeError("503 UNAVAILABLE")
    prompt = llm_request.contents[0].parts[0].text
    text = (
        _SUPPORTED_VALIDATOR_REPLY
        if "Sentences Begin" in prompt and "Context Begin" in prompt
        else "<sentence>Final response.</sentence>"
    )
    yield LlmResponse(
        content=genai_types.Content(parts=[genai_types.Part(text=text)])
    )

  mock_llm_registry.generate_content_async = mock_generate

  with caplog.at_level("INFO"):
    result = await metric.evaluate_invocations([_single_response_invocation()])

  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[0].rubric_scores == []
  # 1 failed segmenter call + segmenter + validator on the retry.
  assert calls == 3
  assert "attempt 1/3 produced no score, retrying" in caplog.text
  assert "503 UNAVAILABLE" in caplog.text
  # A recovered failure is not reported as a lost score.
  assert "could not score" not in caplog.text


async def test_evaluate_invocations_reports_reason_when_all_attempts_fail(
    mock_llm_registry, caplog
):
  """Exhausted attempts yield NOT_EVALUATED with the reason attached.

  Setup: num_samples=2. The segmenter never returns <sentence> tags.
  Act: Evaluate one invocation.
  Assert: NOT_EVALUATED, an "error" RubricScore carries the reason, the judge
    was tried num_samples times, and a WARNING is logged.
  """
  metric = _make_metric_with_num_samples(2)
  calls = 0

  async def mock_generate(llm_request):
    del llm_request
    nonlocal calls
    calls += 1
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="```\nFinal response.\n```")]
        )
    )

  mock_llm_registry.generate_content_async = mock_generate

  with caplog.at_level("WARNING"):
    result = await metric.evaluate_invocations([_single_response_invocation()])

  assert calls == 2
  assert result.overall_score is None
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.eval_status == EvalStatus.NOT_EVALUATED
  assert len(per_invocation_result.rubric_scores) == 1
  error_score = per_invocation_result.rubric_scores[0]
  assert error_score.rubric_id == "error"
  assert error_score.score is None
  assert "No score after 2 attempt(s)" in error_score.rationale
  assert "No sentences produced by segmenter." in error_score.rationale
  assert "could not score a response after 2 attempt(s)" in caplog.text


async def test_evaluate_invocations_does_not_retry_without_response(
    mock_llm_registry,
):
  """An invocation with no response text is not sent to the judge at all."""
  metric = _make_metric_with_num_samples(3)
  calls = 0

  async def mock_generate(llm_request):
    del llm_request
    nonlocal calls
    calls += 1
    yield LlmResponse()

  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([
      Invocation(
          user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
          final_response=None,
      )
  ])

  assert calls == 0
  assert result.per_invocation_results[0].rubric_scores == []


async def test_evaluate_invocations_attaches_reason_per_unscored_step(
    hallucinations_metric, mocker
):
  """Each unscored step contributes its own "error" RubricScore."""
  metric = hallucinations_metric
  actual_invocation = Invocation(
      user_content=genai_types.Content(parts=[genai_types.Part(text="q")]),
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="root",
                  content=genai_types.Content(
                      parts=[genai_types.Part(text="Intermediate.")]
                  ),
              )
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final.")]
      ),
  )

  async def mock_evaluate_nl_response(nl_response, context):
    del context
    if nl_response == "Intermediate.":
      return 1.0, "[]"
    return None, "Sentence validator returned no text."

  mocker.patch.object(
      metric, "_evaluate_nl_response", side_effect=mock_evaluate_nl_response
  )
  result = await metric.evaluate_invocations([actual_invocation])

  rubric_scores = result.per_invocation_results[0].rubric_scores
  assert [s.rubric_id for s in rubric_scores] == ["error"]
  assert rubric_scores[0].rationale.startswith("Response 2 of 2")
  assert "Sentence validator returned no text." in rubric_scores[0].rationale


def _counting_judge(responses_or_errors):
  """Returns a judge that yields/raises the given items, then repeats the last."""
  calls = []

  async def mock_generate(llm_request):
    del llm_request
    item = responses_or_errors[min(len(calls), len(responses_or_errors) - 1)]
    calls.append(item)
    if isinstance(item, Exception):
      raise item
    yield item

  return mock_generate, calls


def _text_response(text, finish_reason=None):
  return LlmResponse(
      content=genai_types.Content(parts=[genai_types.Part(text=text)]),
      finish_reason=finish_reason,
  )


async def test_evaluate_invocations_caps_attempts(mock_llm_registry):
  """num_samples=5 still makes at most 3 attempts on a judge that keeps failing."""
  metric = _make_metric_with_num_samples(5)
  mock_generate, calls = _counting_judge([RuntimeError("503 UNAVAILABLE")])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  assert len(calls) == 3
  rationale = result.per_invocation_results[0].rubric_scores[0].rationale
  assert "No score after 3 attempt(s)" in rationale


@pytest.mark.parametrize(
    "error",
    [
        genai_errors.ClientError(400, {"error": {"message": "judge error"}}),
        genai_errors.ClientError(429, {"error": {"message": "judge error"}}),
        genai_errors.ServerError(503, {"error": {"message": "judge error"}}),
    ],
    ids=["400", "429", "503"],
)
async def test_evaluate_invocations_does_not_retry_api_errors(
    mock_llm_registry, error
):
  """API errors are left to the HTTP retry options and not retried again."""
  metric = _make_metric_with_num_samples(3)
  mock_generate, calls = _counting_judge([error])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  assert len(calls) == 1
  assert result.overall_eval_status == EvalStatus.NOT_EVALUATED
  rationale = result.per_invocation_results[0].rubric_scores[0].rationale
  assert "No score after 1 attempt(s)" in rationale
  assert str(error.code) in rationale


def _judge_by_stage(validator_replies):
  """Returns a judge that segments correctly and replays validator replies."""
  calls = []

  async def mock_generate(llm_request):
    prompt = llm_request.contents[0].parts[0].text
    is_validator = "Context Begin" in prompt
    calls.append("validator" if is_validator else "segmenter")
    if not is_validator:
      yield _text_response("<sentence>Final response.</sentence>")
      return
    index = min(calls.count("validator"), len(validator_replies)) - 1
    yield validator_replies[index]

  return mock_generate, calls


async def test_evaluate_invocations_retries_truncated_judge_output(
    mock_llm_registry,
):
  """Truncation varies between calls with thinking models, so it is retried."""
  metric = _make_metric_with_num_samples(3)
  mock_generate, calls = _judge_by_stage([
      _text_response(
          "sentence: Final response.\nlabel: supp",
          finish_reason=genai_types.FinishReason.MAX_TOKENS,
      ),
      _text_response(_SUPPORTED_VALIDATOR_REPLY),
  ])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  assert calls == ["segmenter", "validator", "segmenter", "validator"]
  assert result.overall_score == 1.0
  assert result.per_invocation_results[0].rubric_scores == []


async def test_evaluate_invocations_reports_finish_reason_of_truncation(
    mock_llm_registry,
):
  """When truncation persists, the reason says why the judge stopped."""
  metric = _make_metric_with_num_samples(3)
  mock_generate, calls = _judge_by_stage([
      _text_response(
          "sentence: Final response.\nlabel: supp",
          finish_reason=genai_types.FinishReason.MAX_TOKENS,
      ),
  ])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  assert calls.count("validator") == 3
  rationale = result.per_invocation_results[0].rubric_scores[0].rationale
  assert "No score after 3 attempt(s)" in rationale
  assert "returned 0 blocks for 1 sentences" in rationale
  assert "finish_reason=MAX_TOKENS" in rationale


async def test_evaluate_invocations_reports_readable_partial_parse_reason(
    mock_llm_registry,
):
  """A partial parse reports what went wrong, not the raw validator output."""
  metric = _make_metric_with_num_samples(1)
  mock_generate, _ = _counting_judge([
      _text_response("<sentence>Final response.</sentence>"),
      _text_response(
          _SUPPORTED_VALIDATOR_REPLY + "\n" + _SUPPORTED_VALIDATOR_REPLY
      ),
  ])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  rationale = result.per_invocation_results[0].rubric_scores[0].rationale
  assert "Sentence validator returned 2 blocks for 1 sentences." in rationale
  assert "supporting_excerpt" not in rationale


async def test_evaluate_invocations_reports_reasons_at_case_level(
    mock_llm_registry,
):
  """Failure reasons are repeated in overall_rubric_scores for eval reports."""
  metric = _make_metric_with_num_samples(1)
  mock_generate, _ = _counting_judge([RuntimeError("503 UNAVAILABLE")])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations(
      [_single_response_invocation(), _single_response_invocation()]
  )

  overall = result.overall_rubric_scores
  assert [s.rubric_id for s in overall] == ["error", "error"]
  assert overall[0].rationale.startswith("Invocation 1 of 2: Response 1 of 1")
  assert overall[1].rationale.startswith("Invocation 2 of 2: Response 1 of 1")
  assert "503 UNAVAILABLE" in overall[0].rationale


async def test_evaluate_invocations_bounds_rationale_length(mock_llm_registry):
  """A huge judge error message does not bloat the stored rationale."""
  metric = _make_metric_with_num_samples(1)
  mock_generate, _ = _counting_judge([RuntimeError("x" * 10_000)])
  mock_llm_registry.generate_content_async = mock_generate

  result = await metric.evaluate_invocations([_single_response_invocation()])

  rationale = result.per_invocation_results[0].rubric_scores[0].rationale
  assert len(rationale) < 2100
  assert rationale.endswith("... [truncated]")
