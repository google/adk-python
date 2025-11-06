# Copyright 2025 Google LLC
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
from google.genai import types as genai_types
import pytest


@pytest.fixture
def mock_llm_registry(mocker):
  """Mocks LLMRegistry to avoid actual model loading during tests."""
  MockLLMRegistry = mocker.patch(
      "google.adk.evaluation.hallucinations_v1.LLMRegistry"
  )
  MockLLMRegistry.return_value.resolve.return_value = mocker.MagicMock()
  yield


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
}"""
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

  assert result.overall_score == 0.8
  assert len(result.per_invocation_results) == 1
  per_invocation_result = result.per_invocation_results[0]
  assert per_invocation_result.score == 0.8
