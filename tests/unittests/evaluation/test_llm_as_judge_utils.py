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

from __future__ import annotations

import json

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.llm_as_judge_utils import get_average_rubric_score
from google.adk.evaluation.llm_as_judge_utils import get_eval_status
from google.adk.evaluation.llm_as_judge_utils import get_text_from_content
from google.adk.evaluation.llm_as_judge_utils import get_tool_calls_and_responses_as_json_str
from google.adk.evaluation.llm_as_judge_utils import get_tool_declarations_as_json_str
from google.genai import types as genai_types


def test_get_text_from_content_with_none():
  """Tests get_text_from_content with None as input."""
  assert get_text_from_content(None) is None


def test_get_text_from_content_with_content_and_none_parts():
  """Tests get_text_from_content with Content that has None for parts."""
  content = genai_types.Content(parts=None)
  assert get_text_from_content(content) is None


def test_get_text_from_content_with_empty_parts():
  """Tests get_text_from_content with an empty parts list."""
  content = genai_types.Content(parts=[])
  assert get_text_from_content(content) == None


def test_get_text_from_content_with_parts_but_no_text():
  """Tests get_text_from_content with parts that do not contain text."""
  content = genai_types.Content(
      parts=[
          genai_types.Part(
              function_call=genai_types.FunctionCall(name="test_func")
          )
      ]
  )
  assert get_text_from_content(content) == ""


def test_get_text_from_content_with_single_text_part():
  """Tests get_text_from_content with a single text part."""
  content = genai_types.Content(parts=[genai_types.Part(text="Hello")])
  assert get_text_from_content(content) == "Hello"


def test_get_text_from_content_with_multiple_text_parts():
  """Tests get_text_from_content with multiple text parts."""
  content = genai_types.Content(
      parts=[genai_types.Part(text="Hello"), genai_types.Part(text="World")]
  )
  assert get_text_from_content(content) == "Hello\nWorld"


def test_get_text_from_content_with_mixed_parts():
  """Tests get_text_from_content with a mix of text and non-text parts."""
  content = genai_types.Content(
      parts=[
          genai_types.Part(text="Hello"),
          genai_types.Part(
              function_call=genai_types.FunctionCall(name="test_func")
          ),
          genai_types.Part(text="World"),
      ]
  )
  assert get_text_from_content(content) == "Hello\nWorld"


def test_get_eval_status_with_none_score():
  """Tests get_eval_status returns NOT_EVALUATED for a None score."""
  assert get_eval_status(score=None, threshold=0.5) == EvalStatus.NOT_EVALUATED


def test_get_eval_status_when_score_is_greater_than_threshold():
  """Tests get_eval_status returns PASSED when score > threshold."""
  assert get_eval_status(score=0.8, threshold=0.5) == EvalStatus.PASSED


def test_get_eval_status_when_score_is_equal_to_threshold():
  """Tests get_eval_status returns PASSED when score == threshold."""
  assert get_eval_status(score=0.5, threshold=0.5) == EvalStatus.PASSED


def test_get_eval_status_when_score_is_less_than_threshold():
  """Tests get_eval_status returns FAILED when score < threshold."""
  assert get_eval_status(score=0.4, threshold=0.5) == EvalStatus.FAILED


def test_get_average_rubric_score_with_empty_list():
  """Tests get_average_rubric_score returns None for an empty list."""
  assert get_average_rubric_score([]) is None


def test_get_average_rubric_score_with_all_none_scores():
  """Tests get_average_rubric_score returns None when all scores are None."""
  rubric_scores = [
      RubricScore(rubric_id="1", score=None),
      RubricScore(rubric_id="2", score=None),
  ]
  assert get_average_rubric_score(rubric_scores) is None


def test_get_average_rubric_score_with_single_score():
  """Tests get_average_rubric_score with a single valid score."""
  rubric_scores = [RubricScore(rubric_id="1", score=0.8)]
  assert get_average_rubric_score(rubric_scores) == 0.8


def test_get_average_rubric_score_with_multiple_scores():
  """Tests get_average_rubric_score with multiple valid scores."""
  rubric_scores = [
      RubricScore(rubric_id="1", score=0.8),
      RubricScore(rubric_id="2", score=0.6),
  ]
  assert get_average_rubric_score(rubric_scores) == 0.7


def test_get_average_rubric_score_with_mixed_scores():
  """Tests get_average_rubric_score with a mix of valid and None scores."""
  rubric_scores = [
      RubricScore(rubric_id="1", score=0.8),
      RubricScore(rubric_id="2", score=None),
      RubricScore(rubric_id="3", score=0.6),
  ]
  assert get_average_rubric_score(rubric_scores) == 0.7


def test_get_tool_declarations_as_json_str_with_no_agents():
  """Tests get_tool_declarations_as_json_str with no agents."""
  app_details = AppDetails(agent_details={})
  expected_json = {"tool_declarations": {}}
  actual_json_str = get_tool_declarations_as_json_str(app_details)
  assert json.loads(actual_json_str) == expected_json


def test_get_tool_declarations_as_json_str_with_agent_no_tools():
  """Tests get_tool_declarations_as_json_str with an agent that has no tools."""
  agent_details = {"agent1": AgentDetails(name="agent1", tool_declarations=[])}
  app_details = AppDetails(agent_details=agent_details)
  expected_json = {"tool_declarations": {"agent1": []}}
  actual_json_str = get_tool_declarations_as_json_str(app_details)
  assert json.loads(actual_json_str) == expected_json


def test_get_tool_declarations_as_json_str_with_agent_with_tools():
  """Tests get_tool_declarations_as_json_str with an agent that has tools."""
  tool1 = genai_types.Tool(
      function_declarations=[
          genai_types.FunctionDeclaration(
              name="test_func", description="A test function."
          )
      ]
  )
  agent_details = {
      "agent1": AgentDetails(name="agent1", tool_declarations=[tool1])
  }
  app_details = AppDetails(agent_details=agent_details)
  expected_json = {
      "tool_declarations": {
          "agent1": [{
              "function_declarations": [{
                  "name": "test_func",
                  "description": "A test function.",
              }]
          }]
      }
  }
  actual_json_str = get_tool_declarations_as_json_str(app_details)
  assert json.loads(actual_json_str) == expected_json


def test_get_tool_declarations_as_json_str_with_multiple_agents():
  """Tests get_tool_declarations_as_json_str with multiple agents."""
  tool1 = genai_types.Tool(
      function_declarations=[
          genai_types.FunctionDeclaration(
              name="test_func1", description="A test function 1."
          )
      ]
  )
  agent_details = {
      "agent1": AgentDetails(name="agent1", tool_declarations=[tool1]),
      "agent2": AgentDetails(name="agent2", tool_declarations=[]),
  }
  app_details = AppDetails(agent_details=agent_details)
  expected_json = {
      "tool_declarations": {
          "agent1": [{
              "function_declarations": [{
                  "name": "test_func1",
                  "description": "A test function 1.",
              }]
          }],
          "agent2": [],
      }
  }
  actual_json_str = get_tool_declarations_as_json_str(app_details)
  assert json.loads(actual_json_str) == expected_json


def test_get_tool_calls_and_responses_as_json_str_with_none():
  """Tests get_tool_calls_and_responses_as_json_str with None."""
  assert (
      get_tool_calls_and_responses_as_json_str(None)
      == "No intermediate steps were taken."
  )


def test_get_tool_calls_and_responses_as_json_str_with_intermediate_data_no_tools():
  """Tests get_tool_calls_and_responses_as_json_str with IntermediateData and no tools."""
  intermediate_data = IntermediateData(tool_uses=[], tool_responses=[])
  assert (
      get_tool_calls_and_responses_as_json_str(intermediate_data)
      == "No intermediate steps were taken."
  )

  intermediate_data = InvocationEvents(invocation_events=[])
  assert (
      get_tool_calls_and_responses_as_json_str(intermediate_data)
      == "No intermediate steps were taken."
  )


def test_get_tool_calls_and_responses_as_json_str_with_invocation_events_multiple_calls():
  """Tests get_tool_calls_and_responses_as_json_str with multiple calls in InvocationEvents."""
  tool_call1 = genai_types.FunctionCall(name="func1", args={}, id="call1")
  tool_call2 = genai_types.FunctionCall(name="func2", args={}, id="call2")
  tool_response1 = genai_types.FunctionResponse(
      name="func1", response={"status": "ok"}, id="call1"
  )
  invocation_event1 = InvocationEvent(
      author="agent",
      content=genai_types.Content(
          parts=[
              genai_types.Part(function_call=tool_call1),
              genai_types.Part(function_call=tool_call2),
          ]
      ),
  )
  invocation_event2 = InvocationEvent(
      author="tool",
      content=genai_types.Content(
          parts=[genai_types.Part(function_response=tool_response1)]
      ),
  )
  intermediate_data = InvocationEvents(
      invocation_events=[invocation_event1, invocation_event2]
  )
  json_str = get_tool_calls_and_responses_as_json_str(intermediate_data)
  expected_json = {
      "tool_calls_and_response": [
          {
              "step": 0,
              "tool_call": {"name": "func1", "args": {}, "id": "call1"},
              "tool_response": {
                  "name": "func1",
                  "response": {"status": "ok"},
                  "id": "call1",
              },
          },
          {
              "step": 1,
              "tool_call": {"name": "func2", "args": {}, "id": "call2"},
              "tool_response": "None",
          },
      ]
  }
  assert json.loads(json_str) == expected_json
