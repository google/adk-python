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

from __future__ import annotations

import json

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.llm_as_judge_utils import get_average_rubric_score
from google.adk.evaluation.llm_as_judge_utils import get_eval_status
from google.adk.evaluation.llm_as_judge_utils import get_final_response_author
from google.adk.evaluation.llm_as_judge_utils import get_grounding_metadata_as_json_str
from google.adk.evaluation.llm_as_judge_utils import get_text_from_content
from google.adk.evaluation.llm_as_judge_utils import get_tool_calls_and_responses_as_json_str
from google.adk.evaluation.llm_as_judge_utils import get_tool_declarations_as_json_str
from google.adk.events.event import Event
from google.genai import types as genai_types
import pytest


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


def test_get_text_from_content_with_invocation_include_intermediate_responses_in_final():
  """Tests get_text_from_content on an Invocation with and without the flag."""
  intermediate_text = "Let me check."
  final_response_text = "Done."
  invocation = Invocation(
      user_content=genai_types.Content(parts=[genai_types.Part(text="user")]),
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="agent",
                  content=genai_types.Content(
                      parts=[genai_types.Part(text=intermediate_text)]
                  ),
              ),
              InvocationEvent(
                  author="tool",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(
                              function_call=genai_types.FunctionCall(name="t")
                          )
                      ]
                  ),
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=final_response_text)]
      ),
  )

  # Flag off (default): only the final response text is returned.
  assert get_text_from_content(invocation) == final_response_text

  # Flag on: intermediate text is concatenated before the final response.
  assert (
      get_text_from_content(
          invocation, include_intermediate_responses_in_final=True
      )
      == f"{intermediate_text}\n{final_response_text}"
  )


def test_get_text_from_content_with_intermediate_data_full_response():
  invocation = Invocation(
      user_content=genai_types.Content(parts=[genai_types.Part(text="user")]),
      intermediate_data=IntermediateData(
          intermediate_responses=[
              ("agent", [genai_types.Part(text="legacy intro")]),
              (
                  "tool",
                  [
                      genai_types.Part(
                          function_call=genai_types.FunctionCall(name="lookup")
                      )
                  ],
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="final answer")]
      ),
  )

  assert get_text_from_content(invocation) == "final answer"
  assert (
      get_text_from_content(
          invocation, include_intermediate_responses_in_final=True
      )
      == "legacy intro\nfinal answer"
  )


@pytest.mark.parametrize(
    "parts, expected_text",
    [
        pytest.param(
            [
                genai_types.Part(text="The user said hi.", thought=True),
                genai_types.Part(text="Hello"),
            ],
            "Hello",
            id="thought-and-answer",
        ),
        pytest.param(
            [genai_types.Part(text="The user said hi.", thought=True)],
            "",
            id="thought-only",
        ),
    ],
)
def test_get_text_from_content_skips_thought_parts(
    parts: list[genai_types.Part], expected_text: str
):
  """Thought parts are left out of the extracted text."""
  content = genai_types.Content(parts=parts)

  assert get_text_from_content(content) == expected_text


def test_get_text_from_content_skips_thought_parts_in_intermediate_responses():
  """Thought parts of intermediate events are left out of the response text."""
  invocation = Invocation(
      user_content=genai_types.Content(parts=[genai_types.Part(text="user")]),
      intermediate_data=InvocationEvents(
          invocation_events=[
              InvocationEvent(
                  author="agent",
                  content=genai_types.Content(
                      parts=[
                          genai_types.Part(
                              text="I need to look this up.", thought=True
                          ),
                          genai_types.Part(text="Let me check."),
                      ]
                  ),
              ),
          ]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Done.")]
      ),
  )

  text = get_text_from_content(
      invocation, include_intermediate_responses_in_final=True
  )

  assert text == "Let me check.\nDone."


def _make_invocation_with_reply(
    intermediate_data: IntermediateData | InvocationEvents | None,
) -> Invocation:
  """Returns an invocation that has a final response."""
  return Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="How much do I owe?")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="You owe $42.")]
      ),
      intermediate_data=intermediate_data,
  )


def _make_tool_call_event(author: str) -> InvocationEvent:
  """Returns an event in which `author` calls a tool."""
  return InvocationEvent(
      author=author,
      content=genai_types.Content(
          parts=[
              genai_types.Part(
                  function_call=genai_types.FunctionCall(name="a_tool", args={})
              )
          ]
      ),
  )


def test_get_final_response_author_after_transfer():
  """The reply belongs to the sub-agent, not the agent that handed off."""
  invocation = _make_invocation_with_reply(
      InvocationEvents(
          invocation_events=[
              _make_tool_call_event("router"),
              # `convert_events_to_eval_invocations` keeps the final
              # response's event, with its content moved to `final_response`.
              InvocationEvent(author="billing_agent"),
          ]
      )
  )

  assert get_final_response_author(invocation) == "billing_agent"


@pytest.mark.parametrize(
    "later_event",
    [
        InvocationEvent(
            author="billing_agent",
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                total_token_count=12
            ),
        ),
        _make_tool_call_event("audit_agent"),
    ],
    ids=["usage_report_of_the_same_agent", "tool_call_of_another_agent"],
)
def test_get_final_response_author_with_later_event(later_event):
  """A later event of these two kinds leaves the reply with its agent.

  The usage report is itself an event without content, so it is the one the
  helper picks. The label stays the same only because the same agent wrote it.
  """
  invocation = _make_invocation_with_reply(
      InvocationEvents(
          invocation_events=[
              _make_tool_call_event("router"),
              InvocationEvent(author="billing_agent"),
              later_event,
          ]
      )
  )

  assert get_final_response_author(invocation) == "billing_agent"


_USAGE = genai_types.GenerateContentResponseUsageMetadata(total_token_count=12)


def _make_session_event(
    author: str, *parts: genai_types.Part, **kwargs
) -> Event:
  """Returns a session event of `author` in one invocation."""
  return Event(
      invocation_id="inv1",
      author=author,
      content=genai_types.Content(role="model", parts=list(parts)),
      **kwargs,
  )


def _make_transfer_events() -> list[Event]:
  """Returns the events of a router handing the turn to billing_agent."""
  return [
      _make_session_event(
          "router",
          genai_types.Part(
              function_call=genai_types.FunctionCall(
                  name="transfer_to_agent", args={"agent_name": "billing_agent"}
              )
          ),
      ),
      _make_session_event(
          "router",
          genai_types.Part(
              function_response=genai_types.FunctionResponse(
                  name="transfer_to_agent", response={"result": None}
              )
          ),
      ),
  ]


def _make_reply_event(author: str) -> Event:
  """Returns the session event in which `author` answers the user."""
  return _make_session_event(author, genai_types.Part(text="You owe $42."))


def _make_reply_event_with_long_running_call(author: str) -> Event:
  """Returns a reply of `author` that also starts a long-running tool.

  The conversion keeps the content of a reply that carries a function call.
  """
  return _make_session_event(
      author,
      genai_types.Part(text="Checking, you owe $42."),
      genai_types.Part(
          function_call=genai_types.FunctionCall(
              id="call1", name="lookup", args={}
          )
      ),
      long_running_tool_ids={"call1"},
  )


def _make_failed_call_event(author: str) -> Event:
  """Returns the session event of a model call of `author` that failed."""
  return Event(
      invocation_id="inv1",
      author=author,
      error_code="MALFORMED_FUNCTION_CALL",
      usage_metadata=_USAGE,
  )


@pytest.mark.parametrize(
    "agent_events, expected_author",
    [
        ([_make_reply_event("root_agent")], "root_agent"),
        (
            [*_make_transfer_events(), _make_reply_event("billing_agent")],
            "billing_agent",
        ),
        (
            [
                *_make_transfer_events(),
                _make_reply_event("billing_agent"),
                _make_session_event(
                    "audit_agent",
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name="log_reply", args={}
                        )
                    ),
                ),
                _make_session_event(
                    "audit_agent",
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="log_reply", response={"result": None}
                        )
                    ),
                ),
            ],
            "billing_agent",
        ),
        (
            [
                _make_reply_event("drafter"),
                # The model of the agent that runs next returns no parts.
                _make_session_event("critic", usage_metadata=_USAGE),
            ],
            "drafter",
        ),
        (
            [
                _make_session_event("fetcher", usage_metadata=_USAGE),
                _make_reply_event("writer"),
            ],
            "writer",
        ),
        (
            [_make_failed_call_event("fetcher"), _make_reply_event("writer")],
            "writer",
        ),
        (
            # Two agents answer with the same text; the later reply is final.
            [_make_reply_event("drafter"), _make_reply_event("reviewer")],
            "reviewer",
        ),
        (
            [
                _make_reply_event("drafter"),
                # The next agent streams the same text and never completes.
                _make_session_event(
                    "critic",
                    genai_types.Part(text="You owe $42."),
                    partial=True,
                ),
            ],
            "drafter",
        ),
        (
            [
                *_make_transfer_events(),
                _make_reply_event_with_long_running_call("billing_agent"),
            ],
            "billing_agent",
        ),
        (
            [
                _make_failed_call_event("fetcher"),
                _make_reply_event_with_long_running_call("writer"),
            ],
            "writer",
        ),
        (
            [
                _make_reply_event_with_long_running_call("alpha"),
                # A parallel agent's model returns no parts after the reply.
                _make_session_event("beta", usage_metadata=_USAGE),
            ],
            "alpha",
        ),
        (
            [
                _make_reply_event_with_long_running_call("alpha"),
                # A parallel agent calls a tool after the reply.
                _make_session_event(
                    "beta",
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name="log_reply", args={}
                        )
                    ),
                ),
                _make_session_event(
                    "beta",
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="log_reply", response={"result": None}
                        )
                    ),
                ),
            ],
            "alpha",
        ),
    ],
    ids=[
        "single_agent",
        "transfer",
        "tool_call_of_a_later_agent",
        "empty_reply_of_a_later_agent",
        "empty_reply_of_an_earlier_agent",
        "failed_call_of_an_earlier_agent",
        "same_text_from_two_agents",
        "same_text_in_a_partial_event_of_a_later_agent",
        "reply_with_a_long_running_call",
        "failed_call_before_a_reply_with_a_long_running_call",
        "empty_reply_of_a_parallel_agent_after_a_long_running_call",
        "tool_call_of_a_parallel_agent_after_a_long_running_call",
    ],
)
def test_get_final_response_author_of_converted_session_events(
    agent_events, expected_author
):
  """The reply's author is found in what the conversion really produces.

  It is found again after the invocation has been written to JSON and read
  back, as a stored eval case is.
  """
  user_event = Event(
      invocation_id="inv1",
      author="user",
      content=genai_types.Content(
          parts=[genai_types.Part(text="How much do I owe?")]
      ),
  )
  (invocation,) = EvaluationGenerator.convert_events_to_eval_invocations(
      [user_event, *agent_events]
  )

  stored_invocation = Invocation.model_validate_json(
      invocation.model_dump_json()
  )

  assert invocation.final_response is not None
  assert get_final_response_author(invocation) == expected_author
  assert get_final_response_author(stored_invocation) == expected_author


def test_get_final_response_author_without_event_for_the_reply():
  """A hand-built invocation falls back to its last agent event."""
  invocation = _make_invocation_with_reply(
      InvocationEvents(
          invocation_events=[
              _make_tool_call_event("router"),
              _make_tool_call_event("billing_agent"),
              InvocationEvent(
                  author="user",
                  content=genai_types.Content(
                      parts=[genai_types.Part(text="Thanks.")]
                  ),
              ),
          ]
      )
  )

  assert get_final_response_author(invocation) == "billing_agent"


@pytest.mark.parametrize(
    "intermediate_data",
    [None, IntermediateData(), InvocationEvents(invocation_events=[])],
    ids=["no_intermediate_data", "legacy_intermediate_data", "no_events"],
)
def test_get_final_response_author_without_agent_events(intermediate_data):
  """There is no author to report when the invocation has no agent events."""
  invocation = _make_invocation_with_reply(intermediate_data)

  assert get_final_response_author(invocation) is None


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


def test_get_grounding_metadata_as_json_str_with_invocation_events():
  """Tests grounding metadata is serialized for LLM-as-judge prompts."""
  grounding_metadata = genai_types.GroundingMetadata(
      web_search_queries=["recent AI news"]
  )
  intermediate_data = InvocationEvents(
      invocation_events=[
          InvocationEvent(
              author="agent",
              content=None,
              grounding_metadata=grounding_metadata,
          )
      ]
  )

  json_str = get_grounding_metadata_as_json_str(intermediate_data)
  parsed = json.loads(json_str)

  assert parsed["grounding_metadata"][0]["step"] == 0
  assert parsed["grounding_metadata"][0]["author"] == "agent"
  assert parsed["grounding_metadata"][0]["grounding_metadata"][
      "web_search_queries"
  ] == ["recent AI news"]


def test_get_grounding_metadata_as_json_str_without_metadata():
  """Tests empty grounding metadata serialization."""
  assert (
      get_grounding_metadata_as_json_str(InvocationEvents())
      == "No grounding metadata was provided."
  )
