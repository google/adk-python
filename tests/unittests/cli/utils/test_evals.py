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

"""Tests for utilities in eval."""


from google.adk.cli.utils.evals import convert_session_to_eval_format
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types


def build_event(author: str, parts_content: list[dict]) -> Event:
  """Builds an Event object with specified parts."""
  parts = []
  for p_data in parts_content:
    part_args = {}
    if "text" in p_data:
      part_args["text"] = p_data["text"]
    if "func_name" in p_data:
      part_args["function_call"] = types.FunctionCall(
          name=p_data.get("func_name"), args=p_data.get("func_args")
      )
    # Add other part types here if needed for future tests
    parts.append(types.Part(**part_args))
  return Event(author=author, content=types.Content(parts=parts))


def test_convert_empty_session():
  """Test conversion function with empty events list in Session."""
  # Pydantic models require mandatory fields for instantiation
  session_empty_events = Session(
      id="s1", app_name="app", user_id="u1", events=[]
  )
  assert not convert_session_to_eval_format(session_empty_events)


def test_convert_none_session():
  """Test conversion function with None Session."""
  assert not convert_session_to_eval_format(None)


def test_convert_session_skips_initial_non_user_events():
  """Test conversion function with only user events."""
  events = [
      build_event("model", [{"text": "Hello"}]),
      build_event("user", [{"text": "How are you?"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [
      {
          "query": "How are you?",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "",
      },
  ]
  assert convert_session_to_eval_format(session) == expected


def test_convert_single_turn_text_only():
  """Test a single user query followed by a single agent text response."""
  events = [
      build_event("user", [{"text": "What is the time?"}]),
      build_event("root_agent", [{"text": "It is 3 PM."}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "What is the time?",
      "expected_tool_use": [],
      "expected_intermediate_agent_responses": [],
      "reference": "It is 3 PM.",
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_single_turn_tool_only():
  """Test a single user query followed by a single agent tool call."""
  events = [
      build_event("user", [{"text": "Get weather for Seattle"}]),
      build_event(
          "root_agent",
          [{"func_name": "get_weather", "func_args": {"city": "Seattle"}}],
      ),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "Get weather for Seattle",
      "expected_tool_use": [
          {"tool_name": "get_weather", "tool_input": {"city": "Seattle"}}
      ],
      "expected_intermediate_agent_responses": [],
      "reference": "",
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_single_turn_multiple_tools_and_texts():
  """Test a turn with multiple agent responses (tools and text)."""
  events = [
      build_event("user", [{"text": "Do task A then task B"}]),
      build_event(
          "root_agent", [{"text": "Okay, starting task A."}]
      ),  # Intermediate Text 1
      build_event(
          "root_agent", [{"func_name": "task_A", "func_args": {"param": 1}}]
      ),  # Tool 1
      build_event(
          "root_agent", [{"text": "Task A done. Now starting task B."}]
      ),  # Intermediate Text 2
      build_event(
          "another_agent", [{"func_name": "task_B", "func_args": {}}]
      ),  # Tool 2
      build_event(
          "root_agent", [{"text": "All tasks completed."}]
      ),  # Final Text (Reference)
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "Do task A then task B",
      "expected_tool_use": [
          {"tool_name": "task_A", "tool_input": {"param": 1}},
          {"tool_name": "task_B", "tool_input": {}},
      ],
      "expected_intermediate_agent_responses": [
          {"author": "root_agent", "text": "Okay, starting task A."},
          {
              "author": "root_agent",
              "text": "Task A done. Now starting task B.",
          },
      ],
      "reference": "All tasks completed.",
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_multi_turn_session():
  """Test a session with multiple user/agent turns."""
  events = [
      build_event("user", [{"text": "Query 1"}]),
      build_event("agent", [{"text": "Response 1"}]),
      build_event("user", [{"text": "Query 2"}]),
      build_event("agent", [{"func_name": "tool_X", "func_args": {}}]),
      build_event("agent", [{"text": "Response 2"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [
      {  # Turn 1
          "query": "Query 1",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 1",
      },
      {  # Turn 2
          "query": "Query 2",
          "expected_tool_use": [{"tool_name": "tool_X", "tool_input": {}}],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 2",
      },
  ]
  assert convert_session_to_eval_format(session) == expected


def test_convert_agent_event_multiple_parts():
  """Test an agent event with both text and tool call parts."""
  events = [
      build_event("user", [{"text": "Do something complex"}]),
      # Build event with multiple dicts in parts_content list
      build_event(
          "agent",
          [
              {"text": "Okay, doing it."},
              {"func_name": "complex_tool", "func_args": {"value": True}},
          ],
      ),
      build_event("agent", [{"text": "Finished."}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "Do something complex",
      "expected_tool_use": [
          {"tool_name": "complex_tool", "tool_input": {"value": True}}
      ],
      "expected_intermediate_agent_responses": [{
          "author": "agent",
          "text": "Okay, doing it.",
      }],  # Text from first part of agent event
      "reference": "Finished.",  # Text from second agent event
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_handles_missing_content_or_parts():
  """Test that events missing content or parts are skipped gracefully."""
  events = [
      build_event("user", [{"text": "Query 1"}]),
      Event(author="agent", content=None),  # Agent event missing content
      build_event("agent", [{"text": "Response 1"}]),
      Event(author="user", content=None),  # User event missing content
      build_event("user", [{"text": "Query 2"}]),
      Event(
          author="agent", content=types.Content(parts=[])
      ),  # Agent event with empty parts list
      build_event("agent", [{"text": "Response 2"}]),
      # User event with content but no parts (or None parts)
      Event(author="user", content=types.Content(parts=None)),
      build_event("user", [{"text": "Query 3"}]),
      build_event("agent", [{"text": "Response 3"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [
      {  # Turn 1 (from Query 1)
          "query": "Query 1",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 1",
      },
      {  # Turn 2 (from Query 2 - user event with None content was skipped)
          "query": "Query 2",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 2",
      },
      {  # Turn 3 (from Query 3 - user event with None parts was skipped)
          "query": "Query 3",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 3",
      },
  ]
  assert convert_session_to_eval_format(session) == expected


def test_convert_handles_missing_tool_name_or_args():
  """Test tool calls with missing name or args."""
  events = [
      build_event("user", [{"text": "Call tools"}]),
      # Event where FunctionCall has name=None
      Event(
          author="agent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(name=None, args={"a": 1})
                  )
              ]
          ),
      ),
      # Event where FunctionCall has args=None
      Event(
          author="agent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(name="tool_B", args=None)
                  )
              ]
          ),
      ),
      # Event where FunctionCall part exists but FunctionCall object is None
      # (should skip)
      Event(
          author="agent",
          content=types.Content(
              parts=[types.Part(function_call=None, text="some text")]
          ),
      ),
      build_event("agent", [{"text": "Done"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "Call tools",
      "expected_tool_use": [
          {"tool_name": "", "tool_input": {"a": 1}},  # Defaults name to ""
          {"tool_name": "tool_B", "tool_input": {}},  # Defaults args to {}
      ],
      "expected_intermediate_agent_responses": [{
          "author": "agent",
          "text": "some text",
      }],  # Text part from the event where function_call was None
      "reference": "Done",
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_handles_missing_user_query_text():
  """Test user event where the first part has no text."""
  events = [
      # Event where user part has text=None
      Event(
          author="user", content=types.Content(parts=[types.Part(text=None)])
      ),
      build_event("agent", [{"text": "Response 1"}]),
      # Event where user part has text=""
      build_event("user", [{"text": ""}]),
      build_event("agent", [{"text": "Response 2"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [
      {
          "query": "",  # Defaults to "" if text is None
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 1",
      },
      {
          "query": "",  # Defaults to "" if text is ""
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "Response 2",
      },
  ]
  assert convert_session_to_eval_format(session) == expected


def test_convert_handles_empty_agent_text():
  """Test agent responses with empty string text."""
  events = [
      build_event("user", [{"text": "Query"}]),
      build_event("agent", [{"text": "Okay"}]),
      build_event("agent", [{"text": ""}]),  # Empty text
      build_event("agent", [{"text": "Done"}]),
  ]
  session = Session(id="s1", app_name="app", user_id="u1", events=events)
  expected = [{
      "query": "Query",
      "expected_tool_use": [],
      "expected_intermediate_agent_responses": [
          {"author": "agent", "text": "Okay"},
      ],
      "reference": "Done",
  }]
  assert convert_session_to_eval_format(session) == expected


def test_convert_complex_sample_session():
  """Test using the complex sample session provided earlier."""
  events = [
      build_event("user", [{"text": "What can you do?"}]),
      build_event(
          "root_agent",
          [{"text": "I can roll dice and check if numbers are prime. \n"}],
      ),
      build_event(
          "user",
          [{
              "text": (
                  "Roll a 8 sided dice and then check if 90 is a prime number"
                  " or not."
              )
          }],
      ),
      build_event(
          "root_agent",
          [{
              "func_name": "transfer_to_agent",
              "func_args": {"agent_name": "roll_agent"},
          }],
      ),
      # Skipping FunctionResponse events as they don't have text/functionCall
      # parts used by converter
      build_event(
          "roll_agent", [{"func_name": "roll_die", "func_args": {"sides": 8}}]
      ),
      # Skipping FunctionResponse
      build_event(
          "roll_agent",
          [
              {"text": "I rolled a 2. Now, I'll check if 90 is prime. \n\n"},
              {
                  "func_name": "transfer_to_agent",
                  "func_args": {"agent_name": "prime_agent"},
              },
          ],
      ),
      # Skipping FunctionResponse
      build_event(
          "prime_agent",
          [{"func_name": "check_prime", "func_args": {"nums": [90]}}],
      ),
      # Skipping FunctionResponse
      build_event("prime_agent", [{"text": "90 is not a prime number. \n"}]),
  ]
  session = Session(
      id="some_id",
      app_name="hello_world_ma",
      user_id="user",
      events=events,
  )
  expected = [
      {
          "query": "What can you do?",
          "expected_tool_use": [],
          "expected_intermediate_agent_responses": [],
          "reference": "I can roll dice and check if numbers are prime. \n",
      },
      {
          "query": (
              "Roll a 8 sided dice and then check if 90 is a prime number or"
              " not."
          ),
          "expected_tool_use": [
              {
                  "tool_name": "transfer_to_agent",
                  "tool_input": {"agent_name": "roll_agent"},
              },
              {"tool_name": "roll_die", "tool_input": {"sides": 8}},
              {
                  "tool_name": "transfer_to_agent",
                  "tool_input": {"agent_name": "prime_agent"},
              },  # From combined event
              {"tool_name": "check_prime", "tool_input": {"nums": [90]}},
          ],
          "expected_intermediate_agent_responses": [{
              "author": "roll_agent",
              "text": "I rolled a 2. Now, I'll check if 90 is prime. \n\n",
          }],  # Text from combined event
          "reference": "90 is not a prime number. \n",
      },
  ]

  actual = convert_session_to_eval_format(session)
  assert actual == expected
