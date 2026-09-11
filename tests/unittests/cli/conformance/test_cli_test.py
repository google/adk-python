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

"""Tests for ConformanceTestRunner in cli_test.py."""

from unittest.mock import MagicMock

from google.adk.agents.run_config import StreamingMode
from google.adk.cli.conformance.cli_test import ConformanceTestRunner
from google.adk.cli.conformance.test_case import TestCase
from google.adk.cli.conformance.test_case import TestSpec
from google.adk.cli.conformance.test_case import UserMessage
from google.adk.events.event import Event
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_run_user_messages_sse_does_not_duplicate_function_call_ids():
  client = MagicMock()
  fc1 = types.Part(
      function_call=types.FunctionCall(name="long_tool", id="id-1")
  )
  event1_partial = Event(partial=True, content=types.Content(parts=[fc1]))
  event1_final = Event(partial=False, content=types.Content(parts=[fc1]))

  fc2 = types.Part(
      function_call=types.FunctionCall(name="long_tool", id="id-2")
  )
  event2_partial = Event(partial=True, content=types.Content(parts=[fc2]))
  event2_final = Event(partial=False, content=types.Content(parts=[fc2]))

  captured_requests = []

  async def fake_run_agent(req, **kwargs):
    captured_requests.append(req)
    if (
        req.new_message.parts
        and getattr(req.new_message.parts[0], "text", None) == "turn0"
    ):
      yield event1_partial
      yield event1_final
    else:
      yield event2_partial
      yield event2_final

  client.run_agent = fake_run_agent

  runner = ConformanceTestRunner([], client, streaming_mode=StreamingMode.SSE)
  test_case = TestCase(
      category="cat",
      name="tc",
      dir=None,
      test_spec=TestSpec(
          description="test sse function call id mapping",
          agent="agent",
          user_messages=[
              UserMessage(text="turn0"),
              UserMessage(
                  content=types.UserContent(
                      parts=[
                          types.Part(
                              function_response=types.FunctionResponse(
                                  name="long_tool"
                              )
                          )
                      ]
                  )
              ),
              UserMessage(
                  content=types.UserContent(
                      parts=[
                          types.Part(
                              function_response=types.FunctionResponse(
                                  name="long_tool"
                              )
                          )
                      ]
                  )
              ),
          ],
      ),
  )
  await runner._run_user_messages("sess1", test_case)

  assert len(captured_requests) == 3
  assert (
      captured_requests[1].new_message.parts[0].function_response.id == "id-1"
  )
  assert (
      captured_requests[2].new_message.parts[0].function_response.id == "id-2"
  )


@pytest.mark.asyncio
async def test_run_user_messages_sse_partial_event_without_id_does_not_mask_final_id():
  client = MagicMock()
  fc1_partial = types.Part(
      function_call=types.FunctionCall(name="long_tool", id=None)
  )
  event1_partial = Event(
      partial=True, content=types.Content(parts=[fc1_partial])
  )
  fc1_final = types.Part(
      function_call=types.FunctionCall(name="long_tool", id="id-1")
  )
  event1_final = Event(partial=False, content=types.Content(parts=[fc1_final]))

  captured_requests = []

  async def fake_run_agent(req, **kwargs):
    captured_requests.append(req)
    yield event1_partial
    yield event1_final

  client.run_agent = fake_run_agent

  runner = ConformanceTestRunner([], client, streaming_mode=StreamingMode.SSE)
  test_case = TestCase(
      category="cat",
      name="tc",
      dir=None,
      test_spec=TestSpec(
          description=(
              "test sse partial event without id does not mask final id"
          ),
          agent="agent",
          user_messages=[
              UserMessage(text="turn0"),
              UserMessage(
                  content=types.UserContent(
                      parts=[
                          types.Part(
                              function_response=types.FunctionResponse(
                                  name="long_tool"
                              )
                          )
                      ]
                  )
              ),
          ],
      ),
  )
  await runner._run_user_messages("sess1", test_case)

  assert len(captured_requests) == 2
  assert (
      captured_requests[1].new_message.parts[0].function_response.id == "id-1"
  )


@pytest.mark.asyncio
async def test_run_user_messages_function_call_without_id_matches_response():
  client = MagicMock()
  fc = types.Part(function_call=types.FunctionCall(name="long_tool", id=None))
  event = Event(partial=False, content=types.Content(parts=[fc]))
  captured_requests = []

  async def fake_run_agent(req, **kwargs):
    captured_requests.append(req)
    yield event

  client.run_agent = fake_run_agent

  runner = ConformanceTestRunner([], client)
  test_case = TestCase(
      category="cat",
      name="tc",
      dir=None,
      test_spec=TestSpec(
          description="test function call without id",
          agent="agent",
          user_messages=[
              UserMessage(text="turn0"),
              UserMessage(
                  content=types.UserContent(
                      parts=[
                          types.Part.from_text(text="prior text"),
                          types.Part(
                              function_response=types.FunctionResponse(
                                  name="long_tool",
                                  id="initial-placeholder-id",
                              )
                          ),
                      ]
                  )
              ),
          ],
      ),
  )
  await runner._run_user_messages("sess1", test_case)

  assert len(captured_requests) == 2
  assert captured_requests[1].new_message.parts[1].function_response.id is None


@pytest.mark.asyncio
async def test_run_user_messages_sse_ignores_partial_event_transient_id():
  client = MagicMock()
  fc1_partial = types.Part(
      function_call=types.FunctionCall(name="long_tool", id="transient-id")
  )
  event1_partial = Event(
      partial=True, content=types.Content(parts=[fc1_partial])
  )
  fc1_final = types.Part(
      function_call=types.FunctionCall(name="long_tool", id="final-id")
  )
  event1_final = Event(partial=False, content=types.Content(parts=[fc1_final]))

  captured_requests = []

  async def fake_run_agent(req, **kwargs):
    captured_requests.append(req)
    yield event1_partial
    yield event1_final

  client.run_agent = fake_run_agent

  runner = ConformanceTestRunner([], client, streaming_mode=StreamingMode.SSE)
  test_case = TestCase(
      category="cat",
      name="tc",
      dir=None,
      test_spec=TestSpec(
          description="test sse ignores partial event transient id",
          agent="agent",
          user_messages=[
              UserMessage(text="turn0"),
              UserMessage(
                  content=types.UserContent(
                      parts=[
                          types.Part(
                              function_response=types.FunctionResponse(
                                  name="long_tool"
                              )
                          )
                      ]
                  )
              ),
          ],
      ),
  )
  await runner._run_user_messages("sess1", test_case)

  assert len(captured_requests) == 2
  assert (
      captured_requests[1].new_message.parts[0].function_response.id
      == "final-id"
  )
