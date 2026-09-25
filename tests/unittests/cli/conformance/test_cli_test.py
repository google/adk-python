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

"""Tests for cli_test.py."""

from typing import Optional
from unittest.mock import MagicMock

import click
from click.testing import CliRunner
from google.adk.agents.run_config import StreamingMode
from google.adk.cli.conformance.cli_test import _ConformanceTestSummary
from google.adk.cli.conformance.cli_test import _print_test_summary
from google.adk.cli.conformance.cli_test import _TestResult
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


def _summary(
    streaming_mode: StreamingMode, passed: int, failed: int
) -> _ConformanceTestSummary:
  results = [
      _TestResult(category="cat", name=f"passing_{i}", success=True)
      for i in range(passed)
  ] + [
      _TestResult(
          category="cat",
          name=f"failing_{i}",
          success=False,
          error_message="event mismatch",
      )
      for i in range(failed)
  ]
  return _ConformanceTestSummary(
      total_tests=len(results),
      passed_tests=passed,
      failed_tests=failed,
      results=results,
      streaming_mode=streaming_mode,
  )


def _run(
    summaries: list[_ConformanceTestSummary],
    selected_streaming_mode: Optional[StreamingMode] = None,
):
  @click.command()
  def _command():
    _print_test_summary(summaries, selected_streaming_mode)

  return CliRunner().invoke(_command)


def test_summary_reports_every_streaming_mode_when_first_mode_has_no_tests():
  result = _run([
      _summary(StreamingMode.NONE, passed=0, failed=0),
      _summary(StreamingMode.SSE, passed=1, failed=2),
  ])

  assert "STREAMING MODE: StreamingMode.SSE" in result.output
  assert "Total tests: 3" in result.output
  assert result.exit_code != 0
  assert (
      "2 test(s) failed for streaming mode StreamingMode.SSE" in result.output
  )


def test_summary_reports_every_streaming_mode_when_first_mode_fails():
  result = _run([
      _summary(StreamingMode.NONE, passed=0, failed=1),
      _summary(StreamingMode.SSE, passed=2, failed=0),
  ])

  assert "STREAMING MODE: StreamingMode.NONE" in result.output
  assert "STREAMING MODE: StreamingMode.SSE" in result.output
  assert result.exit_code != 0
  assert "1 test(s) failed for streaming mode StreamingMode.NONE" in (
      result.output
  )


def test_summary_fails_when_no_test_cases_were_discovered():
  result = _run([_summary(StreamingMode.NONE, passed=0, failed=0)])

  assert result.exit_code != 0
  assert "No test cases were found for streaming mode StreamingMode.NONE" in (
      result.output
  )


def test_summary_fails_when_there_is_nothing_to_summarize():
  result = _run([])

  assert result.exit_code != 0
  assert "No conformance tests were run" in result.output


def test_summary_ignores_an_unrecorded_streaming_mode():
  result = _run([
      _summary(StreamingMode.NONE, passed=2, failed=0),
      _summary(StreamingMode.SSE, passed=0, failed=0),
  ])

  assert result.exit_code == 0
  assert "No tests were run." in result.output
  assert "No test cases were found" not in result.output


def test_summary_fails_when_the_requested_streaming_mode_has_no_tests():
  result = _run(
      [_summary(StreamingMode.SSE, passed=0, failed=0)],
      selected_streaming_mode=StreamingMode.SSE,
  )

  assert result.exit_code != 0
  assert "No test cases were found for streaming mode StreamingMode.SSE" in (
      result.output
  )


def test_summary_succeeds_when_every_streaming_mode_passes():
  result = _run([
      _summary(StreamingMode.NONE, passed=2, failed=0),
      _summary(StreamingMode.SSE, passed=3, failed=0),
  ])

  assert result.exit_code == 0
  assert result.output.count("All tests passed!") == 2
