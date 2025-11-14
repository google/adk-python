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

"""Tests for function_response sent in separate request (async tools)."""

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_function_response_in_separate_request():
  """Test function_response sent separately finds function_call in history."""
  agent = Agent(model="gemini-2.5-flash", name="test_agent")
  llm_request = LlmRequest(model="gemini-2.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  function_call = types.FunctionCall(
      id="async_call_123", name="async_tool", args={"job": "start"}
  )
  function_response = types.FunctionResponse(
      id="async_call_123",
      name="async_tool",
      response={"status": "completed"},
  )

  # Simulate async job: function_call in early session history,
  # function_response arrives much later in separate request
  events = [
      Event(
          invocation_id="inv1",
          author="user",
          content=types.UserContent("Start async job"),
      ),
      Event(
          invocation_id="inv2",
          author="test_agent",
          content=types.ModelContent([types.Part(function_call=function_call)]),
      ),
      Event(
          invocation_id="inv3",
          author="test_agent",
          content=types.ModelContent("Job started, waiting for completion..."),
      ),
      # Much later: function_response arrives (separate SSE request)
      Event(
          invocation_id="inv4",
          author="user",
          content=types.UserContent(
              [types.Part(function_response=function_response)]
          ),
      ),
  ]
  invocation_context.session.events = events

  # Should not raise ValueError
  async for _ in contents.request_processor.run_async(
      invocation_context, llm_request
  ):
    pass

  # Verify function_response is processed
  assert any(
      hasattr(part, "function_response")
      and part.function_response
      and part.function_response.id == "async_call_123"
      for content in llm_request.contents
      for part in content.parts
  )
