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

import asyncio
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools import ProgressiveTool
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_iter_progressive_streams_then_final():
  async def gen(country: str):
    yield {"p": 1}
    yield {"p": 2}

  tool = ProgressiveTool(gen)

  agent = LlmAgent(name="agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(agent)

  llm_request = LlmRequest()
  llm_request.tools_dict[tool.name] = tool

  # Build function call event with stable id to allow ProgressiveTool caching
  fc_part = types.Part.from_function_call(
      name=tool.name, args={"country": "fr"}
  )
  fc_part.function_call.id = "fc-1"
  function_call_event = Event(
      author=agent.name,
      content=types.Content(role="model", parts=[fc_part]),
  )

  events = []
  async for e in functions.iter_progressive_function_calls_async(
      invocation_context, function_call_event, llm_request.tools_dict
  ):
    events.append(e)

  assert events, "Expected events from progressive iterator"
  assert any(e.partial for e in events), "Should have partial progress events"
  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert finals, "Expected final function_response event"


@pytest.mark.asyncio
async def test_iter_progressive_error_handled_by_plugin():
  class ToolErrorPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="tool_error")

    async def on_tool_error_callback(
        self, *, tool, tool_args, tool_context, error
    ):
      return {"handled": True, "error": str(error)}

  async def faulty():
    yield {"start": True}
    raise RuntimeError("boom")

  tool = ProgressiveTool(faulty)

  agent = LlmAgent(name="agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(
      agent, plugins=[ToolErrorPlugin()]
  )

  llm_request = LlmRequest()
  llm_request.tools_dict[tool.name] = tool

  fc_part = types.Part.from_function_call(name=tool.name, args={})
  fc_part.function_call.id = "fc-2"
  function_call_event = Event(
      author=agent.name,
      content=types.Content(role="model", parts=[fc_part]),
  )

  events = []
  async for e in functions.iter_progressive_function_calls_async(
      invocation_context, function_call_event, llm_request.tools_dict
  ):
    events.append(e)

  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert (
      finals
      and finals[-1].content.parts[0].function_response.response["handled"]
      is True
  )


def test_merge_parallel_function_response_events_merges_parts():
  # Create two simple function_response Events and merge them
  def make_event(name: str, payload: dict):
    fr_part = types.Part.from_function_response(name=name, response=payload)
    return Event(
        author="agent", content=types.Content(role="user", parts=[fr_part])
    )

  e1 = make_event("t1", {"a": 1})
  e2 = make_event("t2", {"b": 2})

  merged = functions.merge_parallel_function_response_events([e1, e2])

  assert (
      merged.content and merged.content.parts and len(merged.content.parts) == 2
  )
  names = [p.function_response.name for p in merged.content.parts]
  assert set(names) == {"t1", "t2"}
