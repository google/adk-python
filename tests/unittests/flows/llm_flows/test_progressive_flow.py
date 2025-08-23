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
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import FunctionTool
from google.adk.tools import ProgressiveTool
from google.adk.tools.progressive_function_tool import ProgressiveFunctionTool
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_base_flow_progressive_tool_streams_and_final():
  # Progressive async generator tool
  async def export_report(country: str):
    yield {"s": "started", "c": country}
    await asyncio.sleep(0)
    yield {"s": "progress", "p": 50}
    yield {"s": "completed", "url": f"https://example.com/{country}.pdf"}

  tool = ProgressiveTool(export_report)

  agent = LlmAgent(name="test_agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(agent)
  flow = BaseLlmFlow()

  llm_request = LlmRequest()
  llm_request.tools_dict[tool.name] = tool

  function_call_event = Event(
      author=agent.name,
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name=tool.name, args={"country": "france"}
              )
          ],
      ),
  )

  events = []
  async for e in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(e)

  assert events, "Expected events from progressive tool"
  partials = [
      e
      for e in events
      if e.partial
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert (
      partials and finals
  ), "Expected partial and final function_response events"
  # Verify final carries completed payload
  assert (
      finals[-1].content.parts[0].function_response.response.get("s")
      == "completed"
  )


@pytest.mark.asyncio
async def test_base_flow_with_progressive_function_tool_subclass():
  class MyProgTool(ProgressiveFunctionTool):

    def __init__(self):
      super().__init__(func=lambda: None)
      self.name = "my_prog"
      self.description = ""

    async def progress_stream(self, *, args: dict[str, Any], tool_context):
      yield {"tick": 1}
      yield {"tick": 2}

    async def run_async(self, *, args: dict[str, Any], tool_context):
      return {"final": True}

  tool = MyProgTool()

  agent = LlmAgent(name="test_agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(agent)
  flow = BaseLlmFlow()

  llm_request = LlmRequest()
  llm_request.tools_dict[tool.name] = tool

  function_call_event = Event(
      author=agent.name,
      content=types.Content(
          role="model",
          parts=[types.Part.from_function_call(name=tool.name, args={})],
      ),
  )

  events = []
  async for e in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(e)

  assert any(
      e.partial for e in events
  ), "Expected partial events from subclass tool"
  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert finals and finals[-1].content.parts[0].function_response.response == {
      "final": True
  }


@pytest.mark.asyncio
async def test_base_flow_non_progressive_only_path():
  # A normal FunctionTool should go through default handler path
  def add(x: int, y: int) -> dict[str, int]:
    return {"sum": x + y}

  agent = LlmAgent(name="test_agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(agent)
  flow = BaseLlmFlow()

  tool = FunctionTool(add)
  llm_request = LlmRequest()
  llm_request.tools_dict[tool.name] = tool

  function_call_event = Event(
      author=agent.name,
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name=tool.name, args={"x": 1, "y": 2}
              )
          ],
      ),
  )

  events = []
  async for e in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(e)

  # Only one final function_response expected, and no partials
  assert len(events) == 1
  assert not events[0].partial
  fr = events[0].content.parts[0].function_response
  assert fr.name == tool.name and fr.response == {"sum": 3}


@pytest.mark.asyncio
async def test_base_flow_progressive_present_but_not_called_uses_fallback():
  # When ProgressiveTool exists on agent but model calls a normal tool, fallback is used
  async def prog():
    yield {"tick": 1}
    yield {"tick": 2}

  def mul(a: int, b: int) -> dict[str, int]:
    return {"prod": a * b}

  agent = LlmAgent(name="test_agent", model="gemini-1.5-flash")
  invocation_context = await testing_utils.create_invocation_context(agent)
  flow = BaseLlmFlow()

  prog_tool = ProgressiveTool(prog)
  mul_tool = FunctionTool(mul)
  llm_request = LlmRequest()
  llm_request.tools_dict[prog_tool.name] = prog_tool
  llm_request.tools_dict[mul_tool.name] = mul_tool

  # Model only calls non-progressive 'mul'
  function_call_event = Event(
      author=agent.name,
      content=types.Content(
          role="model",
          parts=[
              types.Part.from_function_call(
                  name=mul_tool.name, args={"a": 3, "b": 4}
              )
          ],
      ),
  )

  events = []
  async for e in flow._postprocess_handle_function_calls_async(
      invocation_context, function_call_event, llm_request
  ):
    events.append(e)

  assert len(events) == 1 and not events[0].partial
  fr = events[0].content.parts[0].function_response
  assert fr.name == mul_tool.name and fr.response == {"prod": 12}
