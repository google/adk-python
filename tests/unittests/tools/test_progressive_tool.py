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
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions.session import Session
from google.adk.tools import ProgressiveTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


@pytest.mark.asyncio
async def test_progressive_tool_streams_partial_and_final():
  async def export_report(country: str):
    yield {"status": "started", "country": country}
    for i in range(1, 6):
      await asyncio.sleep(0)
      yield {"status": "progress", "percent": i * 20}
    yield {"status": "completed", "url": f"https://example.com/{country}.pdf"}

  tool = ProgressiveTool(export_report)

  # Model first asks to call the tool, then later provides a summary text
  function_call = types.Part.from_function_call(
      name=tool.name, args={"country": "france"}
  )
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model",
          parts=[types.Part.from_text(text="The report for France is ready.")],
      )
  )

  mock_model = testing_utils.MockModel.create([response1, response2])

  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("Please export the report for France.")

  # Expect at least one partial event and one final function_response event
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

  assert partials, "Expected progressive partial events"
  assert finals, "Expected a final function_response event"

  # Check order of progress percentages
  percents = [
      fr.function_response.response.get("percent")
      for e in partials
      for fr in e.content.parts
      if fr.function_response and "percent" in fr.function_response.response
  ]
  assert percents == sorted(
      percents
  ), "Progress percentage should be non-decreasing"

  # Ensure a concluding model text arrived
  model_texts = [
      p.text
      for e in events
      if e.content and e.content.parts
      for p in e.content.parts
      if getattr(p, "text", None)
  ]
  assert any(
      "ready" in (t or "").lower() for t in model_texts
  ), "Expected model summary text"


def test_progressive_tool_init_sets_name_and_doc():
  async def sample_func():
    """Doc string for progressive tool."""
    yield {"x": 1}

  tool = ProgressiveTool(sample_func)
  assert tool.name == "sample_func"
  assert tool.description == "Doc string for progressive tool."


@pytest.mark.asyncio
async def test_progressive_tool_run_async_generator_returns_last_yield():
  async def gen():
    yield {"a": 1}
    yield {"a": 2}

  tool = ProgressiveTool(gen)
  # Direct run_async uses fallback that consumes generator and returns last item
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result == {"a": 2}


@pytest.mark.asyncio
async def test_progressive_tool_final_equals_last_yield():
  final_payload = {"status": "completed", "value": 123}

  async def export_report(country: str):
    yield {"status": "started"}
    yield final_payload

  tool = ProgressiveTool(export_report)

  # Model triggers function call then emits a follow-up text
  function_call = types.Part.from_function_call(
      name=tool.name, args={"country": "x"}
  )
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )

  mock_model = testing_utils.MockModel.create([response1, response2])
  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("run tool")

  # Locate the final (non-partial) function_response
  final_fn_events = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert final_fn_events, "Expected a final function_response event"
  fr = final_fn_events[-1].content.parts[0].function_response
  assert (
      fr.response == final_payload
  ), "Final function response must equal the last yielded payload"


@pytest.mark.asyncio
async def test_progressive_tool_error_converted_by_plugin():
  class ToolErrorToResultPlugin(BasePlugin):

    def __init__(self):
      super().__init__(name="tool_error_to_result")

    async def on_tool_error_callback(
        self, *, tool, tool_args, tool_context, error
    ):
      return {"status": "error_handled", "message": str(error)}

  async def faulty_tool(x: int):
    yield {"status": "started"}
    raise RuntimeError("boom")

  tool = ProgressiveTool(faulty_tool)

  function_call = types.Part.from_function_call(name=tool.name, args={"x": 1})
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="done")]
      )
  )

  mock_model = testing_utils.MockModel.create([response1, response2])
  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(
      root_agent=agent, plugins=[ToolErrorToResultPlugin()]
  )

  events = await runner.run_async("trigger faulty")

  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert finals, "Expected final function_response produced by plugin"
  fr = finals[-1].content.parts[0].function_response
  assert fr.response.get("status") == "error_handled"


@pytest.mark.asyncio
async def test_multiple_progressive_tools_sequential_progress():
  async def tool_a():
    yield {"tool": "a", "step": 1}
    yield {"tool": "a", "step": 2}

  async def tool_b():
    yield {"tool": "b", "step": 1}
    yield {"tool": "b", "step": 2}

  ta = ProgressiveTool(tool_a)
  tb = ProgressiveTool(tool_b)

  fc_a = types.Part.from_function_call(name=ta.name, args={})
  fc_b = types.Part.from_function_call(name=tb.name, args={})
  # Model requests both tools in the same turn
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[fc_a, fc_b])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )

  mock_model = testing_utils.MockModel.create([response1, response2])
  agent = LlmAgent(name="root_agent", model=mock_model, tools=[ta, tb])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("call both")

  # Ensure both tools produced progress
  progress = [
      e.content.parts[0].function_response.response
      for e in events
      if e.partial
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  tools_seen = {p.get("tool") for p in progress if isinstance(p, dict)}
  assert tools_seen == {"a", "b"}


@pytest.mark.asyncio
async def test_non_progressive_tool_unaffected():
  # regular function tool (non-progressive)
  def add(x: int, y: int) -> dict[str, int]:
    return {"sum": x + y}

  # Progressive one
  async def p():
    yield {"p": 1}
    yield {"p": 2}

  add_part = types.Part.from_function_call(name="add", args={"x": 1, "y": 2})
  p_tool = ProgressiveTool(p)

  # The framework will wrap bare callables into FunctionTool automatically
  response1 = LlmResponse(content=types.Content(role="model", parts=[add_part]))
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )
  mock_model = testing_utils.MockModel.create([response1, response2])

  agent = LlmAgent(name="root_agent", model=mock_model, tools=[add, p_tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("add and progress")

  # add tool should only have a final function_response (no partials)
  add_fn_events = [
      e
      for e in events
      if e.content
      and e.content.parts
      and e.content.parts[0].function_response
      and e.content.parts[0].function_response.name == "add"
  ]
  assert add_fn_events, "Expected add tool function_response"
  assert not any(
      e.partial for e in add_fn_events
  ), "Non-progressive tool should not emit partials"


@pytest.mark.asyncio
async def test_progressive_tool_with_progress_param_streams_and_final():
  async def long_task(x: int, progress=None):
    if progress:
      await progress({"step": 1})
    await asyncio.sleep(0)
    if progress:
      await progress({"step": 2})
    return {"done": True}

  tool = ProgressiveTool(long_task)

  function_call = types.Part.from_function_call(name=tool.name, args={"x": 7})
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )
  mock_model = testing_utils.MockModel.create([response1, response2])

  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("run")

  partial_payloads = [
      e.content.parts[0].function_response.response
      for e in events
      if e.partial
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert {"step": 1} in partial_payloads and {"step": 2} in partial_payloads

  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert finals and finals[-1].content.parts[0].function_response.response == {
      "done": True
  }


@pytest.mark.asyncio
async def test_progressive_tool_with_progress_callback_param_streams_and_final():
  async def long_task_2(y: int, progress_callback=None):
    if progress_callback:
      await progress_callback({"stage": "init"})
    await asyncio.sleep(0)
    if progress_callback:
      await progress_callback({"stage": "mid"})
    return {"result": y * 2}

  tool = ProgressiveTool(long_task_2)

  function_call = types.Part.from_function_call(name=tool.name, args={"y": 3})
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )
  mock_model = testing_utils.MockModel.create([response1, response2])

  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("go")

  progress = [
      e.content.parts[0].function_response.response
      for e in events
      if e.partial
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert {"stage": "init"} in progress and {"stage": "mid"} in progress

  finals = [
      e
      for e in events
      if (not e.partial)
      and e.content
      and e.content.parts
      and e.content.parts[0].function_response
  ]
  assert finals and finals[-1].content.parts[0].function_response.response == {
      "result": 6
  }


@pytest.mark.asyncio
async def test_progressive_tool_coroutine_without_progress_param_no_partials():
  async def compute(z: int):
    await asyncio.sleep(0)
    return {"ok": z}

  tool = ProgressiveTool(compute)
  function_call = types.Part.from_function_call(name=tool.name, args={"z": 5})
  response1 = LlmResponse(
      content=types.Content(role="model", parts=[function_call])
  )
  response2 = LlmResponse(
      content=types.Content(
          role="model", parts=[types.Part.from_text(text="ok")]
      )
  )
  mock_model = testing_utils.MockModel.create([response1, response2])

  agent = LlmAgent(name="root_agent", model=mock_model, tools=[tool])
  runner = testing_utils.InMemoryRunner(root_agent=agent)

  events = await runner.run_async("compute")

  assert not any(
      e.partial
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  )
  finals = [
      e
      for e in events
      if e.content and e.content.parts and e.content.parts[0].function_response
  ]
  assert finals and finals[-1].content.parts[0].function_response.response == {
      "ok": 5
  }
