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

"""Resolves the agent's tools into ``llm_request.tools_dict``."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from typing import cast
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ....agents.invocation_context import InvocationContext
from ....agents.readonly_context import ReadonlyContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from ....tools.base_toolset import BaseToolset
from ....tools.tool_context import ToolContext
from .._base_llm_processor import BaseLlmRequestProcessor
from ._batch_executor import _is_non_blocking_tool
from ._batch_executor import _is_streaming_tool

if TYPE_CHECKING:
  from ....agents.llm_agent import LlmAgent


async def process_agent_tools(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Process the agent's tools and populate ``llm_request.tools_dict``.

  Iterates over the agent's ``tools`` list, converts each tool union
  (callable, BaseTool, or BaseToolset) into resolved ``BaseTool``
  instances, and calls ``process_llm_request`` on each to register
  tool declarations in the request.

  Tool-union resolution is dispatched concurrently via ``asyncio.gather``
  to overlap I/O-bound listings (e.g. MCP ``list_tools`` over the
  network). The subsequent ``process_llm_request`` calls are kept
  serial in the original ``agent.tools`` order: some tools read/write
  ``llm_request`` state (e.g. ``GoogleSearchTool`` writes
  ``llm_request.model``; ``ComputerUseToolset`` performs an idempotency
  check on ``llm_request.config.tools``) and rely on observing the
  post-state of earlier tools.

  After this function returns, ``llm_request.tools_dict`` maps tool
  names to ``BaseTool`` instances ready for function call dispatch.

  Args:
    invocation_context: The invocation context (``agent`` is read from
      ``invocation_context.agent``).
    llm_request: The LLM request to populate with tool declarations.
  """
  raw_agent = invocation_context.agent
  if (
      raw_agent is None
      or not hasattr(raw_agent, 'tools')
      or not raw_agent.tools
  ):
    invocation_context.canonical_tools_cache = []
    return
  agent = cast('LlmAgent', raw_agent)

  from ..extensions._agent_transfer import _get_transfer_targets

  multiple_tools = len(agent.tools) > 1 or bool(_get_transfer_targets(agent))
  model = agent.canonical_model

  from ....agents.llm_agent import _convert_tool_union_to_tools

  # Resolve tool_unions in parallel. ``asyncio.gather`` preserves
  # input order in the returned list, so the serial commit phase below
  # still observes ``agent.tools`` order. If any resolution raises,
  # gather cancels the siblings and propagates -- same observable
  # behavior as the previous serial loop, which would propagate the
  # first exception and abandon the rest.
  resolved_tools_per_union = await asyncio.gather(*(
      _convert_tool_union_to_tools(
          tool_union,
          ReadonlyContext(invocation_context),
          model,
          multiple_tools,
      )
      for tool_union in agent.tools
  ))

  # Serial commit phase, in original ``agent.tools`` order. Mutations
  # to ``llm_request`` and reads of its state (model, config.tools,
  # tools_dict) preserve today's ordering semantics exactly.
  for tool_union, tools in zip(agent.tools, resolved_tools_per_union):
    tool_context = ToolContext(invocation_context)

    # If it's a toolset, process it first
    if isinstance(tool_union, BaseToolset):
      await tool_union.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

    # Then process all tools from this tool union
    for tool in tools:
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  if invocation_context.live_request_queue is not None:
    mark_live_async_tools_non_blocking(llm_request)

  # Reuse this exact, current-step resolution in after-model processing. Tool
  # sets can change between model steps, so the cache is refreshed each time.
  invocation_context.canonical_tools_cache = [
      tool for tools in resolved_tools_per_union for tool in tools
  ]


def mark_live_async_tools_non_blocking(llm_request: LlmRequest) -> None:
  """Marks live streaming and response-scheduling tools as NON_BLOCKING.

  These tools emit asynchronous FunctionResponses, which the Live API only
  accepts for NON_BLOCKING declarations.
  """
  if not llm_request.config.tools:
    return
  for gemini_tool in llm_request.config.tools:
    if not isinstance(gemini_tool, types.Tool):
      continue
    for declaration in gemini_tool.function_declarations or []:
      declaration_name = declaration.name
      if declaration_name is None:
        continue
      tool = llm_request.tools_dict.get(declaration_name)
      if tool is None:
        continue
      if _is_streaming_tool(tool) or _is_non_blocking_tool(tool):
        declaration.behavior = types.Behavior.NON_BLOCKING
      elif tool.behavior is types.Behavior.BLOCKING:
        declaration.behavior = tool.behavior


class _AgentToolsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Populates ``llm_request.tools_dict`` from the agent's tools.

  Every processor placed after this one sees a fully resolved
  ``tools_dict``; every processor before it sees an empty one.
  """

  name = 'agent_tools'

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    await process_agent_tools(invocation_context, llm_request)

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _AgentToolsLlmRequestProcessor()
