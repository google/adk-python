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

"""Tests for `BaseLlmFlow.tool_request_processors`."""

from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.prompt import _dynamic_instructions
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.flows.llm_flows.tools import _agent_tools
from google.adk.flows.llm_flows.tools import _toolset_auth
from google.adk.models.llm_request import LlmRequest
import pytest

from ... import testing_utils


class _FlowForTesting(BaseLlmFlow):
  """A flow that installs nothing of its own, like an out-of-tree subclass."""

  pass


class _ToolsDictProbe(BaseLlmRequestProcessor):
  """Records the tool names visible to it at the point where it runs."""

  def __init__(self) -> None:
    self.seen_tool_names: Optional[list[str]] = None

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del invocation_context
    self.seen_tool_names = sorted(llm_request.tools_dict)
    if False:  # pylint: disable=using-constant-test
      yield  # Keeps this an async generator without emitting events.


def _a_tool(query: str) -> str:
  """A tool."""
  return query


async def _run_preprocess(flow: BaseLlmFlow, agent: Agent) -> LlmRequest:
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test message'
  )
  llm_request = LlmRequest()
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass
  return llm_request


def test_default_tool_request_processors_are_ordered():
  assert _FlowForTesting().tool_request_processors == [
      _toolset_auth.request_processor,
      _agent_tools.request_processor,
      _dynamic_instructions.request_processor,
  ]


@pytest.mark.asyncio
async def test_a_flow_that_installs_nothing_still_resolves_tools():
  """Tool resolution does not depend on what a subclass puts in its own list.

  A flow that assembles `request_processors` by hand, which out-of-tree
  subclasses do, must not silently end up with no tools at all.
  """
  llm_request = await _run_preprocess(
      _FlowForTesting(), Agent(name='test_agent', tools=[_a_tool])
  )

  assert '_a_tool' in llm_request.tools_dict


@pytest.mark.asyncio
async def test_processor_inserted_after_agent_tools_sees_the_resolved_tools():
  flow = SingleFlow()
  probe = _ToolsDictProbe()

  flow.tool_request_processors.insert(2, probe)
  await _run_preprocess(flow, Agent(name='test_agent', tools=[_a_tool]))

  assert probe.seen_tool_names == ['_a_tool']


@pytest.mark.asyncio
async def test_processor_appended_to_request_processors_sees_no_tools_yet():
  """The boundary between the two lists is where `tools_dict` gets filled."""
  flow = SingleFlow()
  probe = _ToolsDictProbe()

  flow.request_processors.append(probe)
  await _run_preprocess(flow, Agent(name='test_agent', tools=[_a_tool]))

  assert probe.seen_tool_names == []
