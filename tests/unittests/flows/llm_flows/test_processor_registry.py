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

"""Tests for the `name` declaration on flow processors."""

from typing import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmResponseProcessor
from google.adk.flows.llm_flows.auto_flow import AutoFlow
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_request import LlmRequest


class _AnonymousRequestProcessor(BaseLlmRequestProcessor):
  """A processor that declares no name."""

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del invocation_context, llm_request
    if False:  # pylint: disable=using-constant-test
      yield  # Keeps this an async generator without emitting events.


def test_auto_flow_request_processors_are_all_named():
  """Every built-in request processor is addressable by name."""
  flow = AutoFlow()
  processors = flow.request_processors + flow.tool_request_processors

  names = [p.name for p in processors]

  assert all(names), f'Unnamed built-in request processor in {names}'


def test_single_flow_response_processors_are_all_named():
  processors = SingleFlow().response_processors

  names = [p.name for p in processors]

  assert all(names), f'Unnamed built-in response processor in {names}'


def test_auto_flow_request_processor_names_are_unique():
  """Names must be unique so a name-based lookup is unambiguous."""
  flow = AutoFlow()
  names = [
      p.name for p in flow.request_processors + flow.tool_request_processors
  ]

  assert len(names) == len(set(names)), f'Duplicate name in {names}'


def test_single_flow_response_processor_names_are_unique():
  names = [p.name for p in SingleFlow().response_processors]

  assert len(names) == len(set(names)), f'Duplicate name in {names}'


def test_processor_without_declarations_keeps_working():
  """A processor that ignores `name` behaves exactly as it did before.

  Third-party processors predate this attribute, so the default must leave
  them anonymous rather than forcing them to opt in.
  """
  processor = _AnonymousRequestProcessor()

  assert processor.name == ''


def test_anonymous_processor_can_be_appended_to_a_flow():
  """Appending to the plain list still works and does not disturb the rest."""
  flow = SingleFlow()
  original = list(flow.request_processors)
  processor = _AnonymousRequestProcessor()

  flow.request_processors.append(processor)

  assert flow.request_processors == original + [processor]


def test_base_processor_defaults_are_anonymous():
  assert BaseLlmRequestProcessor.name == ''
  assert BaseLlmResponseProcessor.name == ''


def test_default_tool_request_processors_are_named_and_ordered():
  """Pins the tool pipeline by name, in order.

  Order carries meaning here -- auth resolves before the tools that need it --
  so this list is written out deliberately rather than derived. Updating it is
  the point: a change to this assertion should be a change someone meant to
  make, not one they absorbed.
  """
  assert [p.name for p in SingleFlow().tool_request_processors] == [
      'toolset_auth',
      'agent_tools',
      'dynamic_instructions',
  ]
