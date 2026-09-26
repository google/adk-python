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

"""Tests for the name-based request processor helpers on BaseLlmFlow."""

from typing import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.flows.llm_flows.context import _contents
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_request import LlmRequest
import pytest


class _FakeRequestProcessor(BaseLlmRequestProcessor):
  """A stand-in processor whose name can be set per instance."""

  def __init__(self, name: str = '') -> None:
    # Sets an instance-level name so a single test can build several processors
    # with different names.
    self.name = name

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del invocation_context, llm_request
    if False:  # pylint: disable=using-constant-test
      yield  # Keeps this an async generator without emitting events.


def test_get_request_processor_returns_the_named_processor():
  flow = SingleFlow()

  assert flow.get_request_processor('contents') is _contents.request_processor


def test_replace_request_processor_keeps_the_original_position_and_returns_old():
  flow = SingleFlow()
  original_index = flow.request_processors.index(_contents.request_processor)
  replacement = _FakeRequestProcessor(name='contents')

  old = flow.replace_request_processor('contents', replacement)

  assert old is _contents.request_processor
  assert flow.request_processors[original_index] is replacement
  assert _contents.request_processor not in flow.request_processors


def test_replace_request_processor_does_not_mutate_anonymous_processor():
  """Replacing with an anonymous processor leaves its name untouched."""
  flow = SingleFlow()
  original_index = flow.request_processors.index(_contents.request_processor)
  anonymous_replacement = _FakeRequestProcessor()

  old = flow.replace_request_processor('contents', anonymous_replacement)

  assert old is _contents.request_processor
  assert anonymous_replacement.name == ''
  assert flow.request_processors[original_index] is anonymous_replacement


def test_replace_request_processor_does_not_change_list_length():
  flow = SingleFlow()
  original_length = len(flow.request_processors)

  flow.replace_request_processor(
      'contents', _FakeRequestProcessor(name='contents')
  )

  assert len(flow.request_processors) == original_length


def test_replace_request_processor_raises_if_new_name_already_exists():
  flow = SingleFlow()
  conflicting = _FakeRequestProcessor(name='compaction')

  with pytest.raises(ValueError, match='already exists'):
    flow.replace_request_processor('contents', conflicting)


def test_insert_request_processor_before_puts_it_ahead_of_the_named_one():
  flow = SingleFlow()
  inserted = _FakeRequestProcessor(name='custom')

  flow.insert_request_processor_before('contents', inserted)

  inserted_index = flow.request_processors.index(inserted)
  contents_index = flow.request_processors.index(_contents.request_processor)
  assert inserted_index == contents_index - 1


def test_insert_request_processor_before_raises_for_duplicate_name():
  flow = SingleFlow()
  duplicate = _FakeRequestProcessor(name='contents')

  with pytest.raises(ValueError, match='already exists'):
    flow.insert_request_processor_before('contents', duplicate)


def test_insert_request_processor_after_puts_it_behind_the_named_one():
  flow = SingleFlow()
  inserted = _FakeRequestProcessor(name='custom')

  flow.insert_request_processor_after('contents', inserted)

  inserted_index = flow.request_processors.index(inserted)
  contents_index = flow.request_processors.index(_contents.request_processor)
  assert inserted_index == contents_index + 1


def test_insert_request_processor_after_raises_for_duplicate_name():
  flow = SingleFlow()
  duplicate = _FakeRequestProcessor(name='contents')

  with pytest.raises(ValueError, match='already exists'):
    flow.insert_request_processor_after('contents', duplicate)


def test_remove_request_processor_returns_and_drops_it():
  flow = SingleFlow()

  removed = flow.remove_request_processor('contents')

  assert removed is _contents.request_processor
  assert _contents.request_processor not in flow.request_processors


def test_unknown_name_raises_and_names_what_is_available():
  flow = SingleFlow()

  with pytest.raises(ValueError, match='No request processor'):
    flow.get_request_processor('does_not_exist')


def test_empty_name_is_rejected():
  """Anonymous processors must not be reachable by passing the empty name."""
  flow = SingleFlow()

  with pytest.raises(ValueError, match='non-empty'):
    flow.get_request_processor('')


def test_duplicate_name_is_reported_as_ambiguous():
  """Picking the first of several same-named processors would be a silent bug."""
  flow = SingleFlow()
  # Manually simulate inconsistent list state
  flow.request_processors.append(_FakeRequestProcessor(name='contents'))

  with pytest.raises(ValueError, match='ambiguous'):
    flow.replace_request_processor(
        'contents', _FakeRequestProcessor(name='contents')
    )


def test_anonymous_processors_are_not_matched():
  """An unnamed processor in the list must not be hit by a name lookup."""
  flow = SingleFlow()
  flow.request_processors.append(_FakeRequestProcessor())

  with pytest.raises(ValueError, match='No request processor'):
    flow.get_request_processor('anonymous')


def test_helpers_operate_on_the_same_list_as_direct_manipulation():
  """The helpers are sugar over the list, not a separate container."""
  by_helper = SingleFlow()
  by_hand = SingleFlow()
  replacement = _FakeRequestProcessor(name='contents')

  by_helper.replace_request_processor('contents', replacement)
  by_hand.request_processors[
      by_hand.request_processors.index(_contents.request_processor)
  ] = replacement

  assert by_helper.request_processors == by_hand.request_processors


def test_name_lookups_reach_into_tool_request_processors():
  flow = SingleFlow()

  # Reachable in the second list.
  assert flow.get_request_processor('agent_tools').name == 'agent_tools'
  # Still reachable in the first.
  assert flow.get_request_processor('contents').name == 'contents'


def test_a_tool_request_processor_can_be_removed_by_name():
  """A flow that does its own auth drops the built-in rather than forking it."""
  flow = SingleFlow()

  removed = flow.remove_request_processor('toolset_auth')

  assert removed.name == 'toolset_auth'
  assert [p.name for p in flow.tool_request_processors] == [
      'agent_tools',
      'dynamic_instructions',
  ]


def test_a_tool_request_processor_can_be_replaced_by_name():
  flow = SingleFlow()
  replacement = _FakeRequestProcessor(name='agent_tools')

  old = flow.replace_request_processor('agent_tools', replacement)

  assert old.name == 'agent_tools'
  assert flow.get_request_processor('agent_tools') is replacement
  assert flow.tool_request_processors[1] is replacement


def test_duplicate_names_across_the_two_lists_are_rejected():
  flow = SingleFlow()
  duplicate = _FakeRequestProcessor(name='agent_tools')

  with pytest.raises(ValueError, match='already exists'):
    flow.insert_request_processor_before('contents', duplicate)
