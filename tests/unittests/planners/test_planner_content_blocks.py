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

"""Tests for standardized planner content blocks."""

from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import ACTION_TAG
from google.adk.planners.plan_re_act_planner import FINAL_ANSWER_TAG
from google.adk.planners.plan_re_act_planner import PLANNING_TAG
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.planners.plan_re_act_planner import REASONING_TAG
from google.adk.planners.plan_re_act_planner import REPLANNING_TAG
from google.adk.planners.planner_content_blocks import part_to_content_block
from google.adk.planners.planner_content_blocks import parts_to_content_blocks
from google.genai import types


# ---------------------------------------------------------------------------
# part_to_content_block
# ---------------------------------------------------------------------------


def test_plain_text_part_becomes_text_block():
  part = types.Part(text='the final answer')
  assert part_to_content_block(part) == {
      'type': 'text',
      'text': 'the final answer',
  }


def test_thought_part_becomes_reasoning_block():
  part = types.Part(text='some thinking', thought=True)
  assert part_to_content_block(part) == {
      'type': 'reasoning',
      'reasoning': 'some thinking',
      'reasoning_kind': None,
  }


def test_function_call_part_becomes_tool_call_block():
  part = types.Part.from_function_call(name='get_weather', args={'city': 'SF'})
  assert part_to_content_block(part) == {
      'type': 'tool_call',
      'name': 'get_weather',
      'args': {'city': 'SF'},
      'id': None,
  }


def test_function_call_preserves_id():
  part = types.Part.from_function_call(name='get_weather', args={})
  part.function_call.id = 'call_123'
  block = part_to_content_block(part)
  assert block['id'] == 'call_123'
  assert block['args'] == {}


def test_empty_function_call_name_is_skipped():
  # A function call with no name carries no usable content.
  part = types.Part.from_function_call(name='', args={'x': 1})
  assert part_to_content_block(part) is None


def test_empty_text_part_is_skipped():
  assert part_to_content_block(types.Part(text='')) is None
  assert part_to_content_block(types.Part()) is None


def test_redacted_thought_without_text_is_skipped():
  # e.g. an Anthropic redacted-thinking part that only carries a signature.
  part = types.Part(thought=True, thought_signature=b'sig')
  assert part_to_content_block(part) is None


def test_function_call_takes_precedence_over_text():
  part = types.Part.from_function_call(name='do_it', args={})
  part.text = 'incidental text'
  block = part_to_content_block(part)
  assert block['type'] == 'tool_call'
  assert block['name'] == 'do_it'


# ---------------------------------------------------------------------------
# PlanReActPlanner tag -> reasoning_kind mapping
# ---------------------------------------------------------------------------


def test_planning_tag_maps_to_planning_kind():
  part = types.Part(text=f'{PLANNING_TAG}\n1. do a thing', thought=True)
  block = part_to_content_block(part)
  assert block['type'] == 'reasoning'
  assert block['reasoning_kind'] == 'planning'
  # The leading tag is stripped from the standardized reasoning text.
  assert block['reasoning'] == '1. do a thing'


def test_each_reasoning_tag_maps_to_expected_kind():
  cases = {
      PLANNING_TAG: 'planning',
      REPLANNING_TAG: 'replanning',
      REASONING_TAG: 'reasoning',
      ACTION_TAG: 'action',
  }
  for tag, expected_kind in cases.items():
    part = types.Part(text=f'{tag} body', thought=True)
    block = part_to_content_block(part)
    assert block['reasoning_kind'] == expected_kind, tag
    assert block['reasoning'] == 'body', tag


def test_thought_without_recognized_tag_has_none_kind():
  part = types.Part(text='free-form thought', thought=True)
  block = part_to_content_block(part)
  assert block['reasoning_kind'] is None
  assert block['reasoning'] == 'free-form thought'


def test_trailing_final_answer_tag_stripped_from_reasoning_block():
  # PlanReActPlanner keeps the FINAL_ANSWER separator on the reasoning part; it
  # should not leak into the standardized reasoning text.
  part = types.Part(text=f'{REASONING_TAG} got it{FINAL_ANSWER_TAG}', thought=True)
  block = part_to_content_block(part)
  assert block['type'] == 'reasoning'
  assert block['reasoning_kind'] == 'reasoning'
  assert block['reasoning'] == 'got it'


def test_final_answer_tag_stripped_from_text_block():
  part = types.Part(text=f'{FINAL_ANSWER_TAG} here it is')
  block = part_to_content_block(part)
  assert block['type'] == 'text'
  assert block['text'] == 'here it is'


# ---------------------------------------------------------------------------
# parts_to_content_blocks
# ---------------------------------------------------------------------------


def test_parts_to_content_blocks_empty_and_none():
  assert parts_to_content_blocks(None) == []
  assert parts_to_content_blocks([]) == []


def test_parts_to_content_blocks_skips_unmappable_parts():
  parts = [
      types.Part(text='reasoning', thought=True),
      types.Part(),  # unmappable -> skipped
      types.Part(text='answer'),
  ]
  blocks = parts_to_content_blocks(parts)
  assert [b['type'] for b in blocks] == ['reasoning', 'text']


def test_parts_to_content_blocks_preserves_order():
  parts = [
      types.Part(text=f'{PLANNING_TAG} plan', thought=True),
      types.Part.from_function_call(name='search', args={'q': 'x'}),
      types.Part(text=f'{REASONING_TAG} summary', thought=True),
      types.Part(text='final'),
  ]
  blocks = parts_to_content_blocks(parts)
  assert blocks == [
      {'type': 'reasoning', 'reasoning': 'plan', 'reasoning_kind': 'planning'},
      {'type': 'tool_call', 'name': 'search', 'args': {'q': 'x'}, 'id': None},
      {
          'type': 'reasoning',
          'reasoning': 'summary',
          'reasoning_kind': 'reasoning',
      },
      {'type': 'text', 'text': 'final'},
  ]


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_conversion_does_not_mutate_parts():
  thought = types.Part(text=f'{PLANNING_TAG} plan', thought=True)
  text = types.Part(text='answer')
  fc = types.Part.from_function_call(name='f', args={'a': 1})
  before = [p.model_copy(deep=True) for p in (thought, text, fc)]

  parts_to_content_blocks([thought, text, fc])

  for original, after in zip(before, (thought, text, fc)):
    assert original == after


def test_returned_args_is_a_copy():
  fc = types.Part.from_function_call(name='f', args={'a': 1})
  block = part_to_content_block(fc)
  block['args']['a'] = 999
  # Mutating the returned block must not affect the source part.
  assert fc.function_call.args == {'a': 1}


# ---------------------------------------------------------------------------
# Planner method integration
# ---------------------------------------------------------------------------


def test_plan_re_act_planner_round_trip_plan_then_tool_call():
  """process_planning_response output converts to faithful content blocks.

  When a function call is present, the planner intentionally keeps the leading
  plan plus the function-call group and drops trailing prose, so the
  standardized view contains exactly those blocks.
  """
  planner = PlanReActPlanner()
  raw_parts = [
      types.Part(text=f'{PLANNING_TAG}\n1. call the tool'),
      types.Part.from_function_call(name='get_weather', args={'city': 'SF'}),
      types.Part(text='trailing prose that the planner drops'),
  ]

  processed = planner.process_planning_response(
      callback_context=None, response_parts=raw_parts
  )
  blocks = planner.to_content_blocks(processed)

  assert blocks == [
      {
          'type': 'reasoning',
          'reasoning': '1. call the tool',
          'reasoning_kind': 'planning',
      },
      {
          'type': 'tool_call',
          'name': 'get_weather',
          'args': {'city': 'SF'},
          'id': None,
      },
  ]


def test_plan_re_act_planner_round_trip_reasoning_then_final_answer():
  """A single reasoning/final-answer part splits into reasoning + text blocks."""
  planner = PlanReActPlanner()
  raw_parts = [
      types.Part(text=f'{REASONING_TAG} got it{FINAL_ANSWER_TAG} It is sunny.'),
  ]

  processed = planner.process_planning_response(
      callback_context=None, response_parts=raw_parts
  )
  blocks = planner.to_content_blocks(processed)

  assert blocks == [
      {
          'type': 'reasoning',
          'reasoning': 'got it',
          'reasoning_kind': 'reasoning',
      },
      # The text block preserves the planner's verbatim answer text (including
      # the leading space the planner left after the FINAL_ANSWER tag); only the
      # ADK control tag itself is removed, never user-visible content.
      {'type': 'text', 'text': ' It is sunny.'},
  ]


def test_built_in_planner_to_content_blocks():
  planner = BuiltInPlanner(thinking_config=types.ThinkingConfig())
  parts = [
      types.Part(text='internal thinking', thought=True),
      types.Part(text='visible answer'),
  ]
  blocks = planner.to_content_blocks(parts)
  assert blocks == [
      {
          'type': 'reasoning',
          'reasoning': 'internal thinking',
          'reasoning_kind': None,
      },
      {'type': 'text', 'text': 'visible answer'},
  ]
