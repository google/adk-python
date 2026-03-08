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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Tests for the PlanReActPlanner."""

from unittest import mock

from google.adk.models.llm_request import LlmRequest
from google.adk.planners.plan_re_act_planner import ACTION_TAG
from google.adk.planners.plan_re_act_planner import FINAL_ANSWER_TAG
from google.adk.planners.plan_re_act_planner import PLANNING_TAG
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.adk.planners.plan_re_act_planner import REASONING_TAG
from google.adk.planners.plan_re_act_planner import REPLANNING_TAG
from google.genai import types
import pytest


@pytest.fixture
def planner():
  return PlanReActPlanner()


class TestSplitByLastPattern:

  def test_pattern_in_middle(self, planner):
    before, after = planner._split_by_last_pattern(
        'hello/*SEP*/world', '/*SEP*/'
    )
    assert before == 'hello/*SEP*/'
    assert after == 'world'

  def test_pattern_not_found(self, planner):
    before, after = planner._split_by_last_pattern('hello world', '/*SEP*/')
    assert before == 'hello world'
    assert after == ''

  def test_pattern_at_beginning(self, planner):
    before, after = planner._split_by_last_pattern('/*SEP*/rest', '/*SEP*/')
    assert before == '/*SEP*/'
    assert after == 'rest'

  def test_pattern_at_end(self, planner):
    before, after = planner._split_by_last_pattern('text/*SEP*/', '/*SEP*/')
    assert before == 'text/*SEP*/'
    assert after == ''

  def test_multiple_occurrences_splits_on_last(self, planner):
    before, after = planner._split_by_last_pattern(
        'a/*SEP*/b/*SEP*/c', '/*SEP*/'
    )
    assert before == 'a/*SEP*/b/*SEP*/'
    assert after == 'c'

  def test_empty_text(self, planner):
    before, after = planner._split_by_last_pattern('', '/*SEP*/')
    assert before == ''
    assert after == ''


class TestMarkAsThought:

  def test_marks_text_part_as_thought(self, planner):
    part = types.Part(text='some reasoning')
    planner._mark_as_thought(part)
    assert part.thought is True

  def test_does_not_mark_empty_text_part(self, planner):
    part = types.Part(text='')
    planner._mark_as_thought(part)
    assert not part.thought


class TestHandleNonFunctionCallParts:

  def test_text_with_final_answer_tag(self, planner):
    text = f'reasoning text{FINAL_ANSWER_TAG}the final answer'
    part = types.Part(text=text)
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 2
    assert preserved[0].thought is True
    assert FINAL_ANSWER_TAG in preserved[0].text
    assert preserved[1].text == 'the final answer'
    assert not preserved[1].thought

  def test_text_with_final_answer_tag_no_reasoning(self, planner):
    text = f'{FINAL_ANSWER_TAG}only answer'
    part = types.Part(text=text)
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 2
    assert preserved[0].thought is True
    assert preserved[1].text == 'only answer'

  def test_text_with_final_answer_tag_no_answer(self, planner):
    text = f'just reasoning{FINAL_ANSWER_TAG}'
    part = types.Part(text=text)
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 1
    assert preserved[0].thought is True

  @pytest.mark.parametrize(
      'tag',
      [PLANNING_TAG, REASONING_TAG, ACTION_TAG, REPLANNING_TAG],
      ids=['planning', 'reasoning', 'action', 'replanning'],
  )
  def test_text_starting_with_tag_is_marked_as_thought(self, planner, tag):
    text = f'{tag} some content'
    part = types.Part(text=text)
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 1
    assert preserved[0].thought is True

  def test_text_without_any_tag(self, planner):
    part = types.Part(text='plain text response')
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 1
    assert preserved[0].text == 'plain text response'
    assert not preserved[0].thought

  def test_part_without_text(self, planner):
    part = types.Part(
        inline_data=types.Blob(mime_type='image/png', data=b'fakepng')
    )
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 1
    assert not preserved[0].thought

  def test_tag_not_at_start_is_not_marked_as_thought(self, planner):
    text = f'some prefix {PLANNING_TAG} content'
    part = types.Part(text=text)
    preserved = []

    planner._handle_non_function_call_parts(part, preserved)

    assert len(preserved) == 1
    assert not preserved[0].thought


class TestProcessPlanningResponse:

  def test_empty_response_parts(self, planner):
    callback_context = mock.MagicMock()

    result = planner.process_planning_response(callback_context, [])

    assert result is None

  def test_none_response_parts(self, planner):
    callback_context = mock.MagicMock()

    result = planner.process_planning_response(callback_context, None)

    assert result is None

  def test_single_text_part_preserved(self, planner):
    callback_context = mock.MagicMock()
    parts = [types.Part(text='just text')]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 1
    assert result[0].text == 'just text'

  def test_single_function_call(self, planner):
    callback_context = mock.MagicMock()
    fc_part = types.Part(
        function_call=types.FunctionCall(name='my_tool', args={'key': 'value'})
    )
    parts = [fc_part]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 1
    assert result[0].function_call.name == 'my_tool'

  def test_function_call_with_empty_name_is_filtered(self, planner):
    callback_context = mock.MagicMock()
    fc_part = types.Part(function_call=types.FunctionCall(name='', args={}))
    parts = [fc_part]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 0

  def test_text_then_function_calls(self, planner):
    callback_context = mock.MagicMock()
    text_part = types.Part(text=f'{PLANNING_TAG} my plan')
    fc_part_1 = types.Part(
        function_call=types.FunctionCall(name='tool_a', args={})
    )
    fc_part_2 = types.Part(
        function_call=types.FunctionCall(name='tool_b', args={})
    )
    parts = [text_part, fc_part_1, fc_part_2]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 3
    assert result[0].thought is True
    assert result[1].function_call.name == 'tool_a'
    assert result[2].function_call.name == 'tool_b'

  def test_text_after_function_calls_is_dropped(self, planner):
    callback_context = mock.MagicMock()
    text_part = types.Part(text='reasoning')
    fc_part = types.Part(
        function_call=types.FunctionCall(name='tool_a', args={})
    )
    trailing_text = types.Part(text='trailing text')
    parts = [text_part, fc_part, trailing_text]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 2
    assert result[0].text == 'reasoning'
    assert result[1].function_call.name == 'tool_a'

  def test_only_function_calls_keeps_first(self, planner):
    """When all parts are function calls starting at index 0, only the first is kept."""
    callback_context = mock.MagicMock()
    fc_parts = [
        types.Part(function_call=types.FunctionCall(name=f'tool_{i}', args={}))
        for i in range(3)
    ]

    result = planner.process_planning_response(callback_context, fc_parts)

    assert len(result) == 1
    assert result[0].function_call.name == 'tool_0'

  def test_final_answer_in_response(self, planner):
    callback_context = mock.MagicMock()
    text = f'{REASONING_TAG} step 1{FINAL_ANSWER_TAG}The answer is 42'
    parts = [types.Part(text=text)]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 2
    assert result[0].thought is True
    assert result[1].text == 'The answer is 42'
    assert not result[1].thought

  def test_multiple_text_parts_before_function_call(self, planner):
    callback_context = mock.MagicMock()
    parts = [
        types.Part(text=f'{PLANNING_TAG} plan step'),
        types.Part(text=f'{REASONING_TAG} reasoning step'),
        types.Part(function_call=types.FunctionCall(name='tool_a', args={})),
    ]

    result = planner.process_planning_response(callback_context, parts)

    assert len(result) == 3
    assert result[0].thought is True
    assert result[1].thought is True
    assert result[2].function_call.name == 'tool_a'


class TestBuildPlanningInstruction:

  def test_returns_non_empty_string(self, planner):
    readonly_context = mock.MagicMock()
    llm_request = LlmRequest()

    result = planner.build_planning_instruction(readonly_context, llm_request)

    assert isinstance(result, str)
    assert len(result) > 0

  def test_instruction_contains_all_tags(self, planner):
    readonly_context = mock.MagicMock()
    llm_request = LlmRequest()

    result = planner.build_planning_instruction(readonly_context, llm_request)

    assert PLANNING_TAG in result
    assert REPLANNING_TAG in result
    assert REASONING_TAG in result
    assert ACTION_TAG in result
    assert FINAL_ANSWER_TAG in result
