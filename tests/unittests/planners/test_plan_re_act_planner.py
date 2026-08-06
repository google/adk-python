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

"""Tests for PlanReActPlanner.process_planning_response."""

from google.adk.planners.built_in_planner import BuiltInPlanner
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from google.genai import types


def _function_call_names(parts):
  return [p.function_call.name for p in parts if p.function_call]


def test_preserves_all_leading_parallel_function_calls():
  """Parallel function calls at the start of the response must all survive.

  Regression test: the trailing-group guard used ``> 0``, so when the first
  part was a function call (index 0) the loop that collects the rest of the
  parallel call group never ran and every call after the first was dropped.
  """
  planner = PlanReActPlanner()
  response_parts = [
      types.Part.from_function_call(name="get_weather", args={"city": "SF"}),
      types.Part.from_function_call(name="get_time", args={"city": "SF"}),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  assert _function_call_names(result) == ["get_weather", "get_time"]


def test_preserves_parallel_function_calls_after_leading_text():
  """The same parallel group is preserved when text comes first."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(text="Let me look that up."),
      types.Part.from_function_call(name="get_weather", args={"city": "SF"}),
      types.Part.from_function_call(name="get_time", args={"city": "SF"}),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  assert _function_call_names(result) == ["get_weather", "get_time"]


# ---------------------------------------------------------------------------
# Tests for BasePlanner.to_content_blocks (exercised via PlanReActPlanner and
# BuiltInPlanner which are the two concrete subclasses).
# ---------------------------------------------------------------------------


def test_to_content_blocks_text_and_reasoning():
  """Thought parts map to 'reasoning' blocks; plain text maps to 'text' blocks."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(text="I should check the weather first.", thought=True),
      types.Part(text="Here is your answer."),
  ]

  blocks = planner.to_content_blocks(response_parts)

  assert blocks == [
      {"type": "reasoning", "reasoning": "I should check the weather first."},
      {"type": "text", "text": "Here is your answer."},
  ]


def test_to_content_blocks_empty_parts():
  """Empty input returns an empty list."""
  planner = PlanReActPlanner()
  assert planner.to_content_blocks([]) == []


def test_to_content_blocks_skips_non_text_parts():
  """Parts without text (e.g. function calls) are skipped."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(text="Some reasoning.", thought=True),
      types.Part.from_function_call(name="get_weather", args={"city": "NY"}),
      types.Part(text="Final answer."),
  ]

  blocks = planner.to_content_blocks(response_parts)

  assert blocks == [
      {"type": "reasoning", "reasoning": "Some reasoning."},
      {"type": "text", "text": "Final answer."},
  ]


def test_to_content_blocks_built_in_planner():
  """BuiltInPlanner inherits to_content_blocks correctly."""
  planner = BuiltInPlanner(thinking_config=types.ThinkingConfig())
  response_parts = [
      types.Part(text="Thinking step.", thought=True),
      types.Part(text="Response text."),
  ]

  blocks = planner.to_content_blocks(response_parts)

  assert blocks == [
      {"type": "reasoning", "reasoning": "Thinking step."},
      {"type": "text", "text": "Response text."},
  ]
