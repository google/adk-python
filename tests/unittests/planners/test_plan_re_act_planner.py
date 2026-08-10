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

from google.adk.planners.plan_re_act_planner import ACTION_TAG
from google.adk.planners.plan_re_act_planner import FINAL_ANSWER_TAG
from google.adk.planners.plan_re_act_planner import PLANNING_TAG
from google.adk.planners.plan_re_act_planner import REASONING_TAG
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
# to_content_blocks() standardized output
# ---------------------------------------------------------------------------


def test_to_content_blocks_tags_stripped_from_thought_parts():
  """Planning tags must be stripped and thought parts emitted as 'reasoning'."""
  planner = PlanReActPlanner()
  answer_text = "The final answer is 42."

  response_parts = [
      types.Part(text=f"{PLANNING_TAG}Step 1: call search tool."),
      types.Part(text=f"{REASONING_TAG}Analysing results."),
      types.Part(text=f"{ACTION_TAG}Calling tool now."),
      types.Part(text=f"{FINAL_ANSWER_TAG}{answer_text}"),
  ]

  processed = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )
  assert processed is not None

  blocks = planner.to_content_blocks(processed)

  reasoning_blocks = [b for b in blocks if b["type"] == "reasoning"]
  text_blocks = [b for b in blocks if b["type"] == "text"]

  # All thought parts must surface as reasoning blocks with no raw tags.
  assert len(reasoning_blocks) >= 1
  for rb in reasoning_blocks:
    for tag in [PLANNING_TAG, REASONING_TAG, ACTION_TAG, FINAL_ANSWER_TAG]:
      assert tag not in rb["reasoning"]

  # The final answer must become a plain text block.
  assert len(text_blocks) == 1
  assert text_blocks[0]["text"] == answer_text


def test_to_content_blocks_type_keys_present():
  """Every content block must carry a 'type' key."""
  planner = PlanReActPlanner()
  parts = [
      types.Part(text=f"{PLANNING_TAG}my plan"),
      types.Part(text=f"{FINAL_ANSWER_TAG}my answer"),
  ]
  processed = planner.process_planning_response(
      callback_context=None, response_parts=parts
  )
  assert processed is not None
  blocks = planner.to_content_blocks(processed)
  assert all("type" in b for b in blocks)
