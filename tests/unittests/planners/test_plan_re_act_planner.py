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
