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


def test_strips_planning_tag_from_thought_part():
  """Planning/reasoning tags must be stripped from the output text.

  The raw ``/*PLANNING*/``, ``/*REASONING*/``, ``/*ACTION*/`` and
  ``/*REPLANNING*/`` markers are internal prompting artefacts.  After
  processing, the resulting parts should contain clean text, have
  ``thought=True`` set, and preserve other Part metadata such as
  ``thought_signature``.
  """
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(
          text="/*PLANNING*/Step 1: look it up.",
          thought_signature=b"sig1",
      ),
      types.Part(
          text="/*REASONING*/I need to call the tool.",
          thought_signature=b"sig2",
      ),
      types.Part.from_function_call(name="lookup", args={"q": "test"}),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  text_parts = [p for p in result if p.text]
  # Tags must be gone
  for p in text_parts:
    assert "/*PLANNING*/" not in p.text
    assert "/*REASONING*/" not in p.text
  # Thought flag must be set on non-final-answer text parts
  for p in text_parts:
    assert p.thought is True
  # Part metadata like thought_signature must be preserved
  assert text_parts[0].thought_signature == b"sig1"
  assert text_parts[1].thought_signature == b"sig2"
  # Function call must still be present
  assert _function_call_names(result) == ["lookup"]


def test_strips_final_answer_tag_boundary():
  """The /*FINAL_ANSWER*/ tag must not appear in either output block."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(
          text="/*REASONING*/Some reasoning./*FINAL_ANSWER*/The answer is 42."
      ),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  texts = [p.text for p in result if p.text]
  combined = " ".join(texts)
  assert "/*FINAL_ANSWER*/" not in combined
  assert "/*REASONING*/" not in combined
  assert "The answer is 42." in combined


def test_strips_multiple_planning_tags():
  """Embedded planning and reasoning tags must all be stripped."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(
          text=(
              "/*PLANNING*/Initial plan.\n"
              "/*REASONING*/Some reasoning.\n"
              "/*FINAL_ANSWER*/The answer is 42."
          )
      ),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  texts = [p.text for p in result if p.text]
  combined = " ".join(texts)
  assert "/*PLANNING*/" not in combined
  assert "/*REASONING*/" not in combined
  assert "/*FINAL_ANSWER*/" not in combined
  assert "Initial plan." in combined
  assert "Some reasoning." in combined
  assert "The answer is 42." in combined


def test_part_without_leading_tag_not_marked_as_thought():
  """A part without a leading tag (even with stray embedded tag) is not thought."""
  planner = PlanReActPlanner()
  response_parts = [
      types.Part(text="Here is the answer /*PLANNING*/ with stray tag."),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  assert len(result) == 1
  assert result[0].thought is not True
  assert result[0].text == "Here is the answer /*PLANNING*/ with stray tag."


def test_bare_tag_part_is_marked_as_thought():
  """A part containing only planning tags is kept, stripped, and marked as thought."""
  planner = PlanReActPlanner()
  bare_part = types.Part(text="/*ACTION*/")
  response_parts = [
      bare_part,
      types.Part.from_function_call(name="lookup", args={"q": "test"}),
  ]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  assert len(result) == 2
  assert result[0].text == ""
  assert result[0].thought is True
  assert _function_call_names(result) == ["lookup"]


def test_sole_bare_tag_part_is_marked_as_thought():
  """A sole part with only a planning tag is preserved as a thought part."""
  planner = PlanReActPlanner()
  response_parts = [types.Part(text="/*ACTION*/")]

  result = planner.process_planning_response(
      callback_context=None, response_parts=response_parts
  )

  assert len(result) == 1
  assert result[0].text == ""
  assert result[0].thought is True


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
