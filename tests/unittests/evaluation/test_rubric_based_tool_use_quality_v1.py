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

from __future__ import annotations

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.rubric_based_tool_use_quality_v1 import RubricBasedToolUseV1Evaluator
from google.genai import types as genai_types
import pytest


@pytest.fixture
def evaluator() -> RubricBasedToolUseV1Evaluator:
  """Returns a RubricBasedToolUseV1Evaluator."""
  rubrics = [
      Rubric(
          rubric_id="1",
          rubric_content=RubricContent(
              text_property="Did the agent use the correct tool?"
          ),
      ),
      Rubric(
          rubric_id="2",
          rubric_content=RubricContent(
              text_property="Were the tool parameters correct?"
          ),
      ),
  ]
  judge_model_options = JudgeModelOptions(
      judge_model_config=None,
      num_samples=3,
  )
  criterion = RubricsBasedCriterion(
      threshold=0.5, rubrics=rubrics, judge_model_options=judge_model_options
  )
  metric = EvalMetric(
      metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
      threshold=0.5,
      criterion=criterion,
  )
  return RubricBasedToolUseV1Evaluator(metric)


def test_format_auto_rater_prompt_with_basic_invocation(
    evaluator: RubricBasedToolUseV1Evaluator,
):
  """Tests format_auto_rater_prompt with a basic invocation."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "User input here." in prompt
  assert "Did the agent use the correct tool?" in prompt
  assert "Were the tool parameters correct?" in prompt
  assert "<available_tools>\nAgent has no tools.\n</available_tools>" in prompt
  assert "<response>\nNo intermediate steps were taken.\n</response>" in prompt


def test_format_auto_rater_prompt_with_app_details(
    evaluator: RubricBasedToolUseV1Evaluator,
):
  """Tests format_auto_rater_prompt with app_details in invocation."""
  tool = genai_types.Tool(
      function_declarations=[
          genai_types.FunctionDeclaration(
              name="test_func", description="A test function."
          )
      ]
  )
  app_details = AppDetails(
      agent_details={
          "agent1": AgentDetails(
              name="agent1",
              tool_declarations=[tool],
          )
      },
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      app_details=app_details,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"name": "test_func"' in prompt
  assert '"description": "A test function."' in prompt


def test_format_auto_rater_prompt_with_intermediate_data(
    evaluator: RubricBasedToolUseV1Evaluator,
):
  """Tests format_auto_rater_prompt with intermediate_data in invocation."""
  tool_call = genai_types.FunctionCall(
      name="test_func", args={"arg1": "val1"}, id="call1"
  )
  tool_response = genai_types.FunctionResponse(
      name="test_func", response={"result": "ok"}, id="call1"
  )
  intermediate_data = IntermediateData(
      tool_uses=[tool_call], tool_responses=[tool_response]
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      intermediate_data=intermediate_data,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"step": 0' in prompt
  assert '"tool_call":' in prompt
  assert '"name": "test_func"' in prompt
  assert '"tool_response":' in prompt
  assert '"result": "ok"' in prompt
