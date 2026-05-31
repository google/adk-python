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
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge_utils import get_average_rubric_score
from google.adk.evaluation.rubric_based_final_response_quality_v1 import RubricBasedFinalResponseQualityV1Evaluator
from google.genai import types as genai_types
import pytest


@pytest.fixture
def evaluator() -> RubricBasedFinalResponseQualityV1Evaluator:
  """Returns a RubricBasedFinalResponseQualityV1Evaluator."""
  rubrics = [
      Rubric(
          rubric_id="1",
          rubric_content=RubricContent(text_property="Is the response good?"),
      ),
      Rubric(
          rubric_id="2",
          rubric_content=RubricContent(text_property="Is the response bad?"),
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
      metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
      threshold=0.5,
      criterion=criterion,
  )
  return RubricBasedFinalResponseQualityV1Evaluator(metric)


def _create_per_invocation_result(
    rubric_scores: list[RubricScore],
) -> PerInvocationResult:
  """Helper to create a PerInvocationResult."""
  return PerInvocationResult(
      actual_invocation=Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="part_1")]
          )
      ),
      expected_invocation=Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="part_2")]
          )
      ),
      score=get_average_rubric_score(rubric_scores),
      rubric_scores=rubric_scores,
      eval_status=EvalStatus.NOT_EVALUATED,
  )


def test_format_auto_rater_prompt_with_basic_invocation(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with a basic invocation."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "User input here." in prompt
  assert "Final agent response." in prompt
  assert "Is the response good?" in prompt
  assert "Is the response bad?" in prompt
  assert "<developer_instructions>\n  \n  </developer_instructions>" in prompt
  assert (
      "<available_tools>\n  Agent has no tools.\n  </available_tools>" in prompt
  )
  assert (
      "<response_steps>\n  No intermediate steps were taken.\n "
      " </response_steps>"
  ) in prompt


def test_format_auto_rater_prompt_with_app_details(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
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
              instructions="This is an agent instruction.",
              tool_declarations=[tool],
          )
      },
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      app_details=app_details,
      intermediate_data=InvocationEvents(
          invocation_events=[InvocationEvent(author="agent1", content=None)]
      ),
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "This is an agent instruction." in prompt
  assert '"name": "test_func"' in prompt
  assert '"description": "A test function."' in prompt


def test_format_auto_rater_prompt_with_intermediate_data(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
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
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      intermediate_data=intermediate_data,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"step": 0' in prompt
  assert '"tool_call":' in prompt
  assert '"name": "test_func"' in prompt
  assert '"tool_response":' in prompt
  assert '"result": "ok"' in prompt


def test_format_auto_rater_prompt_with_app_details_no_tools(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with app_details but no tools."""
  app_details = AppDetails(
      agent_details={
          "agent1": AgentDetails(name="agent1", tool_declarations=[])
      },
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      app_details=app_details,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"tool_declarations": {\n    "agent1": []\n  }' in prompt


def test_format_auto_rater_prompt_with_intermediate_data_no_tools(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with intermediate_data but no tool calls."""
  intermediate_data = IntermediateData(tool_uses=[], tool_responses=[])
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      intermediate_data=intermediate_data,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "No intermediate steps were taken." in prompt
