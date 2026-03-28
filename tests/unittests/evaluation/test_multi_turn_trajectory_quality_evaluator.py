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

"""Tests for the Multi Turn Trajectory Quality Evaluator."""

from google.adk.dependencies.vertexai import vertexai
from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.multi_turn_trajectory_quality_evaluator import MultiTurnTrajectoryQualityV1Evaluator
from google.genai import types as genai_types

vertexai_types = vertexai.types


class TestMultiTurnTrajectoryQualityV1Evaluator:
  """A class to help organize "patch" that are applicable to all tests."""

  def test_evaluate_invocations_metric_passed(self, mocker):
    """Test evaluate_invocations function for multi-turn trajectory quality metric."""
    mock_perform_eval = mocker.patch(
        "google.adk.evaluation.vertex_ai_eval_facade._VertexAiEvalFacade._perform_eval"
    )
    actual_invocations = [
        Invocation(
            invocation_id="inv1",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q1")]
            ),
            intermediate_data=InvocationEvents(invocation_events=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r1")]
            ),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    )
                }
            ),
        ),
        Invocation(
            invocation_id="inv2",
            user_content=genai_types.Content(
                parts=[genai_types.Part(text="q2")]
            ),
            intermediate_data=InvocationEvents(
                invocation_events=[
                    InvocationEvent(
                        author="agent1",
                        content=genai_types.Content(
                            parts=[genai_types.Part(text="intermediate")]
                        ),
                    )
                ]
            ),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="r2")]
            ),
            app_details=AppDetails(
                agent_details={
                    "agent1": AgentDetails(
                        name="agent1", instructions="instructions1"
                    )
                }
            ),
        ),
    ]
    evaluator = MultiTurnTrajectoryQualityV1Evaluator(
        eval_metric=EvalMetric(
            threshold=0.8, metric_name="multi_turn_trajectory_quality"
        )
    )
    # Mock the return value of _perform_eval
    mock_perform_eval.return_value = vertexai_types.EvaluationResult(
        summary_metrics=[vertexai_types.AggregatedMetricResult(mean_score=0.9)],
        eval_case_results=[],
    )

    evaluation_result = evaluator.evaluate_invocations(
        actual_invocations,
    )

    assert evaluation_result.overall_score == 0.9
    assert evaluation_result.overall_eval_status == EvalStatus.PASSED
    mock_perform_eval.assert_called_once()
    _, mock_kwargs = mock_perform_eval.call_args
    # Compare the names of the metrics.
    assert [m.name for m in mock_kwargs["metrics"]] == [
        vertexai_types.RubricMetric.MULTI_TURN_TRAJECTORY_QUALITY.name
    ]
